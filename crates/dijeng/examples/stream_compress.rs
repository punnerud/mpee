//! Stream-compress a CH-derived travel-time matrix WITHOUT ever materialising
//! the full n² matrix.
//!
//! Each matrix row is produced on demand as one CH one-to-many query
//! (`ch::matrix` with a single source → all targets); `matcodec` pulls rows
//! through the [`matcodec::RowSource`] trait and writes the streamed `MTZT`
//! container (resident gateway index + residual-only frames). Peak memory is
//! the L resident landmark rows + the n×L gateway index + one working row —
//! independent of n². This is the "rest of the world" path: a matrix far
//! larger than RAM can be compressed by computing it block/row at a time from
//! the contraction hierarchy.
//!
//!   cargo run -p dijeng --example stream_compress -- graph.ch nodes.txt out.mtz [--landmarks L]
//!
//! `nodes.txt` is a whitespace-separated list of CH-internal node ids. To go
//! from coordinates, snap each with `RoutingService::nearest_node(lat, lon)`
//! then map through `ch.perm[csr_id]`.
//!
//! Decode the result with `matcodec decompress out.mtz back.json`, or random-
//! access it in RAM with `matcodec::MtzReader`.

use dijeng::ch::{self, ContractionHierarchy};
use matcodec::RowSource as _;
use std::fs;
use std::io::BufWriter;

/// Pulls one matrix row per CH one-to-many query — the full matrix is never held.
struct ChRowSource {
    ch: ContractionHierarchy,
    ids: Vec<u32>,
}

impl matcodec::RowSource for ChRowSource {
    fn n(&self) -> usize {
        self.ids.len()
    }
    fn row(&mut self, i: usize) -> Vec<i32> {
        // single source -> all targets = exactly one matrix row
        let durs = ch::matrix(&self.ch, &[self.ids[i]], &self.ids);
        durs.iter()
            .map(|&s| {
                if s.is_finite() {
                    s.round() as i32
                } else {
                    matcodec::UNREACHABLE
                }
            })
            .collect()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: stream_compress <graph.ch> <nodes.txt> <out.mtz> [--landmarks L]");
        std::process::exit(2);
    }
    let ch = dijeng::cache_ch::load_mmap(&args[1]).expect("load .ch");
    // nodes arg: a path to a whitespace-separated id list, OR "random:K" to pick
    // K distinct random CH-internal node ids from the loaded hierarchy.
    let ids: Vec<u32> = if let Some(k) = args[2].strip_prefix("random:") {
        let k: usize = k.parse().expect("random:K");
        let count = ch.perm.len();
        assert!(count >= 2, "graph too small");
        let mut s: u64 = 0x1234_5678_9abc_def1;
        let mut seen = std::collections::HashSet::new();
        let mut v = Vec::with_capacity(k);
        while v.len() < k.min(count) {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let id = ((s >> 33) as usize % count) as u32;
            if seen.insert(id) {
                v.push(id);
            }
        }
        v
    } else {
        fs::read_to_string(&args[2])
            .expect("read nodes.txt")
            .split_whitespace()
            .map(|t| t.parse::<u32>().expect("node id must be u32"))
            .collect()
    };
    let n = ids.len();
    assert!(n >= 2, "need at least 2 nodes");

    // Landmarks for the bridge model: evenly spread row indices. (No full matrix
    // exists to do farthest-point sampling, so we spread across the point set —
    // good enough; the residual is exact either way, only the ratio shifts.)
    let mut l = 32usize;
    if let Some(p) = args.iter().position(|a| a == "--landmarks") {
        if let Some(v) = args.get(p + 1) {
            l = v.parse().unwrap_or(32);
        }
    }
    l = l.min(n - 1).max(1);
    let lm: Vec<usize> = (0..l).map(|k| (k * n) / l).collect();

    let mut src = ChRowSource { ch, ids };
    let dump = args
        .iter()
        .position(|a| a == "--dump")
        .and_then(|p| args.get(p + 1))
        .cloned();

    let rep = if let Some(dp) = dump {
        // Benchmark/comparison mode: materialise the matrix once, write it as
        // JSON (so `matcodec roundtrip`/`validate` can run the best-of model on
        // the same real CH data), then stream-compress from the buffer.
        let mut full = vec![0i32; n * n];
        for i in 0..n {
            let r = src.row(i);
            full[i * n..i * n + n].copy_from_slice(&r);
        }
        let mut s = String::from("{\"durations\":[");
        for i in 0..n {
            if i > 0 {
                s.push(',');
            }
            s.push('[');
            for j in 0..n {
                if j > 0 {
                    s.push(',');
                }
                s.push_str(&full[i * n + j].to_string());
            }
            s.push(']');
        }
        s.push_str("]}");
        fs::write(&dp, s).expect("write dump");
        eprintln!("dumped full matrix to {dp} (comparison only; streaming itself never materialises)");
        let mut sbuf = matcodec::SliceRows { d: &full, n };
        let mut out = BufWriter::new(fs::File::create(&args[3]).expect("create out.mtz"));
        matcodec::compress_stream(&mut sbuf, &lm, &mut out).expect("compress_stream")
    } else {
        // Pure streaming: rows pulled on demand from the CH, n² never held.
        let mut out = BufWriter::new(fs::File::create(&args[3]).expect("create out.mtz"));
        matcodec::compress_stream(&mut src, &lm, &mut out).expect("compress_stream")
    };

    let sz = fs::metadata(&args[3]).map(|m| m.len()).unwrap_or(0);
    let raw = (n as u64) * (n as u64) * 4;
    println!(
        "stream-compressed {n}x{n} CH road matrix (never materialised): {} -> {} bytes ({:.2}x), L={l}",
        raw,
        sz,
        raw as f64 / sz.max(1) as f64
    );
    for w in rep.warnings() {
        eprintln!("{w}");
    }
    if !rep.metric_ok() {
        eprintln!(
            "  note: flagged non-metric/asymmetric — normal for road nets (one-way streets)."
        );
    }
}
