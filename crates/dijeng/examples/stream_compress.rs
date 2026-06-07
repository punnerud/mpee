//! Stream-compress a CH-derived travel-time matrix WITHOUT ever materialising
//! the full n² matrix.
//!
//! Each matrix row is produced on demand as one CH one-to-many query
//! (`ch::matrix` with a single source → all targets); `matcodec` pulls rows
//! through the [`matcodec::RowSource`] trait and writes the streamed `MTZS`
//! container. Peak memory is the L resident landmark rows + one working row —
//! independent of n. This is the "rest of the world" path: a matrix far larger
//! than RAM can be compressed by computing it block/row at a time from the
//! contraction hierarchy.
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
    let ids: Vec<u32> = fs::read_to_string(&args[2])
        .expect("read nodes.txt")
        .split_whitespace()
        .map(|t| t.parse::<u32>().expect("node id must be u32"))
        .collect();
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
    let mut out = BufWriter::new(fs::File::create(&args[3]).expect("create out.mtz"));
    let rep = matcodec::compress_stream(&mut src, &lm, &mut out).expect("compress_stream");
    drop(out);

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
