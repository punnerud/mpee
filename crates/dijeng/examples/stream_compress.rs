//! Stream-compress a CH-derived travel-time matrix WITHOUT ever materialising
//! the full n² matrix.
//!
//! Each matrix row is produced on demand as one CH one-to-many query
//! (`ch::matrix` with a single source → all targets); `matcodec` pulls rows
//! through the [`matcodec::RowSource`] trait and writes the streamed `MTZU`
//! container (directional path labels + varint residual frames). Peak memory
//! is the candidate rows used for hub mining + the labels + one working row —
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

    // Path-label model (MTZU): hubs are mined from candidate rows fetched via
    // RowSource random access, so no full matrix is ever needed. --landmarks L
    // maps to the hub count for backwards compatibility.
    let mut opts = matcodec::HubOpts::default();
    if let Some(p) = args.iter().position(|a| a == "--landmarks") {
        if let Some(v) = args.get(p + 1) {
            opts.hubs = v.parse().unwrap_or(opts.hubs);
        }
    }

    // --graph-hubs C: use the C most important road-network nodes (top of the
    // contraction order = motorway/trunk junctions) as hub candidates instead
    // of matrix points. Their distance tables come from three chunked CH
    // matrix calls; the MTZU format itself is unchanged.
    let graph_hubs: Option<usize> = args
        .iter()
        .position(|a| a == "--graph-hubs")
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse().ok());

    let mut src = ChRowSource { ch, ids };
    let dump = args
        .iter()
        .position(|a| a == "--dump")
        .and_then(|p| args.get(p + 1))
        .cloned();

    let to_i32 = |v: &[f32]| -> Vec<i32> {
        v.iter()
            .map(|&s| if s.is_finite() { s.round() as i32 } else { matcodec::UNREACHABLE })
            .collect()
    };

    let rep = if let Some(c) = graph_hubs {
        let nn = src.ch.graph_fwd.n;
        let mut top: Vec<u32> = (0..nn as u32).collect();
        top.sort_by_key(|&v| std::cmp::Reverse(src.ch.rank[v as usize]));
        // Physical junctions are node *clusters* (roundabout rings, interchange
        // ramps, dual-carriageway crossings), so the raw rank top wastes
        // candidate slots on near-duplicates. Dedup by graph time: walk in
        // rank order, keep a node only if it is ≥ eps seconds from every kept
        // node (--hub-dedup-s, default 30; 0 disables).
        let dedup_s: f32 = args
            .iter()
            .position(|a| a == "--hub-dedup-s")
            .and_then(|p| args.get(p + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(30.0);
        if dedup_s > 0.0 {
            let pool: Vec<u32> = top.iter().copied().take((c * 6).min(nn)).collect();
            let dm = ch::matrix(&src.ch, &pool, &pool);
            let m = pool.len();
            let mut kept: Vec<usize> = Vec::with_capacity(c);
            for cand in 0..m {
                if kept.len() >= c {
                    break;
                }
                let dup = kept.iter().any(|&k| {
                    let a = dm[k * m + cand].min(dm[cand * m + k]);
                    a.is_finite() && a < dedup_s
                });
                if !dup {
                    kept.push(cand);
                }
            }
            eprintln!(
                "graph-hub candidates: {} distinct junctions (>= {dedup_s}s apart) from top {} ranked nodes",
                kept.len(),
                pool.len()
            );
            top = kept.into_iter().map(|k| pool[k]).collect();
        } else {
            top.truncate(c);
            eprintln!("graph-hub candidates: top {c} CH-rank junctions");
        }
        let rows = to_i32(&ch::matrix(&src.ch, &top, &src.ids));
        let cols = to_i32(&ch::matrix(&src.ch, &src.ids, &top));
        let hub_hub = to_i32(&ch::matrix(&src.ch, &top, &top));
        let ext = matcodec::ExternalHubs {
            ids: top.iter().map(|&v| v as usize).collect(),
            rows,
            cols,
            hub_hub,
        };
        let mut out = BufWriter::new(fs::File::create(&args[3]).expect("create out.mtz"));
        matcodec::compress_stream_hub_ext(&mut src, &opts, &ext, &mut out)
            .expect("compress_stream_hub_ext")
    } else if let Some(dp) = dump {
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
        matcodec::compress_stream_hub(&mut sbuf, &opts, &mut out).expect("compress_stream")
    } else {
        // Pure streaming: rows pulled on demand from the CH, n² never held.
        let mut out = BufWriter::new(fs::File::create(&args[3]).expect("create out.mtz"));
        matcodec::compress_stream_hub(&mut src, &opts, &mut out).expect("compress_stream")
    };

    let sz = fs::metadata(&args[3]).map(|m| m.len()).unwrap_or(0);
    let raw = (n as u64) * (n as u64) * 4;
    println!(
        "stream-compressed {n}x{n} CH road matrix (never materialised): {} -> {} bytes ({:.2}x), H={}",
        raw,
        sz,
        raw as f64 / sz.max(1) as f64,
        opts.hubs
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
