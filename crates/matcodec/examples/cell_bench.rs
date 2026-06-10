//! Random-access benchmark for the MTZT gateway index.
//!
//!   cargo run --release -p matcodec --example cell_bench [n]
//!
//! Builds two worlds — a gateway-structured one (regions joined pairwise by 3
//! distinct roads, the case the index targets) and a smooth Euclidean one
//! (worst case: every row carries residuals) — stream-compresses them, then
//! measures random `cell(i,j)` lookups with the O(L) index fast path on vs
//! off (off = the legacy behaviour: every cold lookup inflates a frame). Also
//! reports the O(L) `cell_bounds` probe rate and the resident index memory.

use matcodec::{compress_stream, pick_landmarks, MtzReader, SliceRows};
use std::time::Instant;

/// Exact integer gateway world (L1 metric, no rounding noise): road k joins
/// gateway `gw[ra][k] ↔ gw[rb][k]`, so cross-region distances are min-plus
/// exact through gateway points that exist in the matrix.
fn gateway_world(n: usize, regions: usize) -> (Vec<i32>, usize) {
    let per = n / regions;
    let n = per * regions;
    let mut s: u64 = 0xC0FFEE;
    let mut rnd = |range: i64| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as i64) % range
    };
    let mut pts = vec![(0i64, 0i64); n];
    for r in 0..regions {
        let c = ((r % 4) as i64 * 100_000, (r / 4) as i64 * 100_000);
        for i in 0..per {
            pts[r * per + i] = (c.0 + rnd(4000), c.1 + rnd(4000));
        }
    }
    let l1 = |a: (i64, i64), b: (i64, i64)| (a.0 - b.0).abs() + (a.1 - b.1).abs();
    let gw = |r: usize, k: usize| r * per + k;
    let road = 30_000i64;
    let mut d = vec![0i32; n * n];
    for i in 0..n {
        for j in 0..n {
            let (ri, rj) = (i / per, j / per);
            let v = if ri == rj {
                l1(pts[i], pts[j])
            } else {
                (0..3)
                    .map(|k| l1(pts[i], pts[gw(ri, k)]) + road + l1(pts[gw(rj, k)], pts[j]))
                    .min()
                    .unwrap()
            };
            d[i * n + j] = v as i32;
        }
    }
    (d, n)
}

fn euclid_world(n: usize) -> Vec<i32> {
    let mut s: u64 = 42;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as f64 / (1u64 << 31) as f64 * 100_000.0
    };
    let pts: Vec<(f64, f64)> = (0..n).map(|_| (rnd(), rnd())).collect();
    let mut d = vec![0i32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = (((pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2)).sqrt())
                .round() as i32;
        }
    }
    d
}

fn bench(name: &str, d: &[i32], n: usize, l: usize, probes: usize) {
    let t = Instant::now();
    let lm = pick_landmarks(d, n, l);
    let pick_t = t.elapsed();
    let mut src = SliceRows { d, n };
    let mut blob = Vec::new();
    compress_stream(&mut src, &lm, &mut blob).expect("compress");
    let raw = n * n * 4;
    let resident = 2 * l * n * 4 + l * n + 2 * n; // dlj + dil + blockmax + cell_of + rowmax
    println!("\n== {name}  n={n} L={l} ==");
    println!(
        "  size: raw {:.1} MB -> blob {:.2} MB ({:.2}x)   resident index {:.2} MB   landmark pick {:.1}s",
        raw as f64 / 1e6,
        blob.len() as f64 / 1e6,
        raw as f64 / blob.len() as f64,
        resident as f64 / 1e6,
        pick_t.as_secs_f64()
    );

    // pseudo-random probe sequence, identical for both passes
    let mk_probes = || {
        let mut s: u64 = 1234567;
        (0..probes)
            .map(move |_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let i = ((s >> 33) as usize) % n;
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = ((s >> 33) as usize) % n;
                (i, j)
            })
            .collect::<Vec<_>>()
    };
    let ps = mk_probes();

    let mut rd = MtzReader::open(blob.clone(), 64).expect("open");
    println!(
        "  index-exact: {:.0}% of blocks, {} of {} rows   blocks within tol 2/5/15/60s: {:.0}%/{:.0}%/{:.0}%/{:.0}%",
        100.0 * rd.exact_index_block_share(),
        rd.exact_index_rows(),
        n,
        100.0 * rd.index_share_within(2),
        100.0 * rd.index_share_within(5),
        100.0 * rd.index_share_within(15),
        100.0 * rd.index_share_within(60),
    );

    let mut sink = 0i64;
    let t = Instant::now();
    for &(i, j) in &ps {
        sink += rd.cell(i, j).expect("cell") as i64;
    }
    let fast = t.elapsed();

    let mut rd2 = MtzReader::open(blob.clone(), 64).expect("open");
    rd2.set_index_fast_path(false);
    let t = Instant::now();
    for &(i, j) in &ps {
        sink -= rd2.cell(i, j).expect("cell") as i64;
    }
    let slow = t.elapsed();
    assert_eq!(sink, 0, "fast/frame paths disagree");

    let mut bsink = 0i64;
    let mut exact = 0usize;
    let t = Instant::now();
    for &(i, j) in &ps {
        let (lo, up) = rd.cell_bounds(i, j);
        bsink += (lo as i64) ^ (up as i64);
        exact += (lo == up) as usize;
    }
    let bounds = t.elapsed();
    std::hint::black_box(bsink);

    let per = |d: std::time::Duration| d.as_nanos() as f64 / probes as f64;
    println!(
        "  cell() random  index on: {:>8.0} ns/op   index off (legacy): {:>8.0} ns/op   speedup {:.1}x",
        per(fast),
        per(slow),
        per(slow) / per(fast)
    );
    println!(
        "  cell_bounds()  {:>8.0} ns/op (O(L), no decompression), exact on {:.0}% of probes",
        per(bounds),
        100.0 * exact as f64 / probes as f64
    );

    // tolerance path: ≤5s overestimate allowed (VRP local-search probing)
    let mut rd3 = MtzReader::open(blob, 64).expect("open");
    let mut wsink = 0i64;
    let t = Instant::now();
    for &(i, j) in &ps {
        wsink += rd3.cell_within(i, j, 5).expect("cell_within") as i64;
    }
    let within = t.elapsed();
    std::hint::black_box(wsink);
    println!("  cell_within(5) {:>8.0} ns/op   speedup vs legacy {:.0}x", per(within), per(slow) / per(within));
}

/// `{"durations": [[...]]}` (as written by stream_compress --dump) or a bare
/// JSON array — parsed crudely to avoid a serde dependency in the example.
fn load_json_matrix(path: &str) -> (Vec<i32>, usize) {
    let txt = std::fs::read_to_string(path).expect("read matrix json");
    let body = txt.trim().trim_start_matches("{\"durations\":").trim_end_matches('}');
    let mut rows: Vec<Vec<i32>> = Vec::new();
    for row in body.split("],") {
        let cells: Vec<i32> = row
            .matches(|c: char| c.is_ascii_digit() || c == '-' || c == ',')
            .collect::<String>()
            .split(',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().expect("cell"))
            .collect();
        if !cells.is_empty() {
            rows.push(cells);
        }
    }
    let n = rows.len();
    let mut d = vec![0i32; n * n];
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r.len(), n, "matrix not square at row {i}");
        d[i * n..i * n + n].copy_from_slice(r);
    }
    (d, n)
}

fn main() {
    let arg = std::env::args().nth(1);
    let l: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(32);
    let probes = 200_000;
    if let Some(path) = arg.as_deref().filter(|a| a.ends_with(".json")) {
        let (d, n) = load_json_matrix(path);
        bench(&format!("real matrix {path}"), &d, n, l, probes);
        return;
    }
    let n: usize = arg.and_then(|a| a.parse().ok()).unwrap_or(3000);
    let (gd, gn) = gateway_world(n, 8);
    bench("gateway world (8 regions x 3 roads)", &gd, gn, l, probes);
    let ed = euclid_world(n);
    bench("smooth euclidean (worst case)", &ed, n, l, probes);
}
