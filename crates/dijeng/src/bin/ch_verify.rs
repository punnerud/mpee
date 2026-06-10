//! Ground-truth verification of the CH cache against bidirectional Dijkstra
//! on the base graph: random (s, d) pairs, distances must agree to ε = 1e-3
//! relative (CH weights are f32 sums in a different association order).
//!
//! Usage: ch_verify [london|england] [n_pairs]
//!        ch_verify [london|england] matrix [n]   — verify the bucket-MMM:
//!          an n×n matrix WITH stall-on-demand vs WITHOUT (elementwise), plus
//!          a sample of cells against the p2p query (itself Dijkstra-verified).

use std::time::Instant;

use dijeng::bidir::bidir_dijeng;
use dijeng::cache_ch;
use dijeng::cache_pp;
use dijeng::ch::{self, PathScratch};
use dijeng::graph::Rng;

fn main() -> std::io::Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "london".to_string());
    let n_pairs: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "2000".to_string())
        .parse()
        .unwrap_or(2000);
    let (pp_path, ch_path) = match dataset.as_str() {
        "london" => ("data/greater-london.osm.pbf.pp", "data/greater-london.osm.pbf.ch"),
        "england" => ("data/england.osm.pbf.pp", "data/england.osm.pbf.ch"),
        other => {
            eprintln!("unknown dataset {other}");
            std::process::exit(1);
        }
    };
    let pp = cache_pp::load_mmap(pp_path).expect("pp cache missing");
    let h = cache_ch::load_mmap(ch_path).expect("ch cache missing");
    let g = &pp.graph;

    if std::env::args().nth(2).as_deref() == Some("matrix") {
        let n_pts: usize = std::env::args()
            .nth(3)
            .unwrap_or_else(|| "500".to_string())
            .parse()
            .unwrap_or(500);
        return verify_matrix(&pp, &h, n_pts);
    }
    println!("[ch_verify] {dataset}: n={} pairs={n_pairs}", g.n);

    let mut rng = Rng(0x5EED_2026);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n_pairs);
    while pairs.len() < n_pairs {
        let s = rng.range(g.n as u32);
        let d = rng.range(g.n as u32);
        if g.head[s as usize + 1] - g.head[s as usize] == 0 {
            continue;
        }
        if g.head[d as usize + 1] - g.head[d as usize] == 0 {
            continue;
        }
        pairs.push((s, d));
    }

    let mut scratch = PathScratch::new(h.graph_fwd.n);
    let mut bad = 0usize;
    let mut checked = 0usize;
    let t = Instant::now();
    for (i, &(s, d)) in pairs.iter().enumerate() {
        let r = bidir_dijeng(g, &pp.reverse, s, d).unwrap_or(f32::INFINITY);
        let q = ch::query_dist_into(&h, h.perm[s as usize], h.perm[d as usize], &mut scratch)
            .unwrap_or(f32::INFINITY);
        let ok = if r.is_infinite() || q.is_infinite() {
            r == q
        } else {
            (r - q).abs() <= 1e-3 * (1.0 + r.abs())
        };
        checked += 1;
        if !ok {
            bad += 1;
            if bad <= 10 {
                println!("  DIFF pair {i}: s={s} d={d}  dijkstra={r}  ch={q}");
            }
        }
    }
    println!(
        "[ch_verify] {checked} pairs in {:.1} s — {bad} mismatches{}",
        t.elapsed().as_secs_f64(),
        if bad == 0 { "  ✓ CH is exact" } else { "  ✗ CH BROKEN" }
    );
    std::process::exit(if bad == 0 { 0 } else { 1 });
}

fn verify_matrix(
    pp: &dijeng::cache_pp::PpFull,
    h: &dijeng::ch::ContractionHierarchy,
    n_pts: usize,
) -> std::io::Result<()> {
    let g = &pp.graph;
    let mut rng = Rng(0x4D41_5452);
    let mut pts: Vec<u32> = Vec::with_capacity(n_pts);
    while pts.len() < n_pts {
        let v = rng.range(g.n as u32);
        if g.head[v as usize + 1] - g.head[v as usize] > 0 {
            pts.push(h.perm[v as usize]);
        }
    }
    println!("[ch_verify/matrix] {n_pts}×{n_pts} cells, stall vs no-stall + p2p anchor");

    let t = Instant::now();
    let was = ch::set_stall(true);
    let (dur_s, dist_s) = ch::matrix_with_dist(h, &pts, &pts);
    let t_stall = t.elapsed().as_secs_f64();
    let t = Instant::now();
    ch::set_stall(false);
    let (dur_n, dist_n) = ch::matrix_with_dist(h, &pts, &pts);
    let t_nostall = t.elapsed().as_secs_f64();
    ch::set_stall(was);

    let mut bad = 0usize;
    let mut max_rel = 0.0f64;
    for i in 0..dur_s.len() {
        for (a, b) in [(dur_s[i], dur_n[i]), (dist_s[i], dist_n[i])] {
            let ok = if a.is_infinite() || b.is_infinite() {
                a == b
            } else {
                (a - b).abs() <= 1e-3 * (1.0 + b.abs())
            };
            if !ok {
                bad += 1;
                if bad <= 5 {
                    println!("  DIFF cell {i}: stall={a} nostall={b}");
                }
            }
            if a.is_finite() && b.is_finite() {
                max_rel = max_rel.max(((a - b).abs() / (1.0 + b.abs())) as f64);
            }
        }
    }

    // Anchor a sample of cells against the (Dijkstra-verified) p2p query.
    let mut scratch = ch::PathScratch::new(h.graph_fwd.n);
    let mut anchor_bad = 0usize;
    let sample = 300.min(n_pts);
    for i in 0..sample {
        let j = (i * 7919) % n_pts; // spread
        let q = ch::query_dist_into(h, pts[i], pts[j], &mut scratch).unwrap_or(f32::INFINITY);
        let m = dur_s[i * n_pts + j];
        let ok = if q.is_infinite() || m.is_infinite() {
            q == m
        } else {
            (q - m).abs() <= 1e-3 * (1.0 + q.abs())
        };
        if !ok {
            anchor_bad += 1;
            if anchor_bad <= 5 {
                println!("  ANCHOR DIFF ({i},{j}): p2p={q} matrix={m}");
            }
        }
    }

    println!(
        "[ch_verify/matrix] stall {t_stall:.2}s vs no-stall {t_nostall:.2}s ({:.2}x) — \
         {bad} elementwise mismatches (max rel {max_rel:.2e}), {anchor_bad} p2p-anchor mismatches{}",
        t_nostall / t_stall.max(1e-9),
        if bad == 0 && anchor_bad == 0 { "  ✓" } else { "  ✗" }
    );
    std::process::exit(if bad == 0 && anchor_bad == 0 { 0 } else { 1 });
}
