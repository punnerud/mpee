//! Ground-truth verification of the CH cache against bidirectional Dijkstra
//! on the base graph: random (s, d) pairs, distances must agree to ε = 1e-3
//! relative (CH weights are f32 sums in a different association order).
//!
//! Usage: ch_verify [london|england] [n_pairs]

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
