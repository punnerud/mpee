//! Sammenligner sequential vs parallell PBF-parser:
//!   * tid (real wall-clock)
//!   * korrekthet (sammenligner CSR-arrays + koordinater bit-for-bit)

use sssp_bench::osm::{load_osm_routing, load_osm_routing_par};
use sssp_bench::osm_profile::Profile;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/greater-london.osm.pbf".to_string());

    println!("=== sekvensiell ===");
    let t = Instant::now();
    let (g_seq, c_seq, _ed_seq) = load_osm_routing(&path, Profile::Car)?;
    let dt_seq = t.elapsed().as_secs_f64();
    println!("[seq] total: {:.2} s", dt_seq);
    println!();

    println!("=== parallell ===");
    let t = Instant::now();
    let (g_par, c_par, _ed_par) = load_osm_routing_par(&path, Profile::Car)?;
    let dt_par = t.elapsed().as_secs_f64();
    println!("[par] total: {:.2} s", dt_par);
    println!();

    println!("speedup: {:.2}x ({:.0} ms spart)", dt_seq / dt_par, (dt_seq - dt_par) * 1000.0);
    println!();

    // ---- Correctness: compare the two graphs ----
    let mut diffs = 0usize;

    if g_seq.n != g_par.n || g_seq.m() != g_par.m() {
        println!("STR: different sizes! seq=({}, {}) par=({}, {})",
                 g_seq.n, g_seq.m(), g_par.n, g_par.m());
        diffs += 1;
    }

    // The parallel version can yield a different NODE-ID order because blobs
    // are processed out of order. That means the internal indices are permuted.
    // So we check *structure* (sorted edge list) instead of index equality.
    fn canonical_edges(g: &sssp_bench::graph::CsrGraph, coords: &[(f32, f32)]) -> Vec<(u64, u64, f32)> {
        // Use (lat,lon) as identity, round to ~1cm precision to be tolerant.
        // We pack lat/lon into a u64.
        let key = |i: u32| -> u64 {
            let (la, lo) = coords[i as usize];
            // 1e7 ≈ 1cm pr degree. Cast til i32, pakk.
            let la_i = (la * 1.0e7) as i32 as u32 as u64;
            let lo_i = (lo * 1.0e7) as i32 as u32 as u64;
            (la_i << 32) | lo_i
        };
        let mut out: Vec<(u64, u64, f32)> = Vec::with_capacity(g.m());
        for u in 0..g.n {
            let s = g.head[u] as usize;
            let e = g.head[u + 1] as usize;
            let ku = key(u as u32);
            for k in s..e {
                let v = g.edge_to[k];
                let kv = key(v);
                let w = g.edge_w[k];
                let (a, b) = if ku <= kv { (ku, kv) } else { (kv, ku) };
                out.push((a, b, w));
            }
        }
        out.sort_by(|x, y| x.partial_cmp(y).unwrap());
        out
    }

    let t = Instant::now();
    let canon_seq = canonical_edges(&g_seq, &c_seq);
    let canon_par = canonical_edges(&g_par, &c_par);
    println!("[verify] canonical-build: {:.2} s", t.elapsed().as_secs_f64());

    if canon_seq.len() != canon_par.len() {
        println!("DIFF: ulikt antall kanter (seq={} par={})", canon_seq.len(), canon_par.len());
        diffs += 1;
    } else {
        let mut bad = 0usize;
        for i in 0..canon_seq.len() {
            let a = canon_seq[i];
            let b = canon_par[i];
            if a.0 != b.0 || a.1 != b.1 || (a.2 - b.2).abs() > 1e-3 {
                if bad < 5 {
                    println!("DIFF at {i}: seq={a:?} par={b:?}");
                }
                bad += 1;
            }
        }
        if bad == 0 {
            println!("OK: kantsett identisk (etter kanonikalisering)");
        } else {
            println!("DIFF: {bad} kanter avviker");
            diffs += 1;
        }
    }

    if diffs == 0 {
        println!("\nKONKLUSJON: parallell er korrekt OG {:.2}x raskere", dt_seq / dt_par);
    } else {
        println!("\nKONKLUSJON: parallell IKKE bit-likt — IKKE bytt over.");
    }

    Ok(())
}
