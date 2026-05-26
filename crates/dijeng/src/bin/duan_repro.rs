// Test delta_stepping AND duan_inspired on multiple delta values on the London graph.
// If we can reproduce FAIL with a specific delta, we can minimize.

use dijeng::delta_step::delta_stepping;
use dijeng::dijeng::{dijeng_binary, INF};
use dijeng::duan::duan_inspired;
use dijeng::osm::load_with_cache;
use dijeng::osm_profile::Profile;

fn count_bad(reference: &[f32], other: &[f32]) -> (usize, Option<(usize, f32, f32)>) {
    let mut bad = 0usize;
    let mut first = None;
    for i in 0..reference.len() {
        let a = reference[i];
        let b = other[i];
        let ok = if a == INF || b == INF {
            a == b
        } else {
            (a - b).abs() <= 1e-3 * (1.0 + a.abs())
        };
        if !ok {
            if first.is_none() {
                first = Some((i, a, b));
            }
            bad += 1;
        }
    }
    (bad, first)
}

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/greater-london.osm.pbf".to_string());
    let cache_path = format!("{}.csr", &path);
    let (g, _coords, _edge_dist) = load_with_cache(&path, &cache_path, Profile::Car)?;
    let src: u32 = 401720;

    let r = dijeng_binary(&g, src);

    // Sweep delta over a wide range.
    let deltas = [
        0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0,
    ];
    println!("delta_stepping sweep (source={src}):");
    for &d in &deltas {
        let s = delta_stepping(&g, src, d);
        let (bad, first) = count_bad(&r, &s);
        if bad == 0 {
            println!("  delta={d:>7.2} : OK");
        } else {
            let (i, a, b) = first.unwrap();
            println!("  delta={d:>7.2} : FAIL {bad} (v={i} ref={a} got={b})");
        }
    }
    println!("\nduan_inspired sweep:");
    for &d in &deltas {
        let bw = 4.0 * d;
        let s = duan_inspired(&g, src, bw);
        let (bad, first) = count_bad(&r, &s);
        if bad == 0 {
            println!("  bw={bw:>7.2} : OK");
        } else {
            let (i, a, b) = first.unwrap();
            println!("  bw={bw:>7.2} : FAIL {bad} (v={i} ref={a} got={b})");
        }
    }
    Ok(())
}
