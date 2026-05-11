// Reproducer for delta_stepping path-bug.

use sssp_bench::delta_step::{delta_stepping, delta_stepping_debug, delta_stepping_v2};
use sssp_bench::dijkstra::{dijkstra_binary, INF};
use sssp_bench::graph::gen_path;

fn main() {
    // Sweep over n to find when bug starts.
    println!("=== Sweep n with delta=0.3 ===");
    for &n in &[1_000usize, 10_000, 50_000, 100_000, 500_000] {
        let g = gen_path(n, 11);
        let r = dijkstra_binary(&g, 0);
        let s = delta_stepping(&g, 0, 0.3);
        let mut bad = 0usize;
        let mut first = None;
        for i in 0..n {
            let a = r[i];
            let b = s[i];
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
        if bad == 0 {
            println!("  n={n}: OK");
        } else {
            let (i, a, b) = first.unwrap();
            println!("  n={n}: FAIL {bad}, første v={i} ref={a} got={b}");
        }
    }
    println!();

    let n = 500_000;
    let g = gen_path(n, 11);
    let r = dijkstra_binary(&g, 0);

    // Trace the watched vertex
    println!("=== TRACE v=143583 + predecessor 143582 on n=500k delta=0.3 ===");
    {
        let g = gen_path(500_000, 11);
        let _ = delta_stepping_debug(&g, 0, 0.3, 143582);
    }
    println!();

    println!("=== v2 sweep n with delta=0.3 ===");
    for &nv in &[100_000usize, 500_000] {
        let g = gen_path(nv, 11);
        let r = dijkstra_binary(&g, 0);
        let s = delta_stepping_v2(&g, 0, 0.3);
        let mut bad = 0usize;
        let mut first = None;
        for i in 0..nv {
            let a = r[i];
            let b = s[i];
            let ok = if a == INF || b == INF { a == b } else { (a - b).abs() <= 1e-3 * (1.0 + a.abs()) };
            if !ok { if first.is_none() { first = Some((i, a, b)); } bad += 1; }
        }
        if bad == 0 { println!("  v2 n={nv}: OK"); } else {
            let (i, a, b) = first.unwrap();
            println!("  v2 n={nv}: FAIL {bad}, første v={i} ref={a} got={b}");
        }
    }
    println!();

    println!("=== Sweep delta with n={n} ===");
    for &delta in &[0.05f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.506, 0.6, 1.0, 2.0] {
        let s = delta_stepping(&g, 0, delta);
        let mut bad = 0usize;
        let mut first = None;
        for i in 0..n {
            let a = r[i];
            let b = s[i];
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
        if bad == 0 {
            println!("delta={delta}: OK");
        } else {
            let (i, a, b) = first.unwrap();
            println!(
                "delta={delta}: FAIL {bad}, første v={i} ref={a} got={b}"
            );
        }
    }

    // Also check tiny path
    let g2 = gen_path(20, 11);
    let r2 = dijkstra_binary(&g2, 0);
    for &delta in &[0.1, 0.5, 1.0] {
        let s2 = delta_stepping(&g2, 0, delta);
        println!(
            "small path n=20 delta={delta}: ref={:?} got={:?}",
            &r2[..6],
            &s2[..6]
        );
    }
}
