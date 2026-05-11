//! Benchmark på ekstra (ikke-kart) grafer:
//!   * RMAT (Graph500-stil syntetisk)
//!   * Word ladder (engelsk ordliste)
//!   * SNAP soc-LiveJournal (sosialt nettverk)
//!   * Rubik's pocket cube (2x2x2)
//!
//! Hver workload kjører alle 5 SSSP-algoritmer fra én fast kilde,
//! verifiserer mot binary-Dijkstra, rapporterer tider.

use std::time::Instant;

use sssp_bench::auto::sssp_auto;
use sssp_bench::delta_step::delta_stepping;
use sssp_bench::dijkstra::{dijkstra_4ary, dijkstra_binary, INF};
use sssp_bench::duan::duan_inspired;
use sssp_bench::graph::CsrGraph;
use sssp_bench::rubik::{build_pocket_cube_graph, SOLVED};
use sssp_bench::snap::load_snap_edge_list;
use sssp_bench::synth::gen_rmat;
use sssp_bench::wordladder::{build_word_ladder, pick_word};

fn count_bad(reference: &[f32], other: &[f32], eps: f32) -> usize {
    let mut bad = 0;
    for i in 0..reference.len() {
        let a = reference[i];
        let b = other[i];
        let ok = if a == INF || b == INF {
            a == b
        } else {
            (a - b).abs() <= eps * (1.0 + a.abs())
        };
        if !ok {
            bad += 1;
        }
    }
    bad
}

fn estimate_avg_weight(g: &CsrGraph) -> f32 {
    if g.edge_w.is_empty() {
        return 1.0;
    }
    let stride = (g.edge_w.len() / 4096).max(1);
    let mut s = 0.0f64;
    let mut c = 0u64;
    let mut i = 0;
    while i < g.edge_w.len() {
        s += g.edge_w[i] as f64;
        c += 1;
        i += stride;
    }
    (s / c as f64) as f32
}

fn run_workload(label: &str, g: &CsrGraph, src: u32) {
    let n = g.n;
    let m = g.m();
    let avg_deg = m as f32 / n.max(1) as f32;
    let avg_w = estimate_avg_weight(g);
    let delta = (avg_w / avg_deg).max(1e-4);
    let bw = 4.0 * delta;

    println!();
    println!("=== {label} ===");
    println!("  n = {n}, m = {m}, avg_deg = {avg_deg:.2}, avg_w = {avg_w:.3}");
    println!("  delta = {delta:.4}, bw = {bw:.4}, src = {src}");

    let mut times: [f64; 5] = [0.0; 5];
    let names = [
        "dijkstra_binary",
        "dijkstra_4ary",
        "delta_stepping",
        "duan_inspired",
        "sssp_auto",
    ];

    let t = Instant::now();
    let r = dijkstra_binary(g, src);
    times[0] = t.elapsed().as_secs_f64() * 1000.0;
    let reachable = r.iter().filter(|&&d| d.is_finite()).count();

    let t = Instant::now();
    let a = dijkstra_4ary(g, src);
    times[1] = t.elapsed().as_secs_f64() * 1000.0;
    let bad_4ary = count_bad(&r, &a, 1e-3);

    let t = Instant::now();
    let s = delta_stepping(g, src, delta);
    times[2] = t.elapsed().as_secs_f64() * 1000.0;
    let bad_dstep = count_bad(&r, &s, 1e-3);

    let t = Instant::now();
    let d = duan_inspired(g, src, bw);
    times[3] = t.elapsed().as_secs_f64() * 1000.0;
    let bad_duan = count_bad(&r, &d, 1e-3);

    let t = Instant::now();
    let au = sssp_auto(g, src);
    times[4] = t.elapsed().as_secs_f64() * 1000.0;
    let bad_auto = count_bad(&r, &au, 1e-3);

    println!("  reachable from src: {reachable}/{n}");
    for i in 0..5 {
        let speedup = times[0] / times[i].max(0.001);
        let bad = match i {
            1 => bad_4ary,
            2 => bad_dstep,
            3 => bad_duan,
            4 => bad_auto,
            _ => 0,
        };
        let mark = if bad == 0 { "OK" } else { "FAIL" };
        println!(
            "  {:<22} {:>8.2} ms  ({:>4.2}x)  {} {}",
            names[i],
            times[i],
            speedup,
            mark,
            if bad > 0 {
                format!("({bad} mismatches)")
            } else {
                String::new()
            }
        );
    }
}

fn run_rmat() {
    println!("\n========== RMAT (Graph500) ==========");
    for &(scale, ef) in &[(18u32, 16usize), (20, 16), (22, 16)] {
        let t = Instant::now();
        let g = gen_rmat(scale, ef, 42);
        println!(
            "[rmat] scale={scale} edge_factor={ef} -> n={} m={} ({:.2} s å generere)",
            g.n,
            g.m(),
            t.elapsed().as_secs_f64()
        );
        // Pick a random reasonably-connected source.
        let mut src = 0u32;
        for u in 0..g.n {
            if g.head[u + 1] - g.head[u] > 10 {
                src = u as u32;
                break;
            }
        }
        run_workload(&format!("RMAT scale={scale}"), &g, src);
    }
}

fn run_word_ladder() {
    println!("\n========== Word ladder ==========");
    let path = "/usr/share/dict/words";
    if !std::path::Path::new(path).exists() {
        println!("Hopper over: {path} finnes ikke");
        return;
    }
    let t = Instant::now();
    let (g, words) = match build_word_ladder(path, 4, 8) {
        Ok(x) => x,
        Err(e) => {
            println!("Feil: {e}");
            return;
        }
    };
    println!(
        "[ladder] bygget på {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let src = pick_word(&words, "cat")
        .or_else(|| pick_word(&words, "love"))
        .or_else(|| pick_word(&words, "rust"))
        .map(|(i, _w)| i)
        .unwrap_or(0);
    let src_word = words.get(src as usize).cloned().unwrap_or_default();
    println!("  starter fra ord: \"{}\" (idx={src})", src_word);
    run_workload("Word ladder (4-8 bokstaver)", &g, src);
}

fn run_snap() {
    println!("\n========== SNAP soc-LiveJournal1 ==========");
    let path = "data/soc-LiveJournal1.txt.gz";
    if !std::path::Path::new(path).exists() {
        println!("Hopper over: {path} finnes ikke. Last ned med:");
        println!("  curl -L -o data/soc-LiveJournal1.txt.gz \\");
        println!("    https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz");
        return;
    }
    let t = Instant::now();
    let (g, nonempty) = match load_snap_edge_list(path) {
        Ok(x) => x,
        Err(e) => {
            println!("Feil: {e}");
            return;
        }
    };
    println!(
        "[snap] bygget på {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let src = nonempty.first().copied().unwrap_or(0);
    run_workload("SNAP soc-LiveJournal", &g, src);
}

fn run_rubik() {
    println!("\n========== Rubik's Pocket Cube (2x2x2) ==========");
    let t = Instant::now();
    let (g, _depth) = build_pocket_cube_graph();
    println!(
        "[rubik] bygget på {:.2} s",
        t.elapsed().as_secs_f64()
    );
    // SOLVED er ID 0.
    let _ = SOLVED;
    run_workload("Rubik 2x2x2 fra SOLVED", &g, 0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let only = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match only {
        "rmat" => run_rmat(),
        "ladder" => run_word_ladder(),
        "snap" => run_snap(),
        "rubik" => run_rubik(),
        _ => {
            run_rmat();
            run_word_ladder();
            run_rubik();
            run_snap();
        }
    }
}
