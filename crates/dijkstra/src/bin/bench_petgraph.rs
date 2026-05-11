//! Full comparison of OUR Dijkstra stack vs petgraph (the Rust ecosystem).
//! Covers all benchmark workloads used elsewhere in the project.

use ordered_float::OrderedFloat;
use petgraph::algo::dijkstra as pg_dijkstra;
use petgraph::graph::{DiGraph, NodeIndex};
use std::time::Instant;

use sssp_bench::auto::sssp_auto;
use sssp_bench::delta_step::delta_stepping;
use sssp_bench::dijkstra::{dijkstra_4ary, dijkstra_8ary, dijkstra_binary, INF};
use sssp_bench::duan::duan_inspired;
use sssp_bench::graph::{
    gen_grid, gen_path, gen_power_law, gen_random_sparse, CsrGraph,
};
use sssp_bench::osm::load_with_cache;
use sssp_bench::osm_profile::Profile;
use sssp_bench::rubik::build_pocket_cube_graph;
use sssp_bench::snap::load_snap_edge_list;
use sssp_bench::synth::gen_rmat;
use sssp_bench::wordladder::build_word_ladder;

fn build_petgraph(g: &CsrGraph) -> (DiGraph<(), OrderedFloat<f32>>, Vec<NodeIndex>) {
    let mut pg = DiGraph::<(), OrderedFloat<f32>>::with_capacity(g.n, g.m());
    let nodes: Vec<NodeIndex> = (0..g.n).map(|_| pg.add_node(())).collect();
    for u in 0..g.n {
        let s = g.head[u] as usize;
        let e = g.head[u + 1] as usize;
        for k in s..e {
            let v = g.edge_to[k] as usize;
            let w = g.edge_w[k];
            pg.add_edge(nodes[u], nodes[v], OrderedFloat(w));
        }
    }
    (pg, nodes)
}

fn pg_to_vec(
    map: &std::collections::HashMap<NodeIndex, OrderedFloat<f32>>,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![INF; n];
    for (&idx, &v) in map {
        out[idx.index()] = v.into_inner();
    }
    out
}

fn count_bad(reference: &[f32], other: &[f32]) -> usize {
    let mut bad = 0;
    for i in 0..reference.len() {
        let a = reference[i];
        let b = other[i];
        let ok = if a == INF || b == INF {
            a == b
        } else {
            (a - b).abs() <= 1e-3 * (1.0 + a.abs())
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

#[derive(Default, Clone, Copy)]
struct Row {
    petgraph: f64,
    binary: f64,
    quad: f64,
    oct: f64,
    dstep: f64,
    duan: f64,
    auto: f64,
    bad_pg: usize,
    bad_4ary: usize,
    bad_8ary: usize,
    bad_dstep: usize,
    bad_duan: usize,
    bad_auto: usize,
}

fn run(label: &str, g: &CsrGraph, src: u32, skip_pg: bool) -> Row {
    let n = g.n;
    let m = g.m();
    let avg_deg = m as f32 / n.max(1) as f32;
    let avg_w = estimate_avg_weight(g);
    let delta = (avg_w / avg_deg).max(1e-4);
    let bw = 4.0 * delta;

    println!();
    println!("=== {label} ===");
    println!(
        "  n = {n}, m = {m}, avg_deg = {avg_deg:.2}, avg_w = {avg_w:.3}, src = {src}"
    );

    // Reference (our binary)
    let t = Instant::now();
    let r = dijkstra_binary(g, src);
    let t_bin = t.elapsed().as_secs_f64() * 1000.0;
    let reachable = r.iter().filter(|&&d| d.is_finite()).count();
    println!("  our dijkstra_binary:   {:>9.2} ms   ({reachable}/{n} reach)", t_bin);

    let t = Instant::now();
    let r4 = dijkstra_4ary(g, src);
    let t_4 = t.elapsed().as_secs_f64() * 1000.0;
    let bad_4 = count_bad(&r, &r4);
    println!("  our dijkstra_4ary:     {:>9.2} ms   {}", t_4,
             if bad_4 == 0 { "OK" } else { "FAIL" });

    let t = Instant::now();
    let r8 = dijkstra_8ary(g, src);
    let t_8 = t.elapsed().as_secs_f64() * 1000.0;
    let bad_8 = count_bad(&r, &r8);
    println!("  our dijkstra_8ary:     {:>9.2} ms   {}", t_8,
             if bad_8 == 0 { "OK" } else { "FAIL" });

    let t = Instant::now();
    let rs = delta_stepping(g, src, delta);
    let t_ds = t.elapsed().as_secs_f64() * 1000.0;
    let bad_ds = count_bad(&r, &rs);
    println!("  our delta_stepping:    {:>9.2} ms   {}", t_ds,
             if bad_ds == 0 { "OK" } else { "FAIL" });

    let t = Instant::now();
    let rd = duan_inspired(g, src, bw);
    let t_du = t.elapsed().as_secs_f64() * 1000.0;
    let bad_du = count_bad(&r, &rd);
    println!("  our duan_inspired:     {:>9.2} ms   {}", t_du,
             if bad_du == 0 { "OK".to_string() } else { format!("FAIL ({bad_du})") });

    let t = Instant::now();
    let ra = sssp_auto(g, src);
    let t_au = t.elapsed().as_secs_f64() * 1000.0;
    let bad_au = count_bad(&r, &ra);
    println!("  our sssp_auto:         {:>9.2} ms   {}", t_au,
             if bad_au == 0 { "OK".to_string() } else { format!("FAIL ({bad_au})") });

    let mut t_pg = f64::NAN;
    let mut bad_pg = 0;
    if !skip_pg {
        let t = Instant::now();
        let (pg, pg_nodes) = build_petgraph(g);
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let pg_map = pg_dijkstra(&pg, pg_nodes[src as usize], None, |e| *e.weight());
        t_pg = t.elapsed().as_secs_f64() * 1000.0;
        let pg_vec = pg_to_vec(&pg_map, n);
        bad_pg = count_bad(&r, &pg_vec);
        println!(
            "  petgraph (build={:>5.0} ms + dijkstra): {:>9.2} ms   {}",
            build_ms,
            t_pg,
            if bad_pg == 0 { "OK" } else { "FAIL" }
        );
    } else {
        println!("  petgraph: skipped (too large)");
    }

    Row {
        petgraph: t_pg,
        binary: t_bin,
        quad: t_4,
        oct: t_8,
        dstep: t_ds,
        duan: t_du,
        auto: t_au,
        bad_pg,
        bad_4ary: bad_4,
        bad_8ary: bad_8,
        bad_dstep: bad_ds,
        bad_duan: bad_du,
        bad_auto: bad_au,
    }
}

fn main() -> std::io::Result<()> {
    let mut rows: Vec<(String, Row)> = Vec::new();

    // ---------- Synthetic ----------
    {
        let g = gen_random_sparse(100_000, 8, 42);
        rows.push(("Sparse rand n=100k deg=8".to_string(), run("Sparse rand n=100k deg=8", &g, 0, false)));
    }
    {
        let g = gen_random_sparse(1_000_000, 8, 42);
        rows.push(("Sparse rand n=1M deg=8".to_string(), run("Sparse rand n=1M deg=8", &g, 0, false)));
    }
    {
        let g = gen_grid(500, 7);
        rows.push(("Grid 500x500".to_string(), run("Grid 500x500", &g, 0, false)));
    }
    {
        let g = gen_grid(2000, 7);
        rows.push(("Grid 2000x2000".to_string(), run("Grid 2000x2000", &g, 0, false)));
    }
    {
        let g = gen_path(500_000, 11);
        rows.push(("Path n=500k".to_string(), run("Path n=500k", &g, 0, false)));
    }
    {
        let g = gen_power_law(200_000, 4, 13);
        rows.push(("Power-law n=200k".to_string(), run("Power-law n=200k", &g, 0, false)));
    }
    // ---------- RMAT ----------
    {
        let g = gen_rmat(20, 16, 42);
        // pick well-connected source
        let mut src = 0u32;
        for u in 0..g.n {
            if g.head[u + 1] - g.head[u] > 10 {
                src = u as u32;
                break;
            }
        }
        rows.push(("RMAT scale=20".to_string(), run("RMAT scale=20", &g, src, false)));
    }
    {
        let g = gen_rmat(22, 16, 42);
        let mut src = 0u32;
        for u in 0..g.n {
            if g.head[u + 1] - g.head[u] > 10 {
                src = u as u32;
                break;
            }
        }
        // RMAT scale=22 has 67M edges; petgraph still doable but slow
        rows.push(("RMAT scale=22".to_string(), run("RMAT scale=22", &g, src, false)));
    }
    // ---------- Word ladder ----------
    if std::path::Path::new("/usr/share/dict/words").exists() {
        match build_word_ladder("/usr/share/dict/words", 4, 8) {
            Ok((g, _)) => {
                rows.push(("Word ladder".to_string(), run("Word ladder", &g, 0, false)));
            }
            Err(e) => println!("(Skipping word ladder: {e})"),
        }
    }
    // ---------- Rubik ----------
    {
        let (g, _depth) = build_pocket_cube_graph();
        rows.push(("Rubik 2x2x2".to_string(), run("Rubik 2x2x2", &g, 0, false)));
    }
    // ---------- SNAP ----------
    if std::path::Path::new("data/soc-LiveJournal1.txt.gz").exists() {
        match load_snap_edge_list("data/soc-LiveJournal1.txt.gz") {
            Ok((g, _)) => {
                // 137M edges -> petgraph build ~1-2s, dijkstra ~3-5s. We accept it.
                rows.push(("SNAP LiveJournal".to_string(), run("SNAP LiveJournal", &g, 0, false)));
            }
            Err(e) => println!("(Skipping SNAP: {e})"),
        }
    }
    // ---------- London OSM ----------
    if std::path::Path::new("data/greater-london.osm.pbf").exists() {
        let cache = "data/greater-london.osm.pbf.csr";
        let (g, _coords, _edge_dist) = load_with_cache("data/greater-london.osm.pbf", cache, Profile::Car)?;
        rows.push(("London OSM".to_string(), run("London OSM", &g, 0, false)));
    }

    // ---------- Summary ----------
    println!();
    println!("=== SUMMARY (all times in ms) ===");
    println!(
        "{:<26} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "Workload", "petgraph", "our_bin", "our_4ar", "our_8ar", "d-step", "duan", "auto"
    );
    for (lbl, r) in &rows {
        let ms = |x: f64| if x.is_nan() { "  -- ".to_string() } else { format!("{:.1}", x) };
        let mark = |b: usize| if b == 0 { "" } else { "x" };
        println!(
            "{:<26} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            lbl, ms(r.petgraph),
            ms(r.binary),
            format!("{}{}", ms(r.quad), mark(r.bad_4ary)),
            format!("{}{}", ms(r.oct), mark(r.bad_8ary)),
            format!("{}{}", ms(r.dstep), mark(r.bad_dstep)),
            format!("{}{}", ms(r.duan), mark(r.bad_duan)),
            format!("{}{}", ms(r.auto), mark(r.bad_auto)),
        );
    }

    println!();
    println!("=== SPEEDUP vs petgraph ===");
    println!(
        "{:<26} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "Workload", "our_bin", "our_4ar", "our_8ar", "d-step", "duan", "auto"
    );
    for (lbl, r) in &rows {
        let s = |x: f64| {
            if r.petgraph.is_nan() {
                "  - ".to_string()
            } else {
                format!("{:.2}x", r.petgraph / x.max(0.001))
            }
        };
        println!(
            "{:<26} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            lbl, s(r.binary), s(r.quad), s(r.oct), s(r.dstep), s(r.duan), s(r.auto)
        );
    }

    Ok(())
}
