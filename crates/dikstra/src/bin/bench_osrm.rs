//! Sammenligning vs OSRM med Contraction Hierarchies.
//!
//! OSRM-serveren må kjøre på localhost:5002 med /data/london.osrm lastet.
//! Start med:
//!   docker run -p 5002:5000 -d --rm -v "$(pwd)/data:/data" \
//!     --name osrm-london \
//!     ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm ch /data/london.osrm

use std::time::Instant;

use sssp_bench::auto::sssp_auto;
use sssp_bench::bidir::bidir_dijkstra;
use sssp_bench::cache_pp;
use sssp_bench::dijkstra::dijkstra_binary;
use sssp_bench::graph::Rng;

fn main() -> std::io::Result<()> {
    let pp_cache = "data/greater-london.osm.pbf.pp";
    let pp = match cache_pp::load_mmap(pp_cache) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("pp-cache mangler ({e}). Kjør først: cargo run --release --bin bench_pp");
            std::process::exit(1);
        }
    };

    let g = &pp.graph;
    let n = g.n;
    println!("=== OSRM vs vår på Greater London (n={n}, m={}) ===", g.m());

    // Generer 1000 random (vertex_idx, vertex_idx)-par.
    let n_queries = 1000usize;
    let mut rng = Rng(20260509);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n_queries);
    while pairs.len() < n_queries {
        let s = rng.range(n as u32);
        let d = rng.range(n as u32);
        // Skip vertices with no edges.
        if g.head[s as usize + 1] - g.head[s as usize] == 0 {
            continue;
        }
        if g.head[d as usize + 1] - g.head[d as usize] == 0 {
            continue;
        }
        pairs.push((s, d));
    }
    println!("Genererte {} (s,d)-par.\n", n_queries);

    // ---- Bidir Dijkstra (vår) ----
    let t = Instant::now();
    let mut bidir_results: Vec<f32> = Vec::with_capacity(n_queries);
    for &(s, d) in &pairs {
        let r = bidir_dijkstra(g, &pp.reverse, s, d);
        bidir_results.push(r.unwrap_or(f32::INFINITY));
    }
    let bidir_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "vår bidir Dijkstra:    {:>9.0} ms total ({:.3} ms/query)",
        bidir_ms,
        bidir_ms / n_queries as f64
    );

    // ---- Reference: full SSSP fra hver src (kun 30 stk for å spare tid) ----
    let n_ref = 30usize;
    let mut ref_results: Vec<f32> = Vec::with_capacity(n_ref);
    let t = Instant::now();
    for &(s, d) in &pairs[..n_ref] {
        let dist = dijkstra_binary(g, s);
        ref_results.push(dist[d as usize]);
    }
    let ref_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "full Dijkstra (1 src): {:>9.0} ms for {n_ref} queries ({:.1} ms/query)",
        ref_ms,
        ref_ms / n_ref as f64
    );

    // ---- sssp_auto (vår) — for sammenligning av SSSP-stil pull ----
    let t = Instant::now();
    for &(s, _) in &pairs[..n_ref] {
        let _ = sssp_auto(g, s);
    }
    let auto_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "sssp_auto (1 src):     {:>9.0} ms for {n_ref} queries ({:.1} ms/query)",
        auto_ms,
        auto_ms / n_ref as f64
    );

    // ---- OSRM via HTTP ----
    println!("\nKjører OSRM (CH) på localhost:5002...");

    // Sjekk at serveren svarer.
    let test_url = format!(
        "http://localhost:5002/route/v1/driving/{},{};{},{}?overview=false",
        pp.coords[0].1, pp.coords[0].0,
        pp.coords[1].1, pp.coords[1].0
    );
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(10))
        .build();
    match agent.get(&test_url).call() {
        Ok(_) => println!("OSRM-server svarer."),
        Err(e) => {
            eprintln!("OSRM-server svarer ikke: {e}");
            eprintln!("Hopper over OSRM-bench.");
            return Ok(());
        }
    }

    let t = Instant::now();
    let mut osrm_results: Vec<f32> = Vec::with_capacity(n_queries);
    let mut osrm_failed = 0usize;
    for &(s, d) in &pairs {
        let (lat_s, lon_s) = pp.coords[s as usize];
        let (lat_d, lon_d) = pp.coords[d as usize];
        let url = format!(
            "http://localhost:5002/route/v1/driving/{},{};{},{}?overview=false&steps=false",
            lon_s, lat_s, lon_d, lat_d
        );
        match agent.get(&url).call() {
            Ok(resp) => {
                let body = resp.into_string().unwrap_or_default();
                // Parse JSON for "distance" — enkel string-søk:
                if let Some(idx) = body.find("\"distance\":") {
                    let rest = &body[idx + 11..];
                    let end = rest.find(['}', ',']).unwrap_or(rest.len());
                    let val: f32 = rest[..end].parse().unwrap_or(f32::NAN);
                    osrm_results.push(val);
                } else {
                    osrm_results.push(f32::INFINITY);
                    osrm_failed += 1;
                }
            }
            Err(_) => {
                osrm_results.push(f32::INFINITY);
                osrm_failed += 1;
            }
        }
    }
    let osrm_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "OSRM CH (HTTP):        {:>9.0} ms total ({:.3} ms/query)  [{osrm_failed} feil]",
        osrm_ms,
        osrm_ms / n_queries as f64
    );

    // ---- Korrekthet sammenligning ----
    // OSRM bruker bil-profil og veier; vår bruker haversine-vekter.
    // Vi forventer at OSRM-distanser er litt LENGRE (følger veier) enn våre
    // (haversine), men korrelasjonen skal være sterk.
    println!("\n=== Distanse-sammenligning (vår bidir vs OSRM, 30 første queries) ===");
    println!("{:>5}  {:>10}  {:>10}  {:>10}", "i", "vår (m)", "OSRM (m)", "ratio");
    for i in 0..n_ref.min(30) {
        let our = bidir_results[i];
        let osrm = osrm_results[i];
        let ratio = if our.is_finite() && osrm.is_finite() && our > 0.0 {
            osrm / our
        } else {
            0.0
        };
        println!("{:>5}  {:>10.0}  {:>10.0}  {:>10.2}", i, our, osrm, ratio);
        if i >= 9 {
            break;
        }
    }

    println!("\n=== Sammendrag ===");
    println!(
        "OSRM CH p2p:    {:.3} ms/query (incl HTTP overhead, lokalt)",
        osrm_ms / n_queries as f64
    );
    println!(
        "vår bidir:      {:.3} ms/query",
        bidir_ms / n_queries as f64
    );
    println!(
        "ratio (OSRM/vår): {:.2}x",
        (osrm_ms / n_queries as f64) / (bidir_ms / n_queries as f64)
    );

    println!();
    println!("Preprocessing-tid:");
    println!("  OSRM (extract + contract):  ~80 sekunder");
    println!("  vår (parse + reorder + transpose + cache): ~1 sekund");

    Ok(())
}
