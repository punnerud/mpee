//! Intercity routing test on the England graph.
//!
//! Pick coordinates of major UK cities, find the nearest vertex by
//! lat/lon distance, then run CH queries between them. Compare with
//! Google Maps / common road distances as a sanity check.

use std::time::Instant;

use sssp_bench::cache_ch;
use sssp_bench::cache_pp;
use sssp_bench::ch;

/// Major UK cities (approximate (lat, lon) of city center).
const CITIES: &[(&str, f32, f32)] = &[
    ("London",        51.5074, -0.1278),
    ("Manchester",    53.4808, -2.2426),
    ("Birmingham",    52.4862, -1.8904),
    ("Liverpool",     53.4084, -2.9916),
    ("Leeds",         53.8008, -1.5491),
    ("Sheffield",     53.3811, -1.4701),
    ("Bristol",       51.4545, -2.5879),
    ("Newcastle",     54.9783, -1.6178),
    ("Nottingham",    52.9548, -1.1581),
    ("Cambridge",     52.2053,  0.1218),
    ("Brighton",      50.8225, -0.1372),
    ("Oxford",        51.7520, -1.2577),
];

fn nearest_vertex(coords: &[(f32, f32)], target_lat: f32, target_lon: f32) -> u32 {
    let mut best_dist = f32::INFINITY;
    let mut best = 0u32;
    for (i, &(lat, lon)) in coords.iter().enumerate() {
        let dlat = lat - target_lat;
        let dlon = (lon - target_lon) * (target_lat.to_radians()).cos();
        let d = dlat * dlat + dlon * dlon;
        if d < best_dist {
            best_dist = d;
            best = i as u32;
        }
    }
    best
}

fn main() -> std::io::Result<()> {
    println!("=== Intercity routing on England ===");
    let pp = cache_pp::load_mmap("data/england.osm.pbf.pp")?;
    println!(
        "England graph: n={}, m={}, avg_deg={:.2}",
        pp.graph.n,
        pp.graph.m(),
        pp.graph.m() as f32 / pp.graph.n.max(1) as f32
    );

    // Map cities → vertex idx via nearest coordinate
    let t = Instant::now();
    let city_verts: Vec<(usize, u32)> = CITIES
        .iter()
        .enumerate()
        .map(|(i, &(_, lat, lon))| (i, nearest_vertex(&pp.coords, lat, lon)))
        .collect();
    println!("Mapping cities to vertices: {:.2} s", t.elapsed().as_secs_f64());

    // Verify mapping by printing chosen coordinates
    println!("\nCity → nearest graph vertex:");
    for &(i, v) in &city_verts {
        let (name, tlat, tlon) = CITIES[i];
        let (lat, lon) = pp.coords[v as usize];
        println!(
            "  {:<14} target ({:.4}, {:.4})  →  vertex {} ({:.4}, {:.4})",
            name, tlat, tlon, v, lat, lon
        );
    }

    // Load CH
    println!("\nLoading CH cache...");
    let h = cache_ch::load_mmap("data/england.osm.pbf.ch")?;

    println!("\nIntercity distances (km, via CH):");
    println!(
        "{:<14}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}",
        "from\\to",
        "London",
        "Manch",
        "Birm",
        "Liv",
        "Leeds",
        "Bristol",
        "Newc",
        "Camb",
    );
    let column_indices: Vec<usize> = vec![0, 1, 2, 3, 4, 6, 7, 9]; // first 8 cities for table
    let t = Instant::now();
    let mut total_queries = 0u32;
    for &(from_i, from_v) in &city_verts {
        let from_name = CITIES[from_i].0;
        if from_name.len() > 14 {
            continue;
        }
        print!("{:<14}", from_name);
        for &col in &column_indices {
            let to_v = city_verts.iter().find(|&&(i, _)| i == col).map(|&(_, v)| v).unwrap();
            let from_int = h.perm[from_v as usize];
            let to_int = h.perm[to_v as usize];
            let d = ch::query(&h, from_int, to_int).unwrap_or(f32::INFINITY);
            total_queries += 1;
            print!("{:>10}", if d.is_finite() { format!("{:.0}", d / 1000.0) } else { "-".to_string() });
        }
        println!();
    }
    println!(
        "\n({} queries in {:.0} ms, avg {:.3} ms/query)",
        total_queries,
        t.elapsed().as_secs_f64() * 1000.0,
        t.elapsed().as_secs_f64() * 1000.0 / total_queries as f64
    );

    Ok(())
}
