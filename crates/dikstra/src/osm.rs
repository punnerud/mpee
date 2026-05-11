//! Last inn en OSM .pbf-fil og bygg en routing-graf (CSR) med haversine-vekter.
//!
//! Strategi: én-pass parsing. PBF lagrer noder først, så ways. Vi samler
//! ALLE node-koordinater i en stor HashMap, deretter materialize-r vi
//! drivable ways direkte til kanter. Det sparer en hel ekstra pass over
//! filen sammenlignet med to-pass-strategien (filtrer ways → finn ut hvilke
//! noder vi trenger → andre pass for å hente koordinater).

use crate::cache;
use crate::graph::CsrGraph;
use crate::osm_profile::{Profile, kmh_to_mps, parse_maxspeed};
use osmpbf::{BlobDecode, BlobReader, Element, ElementReader};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Load (or parse and cache) an OSM-derived CSR graph for the given profile.
/// Returns `(graph, coords, edge_dist)` where `graph.edge_w` is duration in
/// seconds and `edge_dist[k]` is the geometric distance in metres of the same
/// edge (haversine between endpoints).
pub fn load_with_cache<P: AsRef<Path>>(
    pbf_path: P,
    cache_path: P,
    profile: Profile,
) -> std::io::Result<(
    CsrGraph,
    crate::buffer::Buffer<(f32, f32)>,
    crate::buffer::Buffer<f32>,
)> {
    let pbf = pbf_path.as_ref();
    let cache_p = cache_path.as_ref();

    let pbf_meta = std::fs::metadata(pbf)?;
    let cache_meta = std::fs::metadata(cache_p);

    let cache_valid = match cache_meta {
        Ok(m) => match (m.modified(), pbf_meta.modified()) {
            (Ok(c), Ok(p)) => c >= p,
            _ => false,
        },
        Err(_) => false,
    };

    if cache_valid {
        let t = std::time::Instant::now();
        match cache::load_mmap(cache_p) {
            Ok((g, c, ed)) => {
                println!(
                    "[osm/{prof}] mmap-cache: åpnet {} noder + {} kanter på {:.2} ms (instant start)",
                    g.n,
                    g.m(),
                    t.elapsed().as_secs_f64() * 1000.0,
                    prof = profile.name(),
                );
                return Ok((g, c, ed));
            }
            Err(e) => {
                println!("[osm/{}] cache korrupt ({e}) — re-parser", profile.name());
            }
        }
    }

    let (g, coords, edge_dist) = load_osm_routing_par(pbf, profile)?;
    let t = std::time::Instant::now();
    if let Err(e) = cache::save(cache_p, &g, &coords, &edge_dist) {
        eprintln!("[osm/{}] kunne ikke lagre cache: {e}", profile.name());
    } else {
        println!(
            "[osm/{}] cache skrevet ({:.0} ms): {}",
            profile.name(),
            t.elapsed().as_secs_f64() * 1000.0,
            cache_p.display()
        );
    }
    Ok((
        g,
        crate::buffer::Buffer::from(coords),
        crate::buffer::Buffer::from(edge_dist),
    ))
}

/// Parallell parse via BlobReader + rayon: hver blob (~8000 elementer)
/// dekodes på sin egen tråd og produserer en lokal (ways, nodes)-akkumulator.
/// Reducer slår sammen. Etterpå bygges CSR (sekvensielt — det dominerer ikke).
pub fn load_osm_routing_par<P: AsRef<Path>>(
    path: P,
    profile: Profile,
) -> std::io::Result<(CsrGraph, Vec<(f32, f32)>, Vec<f32>)> {
    let path = path.as_ref();
    println!("[osm] åpner {} (parallell)", path.display());

    let blob_reader = BlobReader::from_path(path).map_err(io_err)?;
    let t = std::time::Instant::now();

    type Acc = (Vec<(Vec<i64>, OneWay, f32)>, Vec<(i64, f32, f32)>);

    let (ways, nodes_data) = blob_reader
        .par_bridge()
        .filter_map(|res| res.ok())
        .filter_map(|blob| match blob.decode() {
            Ok(BlobDecode::OsmData(block)) => Some(block),
            _ => None,
        })
        .map(|block| -> Acc {
            // Rimelig forhåndsallokering — typisk PrimitiveBlock har 8000 elementer.
            let mut ways: Vec<(Vec<i64>, OneWay, f32)> = Vec::with_capacity(64);
            let mut nodes: Vec<(i64, f32, f32)> = Vec::with_capacity(8000);
            for elem in block.elements() {
                match elem {
                    Element::Node(n) => {
                        nodes.push((n.id(), n.lat() as f32, n.lon() as f32));
                    }
                    Element::DenseNode(n) => {
                        nodes.push((n.id(), n.lat() as f32, n.lon() as f32));
                    }
                    Element::Way(w) => {
                        let mut hw: Option<&str> = None;
                        let mut maxspeed: Option<f32> = None;
                        let mut oneway: OneWay = OneWay::No;
                        let mut roundabout = false;
                        for (k, v) in w.tags() {
                            match k {
                                "highway" => hw = Some(v),
                                "maxspeed" => maxspeed = parse_maxspeed(v),
                                "oneway" => match v {
                                    "yes" | "true" | "1" => oneway = OneWay::Forward,
                                    "-1" | "reverse" => oneway = OneWay::Backward,
                                    _ => {}
                                },
                                "junction" => {
                                    if v == "roundabout" {
                                        roundabout = true;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if let Some(h) = hw {
                            if profile.accepts(h) {
                                if matches!(oneway, OneWay::No)
                                    && (h == "motorway" || h == "motorway_link" || roundabout)
                                {
                                    oneway = OneWay::Forward;
                                }
                                let refs: Vec<i64> = w.refs().collect();
                                if refs.len() >= 2 {
                                    let speed_kmh = profile.speed_kmh(h, maxspeed);
                                    let speed_mps = kmh_to_mps(speed_kmh).max(0.5);
                                    ways.push((refs, oneway, speed_mps));
                                }
                            }
                        }
                    }
                    Element::Relation(_) => {}
                }
            }
            (ways, nodes)
        })
        .reduce(
            || (Vec::new(), Vec::new()),
            |mut a, mut b| {
                a.0.append(&mut b.0);
                a.1.append(&mut b.1);
                a
            },
        );

    println!(
        "[osm] parallel parse: {:.2} s — {} noder, {} drivable ways",
        t.elapsed().as_secs_f64(),
        nodes_data.len(),
        ways.len()
    );

    // Bygg HashMap for koordinatslå-opp.
    let t2 = std::time::Instant::now();
    let mut node_coords: HashMap<i64, (f32, f32)> = HashMap::with_capacity(nodes_data.len());
    for (id, lat, lon) in nodes_data {
        node_coords.insert(id, (lat, lon));
    }
    println!(
        "[osm] node-hashmap: {:.2} s ({} noder)",
        t2.elapsed().as_secs_f64(),
        node_coords.len()
    );

    finalize_csr(ways, node_coords)
}

/// Single-pass parse: samle alle node-koordinater + drivable ways.
pub fn load_osm_routing<P: AsRef<Path>>(
    path: P,
    profile: Profile,
) -> std::io::Result<(CsrGraph, Vec<(f32, f32)>, Vec<f32>)> {
    let path = path.as_ref();
    println!("[osm] åpner {}", path.display());

    // Vi samler ALLE noder i én stor HashMap. Det er greedy minne-bruk —
    // for London ~5-10M noder × 16 bytes ≈ 80-160 MB — men det lar oss
    // lese filen i én pass.
    let mut node_coords: HashMap<i64, (f32, f32)> = HashMap::with_capacity(8_000_000);
    let mut ways: Vec<(Vec<i64>, OneWay, f32)> = Vec::with_capacity(300_000);
    let mut total_nodes = 0usize;
    let mut total_ways = 0usize;
    let mut kept_ways = 0usize;

    let reader = ElementReader::from_path(path).map_err(io_err)?;
    let t = std::time::Instant::now();
    reader
        .for_each(|elem| match elem {
            Element::Node(n) => {
                total_nodes += 1;
                node_coords.insert(n.id(), (n.lat() as f32, n.lon() as f32));
            }
            Element::DenseNode(n) => {
                total_nodes += 1;
                node_coords.insert(n.id(), (n.lat() as f32, n.lon() as f32));
            }
            Element::Way(w) => {
                total_ways += 1;
                let mut hw: Option<&str> = None;
                let mut maxspeed: Option<f32> = None;
                let mut oneway: OneWay = OneWay::No;
                let mut roundabout = false;
                for (k, v) in w.tags() {
                    match k {
                        "highway" => hw = Some(v),
                        "maxspeed" => maxspeed = parse_maxspeed(v),
                        "oneway" => match v {
                            "yes" | "true" | "1" => oneway = OneWay::Forward,
                            "-1" | "reverse" => oneway = OneWay::Backward,
                            _ => {}
                        },
                        "junction" => {
                            if v == "roundabout" {
                                roundabout = true;
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(h) = hw {
                    if profile.accepts(h) {
                        if matches!(oneway, OneWay::No)
                            && (h == "motorway" || h == "motorway_link" || roundabout)
                        {
                            oneway = OneWay::Forward;
                        }
                        let refs: Vec<i64> = w.refs().collect();
                        if refs.len() >= 2 {
                            let speed_kmh = profile.speed_kmh(h, maxspeed);
                            let speed_mps = kmh_to_mps(speed_kmh).max(0.5);
                            ways.push((refs, oneway, speed_mps));
                            kept_ways += 1;
                        }
                    }
                }
            }
            Element::Relation(_) => {}
        })
        .map_err(io_err)?;
    println!(
        "[osm] én-pass parse: {:.2} s — {} noder lagret, {} ways totalt, {} drivable",
        t.elapsed().as_secs_f64(),
        total_nodes,
        total_ways,
        kept_ways
    );

    finalize_csr(ways, node_coords)
}

fn finalize_csr(
    ways: Vec<(Vec<i64>, OneWay, f32)>,
    node_coords: HashMap<i64, (f32, f32)>,
) -> std::io::Result<(CsrGraph, Vec<(f32, f32)>, Vec<f32>)> {
    let t = std::time::Instant::now();
    let mut id_map: HashMap<i64, u32> = HashMap::with_capacity(1_500_000);
    let mut coords: Vec<(f32, f32)> = Vec::with_capacity(1_500_000);
    // (u, v, duration_s, distance_m). The graph's `edge_w` will be duration;
    // the parallel `edge_dist` array is the metres returned alongside.
    let mut edges: Vec<(u32, u32, f32, f32)> = Vec::with_capacity(2 * ways.len() * 4);
    let mut dropped = 0usize;
    for (refs, oneway, speed_mps) in &ways {
        for win in refs.windows(2) {
            let a_idx = match id_map.get(&win[0]) {
                Some(&i) => i,
                None => match node_coords.get(&win[0]) {
                    Some(&xy) => {
                        let i = coords.len() as u32;
                        coords.push(xy);
                        id_map.insert(win[0], i);
                        i
                    }
                    None => {
                        dropped += 1;
                        continue;
                    }
                },
            };
            let b_idx = match id_map.get(&win[1]) {
                Some(&i) => i,
                None => match node_coords.get(&win[1]) {
                    Some(&xy) => {
                        let i = coords.len() as u32;
                        coords.push(xy);
                        id_map.insert(win[1], i);
                        i
                    }
                    None => {
                        dropped += 1;
                        continue;
                    }
                },
            };
            if a_idx == b_idx {
                continue;
            }
            let (a_lat, a_lon) = coords[a_idx as usize];
            let (b_lat, b_lon) = coords[b_idx as usize];
            let dist_m = haversine_m(a_lat, a_lon, b_lat, b_lon).max(1e-3);
            let dur_s = (dist_m / *speed_mps).max(1e-4);
            match oneway {
                OneWay::No => {
                    edges.push((a_idx, b_idx, dur_s, dist_m));
                    edges.push((b_idx, a_idx, dur_s, dist_m));
                }
                OneWay::Forward => edges.push((a_idx, b_idx, dur_s, dist_m)),
                OneWay::Backward => edges.push((b_idx, a_idx, dur_s, dist_m)),
            }
        }
    }
    println!(
        "[osm] kantkonstruksjon: {:.2} s — {} kanter ({} dropped) (vekt = duration_s, parallel edge_dist_m)",
        t.elapsed().as_secs_f64(),
        edges.len(),
        dropped
    );

    drop(node_coords);

    let (g, edge_dist) = CsrGraph::from_edges_with_dist(coords.len(), &edges);
    println!(
        "[osm] CSR: n = {}, m = {}, avg_deg = {:.2}",
        g.n,
        g.m(),
        g.m() as f32 / g.n.max(1) as f32
    );
    Ok((g, coords, edge_dist))
}

#[derive(Clone, Copy)]
enum OneWay {
    No,
    Forward,
    Backward,
}

#[inline]
fn haversine_m(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let r = 6_371_000.0_f64;
    let l1 = (lat1 as f64).to_radians();
    let l2 = (lat2 as f64).to_radians();
    let dlat = (lat2 as f64 - lat1 as f64).to_radians();
    let dlon = (lon2 as f64 - lon1 as f64).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + l1.cos() * l2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    (r * c) as f32
}

fn io_err(e: osmpbf::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}
