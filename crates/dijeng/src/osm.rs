//! Load an OSM .pbf file and build a routing graph (CSR) with haversine weights.
//!
//! Strategy: one-pass parsing. PBF stores nodes first, then ways. We collect
//! ALL node coordinates in a large HashMap, then materialize drivable ways
//! directly into edges. That saves a whole extra pass over the file compared
//! to the two-pass strategy (filter ways → figure out which nodes we need →
//! second pass to fetch coordinates).

use crate::addresses::AddrRec;
use crate::cache;
use crate::graph::CsrGraph;
use crate::osm_profile::{Profile, kmh_to_mps, parse_maxspeed};
use osmpbf::{BlobDecode, BlobReader, Element, ElementReader};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Address tags collected from one OSM element (`addr:*`).
#[derive(Clone, Default)]
struct NodeAddr {
    housenumber: Option<Box<str>>,
    street: Option<Box<str>>,
    city: Option<Box<str>>,
    postcode: Option<Box<str>>,
    /// `addr:interpolation` value when present (ways only).
    interpolation: Option<Box<str>>,
}

impl NodeAddr {
    fn read(k: &str, v: &str, a: &mut NodeAddr) {
        match k {
            "addr:housenumber" => a.housenumber = Some(v.into()),
            "addr:street" => a.street = Some(v.into()),
            "addr:city" => a.city = Some(v.into()),
            "addr:postcode" => a.postcode = Some(v.into()),
            "addr:interpolation" => a.interpolation = Some(v.into()),
            _ => {}
        }
    }
    fn has_any(&self) -> bool {
        self.housenumber.is_some()
            || self.street.is_some()
            || self.city.is_some()
            || self.postcode.is_some()
    }
}

/// Parse accumulator for one PBF block / the whole sequential pass.
#[derive(Default)]
struct ParseAcc {
    ways: Vec<WayRec>,
    nodes: Vec<(i64, f32, f32)>,
    /// Nodes carrying `addr:*` (id + tags); coords resolved later from the map.
    addr_nodes: Vec<(i64, NodeAddr)>,
    /// Building/area ways with `addr:housenumber`+`addr:street` (centroid later).
    way_addrs: Vec<(Vec<i64>, NodeAddr)>,
    /// `addr:interpolation` ways (refs + rule); endpoints looked up in addr_nodes.
    interp_ways: Vec<(Vec<i64>, Box<str>)>,
}

impl ParseAcc {
    fn merge(&mut self, mut o: ParseAcc) {
        self.ways.append(&mut o.ways);
        self.nodes.append(&mut o.nodes);
        self.addr_nodes.append(&mut o.addr_nodes);
        self.way_addrs.append(&mut o.way_addrs);
        self.interp_ways.append(&mut o.interp_ways);
    }
}

/// OSM packs several house numbers on one feature with `;` (e.g. "4081;4083"
/// for a building covering both). Render that as a clean range "4081–4083"
/// (first–last) instead of the raw semicolon list; a single number is returned
/// trimmed and unchanged.
fn clean_housenumber(s: &str) -> String {
    let t = s.trim();
    if !t.contains(';') {
        return t.to_string();
    }
    let parts: Vec<&str> = t.split(';').map(|p| p.trim()).filter(|p| !p.is_empty()).collect();
    match (parts.first(), parts.last()) {
        (Some(&a), Some(&b)) if a != b => format!("{a}\u{2013}{b}"),
        (Some(&a), _) => a.to_string(),
        _ => t.to_string(),
    }
}

#[inline]
fn leading_int(s: &str) -> Option<i64> {
    let d: String = s.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
    d.parse().ok()
}

/// Resolve collected `addr:*` data into address points. Standalone nodes use
/// their own coord; building ways use a centroid of their refs; interpolation
/// ways synthesize intermediate numbers between consecutive numbered endpoints.
fn resolve_addresses(
    addr_nodes: Vec<(i64, NodeAddr)>,
    way_addrs: Vec<(Vec<i64>, NodeAddr)>,
    interp_ways: Vec<(Vec<i64>, Box<str>)>,
    node_coords: &HashMap<i64, (f32, f32)>,
) -> Vec<AddrRec> {
    let mk = |lat: f32, lon: f32, hn: &str, st: &str, a: &NodeAddr| AddrRec {
        lat,
        lon,
        housenumber: clean_housenumber(hn),
        street: st.to_string(),
        city: a.city.as_deref().map(|s| s.to_string()),
        postcode: a.postcode.as_deref().map(|s| s.to_string()),
    };
    let node_addr: HashMap<i64, &NodeAddr> = addr_nodes.iter().map(|(id, a)| (*id, a)).collect();
    let mut out: Vec<AddrRec> = Vec::new();

    // Standalone address nodes.
    for (id, a) in &addr_nodes {
        if let (Some(hn), Some(st)) = (&a.housenumber, &a.street) {
            if let Some(&(lat, lon)) = node_coords.get(id) {
                out.push(mk(lat, lon, hn, st, a));
            }
        }
    }

    // Building/area ways → centroid of resolved refs.
    for (refs, a) in &way_addrs {
        if let (Some(hn), Some(st)) = (&a.housenumber, &a.street) {
            let (mut sx, mut sy, mut k) = (0f64, 0f64, 0u32);
            let mut last: Option<i64> = None;
            for r in refs {
                if last == Some(*r) {
                    continue; // skip the repeated closing ref
                }
                last = Some(*r);
                if let Some(&(lat, lon)) = node_coords.get(r) {
                    sx += lat as f64;
                    sy += lon as f64;
                    k += 1;
                }
            }
            if k > 0 {
                out.push(mk((sx / k as f64) as f32, (sy / k as f64) as f32, hn, st, a));
            }
        }
    }

    // Interpolation ways → synthesize numbers between numbered endpoints.
    for (refs, rule_s) in &interp_ways {
        let rule = rule_s.trim().to_lowercase();
        let step_num: Option<i64> = rule.parse().ok();
        // numbered endpoints in way order, with coord + addr
        let pts: Vec<(i64, (f32, f32), &NodeAddr)> = refs
            .iter()
            .filter_map(|r| {
                let a = *node_addr.get(r)?;
                let n = leading_int(a.housenumber.as_deref()?)?;
                let c = *node_coords.get(r)?;
                Some((n, c, a))
            })
            .collect();
        for w in pts.windows(2) {
            let (n0, c0, a0) = w[0];
            let (n1, c1, _a1) = w[1];
            if n1 <= n0 || n1 - n0 > 2000 {
                continue; // malformed / decreasing / absurd range
            }
            let st = match a0.street.as_deref() {
                Some(s) => s,
                None => continue,
            };
            let step = match (rule.as_str(), step_num) {
                ("all", _) => 1,
                ("even", _) | ("odd", _) => 2,
                (_, Some(s)) if s > 0 => s,
                _ => 1,
            };
            let mut k = n0;
            while k <= n1 {
                let parity_ok = match rule.as_str() {
                    "even" => k % 2 == 0,
                    "odd" => k % 2 != 0,
                    _ => true,
                };
                if parity_ok {
                    let t = (k - n0) as f32 / (n1 - n0) as f32;
                    let lat = c0.0 + (c1.0 - c0.0) * t;
                    let lon = c0.1 + (c1.1 - c0.1) * t;
                    out.push(mk(lat, lon, &k.to_string(), st, a0));
                }
                k += step;
            }
        }
    }

    out
}

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
                    "[osm/{prof}] mmap cache: opened {} nodes + {} edges in {:.2} ms (instant start)",
                    g.n,
                    g.m(),
                    t.elapsed().as_secs_f64() * 1000.0,
                    prof = profile.name(),
                );
                return Ok((g, c, ed));
            }
            Err(e) => {
                println!("[osm/{}] cache corrupt ({e}) — reparsing", profile.name());
            }
        }
    }

    // The .csr cache stores routing data only; street names are reconstructed
    // by `build::build_cache`, which calls `load_osm_routing_par` directly.
    let parse = load_osm_routing_par(pbf, profile)?;
    let (g, coords, edge_dist) = (parse.graph, parse.coords, parse.edge_dist);
    let t = std::time::Instant::now();
    if let Err(e) = cache::save(cache_p, &g, &coords, &edge_dist) {
        eprintln!("[osm/{}] failed to save cache: {e}", profile.name());
    } else {
        println!(
            "[osm/{}] cache written ({:.0} ms): {}",
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

/// Parallel parse via BlobReader + rayon: each blob (~8000 elements) is
/// decoded on its own thread and produces a local (ways, nodes) accumulator.
/// Reduce merges them. Afterwards CSR is built (sequentially — it doesn't dominate).
pub fn load_osm_routing_par<P: AsRef<Path>>(
    path: P,
    profile: Profile,
) -> std::io::Result<OsmParse> {
    let path = path.as_ref();
    println!("[osm] opening {} (parallel)", path.display());

    let blob_reader = BlobReader::from_path(path).map_err(io_err)?;
    let t = std::time::Instant::now();

    let acc = blob_reader
        .par_bridge()
        .filter_map(|res| res.ok())
        .filter_map(|blob| match blob.decode() {
            Ok(BlobDecode::OsmData(block)) => Some(block),
            _ => None,
        })
        .map(|block| -> ParseAcc {
            // Reasonable pre-allocation — a typical PrimitiveBlock has 8000 elements.
            let mut acc = ParseAcc {
                ways: Vec::with_capacity(64),
                nodes: Vec::with_capacity(8000),
                ..Default::default()
            };
            for elem in block.elements() {
                match elem {
                    Element::Node(n) => {
                        acc.nodes.push((n.id(), n.lat() as f32, n.lon() as f32));
                        let mut a = NodeAddr::default();
                        for (k, v) in n.tags() {
                            NodeAddr::read(k, v, &mut a);
                        }
                        if a.has_any() {
                            acc.addr_nodes.push((n.id(), a));
                        }
                    }
                    Element::DenseNode(n) => {
                        acc.nodes.push((n.id(), n.lat() as f32, n.lon() as f32));
                        let mut a = NodeAddr::default();
                        for (k, v) in n.tags() {
                            NodeAddr::read(k, v, &mut a);
                        }
                        if a.has_any() {
                            acc.addr_nodes.push((n.id(), a));
                        }
                    }
                    Element::Way(w) => {
                        let mut hw: Option<&str> = None;
                        let mut maxspeed: Option<f32> = None;
                        let mut oneway: OneWay = OneWay::No;
                        let mut roundabout = false;
                        let mut name: Option<&str> = None;
                        let mut addr = NodeAddr::default();
                        for (k, v) in w.tags() {
                            match k {
                                "highway" => hw = Some(v),
                                "maxspeed" => maxspeed = parse_maxspeed(v),
                                "name" => name = Some(v),
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
                                _ => NodeAddr::read(k, v, &mut addr),
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
                                    acc.ways.push((refs, oneway, speed_mps, name.map(|s| s.into())));
                                }
                            }
                        }
                        // Address-bearing way (independent of being drivable).
                        if let Some(rule) = addr.interpolation.clone() {
                            acc.interp_ways.push((w.refs().collect(), rule));
                        }
                        if addr.housenumber.is_some() && addr.street.is_some() {
                            acc.way_addrs.push((w.refs().collect(), addr));
                        }
                    }
                    Element::Relation(_) => {}
                }
            }
            acc
        })
        .reduce(ParseAcc::default, |mut a, b| {
            a.merge(b);
            a
        });

    let ParseAcc { ways, nodes, addr_nodes, way_addrs, interp_ways } = acc;
    println!(
        "[osm] parallel parse: {:.2} s — {} nodes, {} drivable ways, {} addr-nodes",
        t.elapsed().as_secs_f64(),
        nodes.len(),
        ways.len(),
        addr_nodes.len(),
    );

    // Build HashMap for coordinate lookup.
    let t2 = std::time::Instant::now();
    let mut node_coords: HashMap<i64, (f32, f32)> = HashMap::with_capacity(nodes.len());
    for (id, lat, lon) in nodes {
        node_coords.insert(id, (lat, lon));
    }
    println!(
        "[osm] node-hashmap: {:.2} s ({} nodes)",
        t2.elapsed().as_secs_f64(),
        node_coords.len()
    );

    let addresses = resolve_addresses(addr_nodes, way_addrs, interp_ways, &node_coords);
    println!("[osm] addresses: {} points (nodes + way centroids + interpolation)", addresses.len());
    finalize_csr(ways, node_coords, addresses)
}

/// Single-pass parse: collect all node coordinates + drivable ways.
pub fn load_osm_routing<P: AsRef<Path>>(
    path: P,
    profile: Profile,
) -> std::io::Result<OsmParse> {
    let path = path.as_ref();
    println!("[osm] opening {}", path.display());

    // We collect ALL nodes in one large HashMap. That's greedy memory use —
    // for London ~5-10M nodes × 16 bytes ≈ 80-160 MB — but it lets us
    // read the file in a single pass.
    let mut node_coords: HashMap<i64, (f32, f32)> = HashMap::with_capacity(8_000_000);
    let mut ways: Vec<WayRec> = Vec::with_capacity(300_000);
    let mut addr_nodes: Vec<(i64, NodeAddr)> = Vec::new();
    let mut way_addrs: Vec<(Vec<i64>, NodeAddr)> = Vec::new();
    let mut interp_ways: Vec<(Vec<i64>, Box<str>)> = Vec::new();
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
                let mut a = NodeAddr::default();
                for (k, v) in n.tags() {
                    NodeAddr::read(k, v, &mut a);
                }
                if a.has_any() {
                    addr_nodes.push((n.id(), a));
                }
            }
            Element::DenseNode(n) => {
                total_nodes += 1;
                node_coords.insert(n.id(), (n.lat() as f32, n.lon() as f32));
                let mut a = NodeAddr::default();
                for (k, v) in n.tags() {
                    NodeAddr::read(k, v, &mut a);
                }
                if a.has_any() {
                    addr_nodes.push((n.id(), a));
                }
            }
            Element::Way(w) => {
                total_ways += 1;
                let mut hw: Option<&str> = None;
                let mut maxspeed: Option<f32> = None;
                let mut oneway: OneWay = OneWay::No;
                let mut roundabout = false;
                let mut name: Option<&str> = None;
                let mut addr = NodeAddr::default();
                for (k, v) in w.tags() {
                    match k {
                        "highway" => hw = Some(v),
                        "maxspeed" => maxspeed = parse_maxspeed(v),
                        "name" => name = Some(v),
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
                        _ => NodeAddr::read(k, v, &mut addr),
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
                            ways.push((refs, oneway, speed_mps, name.map(|s| s.into())));
                            kept_ways += 1;
                        }
                    }
                }
                if let Some(rule) = addr.interpolation.clone() {
                    interp_ways.push((w.refs().collect(), rule));
                }
                if addr.housenumber.is_some() && addr.street.is_some() {
                    way_addrs.push((w.refs().collect(), addr));
                }
            }
            Element::Relation(_) => {}
        })
        .map_err(io_err)?;
    println!(
        "[osm] single-pass parse: {:.2} s — {} nodes stored, {} ways total, {} drivable, {} addr-nodes",
        t.elapsed().as_secs_f64(),
        total_nodes,
        total_ways,
        kept_ways,
        addr_nodes.len(),
    );

    let addresses = resolve_addresses(addr_nodes, way_addrs, interp_ways, &node_coords);
    println!("[osm] addresses: {} points (nodes + way centroids + interpolation)", addresses.len());
    finalize_csr(ways, node_coords, addresses)
}

/// Everything one OSM parse produces: the routing graph + per-edge distances,
/// the street-name data for the `.names` sidecar, and the resolved address
/// points for the `.addr` sidecar (independent of the graph).
pub struct OsmParse {
    pub graph: CsrGraph,
    pub coords: Vec<(f32, f32)>,
    pub edge_dist: Vec<f32>,
    pub node_name: Vec<u32>,
    pub name_pool: Vec<String>,
    pub street_nodes: Vec<Vec<u32>>,
    pub addresses: Vec<AddrRec>,
}

fn finalize_csr(
    ways: Vec<WayRec>,
    node_coords: HashMap<i64, (f32, f32)>,
    addresses: Vec<AddrRec>,
) -> std::io::Result<OsmParse> {
    use crate::names::NO_NAME;
    let t = std::time::Instant::now();
    let mut id_map: HashMap<i64, u32> = HashMap::with_capacity(1_500_000);
    let mut coords: Vec<(f32, f32)> = Vec::with_capacity(1_500_000);
    // Per-node street name: `node_name[csr_id]` indexes into `name_pool`, or
    // `NO_NAME`. Grown in lockstep with `coords`. The pool dedups the distinct
    // street names (a city has only a few thousand). First named way to touch
    // a node wins (matters only at intersections).
    let mut node_name: Vec<u32> = Vec::with_capacity(1_500_000);
    let mut name_pool: Vec<String> = Vec::new();
    let mut name_map: HashMap<String, u32> = HashMap::new();
    // `street_nodes[street_id]` = the road nodes that street touches (CSR ids,
    // with duplicates — deduped at write time). Used for intersection search:
    // the node where two streets meet is in both their node sets.
    let mut street_nodes: Vec<Vec<u32>> = Vec::new();
    // (u, v, duration_s, distance_m). The graph's `edge_w` will be duration;
    // the parallel `edge_dist` array is the metres returned alongside.
    let mut edges: Vec<(u32, u32, f32, f32)> = Vec::with_capacity(2 * ways.len() * 4);
    let mut dropped = 0usize;
    for (refs, oneway, speed_mps, wname) in &ways {
        // Intern this way's street name once.
        let wname_id: u32 = match wname {
            Some(s) => {
                let key: &str = s;
                match name_map.get(key) {
                    Some(&id) => id,
                    None => {
                        let id = name_pool.len() as u32;
                        name_pool.push(key.to_string());
                        name_map.insert(key.to_string(), id);
                        street_nodes.push(Vec::new());
                        id
                    }
                }
            }
            None => NO_NAME,
        };
        for win in refs.windows(2) {
            let a_idx = match id_map.get(&win[0]) {
                Some(&i) => i,
                None => match node_coords.get(&win[0]) {
                    Some(&xy) => {
                        let i = coords.len() as u32;
                        coords.push(xy);
                        node_name.push(NO_NAME);
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
                        node_name.push(NO_NAME);
                        id_map.insert(win[1], i);
                        i
                    }
                    None => {
                        dropped += 1;
                        continue;
                    }
                },
            };
            // Stamp the street name on both endpoints (first named way wins),
            // and record both as members of this street (for intersections).
            if wname_id != NO_NAME {
                if node_name[a_idx as usize] == NO_NAME {
                    node_name[a_idx as usize] = wname_id;
                }
                if node_name[b_idx as usize] == NO_NAME {
                    node_name[b_idx as usize] = wname_id;
                }
                let members = &mut street_nodes[wname_id as usize];
                members.push(a_idx);
                members.push(b_idx);
            }
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
        "[osm] edge construction: {:.2} s — {} edges ({} dropped) (weight = duration_s, parallel edge_dist_m)",
        t.elapsed().as_secs_f64(),
        edges.len(),
        dropped
    );

    drop(node_coords);

    let (g, edge_dist) = CsrGraph::from_edges_with_dist(coords.len(), &edges);
    debug_assert_eq!(node_name.len(), coords.len());
    let named = node_name.iter().filter(|&&id| id != crate::names::NO_NAME).count();
    println!(
        "[osm] CSR: n = {}, m = {}, avg_deg = {:.2} — {} distinct street names, {}/{} nodes named",
        g.n,
        g.m(),
        g.m() as f32 / g.n.max(1) as f32,
        name_pool.len(),
        named,
        g.n,
    );
    Ok(OsmParse {
        graph: g,
        coords,
        edge_dist,
        node_name,
        name_pool,
        street_nodes,
        addresses,
    })
}

#[derive(Clone, Copy)]
enum OneWay {
    No,
    Forward,
    Backward,
}

/// A drivable way as collected during parsing: node refs, direction, speed
/// (m/s) and the OSM `name=*` tag (the street name), used for geocoding.
type WayRec = (Vec<i64>, OneWay, f32, Option<Box<str>>);

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

#[cfg(test)]
mod tests {
    use super::clean_housenumber;

    #[test]
    fn semicolon_lists_become_ranges() {
        assert_eq!(clean_housenumber("4081;4083"), "4081\u{2013}4083");
        assert_eq!(clean_housenumber("2111;2113;2115"), "2111\u{2013}2115");
        assert_eq!(clean_housenumber("151"), "151");
        assert_eq!(clean_housenumber(" 42 "), "42");
        assert_eq!(clean_housenumber("5;5"), "5"); // dup collapses
        assert_eq!(clean_housenumber("42B"), "42B");
    }
}
