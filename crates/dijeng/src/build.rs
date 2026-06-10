//! One-call cache build: an OSM `.pbf` → routable `.pp` + `.ch` caches, run
//! entirely **in-process** (no subprocess, no `cargo run`, no recompilation).
//!
//! Shared by the `mpee` CLI (`mpee build`) and the Python `Router.build`, so
//! neither shells out nor duplicates the pipeline. Cache names APPEND to the
//! full PBF path (matching bench_pp/bench_ch and the `--cache <pbf>`
//! convention): `data/x.osm.pbf` → `data/x.osm.pbf.{csr,pp,ch}`.
//!
//! Two knobs make it friendly to call from a library / an agent:
//!   * `progress` — when false, the engine's stdout progress chatter is
//!     suppressed (default on for an interactive terminal).
//!   * `force`    — when false, an existing `.pp` + `.ch` pair is reused
//!     instead of rebuilt, so repeated calls return instantly.

use std::path::{Path, PathBuf};

use crate::osm_profile::{Profile, ProfileSpec};

/// Paths written (or reused) and graph size, returned by [`build_cache`].
pub struct BuildResult {
    pub csr_path: PathBuf,
    pub pp_path: PathBuf,
    pub ch_path: PathBuf,
    /// Street-name sidecar for offline geocoding (written on a full build).
    pub names_path: PathBuf,
    /// House-number address sidecar for offline address geocoding.
    pub addr_path: PathBuf,
    pub nodes: usize,
    pub edges: usize,
    /// Wall time of the CH-contraction step, in seconds (0.0 when reused).
    pub build_secs: f64,
    /// True if an existing cache was reused instead of rebuilt.
    pub cached: bool,
}

/// RAII guard that redirects this process's stdout to /dev/null while alive,
/// used to silence the engine's progress `println!`s when `progress=false`.
/// A no-op when `active` is false.
struct StdoutSilencer {
    saved_fd: Option<i32>,
}

impl StdoutSilencer {
    fn new(active: bool) -> Self {
        if !active {
            return Self { saved_fd: None };
        }
        // Flush Rust's stdout buffer first so nothing already queued leaks.
        use std::io::Write;
        let _ = std::io::stdout().flush();
        unsafe {
            let saved = libc::dup(1);
            let devnull = libc::open(b"/dev/null\0".as_ptr() as *const libc::c_char, libc::O_WRONLY);
            if saved >= 0 && devnull >= 0 {
                libc::dup2(devnull, 1);
                libc::close(devnull);
                return Self { saved_fd: Some(saved) };
            }
            if saved >= 0 {
                libc::close(saved);
            }
        }
        Self { saved_fd: None }
    }
}

impl Drop for StdoutSilencer {
    fn drop(&mut self) {
        if let Some(saved) = self.saved_fd {
            use std::io::Write;
            let _ = std::io::stdout().flush();
            unsafe {
                libc::dup2(saved, 1);
                libc::close(saved);
            }
        }
    }
}

/// Build (or reuse) `.csr` + `.pp` + `.ch` next to `pbf` for the given routing
/// profile. Runs entirely in-process.
///
/// * `progress` — print the engine's parse/CH progress to stdout.
/// * `force`    — rebuild even if a cache already exists.
/// * `keep_csr` — keep the intermediate `.csr` (a PBF-parse cache that speeds
///   up future rebuilds). Routing/solve never need it, so it is deleted by
///   default to save disk (e.g. ~540 MB for Norway).
pub fn build_cache(
    pbf: &Path,
    profile: impl Into<ProfileSpec>,
    progress: bool,
    force: bool,
    keep_csr: bool,
) -> Result<BuildResult, String> {
    build_cache_dem(pbf, profile, progress, force, keep_csr, None)
}

/// `build_cache` with an optional DEM directory of SRTM `.hgt` tiles. When
/// given: every node's elevation is sampled into a `.elev` sidecar, and for
/// gravity-sensitive profiles (anything except builtin car/motorcycle) edge
/// travel times get the grade factor from `elevation::grade_time_factor` —
/// hill-aware bicycle/foot routing. NOTE: cache reuse does not know whether
/// an existing cache was DEM-built; pass `force = true` when switching.
pub fn build_cache_dem(
    pbf: &Path,
    profile: impl Into<ProfileSpec>,
    progress: bool,
    force: bool,
    keep_csr: bool,
    dem: Option<&Path>,
) -> Result<BuildResult, String> {
    use crate::{cache, cache_ch, cache_pp, ch, names, osm, preprocess::preprocess};

    let profile: ProfileSpec = profile.into();
    let suffix = if matches!(profile, ProfileSpec::Builtin(Profile::Car)) {
        String::new()
    } else {
        format!(".{}", profile.name())
    };
    let base = format!("{}{}", pbf.to_string_lossy(), suffix);
    let csr_path = PathBuf::from(format!("{base}.csr"));
    let pp_path = PathBuf::from(format!("{base}.pp"));
    let ch_path = PathBuf::from(format!("{base}.ch"));
    let names_path = PathBuf::from(format!("{base}.names"));
    let addr_path = PathBuf::from(format!("{base}.addr"));

    // Reuse an existing cache unless asked to rebuild — repeated library calls
    // (e.g. from an agent) then return instantly instead of re-contracting.
    if !force && pp_path.is_file() && ch_path.is_file() {
        let nodes = cache_pp::load_mmap(&pp_path)
            .map(|p| p.coords.as_slice().len())
            .unwrap_or(0);
        return Ok(BuildResult {
            csr_path,
            pp_path,
            ch_path,
            names_path,
            addr_path,
            nodes,
            edges: 0,
            build_secs: 0.0,
            cached: true,
        });
    }

    // Suppress the engine's progress chatter when not wanted.
    let _silencer = StdoutSilencer::new(!progress);

    // Parse directly (not via load_with_cache) so we reconstruct street names
    // in the same pass. A fresh build therefore always re-parses rather than
    // reading an existing `.csr`; the `.csr` is only re-written when keep_csr
    // is set, as a parse accelerator for other tools.
    let osm::OsmParse {
        graph,
        coords,
        edge_dist,
        node_name,
        name_pool,
        street_nodes,
        addresses,
    } = osm::load_osm_routing_par(pbf, profile.clone()).map_err(|e| format!("osm load: {e}"))?;
    if keep_csr {
        if let Err(e) = cache::save(&csr_path, &graph, &coords, edge_dist.as_slice()) {
            eprintln!("[build] failed to write .csr accelerator: {e}");
        }
    }

    // DEM: per-node elevations (for the .elev sidecar) and, for gravity-
    // sensitive profiles, grade-adjusted edge times. Uses the parser's
    // per-edge road distances, so no re-derivation of geometry.
    let mut graph = graph;
    let mut elev_old: Option<Vec<f32>> = None;
    if let Some(dem_dir) = dem {
        let dem = crate::elevation::Dem::open(dem_dir);
        let elev: Vec<f32> = coords
            .iter()
            .map(|&(la, lo)| dem.sample(la as f64, lo as f64).unwrap_or(f32::NAN))
            .collect();
        let covered = elev.iter().filter(|v| v.is_finite()).count();
        if progress {
            println!(
                "[build] DEM: {covered}/{} nodes covered ({:.1}%)",
                elev.len(),
                100.0 * covered as f64 / elev.len().max(1) as f64
            );
        }
        let apply_grade = !matches!(
            profile,
            ProfileSpec::Builtin(Profile::Car) | ProfileSpec::Builtin(Profile::Motorcycle)
        );
        if apply_grade && !edge_dist.is_empty() {
            let mut w: Vec<f32> = graph.edge_w.to_vec();
            for u in 0..graph.n {
                let s = graph.head[u] as usize;
                let e = graph.head[u + 1] as usize;
                for k in s..e {
                    let v = graph.edge_to[k] as usize;
                    let delta = elev[v] - elev[u];
                    w[k] *= crate::elevation::grade_time_factor(delta, edge_dist[k]);
                }
            }
            graph = crate::graph::CsrGraph {
                n: graph.n,
                head: graph.head,
                edge_to: graph.edge_to,
                edge_w: w.into(),
            };
        }
        elev_old = Some(elev);
    }

    // delta = average edge weight / average degree (matches bench_pp / iOS).
    let avg_w = if graph.edge_w.is_empty() {
        1.0
    } else {
        let stride = (graph.edge_w.len() / 4096).max(1);
        let (mut s, mut c, mut i) = (0.0f64, 0u64, 0usize);
        while i < graph.edge_w.len() {
            s += graph.edge_w[i] as f64;
            c += 1;
            i += stride;
        }
        (s / c.max(1) as f64) as f32
    };
    let avg_deg = graph.m() as f32 / graph.n.max(1) as f32;
    let delta = (avg_w / avg_deg).max(1e-4);

    let pre = preprocess(&graph, Some(delta), edge_dist.as_slice());
    let (reverse, rev_edge_dist) =
        crate::bidir::transpose_with_dist(&pre.graph, pre.edge_dist.as_slice());
    let mut new_coords = vec![(0.0f32, 0.0f32); graph.n];
    // Street names follow the same node permutation as coords, so the sidecar
    // stays aligned with the `.pp` coords (and hence the snap grid).
    let mut new_name_id = vec![names::NO_NAME; graph.n];
    for old in 0..graph.n {
        new_coords[pre.new_id[old] as usize] = coords[old];
        new_name_id[pre.new_id[old] as usize] = node_name[old];
    }

    // Per-street node lists for intersection search → a CSR keyed by street id.
    // Remap each node to the new order, sort + dedup so two streets' lists can
    // be merge-intersected at query time. Built in the same pass; no re-parse.
    let k = name_pool.len();
    let mut street_offsets = vec![0u32; k + 1];
    let mut street_node_flat: Vec<u32> = Vec::new();
    for (sid, nodes) in street_nodes.iter().enumerate() {
        street_offsets[sid] = street_node_flat.len() as u32;
        let mut remapped: Vec<u32> = nodes.iter().map(|&old| pre.new_id[old as usize]).collect();
        remapped.sort_unstable();
        remapped.dedup();
        street_node_flat.extend_from_slice(&remapped);
    }
    street_offsets[k] = street_node_flat.len() as u32;

    cache_pp::save(
        pp_path.as_path(),
        &pre.graph,
        &reverse,
        &pre.light_count,
        &pre.new_id,
        &new_coords,
        delta,
        pre.edge_dist.as_slice(),
        &rev_edge_dist,
    )
    .map_err(|e| format!("pp save: {e}"))?;

    let t = std::time::Instant::now();
    let h = ch::build_with_dist(&pre.graph, pre.edge_dist.as_slice());
    let build_secs = t.elapsed().as_secs_f64();
    cache_ch::save(ch_path.as_path(), &h).map_err(|e| format!("ch save: {e}"))?;

    // Street-name + intersection sidecar for offline geocoding (deletable when
    // only routing is needed — it never affects route/solve).
    if let Err(e) = names::save(&names_path, &new_name_id, &name_pool, &street_offsets, &street_node_flat) {
        eprintln!("[build] failed to write names sidecar: {e}");
    }

    // Elevation sidecar, permuted to the same node order as coords/.pp.
    if let Some(elev) = &elev_old {
        let mut new_elev = vec![f32::NAN; graph.n];
        for old in 0..graph.n {
            new_elev[pre.new_id[old] as usize] = elev[old];
        }
        let elev_path = PathBuf::from(format!("{base}.elev"));
        if let Err(e) = crate::elevation::save_elev(&elev_path, &new_elev) {
            eprintln!("[build] failed to write elevation sidecar: {e}");
        } else if progress {
            println!("[build] elevation sidecar: {}", elev_path.display());
        }
    }

    // House-number address sidecar (independent of the routing graph — its own
    // coords + grid, so no node permutation applies). Deletable like .names.
    match crate::addresses::save(&addr_path, &addresses) {
        Ok(n) if progress => println!("[build] address sidecar: {n} points → {}", addr_path.display()),
        Ok(_) => {}
        Err(e) => eprintln!("[build] failed to write address sidecar: {e}"),
    }

    // The .csr is only a PBF-parse accelerator for rebuilds; routing/solve use
    // .pp + .ch. Drop any stale one by default to keep the footprint small.
    if !keep_csr {
        let _ = std::fs::remove_file(&csr_path);
    }

    Ok(BuildResult {
        csr_path,
        pp_path,
        ch_path,
        names_path,
        addr_path,
        nodes: pre.graph.n,
        edges: pre.graph.m(),
        build_secs,
        cached: false,
    })
}
