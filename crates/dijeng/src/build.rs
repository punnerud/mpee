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

use crate::osm_profile::Profile;

/// Paths written (or reused) and graph size, returned by [`build_cache`].
pub struct BuildResult {
    pub csr_path: PathBuf,
    pub pp_path: PathBuf,
    pub ch_path: PathBuf,
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
    profile: Profile,
    progress: bool,
    force: bool,
    keep_csr: bool,
) -> Result<BuildResult, String> {
    use crate::{cache_ch, cache_pp, ch, osm, preprocess::preprocess};

    let suffix = if profile == Profile::Car {
        String::new()
    } else {
        format!(".{}", profile.name())
    };
    let base = format!("{}{}", pbf.to_string_lossy(), suffix);
    let csr_path = PathBuf::from(format!("{base}.csr"));
    let pp_path = PathBuf::from(format!("{base}.pp"));
    let ch_path = PathBuf::from(format!("{base}.ch"));

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
            nodes,
            edges: 0,
            build_secs: 0.0,
            cached: true,
        });
    }

    // Suppress the engine's progress chatter when not wanted.
    let _silencer = StdoutSilencer::new(!progress);

    let (graph, coords, edge_dist) = osm::load_with_cache(pbf, csr_path.as_path(), profile)
        .map_err(|e| format!("osm load: {e}"))?;

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
    for old in 0..graph.n {
        new_coords[pre.new_id[old] as usize] = coords[old];
    }
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

    // The .csr is only a PBF-parse accelerator for rebuilds; routing/solve use
    // .pp + .ch. Drop it by default to keep the on-disk footprint small.
    if !keep_csr {
        let _ = std::fs::remove_file(&csr_path);
    }

    Ok(BuildResult {
        csr_path,
        pp_path,
        ch_path,
        nodes: pre.graph.n,
        edges: pre.graph.m(),
        build_secs,
        cached: false,
    })
}
