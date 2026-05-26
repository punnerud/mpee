//! One-call cache build: an OSM `.pbf` → routable `.pp` + `.ch` caches, run
//! entirely **in-process** (no subprocess, no `cargo run`, no recompilation).
//!
//! Shared by the `mpee` CLI (`mpee build`) and the Python `Router.build`, so
//! neither shells out nor duplicates the pipeline. Cache names APPEND to the
//! full PBF path (matching bench_pp/bench_ch and the `--cache <pbf>`
//! convention): `data/x.osm.pbf` → `data/x.osm.pbf.{csr,pp,ch}`.

use std::path::{Path, PathBuf};

use crate::osm_profile::Profile;

/// Paths written and graph size, returned by [`build_cache`].
pub struct BuildResult {
    pub csr_path: PathBuf,
    pub pp_path: PathBuf,
    pub ch_path: PathBuf,
    pub nodes: usize,
    pub edges: usize,
    /// Wall time of the CH-contraction step (the heavy phase), in seconds.
    pub build_secs: f64,
}

/// Build `.csr` + `.pp` + `.ch` next to `pbf` for the given routing profile.
/// Returns the written paths and graph size, or a human-readable error.
pub fn build_cache(pbf: &Path, profile: Profile) -> Result<BuildResult, String> {
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

    let (graph, coords, edge_dist) = osm::load_with_cache(pbf, csr_path.as_path(), profile)
        .map_err(|e| format!("osm load: {e}"))?;

    // delta = average edge weight / average degree (matches the standalone
    // bench_pp / iOS build paths).
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

    Ok(BuildResult {
        csr_path,
        pp_path,
        ch_path,
        nodes: pre.graph.n,
        edges: pre.graph.m(),
        build_secs,
    })
}
