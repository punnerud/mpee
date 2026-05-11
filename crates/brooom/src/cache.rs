//! Solve-result disk cache keyed by a normalized fingerprint of the problem.
//!
//! The fingerprint is invariant under monotonic scaling of the duration
//! matrix — so two inputs that differ only by a constant multiplier (e.g.
//! durations expressed in seconds vs. milliseconds) collide on the same
//! cache entry. This is the "rainbow-table" use case: VRP optimal routes
//! are scale-invariant, so cached solutions transfer.
//!
//! Cache lives at the directory pointed to by `BROOOM_CACHE_DIR` (env) or
//! the explicit `--cache-dir` CLI flag. Each entry is `<fingerprint>.json`
//! holding the verbatim Vroom-style output. Cache misses fall through to a
//! normal solve.
//!
//! Activated only when the directory is set; default behavior is unchanged.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::embedding::{distance, ProblemEmbedding};
use crate::problem::Problem;

/// Compute a stable fingerprint string (16 hex chars) for a problem +
/// CLI-flag set. The matrix is normalized before hashing so proportional
/// matrices fingerprint identically.
pub fn fingerprint(problem: &Problem, flags: &[(&str, String)]) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();

    // Matrix: gather all positive durations, find GCD-like scale (min
    // positive), divide everything by it, round to integer. This makes
    // [[100, 200], [300, 400]] hash identically to [[1, 2], [3, 4]].
    let mut all_vals: Vec<i64> = Vec::new();
    for (_profile, m) in problem.matrices.iter() {
        for row in &m.durations {
            all_vals.extend(row.iter().copied());
        }
    }
    let min_pos = all_vals
        .iter()
        .copied()
        .filter(|&v| v > 0)
        .min()
        .unwrap_or(1);
    for v in &all_vals {
        let scaled = if min_pos > 0 { (*v * 1_000_000) / min_pos } else { *v };
        scaled.hash(&mut h);
    }

    // Problem structure: jobs, shipments, vehicles. Canonical-ish — relies
    // on Problem being deterministically iterable (it is — Vec ordering).
    problem.jobs.len().hash(&mut h);
    for j in &problem.jobs {
        j.id.hash(&mut h);
        j.delivery.hash(&mut h);
        j.pickup.hash(&mut h);
        j.priority.hash(&mut h);
        j.service.hash(&mut h);
        // Skills set — sort to be order-independent.
        let mut skills: Vec<u32> = j.skills.iter().copied().collect();
        skills.sort_unstable();
        skills.hash(&mut h);
        // Time windows.
        for tw in &j.time_windows {
            tw.start.hash(&mut h);
            tw.end.hash(&mut h);
        }
        // Location: prefer matrix index, fall back to coords.
        if let Some(idx) = j.location.index {
            idx.hash(&mut h);
        } else if let Some([lon, lat]) = j.location.coord {
            (lon.to_bits()).hash(&mut h);
            (lat.to_bits()).hash(&mut h);
        }
    }

    problem.vehicles.len().hash(&mut h);
    for v in &problem.vehicles {
        v.id.hash(&mut h);
        v.capacity.hash(&mut h);
        v.profile.hash(&mut h);
        v.time_window.as_ref().map(|tw| (tw.start, tw.end)).hash(&mut h);
    }

    problem.shipments.len().hash(&mut h);

    // CLI flags that influence the solve. Stable order via sort.
    let mut flags_sorted: Vec<&(&str, String)> = flags.iter().collect();
    flags_sorted.sort_by_key(|(k, _)| *k);
    for (k, v) in flags_sorted {
        k.hash(&mut h);
        v.hash(&mut h);
    }

    format!("{:016x}", h.finish())
}

/// Look up a cached solve output. Returns None on miss or any I/O error.
pub fn load(cache_dir: &Path, fingerprint: &str) -> Option<String> {
    let p = cache_dir.join(format!("{fingerprint}.json"));
    std::fs::read_to_string(p).ok()
}

/// Store a solve output to cache. Best-effort; errors are silently ignored.
pub fn store(cache_dir: &Path, fingerprint: &str, output: &str) {
    let _ = std::fs::create_dir_all(cache_dir);
    let p = cache_dir.join(format!("{fingerprint}.json"));
    let _ = std::fs::write(p, output);
}

/// Resolve the cache directory from CLI flag or env.
pub fn resolve_dir(explicit: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(PathBuf::from(p));
    }
    std::env::var("BROOOM_CACHE_DIR").ok().map(PathBuf::from)
}

/// Sidecar metadata persisted alongside each cached output. The output JSON
/// itself is at `<fp>.json`; the meta is at `<fp>.meta.json` so similarity
/// search can scan only metas (small) rather than reading every output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMeta {
    pub fingerprint: String,
    pub embedding: ProblemEmbedding,
    pub cost: f64,
    pub config: SerializedConfig,
}

/// CLI-flag snapshot. Independent of `SolverConfig` to avoid serializing
/// internal-only fields and to make on-disk format stable across solver
/// changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedConfig {
    pub max_passes: usize,
    pub granular_k: usize,
    pub multi_start: usize,
    pub ils_iters: usize,
    pub ils_kick_size: f64,
    pub time_limit_s: Option<f64>,
}

/// Persist meta alongside an output. Best-effort.
pub fn store_meta(cache_dir: &Path, fingerprint: &str, meta: &CacheMeta) {
    let _ = std::fs::create_dir_all(cache_dir);
    let p = cache_dir.join(format!("{fingerprint}.meta.json"));
    if let Ok(s) = serde_json::to_string(meta) {
        let _ = std::fs::write(p, s);
    }
}

/// Scan the cache directory and return all parsed meta entries.
pub fn list_meta(cache_dir: &Path) -> Vec<CacheMeta> {
    let Ok(entries) = std::fs::read_dir(cache_dir) else { return Vec::new(); };
    let mut out = Vec::new();
    for ent in entries.flatten() {
        let path = ent.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue; };
        if !name.ends_with(".meta.json") { continue; }
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Ok(meta) = serde_json::from_str::<CacheMeta>(&s) {
                out.push(meta);
            }
        }
    }
    out
}

/// Compute the median SolverConfig across a set of cached entries.
/// Median per-field is robust against outlier configs (one entry that ran
/// with extreme settings won't dominate the suggestion).
pub fn median_config(entries: &[CacheMeta]) -> Option<SerializedConfig> {
    if entries.is_empty() { return None; }
    let med_usize = |xs: &mut Vec<usize>| {
        xs.sort_unstable();
        xs[xs.len() / 2]
    };
    let med_f64 = |xs: &mut Vec<f64>| {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        xs[xs.len() / 2]
    };

    let mut max_passes: Vec<usize> = entries.iter().map(|e| e.config.max_passes).collect();
    let mut granular_k: Vec<usize> = entries.iter().map(|e| e.config.granular_k).collect();
    let mut multi_start: Vec<usize> = entries.iter().map(|e| e.config.multi_start).collect();
    let mut ils_iters: Vec<usize> = entries.iter().map(|e| e.config.ils_iters).collect();
    let mut ils_kick: Vec<f64> = entries.iter().map(|e| e.config.ils_kick_size).collect();

    Some(SerializedConfig {
        max_passes: med_usize(&mut max_passes),
        granular_k: med_usize(&mut granular_k),
        multi_start: med_usize(&mut multi_start),
        ils_iters: med_usize(&mut ils_iters),
        ils_kick_size: med_f64(&mut ils_kick),
        time_limit_s: None, // intentionally not transferred — wall-time is task-specific
    })
}

/// Search for the K nearest neighbors of `query` in the cache. Distances are
/// L2 over per-feature z-scored vectors (corpus-relative), so dimensions with
/// different scales contribute proportionally. Returns sorted ascending by
/// distance.
pub fn nearest(
    cache_dir: &Path,
    query: &ProblemEmbedding,
    k: usize,
) -> Vec<(f32, CacheMeta)> {
    let metas = list_meta(cache_dir);
    if metas.is_empty() { return Vec::new(); }

    // Build z-score stats from the corpus so log_n_jobs (range ~3..7) and
    // capacity_utilization (range 0..1) contribute on equal footing.
    let corpus: Vec<ProblemEmbedding> = metas.iter().map(|m| m.embedding.clone()).collect();
    let stats = crate::embedding::CorpusStats::from_corpus(&corpus);

    let mut scored: Vec<(f32, CacheMeta)> = metas
        .into_iter()
        .map(|m| (distance(query, &m.embedding, Some(&stats)), m))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}
