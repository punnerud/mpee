//! In-memory sub-route pattern database for RAG-style lookup.
//!
//! Built from BKS solutions via `benchmarks/extract_subroutes.py`:
//! one JSONL line per sliding-window sub-route, each with a canonical
//! geometric signature (translation/rotation/scale invariant) plus
//! context (customer ids, demand sum, time-window envelope).
//!
//! Loaded once at startup; queries are O(N) linear scans over the
//! corpus. For our N=200 corpus (≈10K patterns × 6-d signature) this
//! costs ≈100µs per query — well below the multi-ms LS step that
//! would consume the result.
//!
//! ## Usage
//! ```ignore
//! let db = PatternDb::load_jsonl("benchmarks/gh_canonical/subroutes_n200_k4.jsonl")?;
//! let neighbors = db.knn(&query_signature, 5);
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubroutePattern {
    pub instance: String,
    pub vehicle: usize,
    pub window_start: usize,
    /// Original customer ids in source-instance ordering.
    pub customers: Vec<usize>,
    /// Translation/rotation/scale invariant geometry signature.
    /// Length = 2 * (window_size - 1) − 2 = 2K−4 for K-window.
    pub signature: Vec<f32>,
    pub demand_sum: i64,
    pub tw_min: i64,
    pub tw_max: i64,
}

/// Threshold above which we build a KD-tree index. Below this, linear
/// scan is faster (cache-friendly contiguous Vec, no tree overhead).
/// Picked as the crossover where O(N) starts to dominate solve time at
/// typical query rates (~1000 queries/solve).
const KD_TREE_THRESHOLD: usize = 50_000;

/// KD-tree over signatures. Stores indices into the parent's flat
/// `signatures` array, partitioned in-place by median-split so that at
/// recursion depth d, indices[mid] is the median along dim (d % sig_dim).
struct KdTree {
    indices: Vec<u32>,
}

/// In-memory index over sub-route patterns. The signature is stored
/// flat (`signatures: Vec<f32>`) for cache locality during scan; the
/// rest of the metadata stays in `patterns`.
pub struct PatternDb {
    pub sig_dim: usize,
    /// Flat (n × sig_dim) row-major signatures.
    signatures: Vec<f32>,
    pub patterns: Vec<SubroutePattern>,
    /// KD-tree built when corpus exceeds KD_TREE_THRESHOLD. None means
    /// linear scan is used (faster for small N, cache-friendly).
    kd: Option<KdTree>,
}

impl PatternDb {
    /// Load from a JSONL file (one pattern per line). All patterns must
    /// share the same signature dimension.
    pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Self, Error> {
        let f = File::open(path.as_ref()).map_err(|e| Error::Other(format!("pattern_db: open: {e}")))?;
        let reader = BufReader::new(f);
        let mut patterns: Vec<SubroutePattern> = Vec::new();
        let mut sig_dim: Option<usize> = None;
        for (lineno, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| Error::Other(format!("pattern_db: read: {e}")))?;
            if line.trim().is_empty() {
                continue;
            }
            let p: SubroutePattern = serde_json::from_str(&line)
                .map_err(|e| Error::Other(format!("pattern_db: parse line {}: {e}", lineno + 1)))?;
            match sig_dim {
                None => sig_dim = Some(p.signature.len()),
                Some(d) if d != p.signature.len() => {
                    return Err(Error::Other(format!(
                        "pattern_db: signature dim mismatch at line {}: {} vs {}",
                        lineno + 1,
                        p.signature.len(),
                        d
                    )));
                }
                _ => {}
            }
            patterns.push(p);
        }
        let sig_dim = sig_dim.unwrap_or(0);
        let mut signatures = Vec::with_capacity(patterns.len() * sig_dim);
        for p in &patterns {
            signatures.extend_from_slice(&p.signature);
        }
        let kd = if patterns.len() > KD_TREE_THRESHOLD && sig_dim > 0 {
            Some(KdTree::build(&signatures, patterns.len(), sig_dim))
        } else {
            None
        };
        Ok(Self { sig_dim, signatures, patterns, kd })
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Squared L2 distance between query and the i-th stored signature.
    fn sq_dist(&self, query: &[f32], i: usize) -> f32 {
        let off = i * self.sig_dim;
        let row = &self.signatures[off..off + self.sig_dim];
        let mut s = 0.0f32;
        for k in 0..self.sig_dim {
            let d = query[k] - row[k];
            s += d * d;
        }
        s
    }

    /// Returns the K nearest patterns by Euclidean distance over the
    /// signature, sorted ascending. Uses KD-tree when corpus is large
    /// (`KD_TREE_THRESHOLD`), otherwise linear scan (cache-friendly for
    /// small N).
    pub fn knn(&self, query: &[f32], k: usize) -> Vec<(f32, &SubroutePattern)> {
        if query.len() != self.sig_dim || self.patterns.is_empty() {
            return Vec::new();
        }
        if let Some(kd) = self.kd.as_ref() {
            return self.knn_kdtree(kd, query, k);
        }
        self.knn_linear(query, k)
    }

    fn knn_linear(&self, query: &[f32], k: usize) -> Vec<(f32, &SubroutePattern)> {
        let mut scored: Vec<(f32, usize)> = (0..self.patterns.len())
            .map(|i| (self.sq_dist(query, i), i))
            .collect();
        let topk = k.min(scored.len());
        scored.select_nth_unstable_by(topk.saturating_sub(1), |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top: Vec<(f32, &SubroutePattern)> = scored[..topk]
            .iter()
            .map(|(d, i)| (d.sqrt(), &self.patterns[*i]))
            .collect();
        top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        top
    }

    fn knn_kdtree(&self, kd: &KdTree, query: &[f32], k: usize) -> Vec<(f32, &SubroutePattern)> {
        // top: kept sorted ascending by squared distance, max k entries.
        let mut top: Vec<(f32, u32)> = Vec::with_capacity(k.max(1));
        kd.search(&kd.indices, query, k, &mut top, &self.signatures, self.sig_dim, 0);
        top.into_iter()
            .map(|(d, i)| (d.sqrt(), &self.patterns[i as usize]))
            .collect()
    }
}

impl KdTree {
    fn build(signatures: &[f32], n: usize, sig_dim: usize) -> Self {
        let mut indices: Vec<u32> = (0..n as u32).collect();
        Self::build_recursive(&mut indices, signatures, sig_dim, 0);
        Self { indices }
    }

    /// In-place median-split partition. After this, indices[mid] is the
    /// median along dim (depth % sig_dim), all entries to the left have
    /// values ≤ median on that dim, and all to the right have ≥.
    fn build_recursive(indices: &mut [u32], sigs: &[f32], sig_dim: usize, depth: usize) {
        if indices.len() <= 1 {
            return;
        }
        let split_dim = depth % sig_dim;
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            let av = sigs[a as usize * sig_dim + split_dim];
            let bv = sigs[b as usize * sig_dim + split_dim];
            av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
        });
        let (left, right_with_mid) = indices.split_at_mut(mid);
        let right = &mut right_with_mid[1..];
        Self::build_recursive(left, sigs, sig_dim, depth + 1);
        Self::build_recursive(right, sigs, sig_dim, depth + 1);
    }

    /// Standard KD-tree KNN with backtracking. `top` is kept sorted
    /// ascending by squared distance, capped at k entries.
    fn search(
        &self,
        indices: &[u32],
        query: &[f32],
        k: usize,
        top: &mut Vec<(f32, u32)>,
        sigs: &[f32],
        sig_dim: usize,
        depth: usize,
    ) {
        if indices.is_empty() {
            return;
        }
        let mid = indices.len() / 2;
        let median_idx = indices[mid];
        // Compute squared distance from query to median, possibly insert.
        let off = median_idx as usize * sig_dim;
        let mut d = 0.0f32;
        for k_ in 0..sig_dim {
            let diff = query[k_] - sigs[off + k_];
            d += diff * diff;
        }
        Self::insert_top(top, k, (d, median_idx));

        // Visit the side closer to the query first; that gives us a
        // tight bound for pruning the far side.
        let split_dim = depth % sig_dim;
        let split_val = sigs[off + split_dim];
        let q_diff = query[split_dim] - split_val;

        let (near, far) = if q_diff < 0.0 {
            (&indices[..mid], &indices[mid + 1..])
        } else {
            (&indices[mid + 1..], &indices[..mid])
        };

        self.search(near, query, k, top, sigs, sig_dim, depth + 1);

        // Visit far side only if it could contain a closer point. The
        // squared perpendicular distance to the split plane is q_diff².
        let bound = if top.len() < k {
            f32::INFINITY
        } else {
            top.last().map(|&(d, _)| d).unwrap_or(f32::INFINITY)
        };
        if q_diff * q_diff < bound {
            self.search(far, query, k, top, sigs, sig_dim, depth + 1);
        }
    }

    /// Insert into `top` keeping it sorted ascending by distance, capped
    /// at k entries. Linear scan — fine for small k (typical 5–32).
    fn insert_top(top: &mut Vec<(f32, u32)>, k: usize, item: (f32, u32)) {
        if top.len() == k && top.last().map(|&(d, _)| d <= item.0).unwrap_or(false) {
            return; // worse than the worst we keep — skip.
        }
        let pos = top.partition_point(|&(d, _)| d < item.0);
        top.insert(pos, item);
        if top.len() > k {
            top.truncate(k);
        }
    }
}
