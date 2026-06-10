//! Map matching: reconstruct the road path a noisy GPS trace was driven on
//! (Newson & Krumm 2009 — the HMM formulation OSRM and Valhalla use).
//!
//! Trellis: layer t = the K nearest road nodes to GPS point t.
//!   * Emission  — Gaussian in the great-circle distance point → candidate
//!     (σ = GPS noise, metres).
//!   * Transition — exponential in |route_distance − great_circle_distance|
//!     between consecutive candidates: a matched path should progress along
//!     the road roughly as far as the raw trace moved. Route distances for a
//!     whole K×K layer pair come from one tiny bucket-MMM call.
//!   * Viterbi picks the jointly most probable candidate sequence.
//!
//! Our 20 µs p2p machinery is what makes the inner loop cheap; a typical
//! trace point costs one K×K matrix (sub-millisecond) plus K emissions.
//!
//! Gaps: when no candidate of layer t can be reached from layer t-1 (after a
//! ferry, a tunnel with no signal, a teleporting trace), the trellis restarts
//! at t and the output marks a discontinuity rather than failing the whole
//! trace.

use crate::ch::{self, ContractionHierarchy};

/// One matched trace point.
pub struct MatchedPoint {
    /// Snapped road node (CSR id).
    pub node: u32,
    /// Snapped coordinate.
    pub lat: f32,
    pub lon: f32,
    /// Great-circle distance from the input point to the match (metres).
    pub snap_distance_m: f32,
    /// False when the HMM had to restart here (unreachable from the previous
    /// point) — the path is discontinuous at this index.
    pub connected: bool,
}

pub struct MatchResult {
    pub points: Vec<MatchedPoint>,
    /// Mean emission weight in [0, 1] — a rough confidence (1 = every point
    /// exactly on the matched road).
    pub confidence: f32,
}

/// Match `trace` ((lat, lon) per ping) against the road graph.
///
/// * `candidates_of(lat, lon)` — K nearest road nodes (csr id, lat, lon).
/// * `to_internal(csr)` — CH-internal id (the `perm` mapping).
/// * `sigma_m` — GPS noise σ in metres (15 is a good urban default).
pub fn match_trace(
    ch: &ContractionHierarchy,
    candidates_of: impl Fn(f32, f32) -> Vec<(u32, f32, f32)>,
    to_internal: impl Fn(u32) -> u32,
    trace: &[(f32, f32)],
    sigma_m: f32,
) -> MatchResult {
    let sigma = sigma_m.max(1.0);
    // Newson-Krumm β: scale of tolerated |route − great-circle| differences.
    let beta = (sigma * 3.0).max(20.0);
    let has_dist = ch.edge_dist_fwd.len() == ch.graph_fwd.m();
    // Without a distance channel, compare durations against an expected
    // duration at a nominal urban speed instead.
    const NOMINAL_MPS: f32 = 13.9; // ~50 km/h

    let mut out: Vec<MatchedPoint> = Vec::with_capacity(trace.len());
    let mut confidence_sum = 0.0f64;

    // Trellis state for the previous layer.
    struct Layer {
        cands: Vec<(u32, f32, f32)>, // csr, lat, lon
        internals: Vec<u32>,
        score: Vec<f64>,                 // best log-prob to each candidate
        backptr: Vec<Vec<usize>>,        // per layer: predecessor index per candidate
        start: usize,                    // trace index where this segment began
    }
    let mut layer: Option<Layer> = None;
    // Finished-segment flusher: walk backpointers, emit matched points.
    let flush =
        |layer: &Layer, upto_choice: usize, out: &mut Vec<MatchedPoint>, trace: &[(f32, f32)]| {
            let depth = layer.backptr.len();
            let mut choices = vec![0usize; depth + 1];
            choices[depth] = upto_choice;
            for d in (0..depth).rev() {
                choices[d] = layer.backptr[d][choices[d + 1]];
            }
            // (cands of intermediate layers were captured at push time)
            let _ = trace;
            choices
        };
    // We need each layer's candidates to emit points after backtracking, so
    // keep them per segment.
    let mut seg_layers: Vec<Vec<(u32, f32, f32)>> = Vec::new();

    let mut flush_segment =
        |layer: &Option<Layer>, seg_layers: &mut Vec<Vec<(u32, f32, f32)>>, out: &mut Vec<MatchedPoint>, trace: &[(f32, f32)], confidence_sum: &mut f64| {
            let Some(l) = layer else { return };
            // Best final candidate.
            let best = l
                .score
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let choices = flush(l, best, out, trace);
            for (d, &ci) in choices.iter().enumerate() {
                let (node, la, lo) = seg_layers[d][ci];
                let t_idx = l.start + d;
                let (pla, plo) = trace[t_idx];
                let snap = haversine_m(pla, plo, la, lo);
                *confidence_sum += (-0.5 * (snap / sigma_m.max(1.0)).powi(2)).exp() as f64;
                out.push(MatchedPoint {
                    node,
                    lat: la,
                    lon: lo,
                    snap_distance_m: snap,
                    connected: d > 0 || out.is_empty(),
                });
            }
            seg_layers.clear();
        };

    for (t, &(pla, plo)) in trace.iter().enumerate() {
        let cands = candidates_of(pla, plo);
        if cands.is_empty() {
            continue;
        }
        let emissions: Vec<f64> = cands
            .iter()
            .map(|&(_, la, lo)| {
                let d = haversine_m(pla, plo, la, lo);
                -0.5 * ((d / sigma) as f64).powi(2)
            })
            .collect();
        let internals: Vec<u32> = cands.iter().map(|&(c, _, _)| to_internal(c)).collect();

        layer = Some(match layer.take() {
            None => {
                seg_layers.push(cands.clone());
                Layer {
                    cands,
                    internals,
                    score: emissions,
                    backptr: Vec::new(),
                    start: t,
                }
            }
            Some(prev) => {
                // Route metric between every (prev, cur) candidate pair.
                let (dur, dist) = ch::matrix_with_dist(ch, &prev.internals, &internals);
                let gc = {
                    let (a, b) = trace[t - 1];
                    haversine_m(a, b, pla, plo)
                };
                let np = prev.cands.len();
                let nc = cands.len();
                let mut score = vec![f64::NEG_INFINITY; nc];
                let mut bp = vec![0usize; nc];
                for j in 0..nc {
                    for i in 0..np {
                        let route_m = if has_dist {
                            dist[i * nc + j]
                        } else {
                            dur[i * nc + j] * NOMINAL_MPS
                        };
                        if !route_m.is_finite() {
                            continue;
                        }
                        let trans = -((route_m - gc).abs() as f64) / beta as f64;
                        let s = prev.score[i] + trans + emissions[j];
                        if s > score[j] {
                            score[j] = s;
                            bp[j] = i;
                        }
                    }
                }
                if score.iter().all(|s| !s.is_finite()) {
                    // Gap: nothing reachable. Flush the finished segment and
                    // restart the trellis here.
                    flush_segment(&Some(prev), &mut seg_layers, &mut out, trace, &mut confidence_sum);
                    if let Some(last) = out.last_mut() {
                        let _ = last; // segment boundary is marked on the NEXT point
                    }
                    seg_layers.push(cands.clone());
                    let mut l = Layer {
                        cands,
                        internals,
                        score: emissions,
                        backptr: Vec::new(),
                        start: t,
                    };
                    // Mark discontinuity on the first point of the new segment
                    // when it gets flushed: handled via `connected` (d > 0 ||
                    // out.is_empty()) — the first flushed point of a non-first
                    // segment gets connected = false.
                    l.backptr.clear();
                    l
                } else {
                    seg_layers.push(cands.clone());
                    let mut prev = prev;
                    prev.backptr.push(bp);
                    Layer {
                        cands,
                        internals,
                        score,
                        backptr: std::mem::take(&mut prev.backptr),
                        start: prev.start,
                    }
                }
            }
        });
    }
    flush_segment(&layer, &mut seg_layers, &mut out, trace, &mut confidence_sum);

    let confidence = if out.is_empty() {
        0.0
    } else {
        (confidence_sum / out.len() as f64) as f32
    };
    MatchResult { points: out, confidence }
}

fn haversine_m(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let r = 6_371_000.0_f64;
    let l1 = (lat1 as f64).to_radians();
    let l2 = (lat2 as f64).to_radians();
    let dlat = (lat2 as f64 - lat1 as f64).to_radians();
    let dlon = (lon2 as f64 - lon1 as f64).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + l1.cos() * l2.cos() * (dlon / 2.0).sin().powi(2);
    (r * 2.0 * a.sqrt().asin()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CsrGraph;

    /// Two parallel east-west roads 60 m apart, connected at both ends.
    /// Nodes 0..10 = road A (lat 51.5000), 11..21 = road B (lat 51.50054),
    /// ~50 m spacing along longitude.
    fn parallel_roads() -> (crate::ch::ContractionHierarchy, Vec<(f32, f32)>) {
        let n = 22;
        let lat_a = 51.5000f32;
        let lat_b = 51.50054f32; // ≈60 m north
        let lon0 = -0.1300f32;
        let dlon = 0.00072f32; // ≈50 m at this latitude
        let mut coords = Vec::with_capacity(n);
        for i in 0..11 {
            coords.push((lat_a, lon0 + i as f32 * dlon));
        }
        for i in 0..11 {
            coords.push((lat_b, lon0 + i as f32 * dlon));
        }
        // Bidirectional edges along each road + the two end connectors.
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for i in 0..10u32 {
            edges.push((i, i + 1));
            edges.push((i + 1, i));
            edges.push((11 + i, 12 + i));
            edges.push((12 + i, 11 + i));
        }
        for &(a, b) in &[(0u32, 11u32), (10u32, 21u32)] {
            edges.push((a, b));
            edges.push((b, a));
        }
        // CSR with weight = travel seconds at ~14 m/s, dist = metres.
        let mut head = vec![0u32; n + 1];
        for &(a, _) in &edges {
            head[a as usize + 1] += 1;
        }
        for i in 0..n {
            head[i + 1] += head[i];
        }
        let m = edges.len();
        let mut edge_to = vec![0u32; m];
        let mut edge_w = vec![0f32; m];
        let mut cursor = head.clone();
        let mut edge_dist = vec![0f32; m];
        for &(a, b) in &edges {
            let (la, lo) = coords[a as usize];
            let (lb, lob) = coords[b as usize];
            let d = haversine_m(la, lo, lb, lob);
            let k = cursor[a as usize] as usize;
            cursor[a as usize] += 1;
            edge_to[k] = b;
            edge_w[k] = d / 14.0;
            edge_dist[k] = d;
        }
        let g = CsrGraph { n, head: head.into(), edge_to: edge_to.into(), edge_w: edge_w.into() };
        let h = crate::ch::build_with_dist(&g, &edge_dist);
        (h, coords)
    }

    #[test]
    fn sticks_to_one_road_despite_outlier() {
        let (h, coords) = parallel_roads();
        let cand = |la: f32, lo: f32| -> Vec<(u32, f32, f32)> {
            // Brute-force K=4 nearest for the test.
            let mut v: Vec<(f32, u32)> = coords
                .iter()
                .enumerate()
                .map(|(i, &(a, b))| (haversine_m(la, lo, a, b), i as u32))
                .collect();
            v.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            v.truncate(4);
            v.into_iter().map(|(_, i)| (i, coords[i as usize].0, coords[i as usize].1)).collect()
        };
        // Trace along road A with small noise; point 3 is an outlier 35 m
        // north (closer to road B than to road A).
        let lat_a = 51.5000f32;
        let lon0 = -0.1300f32;
        let dlon = 0.00072f32;
        let mut trace: Vec<(f32, f32)> = (0..8)
            .map(|i| (lat_a + 0.00003, lon0 + i as f32 * dlon))
            .collect();
        trace[3].0 = lat_a + 0.000315; // ≈35 m north of A, ≈25 m south of B
        let res = match_trace(&h, cand, |c| h.perm[c as usize], &trace, 15.0);
        assert_eq!(res.points.len(), trace.len());
        // Every matched node must lie on road A (ids 0..=10) — the HMM should
        // override the outlier's nearest-road-B emission via transitions.
        for (i, p) in res.points.iter().enumerate() {
            assert!(
                p.node <= 10,
                "point {i} matched to road B node {} (snap {:.0} m)",
                p.node,
                p.snap_distance_m
            );
        }
        assert!(res.confidence > 0.3, "confidence {}", res.confidence);
    }

    #[test]
    fn empty_and_single_point() {
        let (h, coords) = parallel_roads();
        let cand = |la: f32, lo: f32| -> Vec<(u32, f32, f32)> {
            let mut v: Vec<(f32, u32)> = coords
                .iter()
                .enumerate()
                .map(|(i, &(a, b))| (haversine_m(la, lo, a, b), i as u32))
                .collect();
            v.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            v.truncate(2);
            v.into_iter().map(|(_, i)| (i, coords[i as usize].0, coords[i as usize].1)).collect()
        };
        let res = match_trace(&h, &cand, |c| h.perm[c as usize], &[], 15.0);
        assert!(res.points.is_empty());
        let res = match_trace(&h, &cand, |c| h.perm[c as usize], &[(51.50001, -0.1300)], 15.0);
        assert_eq!(res.points.len(), 1);
        assert_eq!(res.points[0].node, 0);
    }
}
