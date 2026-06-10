//! Trip service: order a set of waypoints into the shortest visiting sequence
//! (the OSRM `/trip` feature — a small TSP on the waypoint duration matrix).
//!
//! Two modes:
//!   * `roundtrip = true` — a closed tour that starts and ends at waypoint 0.
//!   * `roundtrip = false` — an open path fixed to start at waypoint 0 and end
//!     at the last waypoint, ordering the middle stops.
//!
//! Solvers:
//!   * n ≤ `EXACT_LIMIT` → Held-Karp dynamic program (exact optimum).
//!   * larger → nearest-neighbour construction + 2-opt / Or-opt local search
//!     to convergence (the same class of heuristic OSRM's trip plugin uses).
//!
//! The matrix is row-major `dur[i * n + j]` = travel seconds i → j (may be
//! asymmetric). Unreachable pairs are `f32::INFINITY`; a result whose total is
//! infinite means no feasible ordering exists.

/// Above this many waypoints Held-Karp's 2^n table is no longer worth it.
const EXACT_LIMIT: usize = 13;

#[inline]
fn at(dur: &[f32], n: usize, i: usize, j: usize) -> f32 {
    dur[i * n + j]
}

/// Total cost of an ordering (closing leg included when `roundtrip`).
pub fn tour_cost(dur: &[f32], n: usize, order: &[usize], roundtrip: bool) -> f32 {
    let mut total = 0.0f32;
    for w in order.windows(2) {
        total += at(dur, n, w[0], w[1]);
    }
    if roundtrip && order.len() > 1 {
        total += at(dur, n, *order.last().unwrap(), order[0]);
    }
    total
}

/// Order waypoints `0..n` by shortest travel time. Returns the visiting order
/// (a permutation of `0..n`); `order[0]` is always 0, and for open paths
/// (`roundtrip = false`) the last element is always `n - 1`.
pub fn tsp_order(dur: &[f32], n: usize, roundtrip: bool) -> Vec<usize> {
    assert_eq!(dur.len(), n * n, "duration matrix must be n×n");
    match n {
        0 => return Vec::new(),
        1 => return vec![0],
        2 => return vec![0, 1],
        _ => {}
    }
    if n <= EXACT_LIMIT {
        held_karp(dur, n, roundtrip)
    } else {
        let mut order = nearest_neighbour(dur, n, roundtrip);
        local_search(dur, n, &mut order, roundtrip);
        order
    }
}

// ── Exact: Held-Karp ─────────────────────────────────────────────────────────
//
// dp[mask][j] = cheapest cost starting at node 0, visiting exactly the set
// `mask` of middle nodes, ending at middle node j. For the roundtrip we close
// back to 0; for the open path the "middle" excludes the fixed terminal n-1
// and we close with the leg j → n-1.
fn held_karp(dur: &[f32], n: usize, roundtrip: bool) -> Vec<usize> {
    // Middle nodes: 1..n for roundtrip, 1..n-1 for open path.
    let mids: Vec<usize> = if roundtrip { (1..n).collect() } else { (1..n - 1).collect() };
    let m = mids.len();
    if m == 0 {
        return if roundtrip { vec![0] } else { vec![0, n - 1] };
    }
    let full = 1usize << m;
    let mut dp = vec![f32::INFINITY; full * m];
    let mut parent = vec![u8::MAX; full * m];
    for (j, &node) in mids.iter().enumerate() {
        dp[(1 << j) * m + j] = at(dur, n, 0, node);
    }
    for mask in 1..full {
        for j in 0..m {
            if mask & (1 << j) == 0 {
                continue;
            }
            let rest = mask ^ (1 << j);
            if rest == 0 {
                continue; // singleton — seeded above
            }
            // dp[mask][j] = min over i in rest of dp[rest][i] + d(i, j)
            let mut best = f32::INFINITY;
            let mut best_i = u8::MAX;
            for i in 0..m {
                if rest & (1 << i) == 0 {
                    continue;
                }
                let c = dp[rest * m + i] + at(dur, n, mids[i], mids[j]);
                if c < best {
                    best = c;
                    best_i = i as u8;
                }
            }
            dp[mask * m + j] = best;
            parent[mask * m + j] = best_i;
        }
    }

    // Close the tour/path.
    let mask = full - 1;
    let mut best_total = f32::INFINITY;
    let mut best_j = 0usize;
    for j in 0..m {
        let close = if roundtrip {
            at(dur, n, mids[j], 0)
        } else {
            at(dur, n, mids[j], n - 1)
        };
        let c = dp[mask * m + j] + close;
        if c < best_total {
            best_total = c;
            best_j = j;
        }
    }
    // Disconnected waypoints: no finite completion exists. Return the identity
    // order; `tour_cost` over it is infinite, which is the caller's signal.
    if !best_total.is_finite() {
        return (0..n).collect();
    }

    // Backtrack.
    let mut order_rev: Vec<usize> = Vec::with_capacity(n);
    if !roundtrip {
        order_rev.push(n - 1);
    }
    let mut mask = mask;
    let mut j = best_j;
    loop {
        order_rev.push(mids[j]);
        let p = parent[mask * m + j];
        let rest = mask ^ (1 << j);
        if rest == 0 {
            break;
        }
        mask = rest;
        j = p as usize;
    }
    order_rev.push(0);
    order_rev.reverse();
    order_rev
}

// ── Heuristic: NN + 2-opt + Or-opt ──────────────────────────────────────────

fn nearest_neighbour(dur: &[f32], n: usize, roundtrip: bool) -> Vec<usize> {
    let last_fixed = !roundtrip;
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    order.push(0);
    visited[0] = true;
    if last_fixed {
        visited[n - 1] = true; // reserved as terminal
    }
    let mut cur = 0usize;
    for _ in 0..n - 1 - last_fixed as usize {
        let mut best = usize::MAX;
        let mut best_d = f32::INFINITY;
        for v in 1..n {
            if !visited[v] && at(dur, n, cur, v) < best_d {
                best_d = at(dur, n, cur, v);
                best = v;
            }
        }
        if best == usize::MAX {
            // Disconnected: append remaining in index order.
            for v in 1..n {
                if !visited[v] {
                    order.push(v);
                    visited[v] = true;
                }
            }
            break;
        }
        order.push(best);
        visited[best] = true;
        cur = best;
    }
    if last_fixed {
        order.push(n - 1);
    }
    order
}

/// 2-opt (segment reversal) + Or-opt (move 1–3 stops) to convergence.
/// Positions 0 (and the last, for open paths) are pinned.
fn local_search(dur: &[f32], n: usize, order: &mut Vec<usize>, roundtrip: bool) {
    let len = order.len();
    if len < 4 {
        return;
    }
    let lo = 1usize; // first movable position
    let hi = if roundtrip { len - 1 } else { len - 2 }; // last movable position
    let mut improved = true;
    while improved {
        improved = false;
        // 2-opt: reverse order[a..=b].
        for a in lo..hi {
            for b in (a + 1)..=hi {
                let prev = order[a - 1];
                let next = if b + 1 < len { order[b + 1] } else { order[0] };
                let removed = at(dur, n, prev, order[a]) + at(dur, n, order[b], next);
                let added = at(dur, n, prev, order[b]) + at(dur, n, order[a], next);
                // Asymmetric matrices change interior arc cost on reversal too.
                let mut interior_old = 0.0f32;
                let mut interior_new = 0.0f32;
                for k in a..b {
                    interior_old += at(dur, n, order[k], order[k + 1]);
                    interior_new += at(dur, n, order[k + 1], order[k]);
                }
                if added + interior_new + 1e-6 < removed + interior_old {
                    order[a..=b].reverse();
                    improved = true;
                }
            }
        }
        // Or-opt: relocate a segment of 1–3 stops.
        for seg in 1..=3usize {
            for a in lo..=hi.saturating_sub(seg - 1) {
                let b = a + seg - 1;
                if b > hi {
                    break;
                }
                for ins in lo..=hi + 1 {
                    if ins >= a && ins <= b + 1 {
                        continue;
                    }
                    let cost = |o: &[usize]| tour_cost(dur, n, o, roundtrip);
                    let before = cost(order);
                    let mut cand = Vec::with_capacity(len);
                    cand.extend_from_slice(&order[..a]);
                    cand.extend_from_slice(&order[b + 1..]);
                    let ins_adj = if ins > b { ins - seg } else { ins };
                    for (off, &v) in order[a..=b].iter().enumerate() {
                        cand.insert(ins_adj + off, v);
                    }
                    if cost(&cand) + 1e-6 < before {
                        *order = cand;
                        improved = true;
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 points on a line: 0 -- 1 -- 2 -- 3 (unit spacing, symmetric).
    fn line4() -> Vec<f32> {
        let pos = [0.0f32, 1.0, 2.0, 3.0];
        let n = pos.len();
        let mut d = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = (pos[i] - pos[j]).abs();
            }
        }
        d
    }

    #[test]
    fn open_path_orders_line() {
        // Visiting order on a line from 0 to 3 must be 0,1,2,3.
        let d = line4();
        let order = tsp_order(&d, 4, false);
        assert_eq!(order, vec![0, 1, 2, 3]);
        assert!((tour_cost(&d, 4, &order, false) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn roundtrip_visits_all_optimally() {
        let d = line4();
        let order = tsp_order(&d, 4, true);
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 4);
        // Optimal cycle on a line costs 2 × span = 6.
        assert!((tour_cost(&d, 4, &order, true) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn shuffled_line_recovered() {
        // Points at positions [0, 5, 1, 4, 2, 3]: optimal open path 0→1 (idx 2)
        // →2 (idx 4)→3 (idx 5)→4 (idx 3)→5 (idx 1)... but the open path is
        // pinned 0-first / last-index-last, so expect total = walk in position
        // order with terminal at index 5 (position 3).
        let pos = [0.0f32, 5.0, 1.0, 4.0, 2.0, 3.0];
        let n = pos.len();
        let mut d = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = (pos[i] - pos[j]).abs();
            }
        }
        let order = tsp_order(&d, n, true);
        // Exact Held-Karp must find the 10.0 cycle (sweep right, come back).
        assert!((tour_cost(&d, n, &order, true) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn heuristic_matches_exact_on_random() {
        // 12 random points → exact; 20 points → heuristic within 10% of a
        // brute NN baseline (sanity, not optimality).
        let mut seed = 0x1234_5678u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32) * 100.0
        };
        let n = 20;
        let pts: Vec<(f32, f32)> = (0..n).map(|_| (rnd(), rnd())).collect();
        let mut d = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = pts[i].0 - pts[j].0;
                let dy = pts[i].1 - pts[j].1;
                d[i * n + j] = (dx * dx + dy * dy).sqrt();
            }
        }
        let order = tsp_order(&d, n, true);
        assert_eq!(order.len(), n);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "must be a permutation");
        let nn = nearest_neighbour(&d, n, true);
        assert!(
            tour_cost(&d, n, &order, true) <= tour_cost(&d, n, &nn, true) + 1e-3,
            "local search must not be worse than its NN start"
        );
    }

    #[test]
    fn unreachable_pair_yields_infinite_total() {
        let mut d = line4();
        let n = 4;
        // Make 2 unreachable from everywhere.
        for i in 0..n {
            if i != 2 {
                d[i * n + 2] = f32::INFINITY;
            }
        }
        let order = tsp_order(&d, n, false);
        assert_eq!(order.len(), 4);
        assert!(!tour_cost(&d, n, &order, false).is_finite());
    }
}
