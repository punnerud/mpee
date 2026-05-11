//! Rubik's pocket cube (2×2×2) state graph — *corner permutation only*.
//!
//! We ignore corner orientation for simplicity. This gives a smaller, but
//! still regular graph:
//!   * 8! = 40 320 positions
//!   * 6 base moves: U, U2, R, R2, F, F2 (clockwise + half turn)
//!   * Each move is its own inverse or a half turn is self-inverse; we
//!     add both move and inverse by making the graph undirected.
//!
//! Diameter: 6 (every position permutation can be solved in ≤ 6 moves).
//! BFS distance from SOLVED gives us the classic "God's number" calculation.
//!
//! Weights: all = 1. SSSP from SOLVED gives the number of moves to each position.

use crate::graph::CsrGraph;
use std::collections::HashMap;

pub type State = [u8; 8];

pub const SOLVED: State = [0, 1, 2, 3, 4, 5, 6, 7];

/// Apply one of 6 base moves (clockwise) to state. Half turns can be obtained
/// by applying clockwise twice; inverses by applying half turn +
/// clockwise, or we generate it explicitly.
#[inline]
pub fn apply_cw(s: State, m: u8) -> State {
    let mut t = s;
    match m {
        // U: rotate top face clockwise. Cycle 0→1→2→3→0.
        0 => {
            t[0] = s[3];
            t[1] = s[0];
            t[2] = s[1];
            t[3] = s[2];
        }
        // R: cycle 0→3→7→4→0
        1 => {
            t[3] = s[0];
            t[7] = s[3];
            t[4] = s[7];
            t[0] = s[4];
        }
        // F: cycle 0→1→5→4→0
        2 => {
            t[1] = s[0];
            t[5] = s[1];
            t[4] = s[5];
            t[0] = s[4];
        }
        _ => {}
    }
    t
}

#[inline]
pub fn apply_ccw(s: State, m: u8) -> State {
    apply_cw(apply_cw(apply_cw(s, m), m), m)
}

#[inline]
pub fn apply_half(s: State, m: u8) -> State {
    apply_cw(apply_cw(s, m), m)
}

/// BFS from SOLVED and build the full graph over all reached states. Returns
/// (CsrGraph, depths).
pub fn build_pocket_cube_graph() -> (CsrGraph, Vec<u8>) {
    let mut id_of: HashMap<State, u32> = HashMap::with_capacity(50_000);
    let mut depth: Vec<u8> = Vec::with_capacity(50_000);
    let mut queue: std::collections::VecDeque<State> = std::collections::VecDeque::new();

    id_of.insert(SOLVED, 0);
    depth.push(0);
    queue.push_back(SOLVED);

    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(500_000);

    while let Some(s) = queue.pop_front() {
        let id_s = *id_of.get(&s).unwrap();
        let d_s = depth[id_s as usize];
        // 9 generators: 3 axes × {cw, ccw, half}.
        for m in 0..3u8 {
            for variant in 0..3u8 {
                let t = match variant {
                    0 => apply_cw(s, m),
                    1 => apply_ccw(s, m),
                    2 => apply_half(s, m),
                    _ => unreachable!(),
                };
                let id_t = match id_of.get(&t) {
                    Some(&i) => i,
                    None => {
                        let i = depth.len() as u32;
                        id_of.insert(t, i);
                        depth.push(d_s + 1);
                        queue.push_back(t);
                        i
                    }
                };
                if id_s != id_t {
                    // En retning; CsrGraph::from_edges trenger eksplisitt symmetri.
                    edges.push((id_s, id_t, 1.0));
                }
            }
        }
    }

    let n = depth.len();
    println!(
        "[rubik] BFS-utforsket {} stillinger, max dybde = {}",
        n,
        depth.iter().max().copied().unwrap_or(0)
    );

    let g = CsrGraph::from_edges(n, &edges);
    println!(
        "[rubik] CSR: n = {}, m = {} (avg_deg = {:.2})",
        g.n,
        g.m(),
        g.m() as f32 / g.n.max(1) as f32
    );
    (g, depth)
}
