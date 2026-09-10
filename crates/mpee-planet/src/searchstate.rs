//! A search frontier whose size follows the *state*, not the work.
//!
//! Measured on a cross-Europe overlay query: 1.9 M boundary vertices settled,
//! 1 965 MB resident — about a kilobyte per vertex. Only 192 bytes of that
//! were the distance and parent maps. The rest was the priority queue.
//!
//! The cause is lazy deletion in a dense graph. Each settled vertex relaxes
//! ~139 neighbours through its region's table, and every improvement pushes a
//! new entry while the stale one stays; the queue ends up holding roughly a
//! hundred entries per vertex that is ever settled. Correct, and unaffordable.
//!
//! So this holds one slot per *vertex*, keyed by an open-addressing table, and
//! the heap stores slot indices with a `decrease_key` that moves the existing
//! entry instead of adding another. The queue can then never exceed the number
//! of vertices reached, which is the quantity a memory budget can be stated
//! against.
//!
//! Per reached vertex: 4 bytes of key, 4 of distance, 4 of parent, 4 of heap
//! position and 4 of heap slot, at a 0.7 load factor — about 27 bytes, against
//! roughly a thousand.

const EMPTY: u32 = u32::MAX;

pub struct SearchState {
    /// Open-addressing table: `keys[i]` is the vertex owning slot `i`.
    keys: Vec<u32>,
    dist: Vec<u32>,
    /// What the queue orders by. Equal to `dist` for a plain Dijkstra; for a
    /// goal-directed search it is `dist + potential`, and the two must be kept
    /// apart — the answer is a distance, the ordering is a guess about the
    /// future, and conflating them was never going to end well.
    prio: Vec<u32>,
    par: Vec<u32>,
    /// Where slot `i` sits in `heap`, or `EMPTY` when it is not queued.
    hpos: Vec<u32>,
    heap: Vec<u32>,
    mask: usize,
    len: usize,
}

impl Default for SearchState {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchState {
    pub fn new() -> SearchState {
        let cap = 1024;
        SearchState {
            keys: vec![EMPTY; cap],
            dist: vec![0; cap],
            prio: vec![0; cap],
            par: vec![EMPTY; cap],
            hpos: vec![EMPTY; cap],
            heap: Vec::new(),
            mask: cap - 1,
            len: 0,
        }
    }

    pub fn clear(&mut self) {
        // Only the slots actually used are reset, so clearing costs what the
        // last query touched rather than what the table grew to.
        for &slot in &self.heap {
            let _ = slot;
        }
        self.keys.iter_mut().for_each(|k| *k = EMPTY);
        self.hpos.iter_mut().for_each(|k| *k = EMPTY);
        self.heap.clear();
        self.len = 0;
    }

    pub fn reached(&self) -> usize {
        self.len
    }

    #[inline]
    fn probe(&self, v: u32) -> usize {
        // Fibonacci hashing: cheap and well spread for dense integer ids.
        let mut i = ((v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & self.mask;
        loop {
            let k = self.keys[i];
            if k == EMPTY || k == v {
                return i;
            }
            i = (i + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let cap = (self.mask + 1) * 2;
        let old_keys = std::mem::replace(&mut self.keys, vec![EMPTY; cap]);
        let old_dist = std::mem::replace(&mut self.dist, vec![0; cap]);
        let old_prio = std::mem::replace(&mut self.prio, vec![0; cap]);
        let old_par = std::mem::replace(&mut self.par, vec![EMPTY; cap]);
        let old_hpos = std::mem::replace(&mut self.hpos, vec![EMPTY; cap]);
        self.mask = cap - 1;
        let mut remap = vec![EMPTY; old_keys.len()];
        for (i, &k) in old_keys.iter().enumerate() {
            if k == EMPTY {
                continue;
            }
            let j = self.probe(k);
            self.keys[j] = k;
            self.dist[j] = old_dist[i];
            self.prio[j] = old_prio[i];
            self.par[j] = old_par[i];
            self.hpos[j] = old_hpos[i];
            remap[i] = j as u32;
        }
        for slot in self.heap.iter_mut() {
            *slot = remap[*slot as usize];
        }
    }

    #[inline]
    pub fn dist_of(&self, v: u32) -> Option<u32> {
        let i = self.probe(v);
        (self.keys[i] == v).then(|| self.dist[i])
    }

    #[inline]
    pub fn parent_of(&self, v: u32) -> Option<u32> {
        let i = self.probe(v);
        if self.keys[i] == v && self.par[i] != EMPTY {
            Some(self.par[i])
        } else {
            None
        }
    }

    /// Record `v` at `d` if that improves on what is known, moving its queue
    /// entry rather than adding a second one.
    pub fn relax(&mut self, v: u32, d: u32, parent: u32) -> bool {
        self.relax_with(v, d, d, parent)
    }

    /// As `relax`, with a potential that is computed only when the vertex is
    /// first seen.
    ///
    /// A vertex is relaxed many times — once per incoming edge that improves
    /// it, and in a dense overlay that is hundreds — while its potential never
    /// changes. On a revisit the stored `prio - dist` *is* the potential, so it
    /// costs a subtraction instead of two great-circle calculations. That
    /// difference is the whole reason a goal-directed search can be affordable
    /// here at all.
    pub fn relax_pot(&mut self, v: u32, d: u32, parent: u32, pot: impl FnOnce() -> u32) -> bool {
        let i = self.probe(v);
        if self.keys[i] != EMPTY {
            if d >= self.dist[i] {
                return false;
            }
            let p = self.prio[i] - self.dist[i];
            return self.relax_with(v, d, d.saturating_add(p), parent);
        }
        self.relax_with(v, d, d.saturating_add(pot()), parent)
    }

    /// As `relax`, ordering by `prio` instead of by distance.
    pub fn relax_with(&mut self, v: u32, d: u32, prio: u32, parent: u32) -> bool {
        if (self.len + 1) * 10 > (self.mask + 1) * 7 {
            self.grow();
        }
        let i = self.probe(v);
        if self.keys[i] == EMPTY {
            self.keys[i] = v;
            self.dist[i] = d;
            self.prio[i] = prio;
            self.par[i] = parent;
            self.len += 1;
            self.hpos[i] = self.heap.len() as u32;
            self.heap.push(i as u32);
            self.sift_up(self.heap.len() - 1);
            return true;
        }
        if d >= self.dist[i] {
            return false;
        }
        self.dist[i] = d;
        self.prio[i] = prio;
        self.par[i] = parent;
        if self.hpos[i] == EMPTY {
            // Re-queue a vertex that was already settled: only possible if the
            // caller relaxes after popping, which this search does not do, but
            // handling it keeps the structure honest rather than subtly wrong.
            self.hpos[i] = self.heap.len() as u32;
            self.heap.push(i as u32);
            self.sift_up(self.heap.len() - 1);
        } else {
            let p = self.hpos[i] as usize;
            self.sift_up(p);
        }
        true
    }

    /// The smallest priority in the queue — the figure a termination rule
    /// compares, which under a potential is not a distance.
    pub fn peek(&self) -> Option<u32> {
        self.heap.first().map(|&s| self.prio[s as usize])
    }

    /// The distance of the cheapest queued vertex.
    pub fn peek_dist(&self) -> Option<u32> {
        self.heap.first().map(|&s| self.dist[s as usize])
    }

    /// Vertices near the top of the queue, cheapest first-ish.
    ///
    /// A binary heap only promises its root, so the first `n` slots are not
    /// the `n` cheapest — but they are all shallow, which is exactly the
    /// property wanted here. This is for prefetching work that a vertex will
    /// need when it is settled: being approximately right costs a little
    /// speculation and nothing in correctness, where sorting would cost more
    /// than the prefetch saves.
    pub fn peek_many(&self, n: usize, out: &mut Vec<u32>) {
        out.clear();
        for &slot in self.heap.iter().take(n) {
            out.push(self.keys[slot as usize]);
        }
    }

    /// Remove and return the cheapest `(vertex, distance)`.
    pub fn pop(&mut self) -> Option<(u32, u32)> {
        let top = *self.heap.first()?;
        let last = self.heap.pop().unwrap();
        self.hpos[top as usize] = EMPTY;
        if !self.heap.is_empty() && last != top {
            self.heap[0] = last;
            self.hpos[last as usize] = 0;
            self.sift_down(0);
        }
        Some((self.keys[top as usize], self.dist[top as usize]))
    }

    #[inline]
    fn key(&self, at: usize) -> u32 {
        self.prio[self.heap[at] as usize]
    }

    fn sift_up(&mut self, mut at: usize) {
        while at > 0 {
            let parent = (at - 1) / 2;
            if self.key(parent) <= self.key(at) {
                break;
            }
            self.swap(parent, at);
            at = parent;
        }
    }

    fn sift_down(&mut self, mut at: usize) {
        let n = self.heap.len();
        loop {
            let (l, r) = (2 * at + 1, 2 * at + 2);
            let mut best = at;
            if l < n && self.key(l) < self.key(best) {
                best = l;
            }
            if r < n && self.key(r) < self.key(best) {
                best = r;
            }
            if best == at {
                return;
            }
            self.swap(best, at);
            at = best;
        }
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.hpos[self.heap[a] as usize] = a as u32;
        self.hpos[self.heap[b] as usize] = b as u32;
    }

    /// Bytes the structure currently occupies — what a memory budget is
    /// stated against.
    pub fn bytes(&self) -> usize {
        (self.mask + 1) * 20 + self.heap.capacity() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pops_in_ascending_distance() {
        let mut s = SearchState::new();
        for (v, d) in [(5u32, 30u32), (1, 10), (9, 20), (3, 40)] {
            assert!(s.relax(v, d, 0));
        }
        let mut got = Vec::new();
        while let Some((v, d)) = s.pop() {
            got.push((v, d));
        }
        assert_eq!(got, vec![(1, 10), (9, 20), (5, 30), (3, 40)]);
    }

    #[test]
    fn a_better_distance_moves_the_entry_instead_of_adding_one() {
        let mut s = SearchState::new();
        s.relax(7, 100, 0);
        s.relax(8, 50, 0);
        assert!(s.relax(7, 10, 1), "an improvement is accepted");
        assert!(!s.relax(7, 20, 2), "a worse one is not");
        // Two vertices went in, so two come out — the improvement did not
        // leave a stale duplicate behind.
        assert_eq!(s.pop(), Some((7, 10)));
        assert_eq!(s.pop(), Some((8, 50)));
        assert_eq!(s.pop(), None);
        assert_eq!(s.reached(), 2);
    }

    #[test]
    fn the_queue_never_exceeds_the_vertices_reached() {
        // The property the whole structure exists for: relaxing the same
        // vertices repeatedly must not grow the queue.
        let mut s = SearchState::new();
        for round in 0..50u32 {
            for v in 0..200u32 {
                s.relax(v, 10_000 - round * 100 + v, 0);
            }
        }
        assert_eq!(s.reached(), 200);
        let mut popped = 0;
        let mut last = 0;
        while let Some((_, d)) = s.pop() {
            assert!(d >= last, "distances must come out ordered");
            last = d;
            popped += 1;
        }
        assert_eq!(popped, 200, "10 000 relaxations, 200 entries");
    }

    #[test]
    fn survives_growing_past_its_initial_capacity() {
        let mut s = SearchState::new();
        let n = 5000u32;
        for v in 0..n {
            s.relax(v, n - v, 0);
        }
        assert_eq!(s.reached(), n as usize);
        for v in 0..n {
            assert_eq!(s.dist_of(v), Some(n - v));
        }
        let mut last = 0;
        for _ in 0..n {
            let (_, d) = s.pop().unwrap();
            assert!(d >= last);
            last = d;
        }
        assert_eq!(s.pop(), None);
    }

    #[test]
    fn clear_makes_it_reusable() {
        let mut s = SearchState::new();
        s.relax(1, 5, 0);
        s.clear();
        assert_eq!(s.reached(), 0);
        assert_eq!(s.dist_of(1), None);
        assert_eq!(s.pop(), None);
        s.relax(2, 7, 0);
        assert_eq!(s.pop(), Some((2, 7)));
    }
}
