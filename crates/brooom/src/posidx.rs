//! Maintained task→(route, position) index + static location→tasks map.
//!
//! `locate()` in local search was a linear scan over every route per probe,
//! and the granular position scans in the fast operators walk every position
//! of every route per probe — O(total tasks) work per probe that does not
//! shrink with granular-K. `PosIndex` keeps task positions current across
//! applied moves (the SlackCache::on_apply index discipline) so a lookup is
//! O(1) and the granular enumeration can be INVERTED: iterate the K
//! neighbours of the probe's location and map each one straight to its
//! (route, position) instead of scanning for it.
//!
//! Route identity: local search only UPDATES or REMOVES routes — it never
//! adds one (the empty-route retain runs after LS returns, when this index
//! is already dead). Each route therefore gets a stable slot id at build
//! time; `slot_of_idx`/`idx_of_slot` translate between the live
//! `sol.routes` index and the slot. Removals keep slots stable, so the
//! per-slot generation counters stay valid for caches keyed by slot
//! (swap*'s top-3 insertion cache).
//!
//! The task SET is invariant during LS (moves relocate tasks, never assign
//! or unassign them), so the location→tasks map is built once per call and
//! never updated.
//!
//! Kill switch: `BROOOM_NO_POSIDX=1` restores the linear `locate()` and the
//! scanning enumeration everywhere (the inverted paths require the index).

use crate::problem::Problem;
use crate::solution::{Solution, TaskRef};

/// Master switch (off via BROOOM_NO_POSIDX, for A/B). Read once per
/// local-search call by `PosIndex::arm` — never in the hot loops.
fn posidx_enabled() -> bool {
    std::env::var("BROOOM_NO_POSIDX").is_err()
}

pub(crate) struct PosIndex {
    /// task key → (slot, pos); slot == u32::MAX means "not in any route"
    /// (mirrors the linear `locate()` returning None).
    pos: Vec<(u32, u32)>,
    /// live `sol.routes` index → stable slot id (kept aligned with
    /// `sol.routes` by `on_apply`, same descending-removal discipline as
    /// SlackCache).
    slot_of_idx: Vec<u32>,
    /// stable slot id → live `sol.routes` index; u32::MAX once removed.
    idx_of_slot: Vec<u32>,
    /// Per-slot generation, bumped on every update or removal of the slot's
    /// route. Lets per-route caches validate entries in O(1).
    gens: Vec<u32>,
    n_jobs: usize,
    /// Static location→tasks map (CSR over matrix indices): every task
    /// present in the solution at build time, grouped by location.
    csr_off: Vec<u32>,
    csr_tasks: Vec<TaskRef>,
    /// Enumeration switches resolved at arm time (env reads are too slow for
    /// the per-probe loops): inverted relocate enumeration, its paranoia
    /// cross-check, and the swap* top-3 cache / r1-side gap trick.
    pub(crate) reloc_inv: bool,
    pub(crate) check_inv: bool,
    pub(crate) swap_top3: bool,
    pub(crate) swap_r1gap: bool,
    /// Inverted pair scans in swap*/exchange/2opt*/cross-exchange.
    pub(crate) pair_inv: bool,
}

impl PosIndex {
    /// Dense task key. Reload markers carry no identity (and no location) —
    /// they are skipped, exactly as the linear scan's `position()` would
    /// still find them but no caller ever probes a Reload.
    #[inline]
    fn key(&self, t: TaskRef) -> Option<usize> {
        match t {
            TaskRef::Job(i) => Some(i),
            TaskRef::Pickup(i) => Some(self.n_jobs + 2 * i),
            TaskRef::Delivery(i) => Some(self.n_jobs + 2 * i + 1),
            TaskRef::Reload => None,
        }
    }

    /// Build the index for the solution as it stands, or `None` when the
    /// kill switch is set. `matrix_n` sizes the location map.
    pub(crate) fn arm(problem: &Problem, matrix_n: usize, sol: &Solution) -> Option<PosIndex> {
        if !posidx_enabled() {
            return None;
        }
        let n_jobs = problem.jobs.len();
        let n_keys = n_jobs + 2 * problem.shipments.len();
        let n_routes = sol.routes.len();
        let mut px = PosIndex {
            pos: vec![(u32::MAX, 0); n_keys],
            slot_of_idx: (0..n_routes as u32).collect(),
            idx_of_slot: (0..n_routes as u32).collect(),
            gens: vec![0; n_routes],
            n_jobs,
            csr_off: Vec::new(),
            csr_tasks: Vec::new(),
            reloc_inv: std::env::var("BROOOM_NO_RELOC_INV").is_err(),
            check_inv: std::env::var("BROOOM_CHECK_INV").is_ok(),
            swap_top3: std::env::var("BROOOM_NO_SWAPSTAR_TOPCACHE").is_err(),
            swap_r1gap: std::env::var("BROOOM_NO_SWAPSTAR_R1GAP").is_err(),
            pair_inv: std::env::var("BROOOM_NO_PAIR_INV").is_err(),
        };
        for (r, route) in sol.routes.iter().enumerate() {
            px.stamp(r as u32, &route.steps);
        }
        // Location→tasks CSR: count, prefix-sum, fill.
        let mut counts = vec![0u32; matrix_n + 1];
        let loc_of = |t: TaskRef| t.description(problem).location.index.filter(|&l| l < matrix_n);
        for route in &sol.routes {
            for &t in &route.steps {
                if let Some(l) = loc_of(t) {
                    counts[l + 1] += 1;
                }
            }
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        px.csr_off = counts;
        px.csr_tasks = vec![TaskRef::Reload; *px.csr_off.last().unwrap() as usize];
        let mut fill = px.csr_off.clone();
        for route in &sol.routes {
            for &t in &route.steps {
                if let Some(l) = loc_of(t) {
                    px.csr_tasks[fill[l] as usize] = t;
                    fill[l] += 1;
                }
            }
        }
        Some(px)
    }

    /// Re-stamp every task in `steps` as living in `slot`.
    fn stamp(&mut self, slot: u32, steps: &[TaskRef]) {
        for (p, &t) in steps.iter().enumerate() {
            if let Some(k) = self.key(t) {
                self.pos[k] = (slot, p as u32);
            }
        }
    }

    /// Current (route index, position) of `task` — O(1). Matches the linear
    /// `locate()` scan exactly (verified by a debug_assert at the call site).
    #[inline]
    pub(crate) fn locate(&self, task: TaskRef) -> Option<(usize, usize)> {
        let k = self.key(task)?;
        let (slot, p) = self.pos[k];
        if slot == u32::MAX {
            return None;
        }
        let r = self.idx_of_slot[slot as usize];
        if r == u32::MAX {
            return None;
        }
        Some((r as usize, p as usize))
    }

    /// Tasks (in any route at build time) whose location is `loc`.
    #[inline]
    pub(crate) fn tasks_at(&self, loc: usize) -> &[TaskRef] {
        if loc + 1 >= self.csr_off.len() {
            return &[];
        }
        &self.csr_tasks[self.csr_off[loc] as usize..self.csr_off[loc + 1] as usize]
    }

    /// Stable slot id of the route currently at index `r`.
    #[inline]
    pub(crate) fn slot_of(&self, r: usize) -> usize {
        self.slot_of_idx[r] as usize
    }

    /// Dense task key (public face of `key`, for slot-keyed caches).
    #[inline]
    pub(crate) fn key_of(&self, t: TaskRef) -> Option<usize> {
        self.key(t)
    }

    /// Size of the dense task-key space.
    #[inline]
    pub(crate) fn n_keys(&self) -> usize {
        self.pos.len()
    }

    /// Generation counter for `slot` (bumped on every update/removal).
    #[inline]
    pub(crate) fn gen_of_slot(&self, slot: usize) -> u32 {
        self.gens[slot]
    }

    /// Number of slots (== route count at build time; never grows).
    #[inline]
    pub(crate) fn n_slots(&self) -> usize {
        self.gens.len()
    }

    /// Mirror an applied move. Same shape and ordering discipline as
    /// `SlackCache::on_apply`: descending route index so removals don't
    /// shift the indices still to be processed.
    pub(crate) fn on_apply<M>(&mut self, route_updates: &[(usize, Option<(Vec<TaskRef>, M)>)]) {
        let mut idxs: Vec<(usize, Option<&[TaskRef]>)> = route_updates
            .iter()
            .map(|(r, p)| (*r, p.as_ref().map(|(s, _)| s.as_slice())))
            .collect();
        idxs.sort_by(|a, b| b.0.cmp(&a.0));
        let mut removed_any = false;
        for (r, steps) in idxs {
            let slot = self.slot_of_idx[r] as usize;
            self.gens[slot] = self.gens[slot].wrapping_add(1);
            match steps {
                Some(steps) => {
                    // Tasks that LEFT this route are re-stamped by the other
                    // updated route in the same move; tasks that stayed get
                    // fresh positions here.
                    self.stamp(slot as u32, steps);
                }
                None => {
                    // Route removed (it ended up empty — its tasks moved to
                    // another route in this same move and were stamped there).
                    self.idx_of_slot[slot] = u32::MAX;
                    self.slot_of_idx.remove(r);
                    removed_any = true;
                }
            }
        }
        if removed_any {
            // Re-derive idx_of_slot for the surviving routes; removed slots
            // keep their u32::MAX sentinel. O(n_routes), removals are rare.
            for (r, &slot) in self.slot_of_idx.iter().enumerate() {
                self.idx_of_slot[slot as usize] = r as u32;
            }
        }
    }
}
