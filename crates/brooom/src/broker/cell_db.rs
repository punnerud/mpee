//! Persistent cell database for the matrix broker (Stage B + temporal E1).
//!
//! Bought cells are keyed by **quantised coordinates** (not matrix index), so a
//! distance learned in one solve is reused by any later solve that touches the
//! same place — the cross-run cost win for live/repeated routing. A per-node
//! "seen" counter accrues across runs: hubs/queue points (high count) are worth
//! buying precisely; rarely-touched houses (low count) can be derived instead
//! (the frequency prune, opt-in via `freq_threshold`). The store is a small
//! best-effort flat file, modelled on `cache.rs`.
//!
//! **Temporal profiles (Stage E1).** The cell key optionally folds in a
//! `(weekday_class, hour)` bucket, and each cell carries **running statistics**
//! instead of a single number: a Welford mean + variance over every observation
//! in that time bucket. The mean is the typical (e.g. rush-hour) travel time;
//! the std-dev is the *uncertainty* — how much this arc deviates from "normal"
//! (a queue/incident signal). The killer cost-saver: observe one representative
//! workday's hourly cells, then reuse those time-of-day patterns OFFLINE for
//! every similar day — the key is a weekday *class* (Mon–Fri merged), not a
//! date, so Tuesday-08:00 data answers Thursday-08:00 with zero new buys.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};

// Bumped from MBKZ1: cells now carry Welford statistics and an optional
// (weekday_class, hour) bucket in the key. MBKZ1 files load as empty (the DB is
// a cache, not a source of truth — a clean break is acceptable).
const MAGIC: &[u8; 5] = b"MBKZ2";

/// A time bucket for a cell observation: `(weekday_class, hour)`. `weekday_class`
/// is `0` for a workday (Mon–Fri merged) and `1` for the weekend; `hour` is the
/// hour-of-day (0–23). Passing `None` keys the cell time-agnostically (Stages
/// A–D behaviour — one bucket for all observations).
pub type TimeBucket = Option<(u8, u8)>;

/// What a cell lookup returns: the running mean duration/distance and the
/// duration's std-dev (the uncertainty signal). `count` is how many
/// observations back it (0 = unknown).
#[derive(Debug, Clone, Copy, Default)]
pub struct CellStat {
    pub mean_dur: i32,
    pub mean_dist: i32,
    /// Std-dev of the duration over all observations in this (cell, time)
    /// bucket — the "deviation from normal" used as a congestion/uncertainty
    /// penalty by the broker.
    pub std_dur: f64,
    pub count: u32,
}

/// Welford running mean + variance for one cell/time bucket.
#[derive(Debug, Clone, Copy, Default)]
struct Welford {
    count: u32,
    mean_dur: f64,
    m2_dur: f64,
    mean_dist: f64,
}

impl Welford {
    #[inline]
    fn observe(&mut self, dur: i32, dist: i32) {
        self.count += 1;
        let n = self.count as f64;
        let d = dur as f64;
        let delta = d - self.mean_dur;
        self.mean_dur += delta / n;
        self.m2_dur += delta * (d - self.mean_dur);
        self.mean_dist += (dist as f64 - self.mean_dist) / n;
    }
    #[inline]
    fn stat(&self) -> CellStat {
        let std_dur = if self.count > 0 {
            (self.m2_dur / self.count as f64).max(0.0).sqrt()
        } else {
            0.0
        };
        CellStat {
            mean_dur: self.mean_dur.round() as i32,
            mean_dist: self.mean_dist.round() as i32,
            std_dur,
            count: self.count,
        }
    }
}

/// Quantise a coordinate component to ~6 decimal degrees (sub-metre).
#[inline]
fn q(c: f64) -> i64 {
    (c * 1_000_000.0).round() as i64
}

fn node_key(c: [f64; 2]) -> u64 {
    let mut h = DefaultHasher::new();
    q(c[0]).hash(&mut h);
    q(c[1]).hash(&mut h);
    h.finish()
}

pub struct CellDb {
    path: PathBuf,
    profile: u64,
    /// cell key → running statistics for that (coord-pair, time-bucket).
    cells: HashMap<u64, Welford>,
    /// node key → seen_count (accrues across runs)
    node_seen: HashMap<u64, u32>,
    dirty: bool,
}

impl CellDb {
    /// Open (or start) a DB at `path` for `profile`. Missing/corrupt file ⇒ empty.
    pub fn open<P: AsRef<Path>>(path: P, profile: &str) -> Self {
        let path = path.as_ref().to_path_buf();
        let mut ph = DefaultHasher::new();
        profile.hash(&mut ph);
        let mut db = CellDb {
            path,
            profile: ph.finish(),
            cells: HashMap::new(),
            node_seen: HashMap::new(),
            dirty: false,
        };
        db.load();
        db
    }

    /// Key for a directed coord pair in a time bucket. `when = None` folds all
    /// observations into one bucket (time-agnostic, Stages A–D). `when =
    /// Some((class, hour))` keys per weekday-class + hour, so a workday-08:00
    /// profile is distinct from a weekend-08:00 one — and reused across all days
    /// of the same class.
    #[inline]
    pub fn cell_key(&self, ci: [f64; 2], cj: [f64; 2], when: TimeBucket) -> u64 {
        let mut h = DefaultHasher::new();
        self.profile.hash(&mut h);
        node_key(ci).hash(&mut h);
        node_key(cj).hash(&mut h);
        // Tag the temporal dimension so a time-keyed cell never aliases a
        // time-agnostic one (the `1u8` discriminant before the bucket).
        match when {
            Some((class, hour)) => {
                1u8.hash(&mut h);
                class.hash(&mut h);
                hour.hash(&mut h);
            }
            None => 0u8.hash(&mut h),
        }
        h.finish()
    }

    /// Known statistics for a directed coord pair in a time bucket, if cached.
    pub fn get(&self, ci: [f64; 2], cj: [f64; 2], when: TimeBucket) -> Option<CellStat> {
        let k = self.cell_key(ci, cj, when);
        self.cells.get(&k).map(|w| w.stat())
    }

    /// Record one observation of a cell (Welford update). Repeated calls in the
    /// same (cell, time) bucket build up the mean (typical travel time) and the
    /// std-dev (uncertainty / deviation from normal).
    pub fn observe(&mut self, ci: [f64; 2], cj: [f64; 2], when: TimeBucket, dur: i32, dist: i32) {
        let k = self.cell_key(ci, cj, when);
        self.cells.entry(k).or_default().observe(dur, dist);
        self.dirty = true;
    }

    /// How many times this node has appeared in a solve's skeleton.
    pub fn node_seen(&self, c: [f64; 2]) -> u32 {
        *self.node_seen.get(&node_key(c)).unwrap_or(&0)
    }

    /// Bump a node's usage counter (call once per node per solve).
    pub fn bump_node(&mut self, c: [f64; 2]) {
        *self.node_seen.entry(node_key(c)).or_insert(0) += 1;
        self.dirty = true;
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    fn load(&mut self) {
        let bytes = match std::fs::read(&self.path) {
            Ok(b) if b.len() >= 5 + 8 && &b[..5] == MAGIC => b,
            _ => return,
        };
        let mut o = 5usize;
        let rd_u64 = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let rd_u32 = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let rd_f64 = |b: &[u8], o: usize| f64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let ncells = rd_u64(&bytes, o) as usize;
        o += 8;
        // Each cell record: key(8) + count(4) + mean_dur(8) + m2_dur(8) + mean_dist(8) = 36 bytes.
        for _ in 0..ncells {
            if o + 36 > bytes.len() {
                return;
            }
            let key = rd_u64(&bytes, o);
            let count = rd_u32(&bytes, o + 8);
            let mean_dur = rd_f64(&bytes, o + 12);
            let m2_dur = rd_f64(&bytes, o + 20);
            let mean_dist = rd_f64(&bytes, o + 28);
            o += 36;
            self.cells.insert(key, Welford { count, mean_dur, m2_dur, mean_dist });
        }
        if o + 8 > bytes.len() {
            return;
        }
        let nnodes = rd_u64(&bytes, o) as usize;
        o += 8;
        for _ in 0..nnodes {
            if o + 12 > bytes.len() {
                return;
            }
            let key = rd_u64(&bytes, o);
            let seen = rd_u32(&bytes, o + 8);
            o += 12;
            self.node_seen.insert(key, seen);
        }
    }

    /// Persist if changed. Best-effort (errors are swallowed, like `cache.rs`).
    pub fn flush(&mut self) {
        if !self.dirty {
            return;
        }
        let mut buf: Vec<u8> =
            Vec::with_capacity(5 + 8 + self.cells.len() * 36 + 8 + self.node_seen.len() * 12);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(self.cells.len() as u64).to_le_bytes());
        for (&k, w) in &self.cells {
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&w.count.to_le_bytes());
            buf.extend_from_slice(&w.mean_dur.to_le_bytes());
            buf.extend_from_slice(&w.m2_dur.to_le_bytes());
            buf.extend_from_slice(&w.mean_dist.to_le_bytes());
        }
        buf.extend_from_slice(&(self.node_seen.len() as u64).to_le_bytes());
        for (&k, &seen) in &self.node_seen {
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&seen.to_le_bytes());
        }
        // Atomic-ish: write to a temp then rename.
        let tmp = self.path.with_extension("tmp");
        if let Ok(mut f) = std::fs::File::create(&tmp) {
            if f.write_all(&buf).is_ok() && f.flush().is_ok() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
        self.dirty = false;
    }
}
