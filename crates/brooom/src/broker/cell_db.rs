//! Persistent cell database for the matrix broker (Stage B).
//!
//! Bought cells are keyed by **quantised coordinates** (not matrix index), so a
//! distance learned in one solve is reused by any later solve that touches the
//! same place — the cross-run cost win for live/repeated routing. A per-node
//! "seen" counter accrues across runs: hubs/queue points (high count) are worth
//! buying precisely; rarely-touched houses (low count) can be derived instead
//! (the frequency prune, opt-in via `freq_threshold`). The store is a small
//! best-effort flat file, modelled on `cache.rs`.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 5] = b"MBKZ1";

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
    /// cell key → (duration_s, distance_m, seen_count)
    cells: HashMap<u64, (i32, i32, u32)>,
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

    #[inline]
    pub fn cell_key(&self, ci: [f64; 2], cj: [f64; 2]) -> u64 {
        let mut h = DefaultHasher::new();
        self.profile.hash(&mut h);
        node_key(ci).hash(&mut h);
        node_key(cj).hash(&mut h);
        h.finish()
    }

    /// Known (duration, distance) for a directed coord pair, if cached.
    pub fn get(&self, ci: [f64; 2], cj: [f64; 2]) -> Option<(i32, i32)> {
        let k = self.cell_key(ci, cj);
        self.cells.get(&k).map(|&(d, m, _)| (d, m))
    }

    /// Record a bought cell.
    pub fn put(&mut self, ci: [f64; 2], cj: [f64; 2], dur: i32, dist: i32) {
        let k = self.cell_key(ci, cj);
        let e = self.cells.entry(k).or_insert((dur, dist, 0));
        e.0 = dur;
        e.1 = dist;
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
            Ok(b) if b.len() >= 5 + 16 && &b[..5] == MAGIC => b,
            _ => return,
        };
        let mut o = 5usize;
        let rd_u64 = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let rd_i32 = |b: &[u8], o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let rd_u32 = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let ncells = rd_u64(&bytes, o) as usize;
        o += 8;
        for _ in 0..ncells {
            if o + 20 > bytes.len() {
                return;
            }
            let key = rd_u64(&bytes, o);
            let dur = rd_i32(&bytes, o + 8);
            let dist = rd_i32(&bytes, o + 12);
            let seen = rd_u32(&bytes, o + 16);
            o += 20;
            self.cells.insert(key, (dur, dist, seen));
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
        let mut buf: Vec<u8> = Vec::with_capacity(5 + 8 + self.cells.len() * 20 + 8 + self.node_seen.len() * 12);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(self.cells.len() as u64).to_le_bytes());
        for (&k, &(dur, dist, seen)) in &self.cells {
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&dur.to_le_bytes());
            buf.extend_from_slice(&dist.to_le_bytes());
            buf.extend_from_slice(&seen.to_le_bytes());
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
