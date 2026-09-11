//! What changed between the planet we built from and a newer one.
//!
//! An update should not mean rebuilding 37 GB. Most of a weekly planet file is
//! byte-for-byte the roads we already have; what matters is the handful of ways
//! that moved, closed, appeared or went away — and which regions of the overlay
//! they fall in, because that is all the recomputation they can possibly cause.
//!
//! Two streaming passes, and neither holds more than a buffer:
//!
//! 1. Read the new file and write one `(way id, content hash)` per routable
//!    way. This is the expensive half, and it parallelises over blobs exactly
//!    as the build does.
//! 2. Merge-join that against the way index. Both sides are sorted by id — a
//!    PBF stores ways ascending, and the index was written in read order — so
//!    a cursor on each side classifies every way and forgets it.
//!
//! What it cannot see is a node that moved without its way being touched.
//! Coordinates arrive in a different pass of the file, and hashing them per way
//! would mean reading the planet twice. Tags and topology are what closures and
//! new roads change, and those are covered exactly.

use crate::build::{encode_way, pack_attr, Dir, Paths};
use crate::dataset::Dataset;
use crate::mmapvec;
use crate::overlay::{Overlay, Row};
use crate::parallel::scan_blobs;
use crate::profile;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::io;
use std::path::Path;

/// FNV-1a over a way record, skipping the leading id varint. Must agree with
/// `contract::way_key`, or every way looks changed.
fn hash_rec(rec: &[u8]) -> u64 {
    let mut n = 0;
    while n < rec.len() && rec[n] & 0x80 != 0 {
        n += 1;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &rec[n + 1..] {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

#[derive(Default)]
pub struct Changes {
    pub unchanged: u64,
    pub changed: u64,
    pub added: u64,
    pub deleted: u64,
    /// Segments belonging to a way that changed or went away. A new way has
    /// none — it needs a rebuild to enter the graph at all, which is a
    /// different decision from recomputing a table.
    pub segments: Vec<u32>,
    /// A few way ids per class, so the classification can be checked against
    /// the files themselves rather than believed.
    pub sample: [Vec<u64>; 4],
}

/// Index into `Changes::sample`.
pub const S_UNCHANGED: usize = 0;
pub const S_CHANGED: usize = 1;
pub const S_ADDED: usize = 2;
pub const S_DELETED: usize = 3;
const SAMPLE_MAX: usize = 60;

/// Pass 1: one `(id, hash)` per routable way of the new file, in id order.
pub fn hash_ways(pbf: &Path, paths: &Paths, dir: &Dir) -> io::Result<u64> {
    let outs = vec![paths.f("diff.wayhash")];
    let counted = scan_blobs(
        pbf,
        &dir.blobs,
        dir.first_way..dir.blobs.len(),
        64,
        &(),
        &outs,
        "diff/hash",
        |blk, _st, bufs| {
            let mut tags: Vec<(&[u8], &[u8])> = Vec::with_capacity(24);
            let mut refs: Vec<i64> = Vec::with_capacity(64);
            let mut rec: Vec<u8> = Vec::with_capacity(1 << 12);
            blk.for_each_way(|w| {
                tags.clear();
                tags.extend(w.tags(blk));
                let Some(a) = profile::classify(tags.iter().copied(), |_| profile::NO_STR) else {
                    return;
                };
                refs.clear();
                refs.extend(w.refs());
                if refs.len() < 2 {
                    return;
                }
                let name = tags.iter().find(|(k, _)| *k == b"name").map(|(_, v)| *v);
                let rf = tags.iter().find(|(k, _)| *k == b"ref").map(|(_, v)| *v);
                rec.clear();
                encode_way(
                    &mut rec,
                    w.id as u64,
                    pack_attr(&a),
                    name.unwrap_or(b""),
                    rf.unwrap_or(b""),
                    &refs,
                );
                let b = &mut bufs[0];
                b.extend_from_slice(&(w.id as u64).to_le_bytes());
                b.extend_from_slice(&hash_rec(&rec).to_le_bytes());
            });
        },
    )?;
    Ok(counted[0] / 16)
}

/// Pass 2: merge-join the new file's hashes against the stored index.
pub fn compare(live: &Path, paths: &Paths) -> io::Result<Changes> {
    let new_map = mmapvec::open(&paths.f("diff.wayhash"))?;
    let new_pairs: &[(u64, u64)] = unsafe { mmapvec::as_slice(&new_map[..]) };
    let wid_map = mmapvec::open(&live.join("way.id"))?;
    let whash_map = mmapvec::open(&live.join("way.hash"))?;
    let whead_map = mmapvec::open(&live.join("way.head"))?;
    let wseg_map = mmapvec::open(&live.join("way.seg"))?;
    let wid: &[u64] = unsafe { mmapvec::as_slice(&wid_map[..]) };
    let whash: &[u64] = unsafe { mmapvec::as_slice(&whash_map[..]) };
    let whead: &[u32] = unsafe { mmapvec::as_slice(&whead_map[..]) };
    let wseg: &[u32] = unsafe { mmapvec::as_slice(&wseg_map[..]) };

    let mut out = Changes::default();
    let mut cur = 0usize;
    for &(id, h) in new_pairs {
        // Everything the index still holds below this id is gone from the new
        // file. The join passes it by, which is what makes a deletion free to
        // detect rather than something to search for.
        while cur < wid.len() && wid[cur] < id {
            out.deleted += 1;
            if out.sample[S_DELETED].len() < SAMPLE_MAX {
                out.sample[S_DELETED].push(wid[cur]);
            }
            out.segments.extend_from_slice(&wseg[whead[cur] as usize..whead[cur + 1] as usize]);
            cur += 1;
        }
        if cur < wid.len() && wid[cur] == id {
            if whash[cur] == h {
                out.unchanged += 1;
                if out.sample[S_UNCHANGED].len() < SAMPLE_MAX && out.unchanged % 9973 == 0 {
                    out.sample[S_UNCHANGED].push(id);
                }
            } else {
                out.changed += 1;
                if out.sample[S_CHANGED].len() < SAMPLE_MAX {
                    out.sample[S_CHANGED].push(id);
                }
                out.segments
                    .extend_from_slice(&wseg[whead[cur] as usize..whead[cur + 1] as usize]);
            }
            cur += 1;
        } else {
            out.added += 1;
            if out.sample[S_ADDED].len() < SAMPLE_MAX {
                out.sample[S_ADDED].push(id);
            }
        }
    }
    while cur < wid.len() {
        out.deleted += 1;
        if out.sample[S_DELETED].len() < SAMPLE_MAX {
            out.sample[S_DELETED].push(wid[cur]);
        }
        out.segments.extend_from_slice(&wseg[whead[cur] as usize..whead[cur + 1] as usize]);
        cur += 1;
    }
    out.segments.sort_unstable();
    out.segments.dedup();
    Ok(out)
}

/// The regions, at every level, that a set of segments falls in.
///
/// The partition is nested, so a segment reaches exactly one region per level —
/// a walk up the ladder rather than a search. That is what bounds the work an
/// update causes: a closed motorway touches a few regions at level 0 and one
/// above each of them, never a cascade.
/// `cap` is the same budget the caller used for everything else, and it has to
/// be enforced *here* as well as during the join. Measured without it: the join
/// held 192-373 MB under a 256 MB cap, and then this walk took the process to
/// 1.6 GB in three seconds. It reads `seg_u`, `seg_v` and `cell.of` — over 3 GB
/// between them — and a governed mapping that nothing enforces is just a
/// mapping.
pub fn regions_of(
    ds: &Dataset,
    ov: &Overlay,
    segments: &[u32],
    mut cap: Option<&mut crate::cachecap::CacheCap>,
) -> Vec<BTreeSet<u32>> {
    let mut per_level: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); ov.levels()];
    for (n, &s) in segments.iter().enumerate() {
        if n % 65_536 == 0 {
            if let Some(c) = cap.as_deref_mut() {
                c.enforce();
            }
        }
        if s as usize >= ds.seg_u.len() {
            continue;
        }
        for v in [ds.seg_u[s as usize], ds.seg_v[s as usize]] {
            if v as usize >= ds.n_vertices() {
                continue;
            }
            let mut c = ov.cell(v);
            for (k, set) in per_level.iter_mut().enumerate() {
                if k > 0 {
                    c = ov.level(k).parent(c);
                }
                set.insert(c);
            }
        }
    }
    per_level
}

/// One level's share of a value-based update.
#[derive(Debug, Default, Clone, Copy)]
pub struct Step {
    pub level: usize,
    /// Regions examined at this level.
    pub regions: usize,
    /// Of those, the ones where at least one number moved.
    pub changed: usize,
    /// Rows recomputed.
    pub rows: usize,
    /// Rows whose numbers moved.
    pub moved: usize,
    /// True when the level stores its values eagerly and cannot be checked this
    /// way, so every region of it is treated as changed.
    pub eager: bool,
}

/// Carry an update up the ladder by measured change rather than by containment.
///
/// The structural alternative — forget every region above anything that was
/// touched — costs 52.8 % of the planet's ladder for nine days of edits, and
/// most of that is regions whose numbers did not move at all. Above rung 1
/// there are only tens of regions, each continental, so *any* edit invalidates
/// all of them.
///
/// This recomputes instead, compares, and carries upward only what actually
/// changed. Two things feed the next level's work, and the second is the one
/// that is easy to forget:
///
/// 1. The parents of regions whose table entries moved.
/// 2. The regions holding a **cut edge** whose cost changed. A level's graph is
///    its tables *plus* the real edges between its regions, so a changed speed
///    limit on a road between two regions moves the level above without
///    touching a single table entry. Leaving this out gives exactly the silent
///    staleness the whole scheme is supposed to avoid.
///
/// # What this is sound for
///
/// Metric changes — costs, closures, speed limits — which is what CRP separates
/// out and what `Overrides` applies. It is **not** sound across a topology
/// change: a new or deleted way alters which segments exist and can alter the
/// partition itself, and no comparison of values can see that. Such a change
/// needs the graph rebuilt, and then nothing here transfers.
pub fn propagate(
    ds: &Dataset,
    ovr: Option<&crate::overrides::Overrides>,
    ov: &Overlay,
    segments: &[u32],
) -> Vec<Step> {
    let endpoints = |s: u32| -> Option<(u32, u32)> {
        let i = s as usize;
        if i >= ds.seg_u.len() {
            return None;
        }
        let (u, v) = (ds.seg_u[i], ds.seg_v[i]);
        if u as usize >= ds.n_vertices() || v as usize >= ds.n_vertices() {
            return None;
        }
        Some((u, v))
    };

    // Level 0's work: the regions holding an end of a changed segment.
    let mut seed: BTreeSet<u32> = BTreeSet::new();
    for &s in segments {
        if let Some((u, v)) = endpoints(s) {
            seed.insert(ov.cell(u));
            seed.insert(ov.cell(v));
        }
    }

    let mut out = Vec::new();
    for k in 0..ov.levels() {
        let eager = !ov.level(k).is_lazy();
        let mut step = Step { level: k, regions: seed.len(), eager, ..Default::default() };
        let changed: BTreeSet<u32> = if eager {
            // Its values live in a packed table whose per-region width was
            // chosen from numbers that are about to move, so it needs
            // rebuilding rather than checking. Assume every region changed.
            seed.iter().copied().collect()
        } else {
            // In parallel over regions. Each row is an independent restricted
            // search and writes only its own bytes, which is the same property
            // the warm relies on. Left sequential this was a tenth the speed of
            // the thing it is meant to be cheaper than, which made the whole
            // comparison meaningless.
            let work: Vec<u32> = seed.iter().copied().collect();
            let found: Vec<(u32, bool, usize, usize)> = work
                .par_iter()
                .map(|&c| {
                    let b = ov.level(k).boundary(c).len();
                    let (mut rows, mut moved, mut any) = (0usize, 0usize, false);
                    for i in 0..b {
                        for fwd in [true, false] {
                            rows += 1;
                            match ov.recheck_row(ds, ovr, k, c, i, fwd) {
                                // Absent counts as moved: there is nothing to
                                // compare against, so assuming it held still
                                // would be a guess.
                                Row::Changed | Row::Written => {
                                    moved += 1;
                                    any = true;
                                }
                                Row::Same | Row::Empty => {}
                            }
                        }
                    }
                    (c, any, rows, moved)
                })
                .collect();
            let mut ch = BTreeSet::new();
            for (c, any, rows, moved) in found {
                step.rows += rows;
                step.moved += moved;
                if any {
                    ch.insert(c);
                }
            }
            ch
        };
        step.changed = changed.len();
        out.push(step);

        if k + 1 >= ov.levels() {
            break;
        }
        let mut next: BTreeSet<u32> = BTreeSet::new();
        for &c in &changed {
            next.insert(ov.level(k + 1).parent(c));
        }
        // Condition 2: cut edges. Cheap to test — a segment crosses a level-k
        // boundary when its ends sit in different level-k regions — and the
        // cost of forgetting it is a stale table nobody notices.
        for &s in segments {
            if let Some((u, v)) = endpoints(s) {
                let (cu, cv) = (ov.cell_at(k, u), ov.cell_at(k, v));
                if cu != cv {
                    next.insert(ov.level(k + 1).parent(cu));
                    next.insert(ov.level(k + 1).parent(cv));
                }
            }
        }
        if next.is_empty() {
            break;
        }
        seed = next;
    }
    out
}
