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
use crate::overlay::Overlay;
use crate::parallel::scan_blobs;
use crate::profile;
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
pub fn regions_of(ds: &Dataset, ov: &Overlay, segments: &[u32]) -> Vec<BTreeSet<u32>> {
    let mut per_level: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); ov.levels()];
    for &s in segments {
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
