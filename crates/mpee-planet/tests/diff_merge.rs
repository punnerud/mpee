//! The merge join that decides what an update has to redo.
//!
//! The self-diff — comparing a dataset against the very file it was built from
//! — proves the two sides hash identically, which is the hard half. But it only
//! ever exercises one branch: everything comes out unchanged. The branches that
//! actually cost something are the other three, and a mistake in any of them is
//! quiet. Missing a `changed` way leaves a stale table that routes around a
//! road that has been closed. Missing a `deleted` way does the same. Treating
//! an unchanged way as changed throws away work but is otherwise harmless,
//! which is exactly why it would go unnoticed.
//!
//! So this drives the join over synthetic index arrays, where every case can be
//! placed on purpose: at the start, in the middle, at the end, and alone.

use mpee_planet::build::Paths;
use mpee_planet::diff;
use std::io::Write;
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mpee-diff-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn put<T: Copy>(dir: &Path, name: &str, v: &[T]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
    };
    std::fs::File::create(dir.join(name)).unwrap().write_all(bytes).unwrap();
}

/// Lay down an index of `ids` where way `i` owns segments `head[i]..head[i+1]`
/// of `seg`, plus the new file's `(id, hash)` stream.
fn fixture(
    dir: &Path,
    ids: &[u64],
    hashes: &[u64],
    head: &[u32],
    seg: &[u32],
    new_pairs: &[(u64, u64)],
) {
    put(dir, "way.id", ids);
    put(dir, "way.hash", hashes);
    put(dir, "way.head", head);
    put(dir, "way.seg", seg);
    put(dir, "diff.wayhash", new_pairs);
}

#[test]
fn the_join_places_every_way_in_exactly_one_class() {
    let d = tmp("classes");
    // Five ways in the index, one segment each, ids deliberately not
    // contiguous so a "next id" assumption would fail.
    fixture(
        &d,
        &[10, 20, 30, 40, 50],
        &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE],
        &[0, 1, 2, 3, 4, 5],
        &[100, 200, 300, 400, 500],
        &[
            (5, 0x11),   // added, before everything in the index
            (10, 0xAA),  // unchanged
            (20, 0x99),  // changed
            // 30 absent -> deleted, in the middle
            (40, 0xDD),  // unchanged
            (45, 0x22),  // added, between two index entries
            (50, 0x88),  // changed, at the end
        ],
    );
    let ch = diff::compare(&d, &Paths::new(&d)).unwrap();
    assert_eq!(ch.unchanged, 2, "10 and 40 are untouched");
    assert_eq!(ch.changed, 2, "20 and 50 have new content");
    assert_eq!(ch.added, 2, "5 and 45 are not in the index at all");
    assert_eq!(ch.deleted, 1, "30 is gone from the new file");
    // Changed and deleted ways carry work; an added one has no segments yet.
    assert_eq!(ch.segments, vec![200, 300, 500]);
}

#[test]
fn a_deletion_at_the_very_end_is_not_missed() {
    // The tail is its own branch: the loop over the new file has ended, so
    // whatever the index still holds has to be drained afterwards. Forgetting
    // that is the easy mistake, and it silently keeps stale tables for every
    // way with an id above the last one in the new file.
    let d = tmp("tail");
    fixture(
        &d,
        &[1, 2, 3],
        &[0x1, 0x2, 0x3],
        &[0, 2, 3, 5],
        &[11, 12, 13, 14, 15],
        &[(1, 0x1)],
    );
    let ch = diff::compare(&d, &Paths::new(&d)).unwrap();
    assert_eq!((ch.unchanged, ch.changed, ch.added, ch.deleted), (1, 0, 0, 2));
    assert_eq!(ch.segments, vec![13, 14, 15], "both tail ways' segments");
}

#[test]
fn an_empty_new_file_deletes_everything_rather_than_reporting_nothing() {
    let d = tmp("empty-new");
    fixture(&d, &[7, 8], &[0x7, 0x8], &[0, 1, 2], &[70, 80], &[]);
    let ch = diff::compare(&d, &Paths::new(&d)).unwrap();
    assert_eq!((ch.unchanged, ch.changed, ch.added, ch.deleted), (0, 0, 0, 2));
    assert_eq!(ch.segments, vec![70, 80]);
}

#[test]
fn an_empty_index_makes_every_way_new() {
    let d = tmp("empty-index");
    fixture(&d, &[], &[], &[0u32], &[], &[(1, 0x1), (2, 0x2)]);
    let ch = diff::compare(&d, &Paths::new(&d)).unwrap();
    assert_eq!((ch.unchanged, ch.changed, ch.added, ch.deleted), (0, 0, 2, 0));
    assert!(ch.segments.is_empty(), "a way that is not in the graph has no segments");
}

#[test]
fn a_way_that_produced_several_segments_invalidates_all_of_them() {
    // One OSM way splits at junctions into a run of segments. Invalidating
    // only the first would leave the rest of the road stale — the failure
    // would be a route that is correct up to a junction and wrong after it.
    let d = tmp("run");
    fixture(
        &d,
        &[100, 200],
        &[0xF, 0xF],
        &[0, 4, 6],
        &[1, 2, 3, 4, 90, 91],
        &[(100, 0xEE), (200, 0xF)],
    );
    let ch = diff::compare(&d, &Paths::new(&d)).unwrap();
    assert_eq!(ch.changed, 1);
    assert_eq!(ch.segments, vec![1, 2, 3, 4], "the whole run, not just its first");
}

#[test]
fn segments_come_back_sorted_and_without_repeats() {
    // Two ways can share a segment id only if the index is wrong, but two
    // *changed* ways certainly produce overlapping work for the regions above
    // them, and the caller invalidates by segment. Duplicates would mean doing
    // the same region twice; unsorted output would defeat the dedup entirely.
    let d = tmp("sorted");
    fixture(
        &d,
        &[1, 2, 3],
        &[0xA, 0xB, 0xC],
        &[0, 2, 4, 6],
        &[50, 10, 30, 10, 20, 40],
        &[(1, 0x0), (2, 0x0), (3, 0x0)],
    );
    let ch = diff::compare(&d, &Paths::new(&d)).unwrap();
    assert_eq!(ch.changed, 3);
    assert_eq!(ch.segments, vec![10, 20, 30, 40, 50]);
}
