//! A lazy level must refuse to open unless it is laid out the way this build
//! reads it.
//!
//! The failure this guards against is silent, which is why it needs a test of
//! its own. A table addressed with the wrong formula does not fault: it returns
//! some other region's entries, which are perfectly valid costs, so the search
//! settles and reports a route. The planet answered Paris-Berlin in 52 seconds
//! — a third of what it charged for Paris-Lyon, half the distance — and no
//! layer between the mmap and the answer noticed anything.
//!
//! These tests build the file shapes by hand rather than through a build, so
//! they run everywhere and do not depend on `MPEE_TEST_DATA` being set. The
//! guard in `tests/overlay.rs` once disabled that whole file without failing
//! anything; this file has nothing to disable.

use mpee_planet::overlay::{fmt_file, lazy_files, verify_lazy_layout, write_lazy_stamp, LAZY_LAYOUT};
use std::path::{Path, PathBuf};

/// A scratch directory that removes itself.
struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Dir {
        let p = std::env::temp_dir().join(format!(
            "mpee-lazy-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Boundary counts -> the `bhead` prefix sum every level carries.
fn bhead_of(bs: &[u32]) -> Vec<u32> {
    let mut v = vec![0u32];
    for b in bs {
        v.push(v.last().unwrap() + b);
    }
    v
}

/// Today's layout: each region gets a self-contained `2b²` block.
fn head_rev2(bs: &[u32]) -> Vec<u64> {
    let mut v = vec![0u64];
    for &b in bs {
        v.push(v.last().unwrap() + 2 * (b as u64) * (b as u64));
    }
    v
}

/// The layout the planet was built in: `b²` per region, with every backward
/// row in a second level-wide area that begins at the level's total. The file
/// is the same size either way — `2Σb²` entries — which is the whole reason
/// this went unnoticed.
fn head_rev1(bs: &[u32]) -> Vec<u64> {
    let mut v = vec![0u64];
    for &b in bs {
        v.push(v.last().unwrap() + (b as u64) * (b as u64));
    }
    v
}

/// Lay down the files a level needs, sized for `entries` table entries.
fn write_level(d: &Path, bs: &[u32], entries: u64) {
    let lf = lazy_files(0);
    let nb: u32 = bs.iter().sum();
    std::fs::File::create(d.join(&lf[2])).unwrap().set_len(entries * 4).unwrap();
    std::fs::File::create(d.join(&lf[3]))
        .unwrap()
        .set_len((2 * nb as u64).div_ceil(8))
        .unwrap();
}

const BS: [u32; 4] = [3, 5, 2, 7];

fn nb() -> usize {
    BS.iter().sum::<u32>() as usize
}

/// `2Σb²` — the size of the table, identical under both revisions.
fn entries() -> u64 {
    BS.iter().map(|&b| 2 * (b as u64) * (b as u64)).sum()
}

#[test]
fn a_level_in_todays_layout_verifies_without_a_stamp() {
    // The migration case: every level built before stamps existed is
    // unstamped, and the sound ones have to keep opening.
    let d = Dir::new("ok");
    write_level(d.path(), &BS, entries());
    let r = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev2(&BS), nb());
    assert!(r.is_ok(), "{r:?}");
}

#[test]
fn the_planet_bug_is_caught_with_no_stamp_to_disagree_with() {
    // The level that broke the planet: `head` written with a b² stride by one
    // build, the table re-sized for 2b² by a later one. Both files are
    // individually well-formed and the total size is right, so nothing but the
    // relation between them gives it away.
    let d = Dir::new("rev1");
    write_level(d.path(), &BS, entries());
    let e = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev1(&BS), nb())
        .expect_err("a b² stride must not be accepted by a build that reads 2b²");
    assert!(e.contains("revision 1"), "{e}");
    assert!(e.contains("Rebuild"), "{e}");
}

#[test]
fn the_first_region_that_overlaps_is_the_one_reported() {
    // Region 0 sits at offset 0 under either revision, so the first region that
    // can overlap is region 0 against region 1 — and the message should name
    // the region a reader can go and look at, not just say "corrupt".
    let d = Dir::new("which");
    write_level(d.path(), &BS, entries());
    let e = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev1(&BS), nb()).unwrap_err();
    assert!(e.contains("region 0"), "{e}");
    assert!(e.contains("3 boundary vertices"), "{e}");
}

#[test]
fn a_stamp_from_an_older_revision_is_refused() {
    let d = Dir::new("oldstamp");
    write_level(d.path(), &BS, entries());
    // A stamp written by a build one revision back.
    let mut b = Vec::new();
    b.extend_from_slice(b"MPEEOVL\0");
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&(BS.len() as u64).to_le_bytes());
    b.extend_from_slice(&(nb() as u64).to_le_bytes());
    b.extend_from_slice(&entries().to_le_bytes());
    std::fs::write(d.path().join(fmt_file(0)), &b).unwrap();
    let e = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev2(&BS), nb()).unwrap_err();
    assert!(e.contains("revision 1"), "{e}");
    assert!(e.contains(&format!("revision {LAZY_LAYOUT}")), "{e}");
}

#[test]
fn a_fresh_build_stamps_what_it_actually_wrote() {
    let d = Dir::new("fresh");
    write_level(d.path(), &BS, entries());
    write_lazy_stamp(&d.path().join(fmt_file(0)), BS.len() as u64, nb() as u64, entries()).unwrap();
    let r = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev2(&BS), nb());
    assert!(r.is_ok(), "{r:?}");
}

#[test]
fn a_stamp_that_disagrees_about_the_region_count_is_refused() {
    // Half a rebuild: the table was rewritten with more regions than the stamp
    // beside it records.
    let d = Dir::new("halfbuilt");
    write_level(d.path(), &BS, entries());
    write_lazy_stamp(&d.path().join(fmt_file(0)), 99, nb() as u64, entries()).unwrap();
    let e = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev2(&BS), nb()).unwrap_err();
    assert!(e.contains("99 regions"), "{e}");
}

#[test]
fn a_stamp_that_disagrees_about_the_table_size_is_refused() {
    let d = Dir::new("wrongsize");
    write_level(d.path(), &BS, entries());
    write_lazy_stamp(&d.path().join(fmt_file(0)), BS.len() as u64, nb() as u64, entries() + 8)
        .unwrap();
    let e = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev2(&BS), nb()).unwrap_err();
    assert!(e.contains("table entries"), "{e}");
}

#[test]
fn a_table_too_small_for_its_last_region_is_refused() {
    // Truncation, which a size check catches even where the strides are right.
    let d = Dir::new("short");
    write_level(d.path(), &BS, entries() - 1);
    let e = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev2(&BS), nb()).unwrap_err();
    assert!(e.contains("the table holds"), "{e}");
}

#[test]
fn presence_bits_too_short_for_the_rows_are_refused() {
    let d = Dir::new("bits");
    write_level(d.path(), &BS, entries());
    let lf = lazy_files(0);
    std::fs::File::create(d.path().join(&lf[3])).unwrap().set_len(1).unwrap();
    let e = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev2(&BS), nb()).unwrap_err();
    assert!(e.contains("presence bits"), "{e}");
}

#[test]
fn offsets_out_of_region_order_are_still_accepted_when_the_blocks_fit() {
    // The layout is log-structured: a split appends a changed region's rows at
    // the end and leaves the unchanged ones where they are, so `head` stops
    // being sorted. That is normal, and an overlap rule that assumed order
    // would reject every dataset that had ever been adapted.
    let d = Dir::new("unsorted");
    // Region 0's block moved to the end; region 3's old space now sits unused.
    let bs = [3u32, 5, 2, 7];
    let mut head = head_rev2(&bs);
    let moved = head[4]; // one past the last block
    head[0] = moved;
    let total = moved + 2 * 3 * 3;
    write_level(d.path(), &bs, total);
    let r = verify_lazy_layout(d.path(), 0, &bhead_of(&bs), &head, nb());
    assert!(r.is_ok(), "{r:?}");
}

#[test]
fn a_level_with_no_table_at_all_is_not_this_check_s_business() {
    // An eager level, or a rung that simply is not there. The caller stops on
    // its own; this must not turn absence into an error.
    let d = Dir::new("absent");
    let r = verify_lazy_layout(d.path(), 0, &bhead_of(&BS), &head_rev2(&BS), nb());
    assert!(r.is_ok(), "{r:?}");
}
