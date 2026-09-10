//! Traffic is keyed by vertex, and that is the whole point.
//!
//! The use counters a level gathers are indexed by position in its `vlist`,
//! which belongs to the partition: rebuild the level and index 400 means a
//! different road. The cut that wants to know where the traffic runs is made
//! *while* the level is being rebuilt, so the signal has to be stored against
//! something that survives — the vertex id.

use mpee_planet::overlay::Traffic;
use std::path::{Path, PathBuf};

struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Dir {
        let p = std::env::temp_dir().join(format!(
            "mpee-traffic-{tag}-{}-{:?}",
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
    fn write(&self, pairs: &[(u32, u32)]) {
        let b: &[u8] =
            unsafe { std::slice::from_raw_parts(pairs.as_ptr() as *const u8, pairs.len() * 8) };
        std::fs::write(self.0.join("traffic.bin"), b).unwrap();
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_dataset_that_was_never_queried_says_nothing_rather_than_failing() {
    // The build path calls this before any query has run. It must degrade to
    // "no opinion", not to an error and not to a wrong opinion.
    let d = Dir::new("absent");
    let t = Traffic::open(d.path());
    assert!(t.is_empty());
    assert_eq!(t.get(0), 0);
    assert_eq!(t.get(4_000_000_000), 0);
}

#[test]
fn a_vertex_gets_back_the_count_it_was_given() {
    let d = Dir::new("roundtrip");
    let pairs = vec![(3u32, 70u32), (9, 1), (17, 560_051), (1_000_000, 42)];
    d.write(&pairs);
    let t = Traffic::open(d.path());
    assert_eq!(t.len(), 4);
    for &(v, c) in &pairs {
        assert_eq!(t.get(v), c, "vertex {v}");
    }
}

#[test]
fn a_vertex_with_no_row_reads_as_untravelled_not_as_the_neighbouring_row() {
    // The lookup is a binary search over a sparse table, so a miss must return
    // zero rather than whatever sits at the insertion point. Getting this wrong
    // would hand a quiet road its busy neighbour's weight.
    let d = Dir::new("miss");
    d.write(&[(10, 111), (20, 222), (30, 333)]);
    let t = Traffic::open(d.path());
    for v in [0u32, 9, 11, 19, 21, 29, 31, 99] {
        assert_eq!(t.get(v), 0, "vertex {v} should be silent");
    }
    assert_eq!(t.get(20), 222);
}

#[test]
fn the_table_is_sparse_because_most_of_the_planet_is_never_settled() {
    // 16 % of the planet's level-0 gates have ever been settled. Storing the
    // pairs is 14 MB; a dense array over every vertex would be 1.08 GB. This
    // pins the shape the size argument rests on.
    let d = Dir::new("sparse");
    let pairs: Vec<(u32, u32)> = (0..1000u32).map(|i| (i * 977, i + 1)).collect();
    d.write(&pairs);
    let t = Traffic::open(d.path());
    assert_eq!(t.len(), 1000);
    assert_eq!(t.get(0), 1);
    assert_eq!(t.get(977 * 999), 1000);
    assert_eq!(t.get(977 * 999 - 1), 0);
}
