//! Hold the hand-rolled PBF reader to a reference implementation.
//!
//! A planet build runs for hours, so a silent parse bug is expensive. This
//! test replays a real extract through both this crate and the `osmpbf` crate
//! and compares aggregate invariants that any disagreement would break:
//! counts, id checksums, coordinate checksums, ref checksums and tag
//! checksums. Set `MPEE_TEST_PBF` to run it; skipped otherwise so `cargo test`
//! stays hermetic.

use mpee_planet::pbf::{self, BlobKind, Block, Inflater};
use std::fs::File;
use std::path::PathBuf;

#[derive(Default, Debug, PartialEq)]
struct Sums {
    nodes: u64,
    node_id_sum: i128,
    lat_sum: i128,
    lon_sum: i128,
    node_tags: u64,
    ways: u64,
    way_id_sum: i128,
    refs: u64,
    ref_sum: i128,
    way_tags: u64,
    tag_bytes: u64,
}

fn test_pbf() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("MPEE_TEST_PBF").ok()?);
    p.exists().then_some(p)
}

fn ours(path: &PathBuf) -> Sums {
    let blobs = pbf::index_blobs(path).expect("index");
    let f = File::open(path).unwrap();
    let mut inf = Inflater::default();
    let mut s = Sums::default();
    for d in blobs.iter().filter(|d| d.kind == BlobKind::Data) {
        inf.load(&f, d).expect("inflate");
        let blk = Block::parse(&inf.out);
        blk.for_each_node(|n| {
            s.nodes += 1;
            s.node_id_sum += n.id as i128;
            s.lat_sum += n.lat_e7 as i128;
            s.lon_sum += n.lon_e7 as i128;
            let mut i = 0;
            while i + 1 < n.kv.len() && n.kv[i] != 0 {
                s.node_tags += 1;
                s.tag_bytes += (blk.s(n.kv[i]).len() + blk.s(n.kv[i + 1]).len()) as u64;
                i += 2;
            }
        });
        blk.for_each_way(|w| {
            s.ways += 1;
            s.way_id_sum += w.id as i128;
            for r in w.refs() {
                s.refs += 1;
                s.ref_sum += r as i128;
            }
            for (k, v) in w.tags(&blk) {
                s.way_tags += 1;
                s.tag_bytes += (k.len() + v.len()) as u64;
            }
        });
    }
    s
}

fn reference(path: &PathBuf) -> Sums {
    use osmpbf::{Element, ElementReader};
    let mut s = Sums::default();
    ElementReader::from_path(path)
        .unwrap()
        .for_each(|e| match e {
            Element::Node(n) => {
                s.nodes += 1;
                s.node_id_sum += n.id() as i128;
                s.lat_sum += (n.lat() * 1e7).round() as i128;
                s.lon_sum += (n.lon() * 1e7).round() as i128;
                for (k, v) in n.tags() {
                    s.node_tags += 1;
                    s.tag_bytes += (k.len() + v.len()) as u64;
                }
            }
            Element::DenseNode(n) => {
                s.nodes += 1;
                s.node_id_sum += n.id() as i128;
                s.lat_sum += (n.lat() * 1e7).round() as i128;
                s.lon_sum += (n.lon() * 1e7).round() as i128;
                for (k, v) in n.tags() {
                    s.node_tags += 1;
                    s.tag_bytes += (k.len() + v.len()) as u64;
                }
            }
            Element::Way(w) => {
                s.ways += 1;
                s.way_id_sum += w.id() as i128;
                for r in w.refs() {
                    s.refs += 1;
                    s.ref_sum += r as i128;
                }
                for (k, v) in w.tags() {
                    s.way_tags += 1;
                    s.tag_bytes += (k.len() + v.len()) as u64;
                }
            }
            Element::Relation(_) => {}
        })
        .unwrap();
    s
}

#[test]
fn matches_the_osmpbf_crate() {
    let Some(path) = test_pbf() else {
        eprintln!("MPEE_TEST_PBF not set — skipping conformance test");
        return;
    };
    let a = ours(&path);
    let b = reference(&path);
    assert_eq!(a.nodes, b.nodes, "node count");
    assert_eq!(a.node_id_sum, b.node_id_sum, "node id checksum");
    assert_eq!(a.lat_sum, b.lat_sum, "latitude checksum (e7)");
    assert_eq!(a.lon_sum, b.lon_sum, "longitude checksum (e7)");
    assert_eq!(a.node_tags, b.node_tags, "node tag count");
    assert_eq!(a.ways, b.ways, "way count");
    assert_eq!(a.way_id_sum, b.way_id_sum, "way id checksum");
    assert_eq!(a.refs, b.refs, "way ref count");
    assert_eq!(a.ref_sum, b.ref_sum, "way ref checksum");
    assert_eq!(a.way_tags, b.way_tags, "way tag count");
    assert_eq!(a.tag_bytes, b.tag_bytes, "tag byte checksum");
    assert!(a.nodes > 1_000_000, "extract looks too small to be meaningful");
}

/// The node→way boundary search must land on the first blob holding ways and
/// must not skip any node blob before it.
#[test]
fn way_boundary_is_exact() {
    let Some(path) = test_pbf() else { return };
    let blobs = pbf::index_blobs(&path).unwrap();
    let first = pbf::find_first_way_blob(&path, &blobs).unwrap();
    let f = File::open(&path).unwrap();
    let mut inf = Inflater::default();
    let mut nodes_after = 0u64;
    let mut ways_before = 0u64;
    for (i, d) in blobs.iter().enumerate() {
        if d.kind != BlobKind::Data {
            continue;
        }
        inf.load(&f, d).unwrap();
        let blk = Block::parse(&inf.out);
        if i < first {
            blk.for_each_way(|_| ways_before += 1);
        } else {
            blk.for_each_node(|_| nodes_after += 1);
        }
    }
    assert_eq!(ways_before, 0, "ways found before the computed boundary");
    assert_eq!(nodes_after, 0, "nodes found after the computed boundary");
}
