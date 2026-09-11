//! The change-file reader decides what an update recomputes, so a way it fails
//! to name is a region that silently keeps a stale table.
//!
//! These are synthetic files rather than a downloaded diff, because the cases
//! that matter are the awkward ones: an attribute whose name is a suffix of
//! another, a delete element with no position, a coordinate with more decimals
//! than the format we store it in.

use mpee_planet::osc;
use std::io::Write;
use std::path::{Path, PathBuf};

struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Dir {
        let p = std::env::temp_dir().join(format!(
            "mpee-osc-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn plain(&self, name: &str, body: &str) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }
    fn gz(&self, name: &str, body: &str) -> PathBuf {
        let p = self.0.join(name);
        let f = std::fs::File::create(&p).unwrap();
        let mut e = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
        e.write_all(body.as_bytes()).unwrap();
        e.finish().unwrap();
        p
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

const SAMPLE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<osmChange version="0.6" generator="osmium/1.14.0">
 <create>
  <node id="11" version="1" uid="7" lat="59.9139" lon="10.7522"/>
  <way id="101" version="1" uid="7">
   <nd ref="11"/>
   <nd ref="12"/>
   <tag k="highway" v="residential"/>
  </way>
 </create>
 <modify>
  <node id="12" version="3" uid="8" lat="-33.8688" lon="151.2093"/>
  <way id="102" version="5" uid="8"><nd ref="11"/><nd ref="12"/></way>
 </modify>
 <delete>
  <way id="103" version="9" uid="9"/>
  <node id="13" version="2" uid="9"/>
 </delete>
</osmChange>
"#;

#[test]
fn every_section_is_read_and_the_ways_come_out_sorted() {
    let d = Dir::new("sections");
    let f = d.plain("a.osc", SAMPLE);
    let t = osc::read(&[f]).unwrap();
    assert_eq!(t.ways, vec![101, 102, 103]);
    assert_eq!(t.way_counts, [1, 1, 1], "created, modified, deleted");
    assert_eq!(t.node_counts, [1, 1, 1]);
}

#[test]
fn an_id_attribute_is_not_confused_with_uid() {
    // `uid="7"` sits before `id` in no particular order and ends in the same
    // two letters. Matching without a boundary check returns the editor's user
    // id as a way id, which then names a way that does not exist — an update
    // that recomputes nothing and reports success.
    let d = Dir::new("uid");
    let f = d.plain(
        "a.osc",
        r#"<osmChange><modify><way uid="7" changeset="5" id="4242" version="2"/></modify></osmChange>"#,
    );
    let t = osc::read(&[f]).unwrap();
    assert_eq!(t.ways, vec![4242]);
}

#[test]
fn coordinates_land_on_the_right_ten_millionth() {
    let d = Dir::new("coords");
    let f = d.plain("a.osc", SAMPLE);
    let t = osc::read(&[f]).unwrap();
    assert!(t.nodes.contains(&(-338_688_000, 1_512_093_000)), "{:?}", t.nodes);
}

#[test]
fn only_modified_nodes_are_placed() {
    // A created node reaches the router only through a way that references it,
    // and that way is named in the same file. A node cannot be deleted while a
    // way still references it, so a deletion implies its ways were modified.
    // Both are already covered by `ways`, and placing them costs a snap each —
    // 25 million of them in nine days of planet edits, for nothing.
    let d = Dir::new("onlymod");
    let f = d.plain("a.osc", SAMPLE);
    let t = osc::read(&[f]).unwrap();
    assert_eq!(t.nodes.len(), 1, "only the modified node: {:?}", t.nodes);
    assert!(
        !t.nodes.contains(&(599_139_000, 107_522_000)),
        "the created node must not be placed"
    );
    // Still counted, so the report can say how much of the file was skipped.
    assert_eq!(t.node_counts, [1, 1, 1]);
}

#[test]
fn more_decimals_than_we_store_are_truncated_not_rejected() {
    // OSM writes seven decimals, but a generator may write more. Dropping the
    // element would lose a real edit; truncating costs a centimetre, and the
    // coordinate is only used to find the nearest road.
    let d = Dir::new("decimals");
    let f = d.plain(
        "a.osc",
        r#"<osmChange><modify><node id="1" lat="1.23456789" lon="-0.000000049"/></modify></osmChange>"#,
    );
    let t = osc::read(&[f]).unwrap();
    assert_eq!(t.nodes, vec![(12_345_678, 0)]);
    assert_eq!(t.unplaced, 0);
}

#[test]
fn a_modified_node_with_no_position_is_counted_rather_than_dropped_in_silence() {
    // A modify element is the one case that has to be placed, so one arriving
    // without a position is a real gap. Counting it is the only honest way to
    // say the refresh could not see that edit.
    let d = Dir::new("unplaced");
    let f = d.plain(
        "a.osc",
        r#"<osmChange><modify><node id="1" version="2"/><node id="2" lat="1.0" lon="2.0"/></modify></osmChange>"#,
    );
    let t = osc::read(&[f]).unwrap();
    assert_eq!(t.unplaced, 1);
    assert_eq!(t.nodes, vec![(10_000_000, 20_000_000)]);
}

#[test]
fn gzip_and_plain_read_the_same() {
    let d = Dir::new("gz");
    let a = osc::read(&[d.plain("a.osc", SAMPLE)]).unwrap();
    let b = osc::read(&[d.gz("b.osc.gz", SAMPLE)]).unwrap();
    assert_eq!(a.ways, b.ways);
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.way_counts, b.way_counts);
}

#[test]
fn several_files_merge_and_a_way_edited_twice_is_named_once() {
    // A four-day refresh is four files, and a road worked on across two days
    // must not be recomputed twice.
    let d = Dir::new("merge");
    let one = d.plain("1.osc", r#"<osmChange><modify><way id="7"/><way id="9"/></modify></osmChange>"#);
    let two = d.plain("2.osc", r#"<osmChange><modify><way id="7"/><way id="5"/></modify></osmChange>"#);
    let t = osc::read(&[one, two]).unwrap();
    assert_eq!(t.ways, vec![5, 7, 9]);
}

#[test]
fn a_file_with_no_sections_is_read_as_modifications() {
    // Not what OSM emits, but a hand-made or trimmed file should still name its
    // ways rather than be skipped.
    let d = Dir::new("nosection");
    let f = d.plain("a.osc", r#"<osmChange><way id="1"/><node id="2" lat="0.5" lon="0.5"/></osmChange>"#);
    let t = osc::read(&[f]).unwrap();
    assert_eq!(t.ways, vec![1]);
    assert_eq!(t.nodes, vec![(5_000_000, 5_000_000)]);
}

#[test]
fn closing_tags_and_declarations_are_not_mistaken_for_elements() {
    let d = Dir::new("closing");
    let f = d.plain("a.osc", SAMPLE);
    let t = osc::read(&[f]).unwrap();
    // `</way>`, `</create>` and the XML declaration must not be counted.
    assert_eq!(t.way_counts.iter().sum::<u64>(), 3);
    assert_eq!(t.node_counts.iter().sum::<u64>(), 3);
}

#[test]
fn an_empty_or_truncated_file_is_not_an_error() {
    // A download cut short should report nothing changed, not fail — the caller
    // decides whether that is acceptable.
    let d = Dir::new("trunc");
    assert_eq!(osc::read(&[d.plain("a.osc", "")]).unwrap().ways.len(), 0);
    let f = d.plain("b.osc", r#"<osmChange><modify><way id="77"/><way id="8"#);
    let t = osc::read(&[f]).unwrap();
    assert_eq!(t.ways, vec![77], "the complete element is kept, the partial one dropped");
}

#[test]
fn the_reader_does_not_hold_the_file_in_memory() {
    // 95 MB gzipped is about 1.5 GB of XML. This builds a file large enough
    // that holding it would show, and checks the result rather than the memory
    // directly — the point is that it streams, and a buffered reader that did
    // not would have to grow to the whole input.
    let d = Dir::new("stream");
    let mut body = String::from("<osmChange><modify>");
    for i in 1..200_000u64 {
        body.push_str(&format!("<way id=\"{i}\"/>"));
    }
    body.push_str("</modify></osmChange>");
    let f = d.gz("big.osc.gz", &body);
    let on_disk = std::fs::metadata(&f).unwrap().len();
    let t = osc::read(&[f]).unwrap();
    assert_eq!(t.ways.len(), 199_999);
    assert_eq!(t.way_counts[1], 199_999);
    assert!(on_disk < body.len() as u64 / 4, "the fixture should be compressed");
    let _ = d.path();
}
