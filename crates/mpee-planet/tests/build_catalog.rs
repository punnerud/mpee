//! Atomic publication and the structural check that guards it.
//!
//! These test the property that made the catalogue worth building: a dataset
//! is either wholly live or wholly invisible. During this project the address
//! pass was killed twice mid-run, and nothing on disk knew — so the cases here
//! are the ones that actually happened, plus the ones that would have been
//! served silently if they had.

use mpee_planet::catalog::{self, Catalog};
use mpee_planet::{dataset, pipeline};
use std::path::{Path, PathBuf};

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tmp(name: &str) -> Tmp {
    let p = std::env::temp_dir().join(format!("mpee-cat-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    Tmp(p)
}

fn w(dir: &Path, name: &str, bytes: usize) {
    std::fs::write(dir.join(name), vec![0u8; bytes]).unwrap();
}
fn w_u32(dir: &Path, name: &str, vals: &[u32]) {
    let mut b = Vec::new();
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join(name), b).unwrap();
}
fn w_u64(dir: &Path, name: &str, vals: &[u64]) {
    let mut b = Vec::new();
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join(name), b).unwrap();
}

/// The smallest dataset whose array lengths all agree, in the format a build
/// produces today: packed edge targets and a dictionary attribute array.
fn minimal_dataset(dir: &Path) {
    minimal_common(dir, true)
}

/// The same, in the pre-packing format — the shape of every dataset built
/// before those arrays existed. `verify` promises to keep opening them.
fn minimal_legacy_dataset(dir: &Path) {
    minimal_common(dir, false)
}

fn minimal_common(dir: &Path, packed: bool) {
    let (v, e, s, a) = (2usize, 2usize, 1usize, 1usize);
    w_u32(dir, "csr.head", &vec![0u32; v + 1]);
    w_u32(dir, "csr.rhead", &vec![0u32; v + 1]);
    for f in ["csr.seg", "csr.rseg"] {
        w(dir, f, e * 4);
    }
    if packed {
        for f in ["to.d16", "rto.d16"] {
            w(dir, f, e * 2);
        }
        for f in ["to.exc", "rto.exc"] {
            w(dir, f, 0);
        }
        w(dir, "seg.attr16", s * 2);
        w(dir, "attr.dict", 4);
    } else {
        for f in ["csr.to", "csr.rto"] {
            w(dir, f, e * 4);
        }
        w(dir, "seg.attr", s * 4);
    }
    w(dir, "vcoord.bin", v * 8);
    for f in ["seg.len", "seg.name", "seg.u", "seg.v"] {
        w(dir, f, s * 4);
    }
    // No intermediate shape points: geom.off ends at 0, so geom.pts is empty.
    w_u32(dir, "geom.off", &vec![0u32; s + 1]);
    w(dir, "geom.pts", 0);
    w(dir, "addr.coord", a * 8);
    for f in ["addr.hn", "addr.street", "addr.place", "fwd.idx"] {
        w(dir, f, a * 4);
    }
    // Cell indexes: one key, two offsets.
    for (k, o, sg) in [
        ("snap.cell", "snap.off", "snap.seg"),
        ("big.cell", "big.off", "big.seg"),
        ("acell.key", "acell.off", ""),
    ] {
        w_u64(dir, k, &[0]);
        w_u32(dir, o, &[0, 1]);
        if !sg.is_empty() {
            w_u32(dir, sg, &[0]);
        }
    }
    // Pools: the last offset is the pool length.
    for (off, pool) in [
        ("name.off", "name.pool"),
        ("street.off", "street.pool"),
        ("hn.off", "hn.pool"),
        ("city.off", "city.pool"),
        ("pc.off", "pc.pool"),
        ("street.normoff", "street.norm"),
    ] {
        w_u64(dir, off, &[0, 4]);
        w(dir, pool, 4);
    }
    for f in ["street.sorted", "hn.num", "fwd.group", "place.tab"] {
        w(dir, f, 4);
    }
}

#[test]
fn a_minimal_dataset_verifies() {
    let t = tmp("minimal");
    minimal_dataset(&t.0);
    let p = dataset::verify(&t.0);
    assert!(p.is_empty(), "unexpected: {:?}", p.iter().map(|x| (&x.file, &x.detail)).collect::<Vec<_>>());
}

#[test]
fn a_pre_packing_dataset_still_verifies() {
    // Backward compatibility is a promise the reader makes; this is what
    // holds it to that.
    let t = tmp("legacy");
    minimal_legacy_dataset(&t.0);
    let p = dataset::verify(&t.0);
    assert!(p.is_empty(), "unexpected: {:?}", p.iter().map(|x| (&x.file, &x.detail)).collect::<Vec<_>>());
}

#[test]
fn a_dataset_with_neither_form_of_an_array_is_rejected() {
    // The either/or rule must not degenerate into "neither is fine".
    let t = tmp("neither");
    minimal_dataset(&t.0);
    std::fs::remove_file(t.0.join("to.d16")).unwrap();
    let p = dataset::verify(&t.0);
    assert!(
        p.iter().any(|x| x.detail.contains("neither form")),
        "expected the missing pair to be reported: {:?}",
        p.iter().map(|x| &x.detail).collect::<Vec<_>>()
    );
}

#[test]
fn verify_catches_a_truncated_array() {
    let t = tmp("trunc");
    minimal_dataset(&t.0);
    // A write that stopped halfway — what a killed pass leaves behind.
    w(&t.0, "rto.d16", 2);
    let p = dataset::verify(&t.0);
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].file, "rto.d16");
    assert!(p[0].detail.contains("implies 4"), "{}", p[0].detail);
}

#[test]
fn verify_catches_a_geometry_pass_that_stopped_early() {
    let t = tmp("geom");
    minimal_dataset(&t.0);
    // The offset table claims three points; the point file holds none.
    w_u32(&t.0, "geom.off", &[0, 3]);
    let p = dataset::verify(&t.0);
    assert!(
        p.iter().any(|x| x.file == "geom.pts"),
        "a geometry file shorter than its own index must be caught"
    );
}

#[test]
fn verify_catches_a_missing_artifact() {
    let t = tmp("missing");
    minimal_dataset(&t.0);
    std::fs::remove_file(t.0.join("addr.coord")).unwrap();
    let p = dataset::verify(&t.0);
    assert!(p.iter().any(|x| x.file == "addr.coord" && x.detail == "missing"));
}

#[test]
fn publish_refuses_a_build_whose_passes_did_not_all_finish() {
    let t = tmp("incomplete");
    let cat = Catalog::open(&t.0).unwrap();
    let id = catalog::new_build_id();
    cat.begin_build(&id, Path::new("planet.osm.pbf"), "car", 1.0).unwrap();
    cat.pass_start(&id, "scan").unwrap();
    cat.pass_end(&id, "scan", true, 1.0).unwrap();
    let e = cat.publish(&id).unwrap_err().to_string();
    assert!(e.contains("not complete"), "{e}");
    assert!(e.contains("contract"), "the missing passes are named: {e}");
    assert!(cat.head().is_none(), "nothing was published");
}

#[test]
fn a_failed_build_leaves_the_live_dataset_alone() {
    let t = tmp("failed");
    let cat = Catalog::open(&t.0).unwrap();

    // A first build that completes and goes live.
    let good = catalog::new_build_id();
    let dir = cat.begin_build(&good, Path::new("planet.osm.pbf"), "car", 1.0).unwrap();
    minimal_dataset(&dir);
    for p in catalog::PASSES {
        cat.pass_start(&good, p).unwrap();
        cat.pass_end(&good, p, true, 0.1).unwrap();
    }
    cat.publish(&good).unwrap();
    assert_eq!(cat.head().as_deref(), Some(good.as_str()));
    assert_eq!(catalog::resolve(&t.0), dir);

    // A second build that dies in the middle.
    let bad = format!("{good}x");
    let bad_dir = cat.begin_build(&bad, Path::new("planet.osm.pbf"), "car", 1.0).unwrap();
    cat.pass_start(&bad, "scan").unwrap();
    cat.pass_end(&bad, "scan", true, 0.1).unwrap();
    cat.pass_start(&bad, "contract").unwrap();
    // …and never ends. Readers must be unaffected.
    assert_eq!(cat.head().as_deref(), Some(good.as_str()), "HEAD did not move");
    assert_eq!(catalog::resolve(&t.0), dir, "readers still resolve to the good build");
    assert!(bad_dir.is_dir(), "the dead build is kept for inspection");
    assert!(cat.publish(&bad).is_err(), "and cannot be published");
}

#[test]
fn publishing_is_a_single_indivisible_step() {
    let t = tmp("swap");
    let cat = Catalog::open(&t.0).unwrap();
    let mut ids = Vec::new();
    for i in 0..2 {
        let id = format!("b{i:04}");
        let dir = cat.begin_build(&id, Path::new("p.pbf"), "car", 1.0).unwrap();
        minimal_dataset(&dir);
        for p in catalog::PASSES {
            cat.pass_start(&id, p).unwrap();
            cat.pass_end(&id, p, true, 0.1).unwrap();
        }
        ids.push(id);
    }
    cat.publish(&ids[0]).unwrap();
    assert_eq!(catalog::resolve(&t.0), catalog::build_dir(&t.0, &ids[0]));
    // The switch is a rename over HEAD: readers see one id or the other, and
    // HEAD is never briefly absent.
    cat.publish(&ids[1]).unwrap();
    assert_eq!(catalog::resolve(&t.0), catalog::build_dir(&t.0, &ids[1]));
    assert!(t.0.join("HEAD").exists());
    assert!(!t.0.join("HEAD.new").exists(), "no temporary is left behind");
}

#[test]
fn provenance_survives_the_build() {
    let t = tmp("prov");
    let cat = Catalog::open(&t.0).unwrap();
    let src = t.0.join("source.osm.pbf");
    std::fs::write(&src, vec![7u8; 1234]).unwrap();
    let id = catalog::new_build_id();
    let dir = cat.begin_build(&id, &src, "car", 2.5).unwrap();
    minimal_dataset(&dir);
    for p in catalog::PASSES {
        cat.pass_start(&id, p).unwrap();
        cat.pass_end(&id, p, true, 0.5).unwrap();
    }
    cat.stat(&id, "vertices", 270_173_819).unwrap();
    cat.record_artifacts(&id, &dir, false).unwrap();
    cat.publish(&id).unwrap();

    // The questions that used to require reading a build log.
    let rows = |sql: &str| match cat.query(sql).unwrap() {
        mpedb::ExecResult::Rows { rows, .. } => rows,
        _ => Vec::new(),
    };
    let b = rows(&format!(
        "SELECT source_bytes, simplify_m, profile FROM build WHERE build_id = '{id}'"
    ));
    assert_eq!(b[0][0], mpedb::Value::Int(1234), "which file produced this");
    assert_eq!(b[0][1], mpedb::Value::Numeric("2.500".into()), "at what tolerance");
    assert_eq!(b[0][2], mpedb::Value::Text("car".into()));
    let s = rows(&format!(
        "SELECT value FROM build_stat WHERE build_id = '{id}' AND key = 'vertices'"
    ));
    assert_eq!(s[0][0], mpedb::Value::Int(270_173_819));
    let arts = rows(&format!(
        "SELECT COUNT(*) FROM build_artifact WHERE build_id = '{id}' AND role = 'runtime'"
    ));
    assert!(
        matches!(arts[0][0], mpedb::Value::Int(n) if n > 40),
        "every runtime artifact is recorded"
    );
}

#[test]
fn prune_never_removes_the_live_build() {
    let t = tmp("prune");
    let cat = Catalog::open(&t.0).unwrap();
    let mut ids = Vec::new();
    for i in 0..3 {
        let id = format!("b{i:04}");
        let dir = cat.begin_build(&id, Path::new("p.pbf"), "car", 1.0).unwrap();
        minimal_dataset(&dir);
        for p in catalog::PASSES {
            cat.pass_start(&id, p).unwrap();
            cat.pass_end(&id, p, true, 0.1).unwrap();
        }
        ids.push(id);
    }
    // Publish the *oldest*, so "keep the newest" and "keep the live one" pull
    // in different directions — which is exactly where a naive prune deletes
    // the dataset out from under its readers.
    cat.publish(&ids[0]).unwrap();
    let (removed, _) = pipeline::prune(&t.0, 0).unwrap();
    assert_eq!(removed, 2, "both dead builds go");
    assert!(catalog::build_dir(&t.0, &ids[0]).is_dir(), "the live build stays");
    assert!(dataset::verify(&catalog::resolve(&t.0)).is_empty(), "and still verifies");
}

#[test]
fn prune_keeps_the_requested_number_of_recent_builds() {
    let t = tmp("prunekeep");
    let cat = Catalog::open(&t.0).unwrap();
    for i in 0..4 {
        let id = format!("b{i:04}");
        let dir = cat.begin_build(&id, Path::new("p.pbf"), "car", 1.0).unwrap();
        minimal_dataset(&dir);
        for p in catalog::PASSES {
            cat.pass_start(&id, p).unwrap();
            cat.pass_end(&id, p, true, 0.1).unwrap();
        }
    }
    cat.publish("b0000").unwrap();
    let (removed, _) = pipeline::prune(&t.0, 2).unwrap();
    assert_eq!(removed, 1, "4 builds, live one excluded, 2 kept ⇒ 1 removed");
    assert!(catalog::build_dir(&t.0, "b0003").is_dir());
    assert!(catalog::build_dir(&t.0, "b0002").is_dir());
    assert!(!catalog::build_dir(&t.0, "b0001").is_dir());
    assert!(catalog::build_dir(&t.0, "b0000").is_dir(), "live");
}
