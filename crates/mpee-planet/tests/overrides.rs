//! Local edits: what they bind to, what they do, and what they refuse.
//!
//! The property worth testing hardest is the one that is easy to get wrong and
//! silent when it is: an override must name a *place*, not a segment id.
//! Segment ids are build-local, so a rule stored against one would quietly
//! start applying to a different road after the next rebuild — a bug that
//! looks like data. Everything here is therefore expressed in coordinates and
//! street names, and the routing tests check the effect rather than the
//! binding.

use mpee_planet::dataset::Dataset;
use mpee_planet::overrides::{self, Effect, OverrideDb, Overrides, KIND_CLOSED, KIND_PENALTY, KIND_SPEED};
use mpee_planet::router::Router;
use std::path::PathBuf;

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tmp(name: &str) -> Tmp {
    let p = std::env::temp_dir().join(format!("mpee-ov-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    Tmp(p)
}

fn data() -> Option<Dataset> {
    let p = mpee_planet::catalog::resolve(&PathBuf::from(std::env::var("MPEE_TEST_DATA").ok()?));
    p.join("csr.head").exists().then(|| Dataset::open(&p).unwrap())
}

/// A place in Oslo with a named street through it, used by the routing cases.
const CLOSURE_LAT: f64 = 59.9232;
const CLOSURE_LON: f64 = 10.7295;
const CLOSURE_STREET: &str = "Hegdehaugsveien";
const FROM: (f64, f64) = (59.9276, 10.7168);
const TO: (f64, f64) = (59.9089, 10.7460);

fn snap(ds: &Dataset, p: (f64, f64)) -> mpee_planet::router::Snap {
    ds.snap((p.0 * 1e7) as i32, (p.1 * 1e7) as i32, 500.0).unwrap()
}

// --------------------------------------------------------------- validation

#[test]
fn a_rule_that_cannot_mean_anything_is_refused() {
    let t = tmp("validate");
    let db = OverrideDb::open(&t.0).unwrap();
    // A speed override with no speed would silently do nothing.
    let e = db
        .add(KIND_SPEED, 59.9, 10.7, 30, None, None, None, "", "t", None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("--speed"), "{e}");
    // So would a penalty with no factor, or a zero one.
    assert!(db.add(KIND_PENALTY, 59.9, 10.7, 30, None, None, None, "", "t", None).is_err());
    assert!(db.add(KIND_PENALTY, 59.9, 10.7, 30, None, None, Some(0.0), "", "t", None).is_err());
    // And a kind nobody implements.
    assert!(db.add("teleport", 59.9, 10.7, 30, None, None, None, "", "t", None).is_err());
    // A closure needs nothing else.
    assert!(db.add(KIND_CLOSED, 59.9, 10.7, 30, None, None, None, "", "t", None).is_ok());
}

#[test]
fn every_write_moves_the_generation() {
    let t = tmp("gen");
    let db = OverrideDb::open(&t.0).unwrap();
    let g0 = db.generation();
    let id = db.add(KIND_CLOSED, 59.9, 10.7, 30, None, None, None, "", "t", None).unwrap();
    let g1 = db.generation();
    assert!(g1 > g0, "adding bumps it: {g0} -> {g1}");
    db.remove(id).unwrap();
    assert!(db.generation() > g1, "so does removing — a server must notice a lifted closure too");
}

#[test]
fn rules_round_trip() {
    let t = tmp("roundtrip");
    let db = OverrideDb::open(&t.0).unwrap();
    db.add(KIND_SPEED, 59.91, 10.75, 120, Some("Storgata"), Some(30), None, "school", "m", None)
        .unwrap();
    let r = &db.list().unwrap()[0];
    assert_eq!(r.kind, KIND_SPEED);
    assert_eq!(r.speed_kmh, Some(30));
    assert_eq!(r.street.as_deref(), Some("Storgata"));
    assert_eq!(r.radius_m, 120);
    assert_eq!(r.note, "school");
    // Coordinates survive the trip through e7 storage.
    assert!((r.lat_e7 as f64 * 1e-7 - 59.91).abs() < 1e-6);
}

// ---------------------------------------------------------------- binding

#[test]
fn a_rule_binds_by_place_and_reports_what_it_matched() {
    let Some(ds) = data() else { return };
    let t = tmp("bind");
    let db = OverrideDb::open(&t.0).unwrap();
    let id = db
        .add(KIND_CLOSED, CLOSURE_LAT, CLOSURE_LON, 300, Some(CLOSURE_STREET), None, None, "", "t", None)
        .unwrap();
    let ov = Overrides::resolve(&ds, &db.list().unwrap(), db.generation());
    let matched = ov.matched.iter().find(|(i, _)| *i == id).unwrap().1;
    assert!(matched > 1, "a named street spans several segments, got {matched}");
    assert_eq!(ov.len(), matched);
}

#[test]
fn a_rule_that_matches_nothing_says_so_instead_of_failing_quietly() {
    let Some(ds) = data() else { return };
    let t = tmp("nomatch");
    let db = OverrideDb::open(&t.0).unwrap();
    let id = db
        .add(KIND_CLOSED, CLOSURE_LAT, CLOSURE_LON, 300, Some("Nonexistent Street"), None, None, "", "t", None)
        .unwrap();
    let ov = Overrides::resolve(&ds, &db.list().unwrap(), db.generation());
    assert_eq!(ov.matched.iter().find(|(i, _)| *i == id).unwrap().1, 0);
    assert!(ov.is_empty(), "and nothing is overridden as a result");
}

#[test]
fn the_street_filter_narrows_the_radius() {
    let Some(ds) = data() else { return };
    let wide = overrides::segments_near(&ds, (CLOSURE_LAT * 1e7) as i32, (CLOSURE_LON * 1e7) as i32, 300.0, None);
    let named = overrides::segments_near(
        &ds,
        (CLOSURE_LAT * 1e7) as i32,
        (CLOSURE_LON * 1e7) as i32,
        300.0,
        Some(CLOSURE_STREET),
    );
    assert!(!named.is_empty());
    assert!(named.len() < wide.len(), "naming a street selects a subset of what is nearby");
    assert!(named.iter().all(|s| wide.contains(s)));
}

#[test]
fn an_expired_rule_does_nothing() {
    let Some(ds) = data() else { return };
    let t = tmp("expired");
    let db = OverrideDb::open(&t.0).unwrap();
    // Expired an hour ago.
    let id = db
        .add(
            KIND_CLOSED, CLOSURE_LAT, CLOSURE_LON, 300, Some(CLOSURE_STREET), None, None,
            "yesterday's roadworks", "t",
            Some(overrides::now_us() - 3_600_000_000),
        )
        .unwrap();
    let ov = Overrides::resolve(&ds, &db.list().unwrap(), db.generation());
    assert!(ov.is_empty(), "an expired closure must not still be closing the road");
    assert_eq!(ov.matched.iter().find(|(i, _)| *i == id).unwrap().1, 0);
}

#[test]
fn a_later_rule_wins_over_an_earlier_one() {
    let Some(ds) = data() else { return };
    let t = tmp("later");
    let db = OverrideDb::open(&t.0).unwrap();
    db.add(KIND_CLOSED, CLOSURE_LAT, CLOSURE_LON, 300, Some(CLOSURE_STREET), None, None, "", "t", None)
        .unwrap();
    db.add(KIND_SPEED, CLOSURE_LAT, CLOSURE_LON, 300, Some(CLOSURE_STREET), Some(20), None, "", "t", None)
        .unwrap();
    let ov = Overrides::resolve(&ds, &db.list().unwrap(), db.generation());
    let sid = overrides::segments_near(
        &ds, (CLOSURE_LAT * 1e7) as i32, (CLOSURE_LON * 1e7) as i32, 300.0, Some(CLOSURE_STREET),
    )[0];
    assert_eq!(
        ov.effect(sid),
        Some(Effect::Speed(20)),
        "a correction should not need the earlier rule deleted first"
    );
}

// ----------------------------------------------------------------- routing

#[test]
fn a_closure_forces_a_detour_without_touching_the_dataset() {
    let Some(ds) = data() else { return };
    let t = tmp("detour");
    let db = OverrideDb::open(&t.0).unwrap();
    let (a, b) = (snap(&ds, FROM), snap(&ds, TO));
    let mut r = Router::new(ds.n_vertices());

    let base = r.route(&ds, &a, &b).expect("baseline route");
    db.add(KIND_CLOSED, CLOSURE_LAT, CLOSURE_LON, 300, Some(CLOSURE_STREET), None, None, "", "t", None)
        .unwrap();
    let ov = Overrides::resolve(&ds, &db.list().unwrap(), db.generation());
    let closed = r.route_with(&ds, &a, &b, Some(&ov)).expect("a detour exists");

    assert!(closed.dist_cm > base.dist_cm, "a detour is longer: {} vs {}", closed.dist_cm, base.dist_cm);
    // And the closed street is genuinely gone from the answer.
    let uses_street = |rt: &mpee_planet::router::Route| {
        rt.legs.iter().any(|l| {
            ds.street_name(l.seg as usize).map(|n| n == CLOSURE_STREET).unwrap_or(false)
        })
    };
    assert!(uses_street(&base), "the baseline really did use it");
    assert!(!uses_street(&closed), "the closure really did remove it");
    // The dataset itself is untouched: routing without the override is
    // bit-identical to the baseline.
    let again = r.route(&ds, &a, &b).unwrap();
    assert_eq!(again.dist_cm, base.dist_cm);
}

#[test]
fn slowing_a_road_costs_time_without_costing_reachability() {
    let Some(ds) = data() else { return };
    let t = tmp("slow");
    let db = OverrideDb::open(&t.0).unwrap();
    let (a, b) = (snap(&ds, FROM), snap(&ds, TO));
    let mut r = Router::new(ds.n_vertices());
    let base = r.route(&ds, &a, &b).unwrap();

    db.add(KIND_SPEED, CLOSURE_LAT, CLOSURE_LON, 300, Some(CLOSURE_STREET), Some(5), None, "", "t", None)
        .unwrap();
    let ov = Overrides::resolve(&ds, &db.list().unwrap(), db.generation());
    let slow = r.route_with(&ds, &a, &b, Some(&ov)).expect("still reachable");
    assert!(slow.dur_ds > base.dur_ds, "5 km/h on the direct road must cost time");
}

#[test]
fn a_penalty_never_rounds_a_road_to_free() {
    let Some(ds) = data() else { return };
    let t = tmp("penalty");
    let db = OverrideDb::open(&t.0).unwrap();
    let (a, b) = (snap(&ds, FROM), snap(&ds, TO));
    let mut r = Router::new(ds.n_vertices());
    let base = r.route(&ds, &a, &b).unwrap();
    // A factor below 1 makes a road cheaper; integer arithmetic must not let
    // it reach zero, or a discouraged road becomes a free one.
    db.add(KIND_PENALTY, CLOSURE_LAT, CLOSURE_LON, 300, Some(CLOSURE_STREET), None, Some(0.0001), "", "t", None)
        .unwrap();
    let ov = Overrides::resolve(&ds, &db.list().unwrap(), db.generation());
    let cheap = r.route_with(&ds, &a, &b, Some(&ov)).unwrap();
    assert!(cheap.dur_ds > 0, "a route always takes some time");
    assert!(cheap.dur_ds <= base.dur_ds);
}

#[test]
fn overrides_do_not_disturb_the_distance_contract() {
    let Some(ds) = data() else { return };
    let t = tmp("contract");
    let db = OverrideDb::open(&t.0).unwrap();
    let (a, b) = (snap(&ds, FROM), snap(&ds, TO));
    let mut r = Router::new(ds.n_vertices());
    db.add(KIND_SPEED, CLOSURE_LAT, CLOSURE_LON, 300, Some(CLOSURE_STREET), Some(15), None, "", "t", None)
        .unwrap();
    let ov = Overrides::resolve(&ds, &db.list().unwrap(), db.generation());
    let rt = r.route_with(&ds, &a, &b, Some(&ov)).unwrap();
    // Overrides change *time*, never *length*: the legs still sum exactly.
    let sum: u64 = rt.legs.iter().map(|l| l.len_cm).sum();
    assert_eq!(sum, rt.dist_cm, "an override must not corrupt the exact-distance sum");
}

/// Concurrent writers must not collide over id allocation.
///
/// Regression test for a real bug, found by running four writer processes
/// against one database: the id was chosen by `SELECT MAX(id)` *outside* the
/// write transaction and used inside it, so under contention 27 of 40 inserts
/// died with a primary-key violation. mpedb refused the duplicates rather than
/// accepting them, which is what made the race visible at all — the fix is to
/// allocate under the writer lock, and this pins it.
#[test]
fn concurrent_writers_do_not_collide_over_ids() {
    let t = tmp("concurrent");
    let root = t.0.clone();
    // Create the database once before the threads race to open it.
    drop(OverrideDb::open(&root).unwrap());

    let failures = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ids = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    std::thread::scope(|sc| {
        for w in 0..4 {
            let (root, failures, ids) = (root.clone(), failures.clone(), ids.clone());
            sc.spawn(move || {
                let db = OverrideDb::open(&root).unwrap();
                for i in 0..15 {
                    match db.add(
                        KIND_CLOSED, 59.92, 10.73, 50, None, None, None,
                        &format!("w{w}-{i}"), "test", None,
                    ) {
                        Ok(id) => ids.lock().unwrap().push(id),
                        Err(_) => {
                            failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });

    let n_fail = failures.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(n_fail, 0, "{n_fail} of 60 concurrent inserts failed");

    let mut got = ids.lock().unwrap().clone();
    assert_eq!(got.len(), 60);
    got.sort_unstable();
    let unique = {
        let mut u = got.clone();
        u.dedup();
        u.len()
    };
    assert_eq!(unique, 60, "every writer got its own id");

    let db = OverrideDb::open(&root).unwrap();
    assert_eq!(db.list().unwrap().len(), 60, "and every row landed");
}

/// Every committed write must be observable, or a running reader keeps serving
/// a stale answer until it restarts — the exact failure the layer exists to
/// avoid.
#[test]
fn the_generation_advances_once_per_committed_write() {
    let t = tmp("genrace");
    let db = OverrideDb::open(&t.0).unwrap();
    let start = db.generation();
    let mut ids = Vec::new();
    for i in 0..10 {
        ids.push(db.add(KIND_CLOSED, 59.9, 10.7, 30, None, None, None, &format!("{i}"), "t", None).unwrap());
    }
    assert_eq!(db.generation(), start + 10, "ten writes, ten increments");
    for id in ids {
        db.remove(id).unwrap();
    }
    assert_eq!(db.generation(), start + 20, "removals count too");
}
