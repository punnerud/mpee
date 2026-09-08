//! The overlay must answer exactly what the full graph answers.
//!
//! That is the entire contract, and it is easy to violate subtly: a region
//! table that accidentally allows paths leaving the region, a boundary
//! definition that misses one direction, a cut edge counted twice. Each of
//! those produces answers that are *close*, which is worse than wrong. So the
//! yardstick here is a plain Dijkstra over the same graph, and the comparison
//! is vertex to vertex so no snapping or partial-segment bookkeeping can
//! account for a difference.

use mpee_planet::dataset::Dataset;
use mpee_planet::overlay::{self, Overlay, OverlayRouter};
use std::path::PathBuf;

fn data() -> Option<(Dataset, Overlay)> {
    let p = mpee_planet::catalog::resolve(&PathBuf::from(std::env::var("MPEE_TEST_DATA").ok()?));
    if !p.join("ov.mat").exists() {
        eprintln!("no overlay in the test dataset — skipping");
        return None;
    }
    Some((Dataset::open(&p).unwrap(), Overlay::open(&p).unwrap()))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn the_overlay_answers_exactly_what_a_plain_dijkstra_does() {
    let Some((ds, ov)) = data() else { return };
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let mut r = OverlayRouter::new();
    let nv = ds.n_vertices();
    let (mut checked, mut settled_ref, mut settled_ov) = (0usize, 0u64, 0u64);
    while checked < 25 {
        let s = (rng.next() as usize) % nv;
        let t = (rng.next() as usize) % nv;
        if s == t || ov.cell(s as u32) == ov.cell(t as u32) {
            continue;
        }
        let (exact, n_ref) = overlay::reference_cost(&ds, s as u32, t as u32);
        let Some(exact) = exact else { continue };
        let got = r.search_vertices(&ds, &ov, s as u32, t as u32);
        assert_eq!(
            got,
            Some(exact),
            "overlay disagreed with the full graph on {s} -> {t}"
        );
        settled_ref += n_ref;
        settled_ov += r.settled;
        checked += 1;
    }
    assert!(checked >= 25);
    assert!(
        settled_ov * 4 < settled_ref,
        "the overlay should settle far fewer vertices: {settled_ov} vs {settled_ref}"
    );
    eprintln!("{checked} pairs exact; {settled_ref} settled vs {settled_ov} ({:.1}x fewer)",
        settled_ref as f64 / settled_ov.max(1) as f64);
}

#[test]
fn every_vertex_belongs_to_exactly_one_region() {
    let Some((ds, ov)) = data() else { return };
    let n = ds.n_vertices();
    assert_eq!(ov.cell_of().len(), n);
    let cells = ov.cells();
    let mut rng = Rng(7);
    for _ in 0..20_000 {
        let v = (rng.next() as usize) % n;
        let c = ov.cell(v as u32);
        assert!((c as usize) < cells, "vertex {v} names region {c} of {cells}");
    }
}

#[test]
fn a_cut_edge_makes_both_of_its_ends_boundary_vertices() {
    // If only one end were marked, the backward search could never enter the
    // region the edge leads into.
    let Some((ds, ov)) = data() else { return };
    let mut rng = Rng(99);
    let n = ds.n_vertices();
    let mut checked = 0;
    for _ in 0..200_000 {
        let u = (rng.next() as usize) % n;
        let (a, e) = (ds.head[u] as usize, ds.head[u + 1] as usize);
        for k in a..e {
            let v = ds.target(u as u32, k);
            if ov.cell(u as u32) == ov.cell(v) {
                continue;
            }
            assert!(
                ov.bindex(ov.cell(u as u32), u as u32).is_some(),
                "{u} has an edge leaving its region but is not a boundary vertex"
            );
            assert!(
                ov.bindex(ov.cell(v), v).is_some(),
                "{v} is entered from another region but is not a boundary vertex"
            );
            checked += 1;
        }
        if checked > 500 {
            break;
        }
    }
    assert!(checked > 100, "only {checked} cut edges seen");
}

#[test]
fn a_region_table_never_undercuts_the_real_graph() {
    // The table holds paths that stay inside a region. It must therefore be
    // >= the true shortest path, which may leave and come back.
    let Some((ds, ov)) = data() else { return };
    let mut rng = Rng(4242);
    let mut checked = 0;
    for _ in 0..4000 {
        let c = (rng.next() as usize % ov.cells()) as u32;
        let b = ov.boundary(c);
        if b.len() < 2 {
            continue;
        }
        let i = (rng.next() as usize) % b.len();
        let j = (rng.next() as usize) % b.len();
        if i == j {
            continue;
        }
        let inside = ov.cost(c, i, j);
        if inside == overlay::UNREACHABLE {
            continue;
        }
        let (exact, _) = overlay::reference_cost(&ds, b[i], b[j]);
        let exact = exact.expect("reachable inside the region implies reachable at large");
        assert!(
            inside >= exact,
            "region {c} claims {inside} between {} and {} but the graph does it in {exact}",
            b[i],
            b[j]
        );
        checked += 1;
        if checked >= 15 {
            break;
        }
    }
    assert!(checked >= 10, "only {checked} region pairs exercised");
}

#[test]
fn regions_are_bounded_in_size() {
    let Some((_ds, ov)) = data() else { return };
    let mut counts = vec![0usize; ov.cells()];
    for &c in ov.cell_of() {
        counts[c as usize] += 1;
    }
    let max = counts.iter().copied().max().unwrap_or(0);
    assert!(
        max <= overlay::TARGET * 2,
        "a region grew to {max} vertices, target is {}",
        overlay::TARGET
    );
}

#[test]
fn the_search_state_stays_far_smaller_than_the_graph() {
    // The point of the indexed queue: state follows the vertices reached, not
    // the relaxations performed. A dense overlay relaxes ~139 neighbours per
    // settle, so a lazy queue holds roughly a hundred entries per vertex.
    let Some((ds, ov)) = data() else { return };
    let mut rng = Rng(31337);
    let nv = ds.n_vertices();
    let mut checked = 0;
    let mut worst = 0.0f64;
    while checked < 10 {
        let s = (rng.next() as usize) % nv;
        let t = (rng.next() as usize) % nv;
        if s == t || ov.cell(s as u32) == ov.cell(t as u32) {
            continue;
        }
        // A fresh router each time: `bytes()` reports capacity, and a reused
        // one legitimately keeps what an earlier, larger query grew.
        let mut r = OverlayRouter::new();
        if r.search_vertices(&ds, &ov, s as u32, t as u32).is_none() {
            continue;
        }
        // Frontier only, and floor removed. `reached()` counts the two
        // frontiers, so measuring `bytes()` against it would charge them for
        // the region-local scratch — which holds a whole level-1 region and,
        // with two levels, is the bigger half. The tables also start at 1024
        // slots each and never shrink, so a short search would otherwise be
        // measuring that floor rather than the growth rate. What this test is
        // about is the *slope*: bytes must follow vertices reached, not
        // relaxations performed.
        const FLOOR: f64 = 2.0 * 1024.0 * 16.0;
        let per = (r.frontier_bytes() as f64 - FLOOR).max(0.0) / r.reached().max(1) as f64;
        worst = worst.max(per);
        assert!(
            per < 96.0,
            "search state grows {per:.0} bytes per reached vertex — the queue is holding \
             duplicates again (a lazy heap costs hundreds)"
        );
        checked += 1;
    }
    assert_eq!(checked, 10);
    eprintln!("worst {worst:.0} bytes per reached vertex");
}

#[test]
fn a_budget_refuses_rather_than_growing_without_limit() {
    let Some((ds, ov)) = data() else { return };
    let mut rng = Rng(555);
    let mut r = OverlayRouter::new();
    let nv = ds.n_vertices();
    // Small enough that any real search trips it.
    r.budget_bytes = 8 * 1024;
    let mut refused = 0;
    for _ in 0..40 {
        let s = (rng.next() as usize) % nv;
        let t = (rng.next() as usize) % nv;
        if s == t || ov.cell(s as u32) == ov.cell(t as u32) {
            continue;
        }
        if r.search_vertices(&ds, &ov, s as u32, t as u32).is_none() && r.over_budget {
            refused += 1;
        }
    }
    assert!(refused > 0, "a tiny budget must actually stop a search");

    // And with no budget the same queries succeed, so the limit is what
    // stopped them, not the graph.
    r.budget_bytes = usize::MAX;
    let mut rng = Rng(555);
    let mut ok = 0;
    for _ in 0..40 {
        let s = (rng.next() as usize) % nv;
        let t = (rng.next() as usize) % nv;
        if s == t || ov.cell(s as u32) == ov.cell(t as u32) {
            continue;
        }
        if r.search_vertices(&ds, &ov, s as u32, t as u32).is_some() {
            assert!(!r.over_budget);
            ok += 1;
        }
    }
    assert!(ok > 0);
}

#[test]
fn a_narrowed_region_reads_back_what_was_stored() {
    // Regions store 2- or 4-byte entries depending on their own diameter. A
    // narrow region must still report `UNREACHABLE` as unreachable rather than
    // as 65 535 tenths of a second, and a wide one must not be read as narrow.
    let Some((_ds, ov)) = data() else { return };
    let (mut narrow, mut wide, mut checked) = (0usize, 0usize, 0usize);
    // Every rung stores the same shape of table and narrows the same way, so
    // every rung has to read back the same way too.
    for k in 0..ov.levels() {
        let lvl = ov.level(k);
        for c in 0..lvl.regions() as u32 {
            let b = lvl.boundary(c).len();
            if b < 4 {
                continue;
            }
            if lvl.width_of(c) == 2 {
                narrow += 1;
            } else {
                wide += 1;
            }
            // A diagonal entry is a vertex to itself: always zero.
            for i in 0..b.min(8) {
                assert_eq!(
                    lvl.cost(c, i, i),
                    0,
                    "level {k} region {c} entry ({i},{i}) should be zero"
                );
            }
            // And anything claiming to be reachable must be a plausible time
            // inside one region, not a misread sentinel.
            for i in 0..b.min(3) {
                for j in 0..b.min(3) {
                    let v = lvl.cost(c, i, j);
                    if v == overlay::UNREACHABLE {
                        continue;
                    }
                    assert!(
                        v < 100_000_000,
                        "level {k} region {c} width {} gives an implausible {v} ds",
                        lvl.width_of(c)
                    );
                }
            }
            checked += 1;
            if checked > 4000 {
                break;
            }
        }
    }
    assert!(narrow > 0 && wide > 0, "both widths should occur: {narrow} narrow, {wide} wide");
    eprintln!("{narrow} narrow regions, {wide} wide");
}

/// Capping the resident cache must not change a single answer.
///
/// This is the property the whole "runs on a small machine" claim rests on:
/// every array is a clean, read-only file mapping, so handing those pages back
/// to the kernel throws away a cache and nothing else. If a route came out
/// different under a cap, something in the reader would be holding state it
/// only *thinks* is backed by the file — a bug that would otherwise surface as
/// a wrong route on the small machine and never on the one that built the data.
#[test]
fn capping_the_page_cache_changes_timing_but_never_the_answer() {
    let Some((ds, ov)) = data() else { return };
    let mut rng = Rng(0x0BAD_F00D_1234_5678);
    let nv = ds.n_vertices();

    let mut plain = OverlayRouter::new();
    let mut capped = OverlayRouter::new();
    // A byte, so the cap is over budget at every single check and the pages
    // are dropped as hard and as often as the mechanism allows.
    let mut cap = mpee_planet::cachecap::CacheCap::new(1);
    cap.govern(ds.maps());
    cap.govern(ov.maps());
    capped.cache = Some(cap);

    let mut checked = 0usize;
    for _ in 0..2000 {
        if checked == 12 {
            break;
        }
        let s = (rng.next() % nv as u64) as u32;
        let t = (rng.next() % nv as u64) as u32;
        let a = plain.search_vertices(&ds, &ov, s, t);
        let b = capped.search_vertices(&ds, &ov, s, t);
        assert_eq!(a, b, "route {s} -> {t} differs once the cache is capped");
        if a.is_some() {
            checked += 1;
        }
    }
    assert!(checked > 0, "no connected pair found — the fixture is too sparse");
    let flushes = capped.cache.as_ref().unwrap().flushes;
    assert!(flushes > 0, "the cap never fired, so nothing was actually tested");
}

/// The second level must not change a single answer.
///
/// It exists to make a long route cheap, not different: a level-2 table holds
/// the shortest path across a whole group of regions, and using it must give
/// exactly what walking every boundary vertex inside that group gives. The
/// dangerous failure is silent — the route stays plausible, just slightly
/// wrong — so the yardstick is the same search with the level switched off,
/// which the existing tests already pin to a plain Dijkstra.
#[test]
fn the_second_level_is_a_shortcut_not_a_different_answer() {
    let Some((ds, ov)) = data() else { return };
    if !ov.has_l2() {
        eprintln!("no second level in the test dataset — skipping");
        return;
    }
    let mut rng = Rng(0xFEED_FACE_5EED_1234);
    let nv = ds.n_vertices();
    let mut one = OverlayRouter::new();
    one.use_l2 = false;
    let mut two = OverlayRouter::new();

    let (mut checked, mut used_l2, mut settled1, mut settled2) = (0usize, 0u64, 0u64, 0u64);
    for _ in 0..4000 {
        if checked == 40 {
            break;
        }
        let s = (rng.next() % nv as u64) as u32;
        let t = (rng.next() % nv as u64) as u32;
        let a = one.search_vertices(&ds, &ov, s, t);
        let b = two.search_vertices(&ds, &ov, s, t);
        assert_eq!(a, b, "route {s} -> {t}: the second level changed the answer");
        if a.is_some() {
            checked += 1;
            used_l2 += two.regions2_entered;
            settled1 += one.settled;
            settled2 += two.settled;
        }
    }
    assert!(checked > 0, "no connected pair found — the fixture is too sparse");
    assert!(
        used_l2 > 0,
        "the second level never fired over {checked} routes, so nothing was tested"
    );
    eprintln!(
        "  {checked} routes: {settled1} settled at one level, {settled2} at two \
         ({:.2}x), {used_l2} level-2 crossings",
        settled1 as f64 / settled2.max(1) as f64
    );
}
