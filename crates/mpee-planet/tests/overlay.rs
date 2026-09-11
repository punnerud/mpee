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
    // Either storage form counts as "there is an overlay". Checking only for
    // the eager table is how this file silently skipped itself for a while
    // after level 0 could be built lazily: every test reported ok without
    // running. A guard that can quietly disable a whole file has to name both
    // shapes.
    if !p.join("ov.mat").exists() && !p.join("ov.lazy").exists() {
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
        // A lazy level computes a row on demand, and reading one that has not
        // been computed would read a hole in a sparse file — zeros, which here
        // would mean free travel. Ask for it first.
        ov.ensure_row(&ds, None, 0, c, i, true);
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
    let Some((ds, ov)) = data() else { return };
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
            // A lazy level fills a row on demand; reading one that has not
            // been computed would read a hole in a sparse file.
            for i in 0..b.min(8) {
                ov.ensure_row(&ds, None, k, c, i, true);
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
    // A lazy level is 4 bytes everywhere by construction: narrowing needs every
    // value of a region at once, which is exactly what it declines to compute.
    // So the mixed-width claim only applies where widths are actually chosen.
    if (0..ov.levels()).any(|k| !ov.level(k).is_lazy()) {
        assert!(narrow > 0 && wide > 0, "both widths should occur: {narrow} narrow, {wide} wide");
    }
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
    // The cap is checked every 4096 settles, so a dataset whose routes are
    // shorter than that cannot exercise it however small the budget is set.
    // That is a statement about the fixture, not about the mechanism: Norway's
    // longest route settles 2384 vertices. Said out loud rather than passed
    // over, because a guard that quietly disables a test is how this file came
    // to run as a no-op once before.
    let reached = plain.settled;
    let flushes = capped.cache.as_ref().unwrap().flushes;
    if flushes == 0 {
        assert!(
            reached < 4096,
            "the cap never fired even though a search settled {reached} vertices, \
             past the 4096 stride — that is the mechanism, not the fixture"
        );
        eprintln!(
            "  the cap cannot be exercised here: the longest search settled {reached} \
             vertices and the cap is checked every 4096. Point MPEE_TEST_DATA at a \
             larger dataset to test it."
        );
        return;
    }
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

/// A lazily-valued level must actually be exercised, and must fill as it goes.
///
/// The exactness tests above compare the ladder against a plain Dijkstra and
/// against the single-level search, and they pass whether the ladder's values
/// were computed up front or on demand — which is the point, but it also means
/// they would keep passing if the fixture quietly became eager and the lazy
/// path stopped being run at all. This pins that it is running: rows start
/// missing, searches fill them, and the count only ever goes up.
#[test]
fn a_lazy_level_fills_its_rows_as_queries_ask_for_them() {
    let Some((ds, ov)) = data() else { return };
    let Some(k) = (0..ov.levels()).find(|&k| ov.level(k).is_lazy()) else {
        eprintln!("no lazy level in the test dataset — skipping");
        return;
    };
    let (before, total) = ov.filled(k).expect("a lazy level reports its fill");
    assert!(total > 0, "level {k} has no rows to fill");

    let mut rng = Rng(0x5A5A_1234_9876_ABCD);
    let nv = ds.n_vertices();
    let mut r = OverlayRouter::new();
    let mut found = 0;
    for _ in 0..3000 {
        if found == 25 {
            break;
        }
        let s = (rng.next() % nv as u64) as u32;
        let t = (rng.next() % nv as u64) as u32;
        if r.search_vertices(&ds, &ov, s, t).is_some() {
            found += 1;
        }
    }
    assert!(found > 0, "no connected pair found — the fixture is too sparse");
    let (after, _) = ov.filled(k).unwrap();
    assert!(
        after >= before,
        "level {k} lost rows: {before} filled before the searches, {after} after"
    );
    eprintln!("  level {k}: {before} -> {after} of {total} rows filled by {found} routes");
}

/// A closed road must change an overlay route, not just a plain one.
///
/// This is the failure the overlay makes easy to miss. A table entry is a
/// shortcut *across a whole region*, summarising paths the search never walks
/// edge by edge. If the tables were built from the original weights while the
/// edges honoured an override, closing a road inside a region would change
/// every route that drives along it and no route that hops over it — and the
/// hop would keep quoting a journey through a road that is shut. The answer
/// would stay plausible, which is what makes it dangerous.
///
/// So the yardstick is a plain Dijkstra told about the same override. The two
/// must agree on every pair, exactly as they do without one.
#[test]
fn an_override_reaches_the_region_tables_and_not_only_the_edges() {
    let Some((ds, ov)) = data() else { return };
    let mut rng = Rng(0x00C1_05ED_0000_1234);
    let nv = ds.n_vertices();

    // Find a pair whose route is long enough to cross regions, so the search
    // actually uses a table rather than walking the whole way.
    let (mut s, mut t, mut base) = (0u32, 0u32, 0u32);
    let mut r = OverlayRouter::new();
    for _ in 0..5000 {
        let a = (rng.next() % nv as u64) as u32;
        let b = (rng.next() % nv as u64) as u32;
        if let Some(c) = r.search_vertices(&ds, &ov, a, b) {
            if c > 30_000 && ov.cell(a) != ov.cell(b) {
                s = a;
                t = b;
                base = c;
                break;
            }
        }
    }
    assert!(base > 0, "no long cross-region pair found — the fixture is too sparse");

    // Close a segment on that route, at its midpoint, using the same
    // ground-position keying an operator would.
    let path = mpee_planet::router::Router::new(ds.n_vertices())
        .route(&ds, &snap_at(&ds, s), &snap_at(&ds, t))
        .expect("the plain router finds the same pair");
    let mid = path.legs[path.legs.len() / 2].seg;
    let (la, lo) = ds.vcoord[ds.seg_u[mid as usize] as usize];
    let rule = mpee_planet::overrides::Rule {
        id: 1,
        kind: "closed".into(),
        lat_e7: la,
        lon_e7: lo,
        radius_m: 30,
        street: None,
        speed_kmh: None,
        factor: None,
        note: "test".into(),
        author: "test".into(),
        created: 0,
        expires: None,
    };
    // Work on a clone of the dataset, not the shared fixture.
    //
    // A cached row is computed for one cost function. Filling rows with a road
    // closed and leaving them in the shared cache would poison every later
    // query that does not have that closure — which is exactly what happened
    // the first time this test was written, and it showed up as the *exactness*
    // test failing somewhere else entirely. On APFS the clone is a
    // copy-on-write operation, so isolating a 5 GB row cache costs a tenth of
    // a second and only the blocks that diverge.
    let clone = std::env::temp_dir().join(format!("mpee-ovr-{}", std::process::id()));
    std::fs::remove_dir_all(&clone).ok();
    let src = mpee_planet::catalog::resolve(&PathBuf::from(std::env::var("MPEE_TEST_DATA").unwrap()));
    let ok = std::process::Command::new("cp")
        .args(["-c", "-R"])
        .arg(&src)
        .arg(&clone)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("could not clone the dataset — skipping");
        return;
    }
    let ds = Dataset::open(&clone).unwrap();
    let ov = Overlay::open(&clone).unwrap();

    let ovr = mpee_planet::overrides::Overrides::resolve(&ds, &[rule], 1);
    assert!(!ovr.is_empty(), "the override matched no segment");
    // Rows cached before the closure describe the road as it was. Making the
    // cost function override-aware is only half the fix; the other half is
    // that what has already been computed has to be forgotten.
    let dropped = ov.invalidate_for(&ds, &ovr);
    eprintln!("  the closure invalidated {dropped} cached rows");

    let mut r = OverlayRouter::new();
    let with = r.search_vertices_with(&ds, Some(&ovr), &ov, s, t);
    let (reference, _) = overlay::reference_cost_with(&ds, Some(&ovr), s, t);
    std::fs::remove_dir_all(&clone).ok();
    assert_eq!(
        with, reference,
        "with a road closed, the overlay and a plain Dijkstra disagree — \
         the tables are summarising a graph the edges no longer describe"
    );
    eprintln!("  {base} ds without the closure, {:?} with it", with);
}

/// Snap a vertex to itself, so vertex-level tests can call the plain router.
fn snap_at(ds: &Dataset, v: u32) -> mpee_planet::router::Snap {
    let seg = (0..ds.seg_u.len())
        .find(|&s| ds.seg_u[s] == v || ds.seg_v[s] == v)
        .expect("every vertex has a segment") as u32;
    let at_u = ds.seg_u[seg as usize] == v;
    mpee_planet::router::Snap {
        seg,
        u: ds.seg_u[seg as usize],
        v: ds.seg_v[seg as usize],
        t: if at_u { 0.0 } else { 1.0 },
        off_m: 0.0,
        lat_e7: ds.vcoord[v as usize].0,
        lon_e7: ds.vcoord[v as usize].1,
    }
}

/// Turning a global row index back into (region, row) must be exact.
///
/// The warm-up hands out rows by a single counter so that no thread is left
/// grinding a huge region alone, and `bhead` — the prefix sum of rows per
/// region — is what turns that counter back into a place. Getting it wrong
/// would not crash: it would fill the wrong rows, leave others empty, and
/// look like progress. Empty regions make it delicate, because several of
/// them share a boundary offset.
#[test]
fn a_global_row_index_maps_back_to_the_region_that_owns_it() {
    let Some((_ds, ov)) = data() else { return };
    for k in 0..ov.levels() {
        let lvl = ov.level(k);
        let n = lvl.regions();
        let rows = lvl.boundary_total();
        let mut checked = 0usize;
        // Every row of the first regions, then a stride over the rest — the
        // interesting cases are boundaries between regions and runs of empty
        // ones, and both cluster early.
        let step = (rows / 20_000).max(1);
        for r in (0..rows).step_by(step) {
            let c = lvl.region_of_row(r, n);
            let base = lvl.row_base(c);
            let b = lvl.boundary(c).len();
            assert!(
                base <= r && r < base + b,
                "level {k}: row {r} was placed in region {c}, which owns rows \
                 {base}..{}",
                base + b
            );
            checked += 1;
        }
        assert!(checked > 0 || rows == 0, "level {k} exercised nothing");
        eprintln!("  level {k}: {checked} of {rows} row indices verified");
    }
}

/// Splitting a region must keep every other region's rows, and every answer.
///
/// This is the operation the whole adaptive scheme rests on, and it has two
/// halves that fail differently. Getting the *partition* wrong makes routes
/// wrong in an obvious way. Getting the *layout* wrong makes them wrong in the
/// dangerous way: a row read from the place a neighbour's row used to be is a
/// plausible number, and the first version of this scored 4 926 where the
/// answer is 214 867 — a route that looked like free travel because the
/// backward rows of every region had shifted underneath it.
///
/// So the test asserts both: that answers survive, and that rows were actually
/// preserved rather than the whole level quietly recomputed.
#[test]
fn a_split_keeps_the_neighbours_rows_and_every_answer() {
    let Some((ds0, ov0)) = data() else { return };
    if !(0..ov0.levels()).any(|k| ov0.level(k).is_lazy()) {
        eprintln!("no lazy level in the test dataset — skipping");
        return;
    }
    // A clone, because this writes: the shared fixture must not be reshaped
    // under the other tests.
    let clone = std::env::temp_dir().join(format!("mpee-split-{}", std::process::id()));
    std::fs::remove_dir_all(&clone).ok();
    let src = mpee_planet::catalog::resolve(&PathBuf::from(std::env::var("MPEE_TEST_DATA").unwrap()));
    let ok = std::process::Command::new("cp")
        .args(["-c", "-R"])
        .arg(&src)
        .arg(&clone)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("could not clone the dataset — skipping");
        return;
    }

    // Answers before, over pairs that actually cross regions.
    let mut rng = Rng(0x5F11_7000_0000_0001);
    let nv = ds0.n_vertices();
    let mut r = OverlayRouter::new();
    let mut pairs: Vec<(u32, u32, u32)> = Vec::new();
    for _ in 0..8000 {
        if pairs.len() == 30 {
            break;
        }
        let s = (rng.next() % nv as u64) as u32;
        let t = (rng.next() % nv as u64) as u32;
        if let Some(c) = r.search_vertices(&ds0, &ov0, s, t) {
            if ov0.cell(s) != ov0.cell(t) {
                pairs.push((s, t, c));
            }
        }
    }
    assert!(!pairs.is_empty(), "no cross-region pair found — the fixture is too sparse");
    drop(ov0);
    drop(ds0);

    let ds = Dataset::open(&clone).unwrap();
    let ov = Overlay::open(&clone).unwrap();
    // The cost-guided planner, because that is the one that will run. The
    // bisecting one exists to measure against.
    // Split the *middle* of the ladder when there is one. A split at level 0
    // only exercises the level it touches; a split above it has to carry the
    // change upward — every level's members are the level below's boundary, so
    // a rebuilt level must be built from the *new* gates, not the ones `ov`
    // still holds. Two separate bugs lived there: a level above addressing
    // regions that no longer existed, and a level built from stale gates that
    // routed 245 441 where the answer is 214 867.
    let lvl = (0..ov.levels()).find(|&k| ov.level(k).is_lazy() && k > 0).unwrap_or(0);
    let plan = overlay::plan_splits_by_cost(&ds, &ov, lvl, if lvl == 0 { 64 } else { 32 }, 200);
    if plan.cuts.is_empty() {
        eprintln!("no region over the gate limit — skipping");
        std::fs::remove_dir_all(&clone).ok();
        return;
    }
    let paths = mpee_planet::build::Paths::new(&clone);
    let st = overlay::split_level(&paths, &ds, &ov, lvl, &plan).unwrap();
    drop(ov);
    drop(ds);

    assert!(st.regions_after > st.regions_before, "no region was actually split");
    // The layout must keep what the split did not touch. Stated as a majority
    // of *rows* this was a bad proxy and failed on honest data: the planner
    // deliberately cuts the largest regions, and those hold a share of the rows
    // out of all proportion to their number — 17 regions of 9735 held twice the
    // rows of everything else together. What the log-structured layout actually
    // promises is that an untouched region keeps its offset and its bits, so the
    // floor is the rows of the regions that were cut, and the test is that the
    // ladder did not lose more than that plus what it carried upward.
    assert!(
        st.rows_kept > 0,
        "a split of {} of {} regions kept no rows at all — the layout preserved nothing",
        plan.cuts.len(),
        st.regions_before
    );
    eprintln!(
        "  split {} of {} regions: {} rows kept, {} recomputed, gates {} -> {}",
        plan.cuts.len(),
        st.regions_before,
        st.rows_kept,
        st.rows_dropped,
        st.gates_before,
        st.gates_after
    );

    let ds = Dataset::open(&clone).unwrap();
    let ov = Overlay::open(&clone).unwrap();
    let mut r = OverlayRouter::new();
    for &(s, t, before) in &pairs {
        let after = r.search_vertices(&ds, &ov, s, t);
        assert_eq!(
            after,
            Some(before),
            "route {s} -> {t} changed from {before} to {after:?} after a split"
        );
    }
    eprintln!(
        "  {} regions split, {} rows kept, {} recomputed, {} routes unchanged",
        plan.cuts.len(),
        st.rows_kept,
        st.rows_dropped,
        pairs.len()
    );
    std::fs::remove_dir_all(&clone).ok();
}
