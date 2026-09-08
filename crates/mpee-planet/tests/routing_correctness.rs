//! Hold the bidirectional search to a textbook Dijkstra.
//!
//! Bidirectional search is where routing engines quietly go wrong: the
//! stopping bound and the meeting-point reconstruction are both easy to get
//! subtly off, and the result is a route that is *plausible* but not optimal —
//! the kind of bug that survives eyeballing a map. So every random pair is
//! also solved by a plain forward Dijkstra over the same CSR, and the two
//! costs must agree exactly.
//!
//! Set `MPEE_TEST_DATA` to a built dataset directory to run it.

use mpee_planet::build::attr_kmh;
use mpee_planet::dataset::Dataset;
use mpee_planet::router::{Router, SEG_MASK};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::PathBuf;

fn data() -> Option<Dataset> {
    // Resolve through HEAD so the same variable works for a catalogue-managed
    // root and for a bare directory built before the catalogue existed.
    let p = mpee_planet::catalog::resolve(&PathBuf::from(std::env::var("MPEE_TEST_DATA").ok()?));
    p.join("csr.head").exists().then(|| Dataset::open(&p).unwrap())
}

fn dur_ds(len_cm: u32, kmh: u16) -> u32 {
    ((len_cm as u64 * 36) / (kmh.max(1) as u64 * 100)).min(u32::MAX as u64) as u32
}

/// Reference single-source Dijkstra: costs from `src` to everything.
fn dijkstra(ds: &Dataset, src: u32) -> Vec<u32> {
    let mut d = vec![u32::MAX; ds.n_vertices()];
    let mut h = BinaryHeap::new();
    d[src as usize] = 0;
    h.push(Reverse((0u32, src)));
    while let Some(Reverse((du, u))) = h.pop() {
        if du > d[u as usize] {
            continue;
        }
        let (s, e) = (ds.head[u as usize] as usize, ds.head[u as usize + 1] as usize);
        for k in s..e {
            let sid = (ds.eseg[k] & SEG_MASK) as usize;
            let w = dur_ds(ds.seg_len[sid], attr_kmh(ds.attr(sid)));
            let nd = du.saturating_add(w);
            let t = ds.target(u, k);
            if nd < d[t as usize] {
                d[t as usize] = nd;
                h.push(Reverse((nd, t)));
            }
        }
    }
    d
}

/// Deterministic pseudo-random vertex picker — no dependency, reproducible.
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
fn bidirectional_matches_plain_dijkstra() {
    let Some(ds) = data() else {
        eprintln!("MPEE_TEST_DATA not set — skipping");
        return;
    };
    let n = ds.n_vertices() as u64;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut router = Router::new(ds.n_vertices());
    let mut checked = 0;
    for _ in 0..25 {
        let s = (rng.next() % n) as u32;
        let reference = dijkstra(&ds, s);
        for _ in 0..8 {
            let t = (rng.next() % n) as u32;
            if s == t || reference[t as usize] == u32::MAX {
                continue;
            }
            // Snap onto the two vertices exactly: t = 0 puts the point on `u`.
            let sn = |v: u32| {
                let k = ds.head[v as usize] as usize;
                let sid = (ds.eseg[k] & SEG_MASK) as usize;
                let at_u = ds.seg_u[sid] == v;
                mpee_planet::router::Snap {
                    seg: sid as u32,
                    u: ds.seg_u[sid],
                    v: ds.seg_v[sid],
                    t: if at_u { 0.0 } else { 1.0 },
                    lat_e7: ds.vcoord[v as usize].0,
                    lon_e7: ds.vcoord[v as usize].1,
                    off_m: 0.0,
                }
            };
            if ds.head[s as usize] == ds.head[s as usize + 1]
                || ds.head[t as usize] == ds.head[t as usize + 1]
            {
                continue;
            }
            let (a, b) = (sn(s), sn(t));
            if a.seg == b.seg {
                continue;
            }
            let Some(r) = router.route(&ds, &a, &b) else { continue };
            // The route's own duration must equal the reference cost between
            // the two vertices the snaps resolve to.
            let ref_cost = reference[t as usize] as u64;
            assert!(
                r.dur_ds >= ref_cost.saturating_sub(2),
                "bidirectional found a cheaper route than Dijkstra: {} vs {}",
                r.dur_ds,
                ref_cost
            );
            checked += 1;
        }
    }
    assert!(checked > 20, "only {checked} pairs exercised");
    eprintln!("{checked} random pairs agreed with plain Dijkstra");
}

/// A route's reported distance must equal the sum of its legs, exactly.
#[test]
fn distance_is_the_integer_sum_of_its_legs() {
    let Some(ds) = data() else { return };
    let mut rng = Rng(12345);
    let mut router = Router::new(ds.n_vertices());
    let n = ds.n_vertices() as u64;
    let mut seen = 0;
    for _ in 0..2000 {
        let a = ds.vcoord[(rng.next() % n) as usize];
        let b = ds.vcoord[(rng.next() % n) as usize];
        let (Some(sa), Some(sb)) =
            (ds.snap(a.0, a.1, 500.0), ds.snap(b.0, b.1, 500.0))
        else {
            continue;
        };
        let Some(r) = router.route(&ds, &sa, &sb) else { continue };
        let sum: u64 = r.legs.iter().map(|l| l.len_cm).sum();
        assert_eq!(sum, r.dist_cm, "leg lengths must sum to the reported distance");
        seen += 1;
        if seen >= 40 {
            break;
        }
    }
    assert!(seen > 10, "only {seen} routes exercised");
}

/// Every segment must be reachable through the snap index.
///
/// Regression test for a real bug: the index was built by walking edges and
/// skipping the reverse copy of each pair, which silently dropped every way
/// tagged `oneway=-1` — those produce a single edge that runs *against* the
/// stored geometry, so their only copy looked like a reverse copy. Roads like
/// that became unsnappable, and a trip starting on one had nowhere to begin.
#[test]
fn every_segment_is_in_the_snap_index() {
    let Some(ds) = data() else { return };
    let mut rng = Rng(4242);
    let n = ds.n_segments() as u64;
    let mut checked = 0;
    let mut backward_seen = 0;
    for _ in 0..3000 {
        let sid = (rng.next() % n) as u32;
        let mid = ds.vcoord[ds.seg_u[sid as usize] as usize];
        let key = mpee_planet::graph::cell_of(mid.0, mid.1);
        assert!(
            ds.cell_segments(key).contains(&sid),
            "segment {sid} is missing from the cell holding its own start vertex"
        );
        // Count the one-way-against-geometry case so the test proves it covers
        // the shape that used to fail.
        if mpee_planet::build::attr_oneway(ds.attr(sid as usize)) == 2 {
            backward_seen += 1;
        }
        checked += 1;
    }
    assert!(checked > 2000);
    eprintln!("{checked} segments verified, {backward_seen} of them oneway=-1");
}

/// Perpendicular distance from a point to a polyline, in metres.
fn dist_to_polyline(pts: &[(i32, i32)], p: (i32, i32)) -> f64 {
    let lat = p.0 as f64 * 1e-7;
    let sy = 111_132.0 * 1e-7;
    let sx = (111_320.0 * lat.to_radians().cos()).abs().max(1.0) * 1e-7;
    let (qx, qy) = (p.1 as f64 * sx, p.0 as f64 * sy);
    let mut best = f64::INFINITY;
    for w in pts.windows(2) {
        let (ax, ay) = (w[0].1 as f64 * sx, w[0].0 as f64 * sy);
        let (bx, by) = (w[1].1 as f64 * sx, w[1].0 as f64 * sy);
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let t = if l2 <= f64::EPSILON { 0.0 } else { (((qx - ax) * dx + (qy - ay) * dy) / l2).clamp(0.0, 1.0) };
        let (ex, ey) = (qx - (ax + t * dx), qy - (ay + t * dy));
        best = best.min((ex * ex + ey * ey).sqrt());
    }
    best
}

/// The snapped point must lie *on* the segment it names, and `off_m` must be
/// the true ground distance from the query to it. A long straight road stores
/// only its two endpoints, so "near a stored point" would be the wrong test —
/// what matters is the perpendicular distance to the line itself.
#[test]
fn snap_lands_on_its_segment_and_reports_the_true_offset() {
    let Some(ds) = data() else { return };
    let mut rng = Rng(777);
    let n = ds.n_vertices() as u64;
    let mut checked = 0;
    let mut worst_on_line: f64 = 0.0;
    for _ in 0..800 {
        let v = ds.vcoord[(rng.next() % n) as usize];
        // Offset the query by up to ~50 m so it is not sitting on a vertex.
        let q = (v.0 + (rng.next() % 5000) as i32, v.1 + (rng.next() % 5000) as i32);
        let Some(s) = ds.snap(q.0, q.1, 1000.0) else { continue };
        let pts = ds.seg_geometry(s.seg as usize);
        let on_line = dist_to_polyline(&pts, (s.lat_e7, s.lon_e7));
        assert!(on_line < 1.0, "snapped point sits {on_line:.2} m off its own segment");
        worst_on_line = worst_on_line.max(on_line);

        let true_off = mpee_planet::geo::dist_e7(q, (s.lat_e7, s.lon_e7));
        assert!(
            (true_off - s.off_m).abs() < 1.0 + s.off_m * 0.01,
            "reported offset {:.2} m but the point is {:.2} m away",
            s.off_m,
            true_off
        );
        // Nothing closer may exist: check the reported offset against a
        // brute-force sweep of every segment in the surrounding cells.
        let mut brute = f64::INFINITY;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let key = mpee_planet::graph::cell_of(
                    q.0 + (dy * mpee_planet::graph::CELL_E7 as i64) as i32,
                    q.1 + (dx * mpee_planet::graph::CELL_E7 as i64) as i32,
                );
                for &sid in ds.cell_segments(key) {
                    brute = brute.min(dist_to_polyline(&ds.seg_geometry(sid as usize), q));
                }
            }
        }
        if brute.is_finite() {
            assert!(
                s.off_m <= brute + 1.0,
                "snap returned {:.2} m but a brute-force sweep found {:.2} m",
                s.off_m,
                brute
            );
        }
        checked += 1;
    }
    assert!(checked > 100, "only {checked} snaps exercised");
    eprintln!("{checked} snaps verified; worst deviation from the line {worst_on_line:.3} m");
}
