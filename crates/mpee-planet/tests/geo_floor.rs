//! The geographic floor must never exceed the truth.
//!
//! Everything the search does with it — the A* potential, the settle-time
//! prune — assumes it is a *lower* bound on travel time. One that overstates
//! does not merely slow the search down: it prunes correct routes and returns
//! a longer path while reporting success. So the property under test is not
//! "close to" but "never above", checked against the great-circle arc the
//! chord replaced.

use mpee_planet::overlay::{geo_floor_between, xyz_of, R_POLAR};

/// The arc the chord replaced: the great circle on a sphere of polar radius.
fn arc_m(a: (i32, i32), b: (i32, i32)) -> f64 {
    let (a1, o1) = (a.0 as f64 * 1e-7, a.1 as f64 * 1e-7);
    let (a2, o2) = (b.0 as f64 * 1e-7, b.1 as f64 * 1e-7);
    let (dlat, dlon) = ((a2 - a1).to_radians(), (o2 - o1).to_radians());
    let h = (dlat / 2.0).sin().powi(2)
        + a1.to_radians().cos() * a2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * R_POLAR * h.sqrt().clamp(0.0, 1.0).asin()
}

fn chord_m(a: (i32, i32), b: (i32, i32)) -> f64 {
    let (p, t) = (xyz_of(a.0, a.1), xyz_of(b.0, b.1));
    ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2) + (p[2] - t[2]).powi(2)).sqrt()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A point anywhere on Earth.
    fn point(&mut self) -> (i32, i32) {
        let lat = (self.next() % 1_800_000_001) as i64 - 900_000_000;
        let lon = (self.next() % 3_600_000_001) as i64 - 1_800_000_000;
        (lat as i32, lon as i32)
    }
}

#[test]
fn the_chord_never_exceeds_the_arc_anywhere_on_earth() {
    let mut r = Rng(0x9E3779B97F4A7C15);
    let mut worst_ratio = 1.0f64;
    for _ in 0..200_000 {
        let (a, b) = (r.point(), r.point());
        let (c, s) = (chord_m(a, b), arc_m(a, b));
        assert!(c <= s + 1e-6, "chord {c} > arc {s} for {a:?} {b:?}");
        if s > 1000.0 {
            worst_ratio = worst_ratio.max(s / c);
        }
    }
    // Antipodal pairs are the worst case: the chord is a diameter where the
    // arc is half the circumference, so pi/2. Real routes are nowhere near.
    assert!(worst_ratio <= std::f64::consts::FRAC_PI_2 + 1e-9, "{worst_ratio}");
}

#[test]
fn the_floor_in_time_never_exceeds_the_arc_in_time() {
    // What the search actually consumes: tenths of a second at the fastest
    // anything travels. The slack subtracted for f32 rounding only ever makes
    // this smaller, so the inequality has to hold with room to spare.
    let mut r = Rng(12345);
    for _ in 0..200_000 {
        let (a, b) = (r.point(), r.point());
        for vmax in [30u16, 90, 130, 400] {
            let got = geo_floor_between(a, b, vmax);
            let truth = (arc_m(a, b) / (vmax as f64 * 1000.0 / 3600.0) * 10.0) as u32;
            assert!(got <= truth, "floor {got} > arc-time {truth} at {vmax} km/h");
        }
    }
}

#[test]
fn a_point_against_itself_is_zero_and_never_negative() {
    let mut r = Rng(99);
    for _ in 0..10_000 {
        let a = r.point();
        assert_eq!(geo_floor_between(a, a, 130), 0);
    }
}

#[test]
fn the_slack_is_not_so_large_that_the_bound_stops_saying_anything() {
    // Four metres of headroom must not swallow short hops. At 130 km/h it is
    // about a tenth of a second, so anything above a few hundred metres should
    // still register.
    let oslo = (599_139_000, 107_522_000);
    let near = (599_200_000, 107_522_000); // ~680 m due north
    assert!(geo_floor_between(oslo, near, 130) > 0);
}

#[test]
fn known_distances_land_where_they_should() {
    // Lisboa-Warszawa: the route the whole measurement turned on. The chord is
    // shorter than the arc by the amount the doc comment claims, and if that
    // ever stops being true the comment is wrong, not the test.
    let lisboa = (387_223_000, -91_393_000);
    let warszawa = (522_297_000, 210_122_000);
    let (c, s) = (chord_m(lisboa, warszawa), arc_m(lisboa, warszawa));
    assert!((s / 1000.0 - 2753.0).abs() < 5.0, "arc {s}");
    let shortfall = 100.0 * (1.0 - c / s);
    assert!((shortfall - 0.77).abs() < 0.05, "chord is {shortfall:.2} % under the arc");
}
