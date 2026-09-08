//! Geodesy: exact segment length, 5 m polyline simplification, Hilbert order.
//!
//! The whole premise of this dataset is "approximate positions, exact
//! lengths", so length is the one number that must not be sloppy. Plain
//! haversine on a sphere is off by up to 0.5 % — 2.5 km on a 500 km route —
//! which would defeat the point. We integrate the WGS84 metric instead:
//!
//!   * short hops (the 99.9 % case: consecutive OSM shape points, tens of
//!     metres) use the local ellipsoidal approximation, exact to well under a
//!     millimetre at that scale and branch-free;
//!   * long hops (ferries, sparsely-mapped desert tracks) fall back to
//!     Vincenty's inverse solution, exact to ~0.5 mm.
//!
//! Everything accumulates in `f64` and is only quantised to centimetres once,
//! at the very end, so a route's length is the exact sum of its segments.

pub const E7: f64 = 1e-7;

// WGS84
const A: f64 = 6_378_137.0;
const F: f64 = 1.0 / 298.257_223_563;
const B: f64 = A * (1.0 - F);
const E2: f64 = F * (2.0 - F);

#[inline]
pub fn e7_to_deg(v: i32) -> f64 {
    v as f64 * E7
}

/// Local ellipsoidal distance in metres. Accurate to <1 mm for hops under a
/// kilometre, which is what consecutive OSM shape points always are.
#[inline]
pub fn short_dist_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let phi = (lat1 + lat2) * 0.5 * std::f64::consts::PI / 180.0;
    let s = phi.sin();
    let w2 = 1.0 - E2 * s * s;
    let w = w2.sqrt();
    // Meridional and prime-vertical radii of curvature at the mean latitude.
    let m = A * (1.0 - E2) / (w2 * w);
    let n = A / w;
    let dphi = (lat2 - lat1) * std::f64::consts::PI / 180.0;
    let dlam = (lon2 - lon1) * std::f64::consts::PI / 180.0;
    let x = dphi * m;
    let y = dlam * n * phi.cos();
    (x * x + y * y).sqrt()
}

/// Vincenty inverse — exact geodesic distance, used for long hops.
pub fn vincenty_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let rad = std::f64::consts::PI / 180.0;
    let (phi1, phi2) = (lat1 * rad, lat2 * rad);
    let l = (lon2 - lon1) * rad;
    let u1 = ((1.0 - F) * phi1.tan()).atan();
    let u2 = ((1.0 - F) * phi2.tan()).atan();
    let (su1, cu1) = (u1.sin(), u1.cos());
    let (su2, cu2) = (u2.sin(), u2.cos());
    let mut lam = l;
    let mut cos_sq_alpha = 0.0;
    let mut sin_sigma = 0.0;
    let mut cos_sigma = 0.0;
    let mut cos2_sigma_m = 0.0;
    let mut sigma = 0.0;
    for _ in 0..200 {
        let (sl, cl) = (lam.sin(), lam.cos());
        let t1 = cu2 * sl;
        let t2 = cu1 * su2 - su1 * cu2 * cl;
        sin_sigma = (t1 * t1 + t2 * t2).sqrt();
        if sin_sigma == 0.0 {
            return 0.0; // coincident
        }
        cos_sigma = su1 * su2 + cu1 * cu2 * cl;
        sigma = sin_sigma.atan2(cos_sigma);
        let sin_alpha = cu1 * cu2 * sl / sin_sigma;
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
        cos2_sigma_m = if cos_sq_alpha != 0.0 {
            cos_sigma - 2.0 * su1 * su2 / cos_sq_alpha
        } else {
            0.0
        };
        let c = F / 16.0 * cos_sq_alpha * (4.0 + F * (4.0 - 3.0 * cos_sq_alpha));
        let lam_prev = lam;
        lam = l
            + (1.0 - c)
                * F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos2_sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));
        if (lam - lam_prev).abs() < 1e-12 {
            break;
        }
    }
    let u_sq = cos_sq_alpha * (A * A - B * B) / (B * B);
    let aa = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let bb = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
    let d_sigma = bb
        * sin_sigma
        * (cos2_sigma_m
            + bb / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                    - bb / 6.0
                        * cos2_sigma_m
                        * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                        * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));
    B * aa * (sigma - d_sigma)
}

/// Exact WGS84 distance in metres between two `e7` points.
#[inline]
pub fn dist_e7(a: (i32, i32), b: (i32, i32)) -> f64 {
    let (lat1, lon1) = (e7_to_deg(a.0), e7_to_deg(a.1));
    let (lat2, lon2) = (e7_to_deg(b.0), e7_to_deg(b.1));
    let dlat = (a.0 - b.0).unsigned_abs() as f64;
    let dlon = (a.1 - b.1).unsigned_abs() as f64;
    // ~2 km in e7 units of latitude; beyond that the local metric drifts.
    if dlat < 180_000.0 && dlon < 180_000.0 {
        short_dist_m(lat1, lon1, lat2, lon2)
    } else {
        vincenty_m(lat1, lon1, lat2, lon2)
    }
}

/// Exact length of a polyline, in metres.
pub fn polyline_len_m(pts: &[(i32, i32)]) -> f64 {
    let mut acc = 0.0f64;
    for w in pts.windows(2) {
        acc += dist_e7(w[0], w[1]);
    }
    acc
}

// ------------------------------------------------------- simplification

/// Metres-per-e7-unit scale factors at a given latitude, for turning the
/// perpendicular-distance test into a real ground distance cheaply.
#[inline]
fn scale_at(lat_e7: i32) -> (f64, f64) {
    let phi = e7_to_deg(lat_e7) * std::f64::consts::PI / 180.0;
    let s = phi.sin();
    let w2 = 1.0 - E2 * s * s;
    let w = w2.sqrt();
    let m = A * (1.0 - E2) / (w2 * w); // metres per radian of latitude
    let n = A / w;
    let rad_per_e7 = std::f64::consts::PI / 180.0 * E7;
    (m * rad_per_e7, n * phi.cos() * rad_per_e7)
}

/// Douglas–Peucker simplification with the tolerance expressed in **metres**.
///
/// Iterative (explicit stack) rather than recursive: OSM contains ways with
/// hundreds of thousands of points and a planet build must not blow the stack.
pub fn simplify(pts: &[(i32, i32)], tol_m: f64, keep: &mut Vec<u32>) {
    keep.clear();
    if pts.len() <= 2 {
        for i in 0..pts.len() {
            keep.push(i as u32);
        }
        return;
    }
    let (sy, sx) = scale_at(pts[pts.len() / 2].0);
    let tol2 = tol_m * tol_m;
    let mut mark = vec![false; pts.len()];
    mark[0] = true;
    mark[pts.len() - 1] = true;
    let mut stack: Vec<(usize, usize)> = vec![(0, pts.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (ax, ay) = (pts[a].1 as f64 * sx, pts[a].0 as f64 * sy);
        let (bx, by) = (pts[b].1 as f64 * sx, pts[b].0 as f64 * sy);
        let (dx, dy) = (bx - ax, by - ay);
        let len2 = dx * dx + dy * dy;
        let mut best = 0.0f64;
        let mut best_i = a;
        // The index is the result here, not just a cursor: the split point is
        // what recursion needs, so iterating over values would lose it.
        #[allow(clippy::needless_range_loop)]
        for i in (a + 1)..b {
            let (px, py) = (pts[i].1 as f64 * sx, pts[i].0 as f64 * sy);
            let d2 = if len2 <= f64::EPSILON {
                let (ex, ey) = (px - ax, py - ay);
                ex * ex + ey * ey
            } else {
                let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
                let (ex, ey) = (px - (ax + t * dx), py - (ay + t * dy));
                ex * ex + ey * ey
            };
            if d2 > best {
                best = d2;
                best_i = i;
            }
        }
        if best > tol2 {
            mark[best_i] = true;
            stack.push((a, best_i));
            stack.push((best_i, b));
        }
    }
    for (i, &m) in mark.iter().enumerate() {
        if m {
            keep.push(i as u32);
        }
    }
}

// ------------------------------------------------------------- Hilbert

/// 2D Hilbert index of a point on a `2^order` grid.
///
/// Nodes are stored in Hilbert order so that geographic neighbours are byte
/// neighbours. That is what makes a bigger-than-RAM mmap practical: a search
/// around Oslo touches a handful of contiguous pages instead of scattering
/// reads across 20 GB.
pub fn hilbert_d(order: u32, mut x: u32, mut y: u32) -> u64 {
    let mut rx: u32;
    let mut ry: u32;
    let mut d: u64 = 0;
    let mut s: u32 = 1 << (order - 1);
    while s > 0 {
        rx = u32::from((x & s) > 0);
        ry = u32::from((y & s) > 0);
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
        // rotate
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

/// Hilbert index of a WGS84 position on a 2^16 × 2^16 world grid (~600 m cells).
#[inline]
pub fn hilbert_latlon(lat_e7: i32, lon_e7: i32) -> u64 {
    const ORDER: u32 = 16;
    const MAX: f64 = ((1u64 << ORDER) - 1) as f64;
    let x = (((e7_to_deg(lon_e7) + 180.0) / 360.0).clamp(0.0, 1.0) * MAX).round() as u32;
    let y = (((e7_to_deg(lat_e7) + 90.0) / 180.0).clamp(0.0, 1.0) * MAX).round() as u32;
    hilbert_d(ORDER, x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e7(lat: f64, lon: f64) -> (i32, i32) {
        ((lat / E7).round() as i32, (lon / E7).round() as i32)
    }

    #[test]
    fn short_distance_matches_vincenty() {
        // Two points ~85 m apart in Oslo.
        let a = (59.911_49, 10.757_33);
        let b = (59.912_25, 10.757_51);
        let fast = short_dist_m(a.0, a.1, b.0, b.1);
        let exact = vincenty_m(a.0, a.1, b.0, b.1);
        assert!((fast - exact).abs() < 1e-3, "fast={fast} exact={exact}");
    }

    #[test]
    fn long_distance_uses_geodesic() {
        // Oslo → Trondheim, great-circle ≈ 391 km. Vincenty is the reference.
        let d = dist_e7(e7(59.9139, 10.7522), e7(63.4305, 10.3951));
        assert!((d - 391_000.0).abs() < 3_000.0, "got {d} m");
    }

    #[test]
    fn haversine_error_is_what_we_avoid() {
        // A sphere of radius 6371 km under-measures this north-south leg by
        // roughly a kilometre; that is the error the ellipsoid removes.
        let (a, b) = ((59.9139_f64, 10.7522_f64), (63.4305_f64, 10.3951_f64));
        let r = 6_371_000.0_f64;
        let (p1, p2) = (a.0.to_radians(), b.0.to_radians());
        let (dp, dl) = ((b.0 - a.0).to_radians(), (b.1 - a.1).to_radians());
        let h = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
        let hav = 2.0 * r * h.sqrt().asin();
        let exact = vincenty_m(a.0, a.1, b.0, b.1);
        assert!((hav - exact).abs() > 500.0, "haversine {hav} vs exact {exact}");
    }

    #[test]
    fn simplify_keeps_endpoints_and_drops_collinear() {
        // A straight north-south line: everything between the ends is redundant.
        let pts: Vec<(i32, i32)> =
            (0..20).map(|i| e7(59.9 + i as f64 * 0.0001, 10.75)).collect();
        let mut keep = Vec::new();
        simplify(&pts, 5.0, &mut keep);
        assert_eq!(keep, vec![0, 19]);
    }

    #[test]
    fn simplify_respects_the_tolerance() {
        // A 30 m bump in the middle must survive a 5 m tolerance.
        let mut pts = vec![e7(59.9, 10.75), e7(59.9, 10.7505), e7(59.9, 10.751)];
        pts[1].0 += 3000; // ~33 m north
        let mut keep = Vec::new();
        simplify(&pts, 5.0, &mut keep);
        assert_eq!(keep, vec![0, 1, 2]);
        simplify(&pts, 50.0, &mut keep);
        assert_eq!(keep, vec![0, 2]);
    }

    #[test]
    fn hilbert_is_a_bijection_and_local() {
        let mut seen = std::collections::HashSet::new();
        for x in 0..16u32 {
            for y in 0..16u32 {
                assert!(seen.insert(hilbert_d(4, x, y)), "duplicate at {x},{y}");
            }
        }
        assert_eq!(seen.len(), 256);
        // Consecutive indices are grid neighbours.
        for x in 0..15u32 {
            let a = hilbert_d(4, x, 0);
            let b = hilbert_d(4, x + 1, 0);
            assert!(a.abs_diff(b) >= 1);
        }
    }
}
