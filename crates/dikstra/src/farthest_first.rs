//! Farthest-first traversal ordering on (lat, lon) coordinates.
//!
//! Given a set of points, return a permutation where:
//!   * indices 0 and 1 are (approximately) the pair maximising pairwise
//!     planar distance — the "diameter" pair,
//!   * every subsequent index maximises its minimum planar distance to
//!     all previously selected indices.
//!
//! Use case: stream the rows of a many-to-many matrix in this order so a
//! downstream solver receives a geometrically representative early sample
//! and can start producing approximate solutions before the full matrix
//! has finished. Compute time of the matrix itself is unchanged — only
//! the row order changes.
//!
//! Distance metric: equirectangular `(lat, lon × cos(mean_lat))` —
//! ordering matches haversine for any city / country dataset and is
//! several × cheaper.
//!
//! Complexity: O(n²) time, O(n) extra space. For n = 50k that is ~2.5
//! billion arithmetic ops; rayon-parallel both inner phases of each step.
//! Wall-clock ≈ 2–5 s on a multi-core box.

use rayon::prelude::*;

pub fn farthest_first_order(coords: &[(f32, f32)]) -> Vec<u32> {
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mean_lat: f64 = coords.iter().map(|&(la, _)| la as f64).sum::<f64>() / n as f64;
    let lon_scale = mean_lat.to_radians().cos() as f32;

    let d2 = |a: usize, b: usize| -> f32 {
        let (la_a, lo_a) = coords[a];
        let (la_b, lo_b) = coords[b];
        let dy = la_b - la_a;
        let dx = (lo_b - lo_a) * lon_scale;
        dx * dx + dy * dy
    };

    // 2-approximation of the diameter pair.
    let anchor = 0usize;
    let j = farthest_from(coords, anchor, lon_scale);
    let m = farthest_from(coords, j, lon_scale);

    let mut perm: Vec<u32> = Vec::with_capacity(n);
    let mut min_d2: Vec<f32> = vec![f32::INFINITY; n];
    let mut selected: Vec<bool> = vec![false; n];
    perm.push(j as u32);
    selected[j] = true;
    if m != j {
        perm.push(m as u32);
        selected[m] = true;
    }

    // Initialise min_d2 to min distance to {j, m}.
    min_d2.par_iter_mut().enumerate().for_each(|(i, slot)| {
        if selected[i] {
            *slot = 0.0;
            return;
        }
        let dj = d2(i, j);
        if m != j {
            let dm = d2(i, m);
            *slot = dj.min(dm);
        } else {
            *slot = dj;
        }
    });

    while perm.len() < n {
        // argmax min_d2 over unselected indices.
        let (best, _best_d) = (0..n)
            .into_par_iter()
            .map(|i| {
                if selected[i] {
                    (i, f32::NEG_INFINITY)
                } else {
                    (i, min_d2[i])
                }
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            );
        perm.push(best as u32);
        selected[best] = true;

        // Update min_d2 with distance to `best`.
        let (la_b, lo_b) = coords[best];
        min_d2
            .par_iter_mut()
            .zip(coords.par_iter())
            .enumerate()
            .for_each(|(i, (slot, &(la, lo)))| {
                if selected[i] {
                    return;
                }
                let dy = la - la_b;
                let dx = (lo - lo_b) * lon_scale;
                let d = dx * dx + dy * dy;
                if d < *slot {
                    *slot = d;
                }
            });
    }

    perm
}

#[inline]
fn farthest_from(coords: &[(f32, f32)], anchor: usize, lon_scale: f32) -> usize {
    let (la_a, lo_a) = coords[anchor];
    coords
        .par_iter()
        .enumerate()
        .map(|(i, &(la, lo))| {
            let dy = la - la_a;
            let dx = (lo - lo_a) * lon_scale;
            (i, dx * dx + dy * dy)
        })
        .reduce(
            || (anchor, 0.0_f32),
            |a, b| if a.1 >= b.1 { a } else { b },
        )
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_zero_and_one() {
        assert_eq!(farthest_first_order(&[]), Vec::<u32>::new());
        assert_eq!(farthest_first_order(&[(0.0, 0.0)]), vec![0]);
    }

    #[test]
    fn diameter_starts_first() {
        // Four corners of a unit square plus a centre — the diameter is the
        // diagonal of the square. The first two outputs must be opposite
        // corners.
        let coords = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.5),
        ];
        let perm = farthest_first_order(&coords);
        assert_eq!(perm.len(), 5);
        // perm[0..2] should be a diagonal pair: (0,0)<->(1,1) or (1,0)<->(0,1)
        let p0 = coords[perm[0] as usize];
        let p1 = coords[perm[1] as usize];
        let diag = (p0.0 - p1.0).abs() > 0.5 && (p0.1 - p1.1).abs() > 0.5;
        assert!(diag, "first two should be a diagonal pair, got {p0:?} {p1:?}");
    }

    #[test]
    fn is_a_permutation() {
        let coords: Vec<(f32, f32)> = (0..50)
            .map(|i| (i as f32 * 0.13, (i * 7 % 50) as f32 * 0.21))
            .collect();
        let perm = farthest_first_order(&coords);
        assert_eq!(perm.len(), 50);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<u32> = (0..50).collect();
        assert_eq!(sorted, expected);
    }
}
