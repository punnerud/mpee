//! Reading side of the address index: street + number → coordinate, and
//! coordinate → address. Both are pure binary searches over mmapped arrays —
//! no text index is loaded, nothing is built at startup.

use crate::addresses::normalize;
use crate::geo;
use crate::graph::{cell_of, CELL_E7};
use crate::mmapvec;
use memmap2::Mmap;
use std::io;
use std::path::Path;

pub struct Geocoder {
    _maps: Vec<Mmap>,
    coord: &'static [(i32, i32)],
    hn: &'static [u32],
    street: &'static [u32],
    place: &'static [u32],
    cell_key: &'static [u64],
    cell_off: &'static [u32],
    fwd_idx: &'static [u32],
    /// Flat triples (street, place, start).
    fwd_group: &'static [u32],
    hn_num: &'static [u32],
    street_pool: &'static [u8],
    street_off: &'static [u64],
    street_sorted: &'static [u32],
    street_norm: &'static [u8],
    street_normoff: &'static [u64],
    city_pool: &'static [u8],
    city_off: &'static [u64],
    pc_pool: &'static [u8],
    pc_off: &'static [u64],
    hn_pool: &'static [u8],
    hn_off: &'static [u64],
    place_tab: &'static [u32],
}

unsafe fn m<T>(dir: &Path, name: &str, keep: &mut Vec<Mmap>) -> io::Result<&'static [T]> {
    let mm = mmapvec::open(&dir.join(name))?;
    let s: &[T] = mmapvec::as_slice(&mm[..]);
    let out = std::slice::from_raw_parts(s.as_ptr(), s.len());
    keep.push(mm);
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub lat: f64,
    pub lon: f64,
    pub street: String,
    pub housenumber: String,
    pub city: Option<String>,
    pub postcode: Option<String>,
    /// True when the exact house number was absent and a neighbour was used.
    pub approximate: bool,
    /// Metres from the query coordinate (reverse lookup only).
    pub distance_m: f64,
}

impl Geocoder {
    pub fn open(dir: &Path) -> io::Result<Geocoder> {
        let mut k = Vec::new();
        unsafe {
            Ok(Geocoder {
                coord: m(dir, "addr.coord", &mut k)?,
                hn: m(dir, "addr.hn", &mut k)?,
                street: m(dir, "addr.street", &mut k)?,
                place: m(dir, "addr.place", &mut k)?,
                cell_key: m(dir, "acell.key", &mut k)?,
                cell_off: m(dir, "acell.off", &mut k)?,
                fwd_idx: m(dir, "fwd.idx", &mut k)?,
                fwd_group: m(dir, "fwd.group", &mut k)?,
                hn_num: m(dir, "hn.num", &mut k)?,
                street_pool: m(dir, "street.pool", &mut k)?,
                street_off: m(dir, "street.off", &mut k)?,
                street_sorted: m(dir, "street.sorted", &mut k)?,
                street_norm: m(dir, "street.norm", &mut k)?,
                street_normoff: m(dir, "street.normoff", &mut k)?,
                city_pool: m(dir, "city.pool", &mut k)?,
                city_off: m(dir, "city.off", &mut k)?,
                pc_pool: m(dir, "pc.pool", &mut k)?,
                pc_off: m(dir, "pc.off", &mut k)?,
                hn_pool: m(dir, "hn.pool", &mut k)?,
                hn_off: m(dir, "hn.off", &mut k)?,
                place_tab: m(dir, "place.tab", &mut k)?,
                _maps: k,
            })
        }
    }

    pub fn len(&self) -> usize {
        self.coord.len()
    }
    pub fn is_empty(&self) -> bool {
        self.coord.is_empty()
    }
    pub fn street_count(&self) -> usize {
        self.street_off.len().saturating_sub(1)
    }

    fn pool<'a>(pool: &'a [u8], off: &[u64], i: u32) -> Option<&'a str> {
        if i == u32::MAX {
            return None;
        }
        let (a, b) = (*off.get(i as usize)? as usize, *off.get(i as usize + 1)? as usize);
        std::str::from_utf8(pool.get(a..b)?).ok()
    }

    /// Normalised name of the `k`-th street in sorted order.
    fn norm_at(&self, k: usize) -> &str {
        let (a, b) = (self.street_normoff[k] as usize, self.street_normoff[k + 1] as usize);
        std::str::from_utf8(&self.street_norm[a..b]).unwrap_or("")
    }

    /// Range of `street.sorted` positions whose normalised name equals `q`.
    fn street_range(&self, q: &str) -> std::ops::Range<usize> {
        let n = self.street_sorted.len();
        let lo = partition_point(n, |i| self.norm_at(i) < q);
        let hi = partition_point(n, |i| self.norm_at(i) <= q);
        lo..hi
    }

    /// Street ids whose normalised name starts with `q` — for autocomplete.
    pub fn suggest(&self, q: &str, limit: usize) -> Vec<String> {
        let q = normalize(q);
        if q.is_empty() {
            return Vec::new();
        }
        let n = self.street_sorted.len();
        let lo = partition_point(n, |i| self.norm_at(i) < q.as_str());
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for k in lo..n {
            if !self.norm_at(k).starts_with(&q) {
                break;
            }
            let sid = self.street_sorted[k];
            if let Some(s) = Self::pool(self.street_pool, self.street_off, sid) {
                if seen.insert(s.to_string()) {
                    out.push(s.to_string());
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        out
    }

    fn group_span(&self, gi: usize) -> (u32, u32, usize, usize) {
        let s = self.fwd_group[gi * 3];
        let p = self.fwd_group[gi * 3 + 1];
        let a = self.fwd_group[gi * 3 + 2] as usize;
        let b = self.fwd_group[(gi + 1) * 3 + 2] as usize;
        (s, p, a, b)
    }

    fn n_groups(&self) -> usize {
        self.fwd_group.len() / 3 - 1
    }

    /// Groups belonging to one street id.
    fn groups_of(&self, sid: u32) -> std::ops::Range<usize> {
        let n = self.n_groups();
        let lo = partition_point(n, |i| self.fwd_group[i * 3] < sid);
        let hi = partition_point(n, |i| self.fwd_group[i * 3] <= sid);
        lo..hi
    }

    fn hit(&self, addr: usize, approximate: bool, distance_m: f64) -> Hit {
        let (lat, lon) = self.coord[addr];
        let pl = self.place[addr];
        let (c, p) = if (pl as usize) < self.place_tab.len() / 2 {
            (self.place_tab[pl as usize * 2], self.place_tab[pl as usize * 2 + 1])
        } else {
            (u32::MAX, u32::MAX)
        };
        Hit {
            lat: geo::e7_to_deg(lat),
            lon: geo::e7_to_deg(lon),
            street: Self::pool(self.street_pool, self.street_off, self.street[addr])
                .unwrap_or("")
                .to_string(),
            housenumber: Self::pool(self.hn_pool, self.hn_off, self.hn[addr])
                .unwrap_or("")
                .to_string(),
            city: Self::pool(self.city_pool, self.city_off, c).map(str::to_string),
            postcode: Self::pool(self.pc_pool, self.pc_off, p).map(str::to_string),
            approximate,
            distance_m,
        }
    }

    /// Street name + house number → coordinates.
    ///
    /// `city` is an optional filter matched against the address's city or
    /// postcode. When the exact number is missing the nearest number on the
    /// same street is returned, flagged `approximate`, so "42 not mapped but
    /// 40 and 44 are" still yields a usable pin.
    pub fn forward(&self, street: &str, number: &str, city: Option<&str>, limit: usize) -> Vec<Hit> {
        let q = normalize(street);
        let want = crate::addresses::hn_number(number);
        let cityq = city.map(normalize);
        let mut out = Vec::new();
        for k in self.street_range(&q) {
            let sid = self.street_sorted[k];
            for gi in self.groups_of(sid) {
                let (_, pl, a, b) = self.group_span(gi);
                if let Some(cq) = &cityq {
                    let (c, p) = (
                        self.place_tab[pl as usize * 2],
                        self.place_tab[pl as usize * 2 + 1],
                    );
                    let cm = Self::pool(self.city_pool, self.city_off, c)
                        .map(|s| normalize(s) == *cq)
                        .unwrap_or(false);
                    let pm = Self::pool(self.pc_pool, self.pc_off, p)
                        .map(|s| normalize(s) == *cq)
                        .unwrap_or(false);
                    if !cm && !pm {
                        continue;
                    }
                }
                if a >= b {
                    continue;
                }
                // Entries in a group are ordered by house number.
                let num_at = |i: usize| self.hn_num[self.hn[self.fwd_idx[i] as usize] as usize];
                let pos = a + partition_point(b - a, |i| num_at(a + i) < want);
                let mut best = pos.min(b - 1);
                if number.is_empty() {
                    best = a;
                }
                let exact = num_at(best) == want && !number.is_empty();
                if !exact && pos > a {
                    // Pick whichever neighbour is numerically closer.
                    let lo = pos - 1;
                    if want.abs_diff(num_at(lo)) <= want.abs_diff(num_at(best)) {
                        best = lo;
                    }
                }
                // Prefer the entry whose full string matches, e.g. "42B".
                let mut chosen = best;
                if exact {
                    let target = normalize(number);
                    for i in a..b {
                        if num_at(i) != want {
                            continue;
                        }
                        let s = Self::pool(
                            self.hn_pool,
                            self.hn_off,
                            self.hn[self.fwd_idx[i] as usize],
                        )
                        .unwrap_or("");
                        if normalize(s) == target {
                            chosen = i;
                            break;
                        }
                    }
                }
                out.push(self.hit(self.fwd_idx[chosen] as usize, !exact, 0.0));
                if out.len() >= limit * 4 {
                    break;
                }
            }
        }
        // A street usually spans several postcodes, so an exact number can
        // surface behind three approximate neighbours from other groups.
        // Rank exact hits first, then by how far off the number is.
        out.sort_by(|a, b| {
            a.approximate.cmp(&b.approximate).then_with(|| {
                let da = crate::addresses::hn_number(&a.housenumber).abs_diff(want);
                let db = crate::addresses::hn_number(&b.housenumber).abs_diff(want);
                da.cmp(&db)
            })
        });
        out.truncate(limit);
        out
    }

    /// Coordinate → nearest address, searching outward by grid cell.
    pub fn reverse(&self, lat: f64, lon: f64, max_m: f64) -> Option<Hit> {
        let lat_e7 = (lat / geo::E7).round() as i32;
        let lon_e7 = (lon / geo::E7).round() as i32;
        let mut best: Option<(f64, usize)> = None;
        // A grid cell spans 0.01° of latitude ≈ 1113 m, which is the distance
        // each additional ring buys us.
        let ring_m = 1113.0;
        let mut ring = 0i64;
        loop {
            for dy in -ring..=ring {
                for dx in -ring..=ring {
                    if dy.abs() != ring && dx.abs() != ring {
                        continue;
                    }
                    let key = cell_of(
                        lat_e7 + (dy * CELL_E7 as i64) as i32,
                        lon_e7 + (dx * CELL_E7 as i64) as i32,
                    );
                    if let Ok(i) = self.cell_key.binary_search(&key) {
                        let (a, b) = (self.cell_off[i] as usize, self.cell_off[i + 1] as usize);
                        for j in a..b {
                            let d = geo::dist_e7(self.coord[j], (lat_e7, lon_e7));
                            if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                                best = Some((d, j));
                            }
                        }
                    }
                }
            }
            if let Some((d, _)) = &best {
                if *d <= ring as f64 * ring_m {
                    break;
                }
            }
            ring += 1;
            if ring as f64 * ring_m > max_m + ring_m {
                break;
            }
        }
        best.filter(|(d, _)| *d <= max_m).map(|(d, i)| self.hit(i, false, d))
    }
}

/// `slice::partition_point` over an index range, without materialising the
/// slice — the arrays it searches are mmapped and indirect.
#[inline]
fn partition_point(n: usize, pred: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if pred(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::partition_point;

    #[test]
    fn partition_point_finds_the_boundary() {
        let v = [1, 3, 3, 5, 9];
        assert_eq!(partition_point(v.len(), |i| v[i] < 3), 1);
        assert_eq!(partition_point(v.len(), |i| v[i] <= 3), 3);
        assert_eq!(partition_point(v.len(), |i| v[i] < 0), 0);
        assert_eq!(partition_point(v.len(), |i| v[i] < 100), 5);
    }
}
