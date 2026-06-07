//! House-number address sidecar for **offline address geocoding**.
//!
//! Where [`crate::names`] resolves *streets*, this resolves full *addresses*
//! ("Karl Johans gate 42" → coord, and coord → "Karl Johans gate 42"). Address
//! points are NOT in the routing graph (they come from OSM `addr:*` nodes,
//! building-way centroids, and `addr:interpolation` ways), so this index is
//! fully independent of the CH/CSR — it carries its own coordinates and its own
//! [`LatLonGrid`], and needs no node-permutation remap.
//!
//!   * **Forward** (street + number → coord): resolve the street by name
//!     (case-insensitive, like the names table), then scan that street's address
//!     list (a small CSR group, sorted by number). Exact number wins; otherwise
//!     the nearest existing number on the same street is returned, flagged
//!     `approximate` (so "42 missing, 40/44 present" still yields a useful pin).
//!   * **Reverse** (coord → address): nearest address point via the grid.
//!
//! On disk, a small sidecar next to `.pp`/`.ch`, independently deletable
//! (missing `.addr` ⇒ today's street-only behaviour):
//! ```text
//! 0:    magic        "MPEEADR1"      (8)
//! 8:    a            u64   address-point count
//! 16:   s            u64   distinct street count
//! 24:   hn_pool_len  u64
//! 32:   st_pool_len  u64
//! 40:   city_pool_len u64  (0 if no city present)
//! 48:   pc_pool_len  u64   (0 if no postcode present)
//! 56:   flags        u64   bit0 = has_city, bit1 = has_postcode
//! 64:   coords       a × (f32,f32)
//! ...:  hn_off       (a+1) × u32     per-address housenumber byte ranges
//! ...:  street_id    a × u32         per-address index into the street pool
//! ...:  st_off       (s+1) × u32
//! ...:  fwd_off      (s+1) × u32     per-street CSR offsets into fwd_idx
//! ...:  fwd_idx      a × u32         address indices, grouped by street, sorted by number
//! ...:  [has_city]   city_off (a+1)×u32
//! ...:  [has_pc]     pc_off   (a+1)×u32
//! ...:  hn_pool      hn_pool_len     UTF-8 housenumbers ("42", "42B", "42-44")
//! ...:  st_pool      st_pool_len     distinct street names
//! ...:  [has_city]   city_pool       city_pool_len
//! ...:  [has_pc]     pc_pool         pc_pool_len
//! ```
//! All fixed-width (u32 / coord) arrays come first so each stays naturally
//! aligned; the variable-length UTF-8 byte pools go last.

use crate::buffer::Buffer;
use crate::geo_index::LatLonGrid;
#[cfg(feature = "native")]
use memmap2::Mmap;
#[cfg(feature = "native")]
use std::collections::HashSet;
#[cfg(feature = "native")]
use std::fs::OpenOptions;
#[cfg(feature = "native")]
use std::io::{BufWriter, Write};
#[cfg(feature = "native")]
use std::path::Path;
#[cfg(feature = "native")]
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"MPEEADR1";
const HEADER_BYTES: usize = 64;
const FLAG_CITY: u64 = 1;
const FLAG_PC: u64 = 2;

/// One address point collected during the OSM build.
#[derive(Clone, Debug)]
pub struct AddrRec {
    pub lat: f32,
    pub lon: f32,
    pub housenumber: String,
    pub street: String,
    pub city: Option<String>,
    pub postcode: Option<String>,
}

/// A resolved address (forward or reverse result).
#[derive(Clone, Debug, PartialEq)]
pub struct AddressHit {
    pub lat: f32,
    pub lon: f32,
    pub street: String,
    pub housenumber: String,
    pub city: Option<String>,
    pub postcode: Option<String>,
    /// True when the exact house number was not present and the nearest number
    /// on the street was returned instead.
    pub approximate: bool,
}

/// Split a house number into (leading integer, lowercase suffix). Non-numeric
/// numbers sort last. "42B"→(42,"b"), "42-44"→(42,"-44"), "C"→(MAX,"c").
fn norm_num(s: &str) -> (i64, String) {
    let t = s.trim();
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n = digits.parse::<i64>().unwrap_or(i64::MAX);
    let suffix = t[digits.len()..].to_lowercase();
    (n, suffix)
}

fn planar2(a: (f32, f32), b: (f32, f32)) -> f32 {
    let scale = (a.0 as f64).to_radians().cos() as f32;
    let dlat = a.0 - b.0;
    let dlon = (a.1 - b.1) * scale.max(1e-6);
    dlat * dlat + dlon * dlon
}

/// A loaded address sidecar. Arrays are mmap-backed (native) or owned (wasm).
pub struct AddressIndex {
    coords: Buffer<(f32, f32)>,
    grid: LatLonGrid,
    hn_off: Buffer<u32>,
    hn_pool: Buffer<u8>,
    street_id: Buffer<u32>,
    st_off: Buffer<u32>,
    st_pool: Buffer<u8>,
    fwd_off: Buffer<u32>,
    fwd_idx: Buffer<u32>,
    city: Option<(Buffer<u32>, Buffer<u8>)>,
    pc: Option<(Buffer<u32>, Buffer<u8>)>,
}

impl AddressIndex {
    /// Number of address points.
    pub fn len(&self) -> usize {
        self.coords.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Number of distinct streets.
    pub fn street_count(&self) -> usize {
        self.st_off.len().saturating_sub(1)
    }

    fn pool_str<'a>(off: &[u32], pool: &'a [u8], i: usize) -> Option<&'a str> {
        let (a, b) = (*off.get(i)? as usize, *off.get(i + 1)? as usize);
        std::str::from_utf8(pool.get(a..b)?).ok()
    }

    fn housenumber(&self, addr: usize) -> Option<&str> {
        Self::pool_str(self.hn_off.as_slice(), self.hn_pool.as_slice(), addr)
    }
    fn street_name(&self, street_id: usize) -> Option<&str> {
        Self::pool_str(self.st_off.as_slice(), self.st_pool.as_slice(), street_id)
    }
    fn city_of(&self, addr: usize) -> Option<String> {
        let (off, pool) = self.city.as_ref()?;
        let s = Self::pool_str(off.as_slice(), pool.as_slice(), addr)?;
        if s.is_empty() { None } else { Some(s.to_string()) }
    }
    fn pc_of(&self, addr: usize) -> Option<String> {
        let (off, pool) = self.pc.as_ref()?;
        let s = Self::pool_str(off.as_slice(), pool.as_slice(), addr)?;
        if s.is_empty() { None } else { Some(s.to_string()) }
    }

    fn hit(&self, addr: usize, approximate: bool) -> AddressHit {
        let (lat, lon) = self.coords.as_slice()[addr];
        let sid = self.street_id.as_slice()[addr] as usize;
        AddressHit {
            lat,
            lon,
            street: self.street_name(sid).unwrap_or("").to_string(),
            housenumber: self.housenumber(addr).unwrap_or("").to_string(),
            city: self.city_of(addr),
            postcode: self.pc_of(addr),
            approximate,
        }
    }

    /// Street id by name (case-insensitive; exact wins, else first substring).
    pub fn find_street_id(&self, query: &str) -> Option<u32> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return None;
        }
        let mut first_contains: Option<u32> = None;
        for id in 0..self.street_count() {
            if let Some(name) = self.street_name(id) {
                let l = name.to_lowercase();
                if l == q {
                    return Some(id as u32);
                }
                if first_contains.is_none() && l.contains(&q) {
                    first_contains = Some(id as u32);
                }
            }
        }
        first_contains
    }

    fn street_group(&self, sid: u32) -> &[u32] {
        let off = self.fwd_off.as_slice();
        match (off.get(sid as usize), off.get(sid as usize + 1)) {
            (Some(&a), Some(&b)) => &self.fwd_idx.as_slice()[a as usize..b as usize],
            _ => &[],
        }
    }

    /// Forward geocode `street` + `number`. Optional `near` disambiguates when a
    /// street name spans several towns (picks the matching/nearest candidate).
    /// Exact number → `approximate=false`; otherwise the nearest number on the
    /// street → `approximate=true`. `None` if the street doesn't resolve.
    pub fn forward(
        &self,
        street: &str,
        number: &str,
        near: Option<(f32, f32)>,
    ) -> Option<AddressHit> {
        let sid = self.find_street_id(street)?;
        let group = self.street_group(sid);
        if group.is_empty() {
            return None;
        }
        let (qn, qs) = norm_num(number);
        let mut exact: Vec<u32> = Vec::new();
        let mut best_near: Option<(i64, f32, u32)> = None; // (|num diff|, ref dist, addr)
        for &ai in group {
            let (n, s) = self.housenumber(ai as usize).map(norm_num).unwrap_or((i64::MAX, String::new()));
            if n == qn && s == qs {
                exact.push(ai);
            }
            let diff = (n - qn).abs();
            let rd = near.map_or(0.0, |r| planar2(r, self.coords.as_slice()[ai as usize]));
            if best_near.map_or(true, |(d, prd, _)| diff < d || (diff == d && rd < prd)) {
                best_near = Some((diff, rd, ai));
            }
        }
        if !exact.is_empty() {
            let pick = match near {
                Some(r) => *exact
                    .iter()
                    .min_by(|&&a, &&b| {
                        planar2(r, self.coords.as_slice()[a as usize])
                            .partial_cmp(&planar2(r, self.coords.as_slice()[b as usize]))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap(),
                None => exact[0],
            };
            return Some(self.hit(pick as usize, false));
        }
        best_near.map(|(_, _, ai)| self.hit(ai as usize, true))
    }

    /// Reverse geocode: the nearest address point to `(lat, lon)`.
    pub fn reverse(&self, lat: f32, lon: f32) -> Option<AddressHit> {
        let id = self.grid.nearest(lat, lon, self.coords.as_slice())?;
        Some(self.hit(id as usize, false))
    }

    fn from_parts(
        coords: Buffer<(f32, f32)>,
        hn_off: Buffer<u32>,
        hn_pool: Buffer<u8>,
        street_id: Buffer<u32>,
        st_off: Buffer<u32>,
        st_pool: Buffer<u8>,
        fwd_off: Buffer<u32>,
        fwd_idx: Buffer<u32>,
        city: Option<(Buffer<u32>, Buffer<u8>)>,
        pc: Option<(Buffer<u32>, Buffer<u8>)>,
    ) -> Self {
        let n = coords.len();
        let cell = if n > 1_000_000 { 0.01 } else { 0.005 };
        let grid = LatLonGrid::from_coords(coords.as_slice(), cell);
        AddressIndex {
            coords,
            grid,
            hn_off,
            hn_pool,
            street_id,
            st_off,
            st_pool,
            fwd_off,
            fwd_idx,
            city,
            pc,
        }
    }

    /// Load from an in-memory byte slice (wasm / no-mmap path).
    pub fn load_bytes(bytes: &[u8]) -> std::io::Result<AddressIndex> {
        let h = parse_header(bytes)?;
        let has_city = h.flags & FLAG_CITY != 0;
        let has_pc = h.flags & FLAG_PC != 0;
        let mut o = HEADER_BYTES;
        let take_u32 = |bytes: &[u8], o: &mut usize, n: usize| {
            let b = Buffer::<u32>::from_bytes_copy(bytes, *o, n);
            *o += 4 * n;
            b
        };
        let coords = Buffer::<(f32, f32)>::from_bytes_copy(bytes, o, h.a);
        o += 8 * h.a;
        let hn_off = take_u32(bytes, &mut o, h.a + 1);
        let street_id = take_u32(bytes, &mut o, h.a);
        let st_off = take_u32(bytes, &mut o, h.s + 1);
        let fwd_off = take_u32(bytes, &mut o, h.s + 1);
        let fwd_idx = take_u32(bytes, &mut o, h.a);
        let city_off = if has_city { Some(take_u32(bytes, &mut o, h.a + 1)) } else { None };
        let pc_off = if has_pc { Some(take_u32(bytes, &mut o, h.a + 1)) } else { None };
        let hn_pool = Buffer::<u8>::from_bytes_copy(bytes, o, h.hn_pool_len);
        o += h.hn_pool_len;
        let st_pool = Buffer::<u8>::from_bytes_copy(bytes, o, h.st_pool_len);
        o += h.st_pool_len;
        let city = city_off.map(|off| {
            let pool = Buffer::<u8>::from_bytes_copy(bytes, o, h.city_pool_len);
            o += h.city_pool_len;
            (off, pool)
        });
        let pc = pc_off.map(|off| {
            let pool = Buffer::<u8>::from_bytes_copy(bytes, o, h.pc_pool_len);
            (off, pool)
        });
        Ok(Self::from_parts(
            coords, hn_off, hn_pool, street_id, st_off, st_pool, fwd_off, fwd_idx, city, pc,
        ))
    }

    /// Load via mmap (native).
    #[cfg(feature = "native")]
    pub fn load_mmap<P: AsRef<Path>>(path: P) -> std::io::Result<AddressIndex> {
        let f = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        let h = parse_header(&mmap)?;
        let has_city = h.flags & FLAG_CITY != 0;
        let has_pc = h.flags & FLAG_PC != 0;
        let arc = Arc::new(mmap);
        let mut o = HEADER_BYTES;
        let take_u32 = |o: &mut usize, n: usize| {
            let b = Buffer::<u32>::from_mmap(arc.clone(), *o, n);
            *o += 4 * n;
            b
        };
        let coords = Buffer::<(f32, f32)>::from_mmap(arc.clone(), o, h.a);
        o += 8 * h.a;
        let hn_off = take_u32(&mut o, h.a + 1);
        let street_id = take_u32(&mut o, h.a);
        let st_off = take_u32(&mut o, h.s + 1);
        let fwd_off = take_u32(&mut o, h.s + 1);
        let fwd_idx = take_u32(&mut o, h.a);
        let city_off = if has_city { Some(take_u32(&mut o, h.a + 1)) } else { None };
        let pc_off = if has_pc { Some(take_u32(&mut o, h.a + 1)) } else { None };
        let hn_pool = Buffer::<u8>::from_mmap(arc.clone(), o, h.hn_pool_len);
        o += h.hn_pool_len;
        let st_pool = Buffer::<u8>::from_mmap(arc.clone(), o, h.st_pool_len);
        o += h.st_pool_len;
        let city = city_off.map(|off| {
            let pool = Buffer::<u8>::from_mmap(arc.clone(), o, h.city_pool_len);
            o += h.city_pool_len;
            (off, pool)
        });
        let pc = pc_off.map(|off| {
            let pool = Buffer::<u8>::from_mmap(arc.clone(), o, h.pc_pool_len);
            (off, pool)
        });
        Ok(Self::from_parts(
            coords, hn_off, hn_pool, street_id, st_off, st_pool, fwd_off, fwd_idx, city, pc,
        ))
    }
}

struct Header {
    a: usize,
    s: usize,
    hn_pool_len: usize,
    st_pool_len: usize,
    city_pool_len: usize,
    pc_pool_len: usize,
    flags: u64,
}

fn parse_header(bytes: &[u8]) -> std::io::Result<Header> {
    if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid magic — address sidecar corrupt or wrong version",
        ));
    }
    let rd = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap()) as usize;
    Ok(Header {
        a: rd(8),
        s: rd(16),
        hn_pool_len: rd(24),
        st_pool_len: rd(32),
        city_pool_len: rd(40),
        pc_pool_len: rd(48),
        flags: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
    })
}

/// Build per-record offset+pool arrays for an optional string field. Returns
/// `(off[a+1], pool)`. Missing values become empty strings.
#[cfg(feature = "native")]
fn build_pool<'a>(strings: impl Iterator<Item = &'a str>, a: usize) -> (Vec<u32>, Vec<u8>) {
    let mut off = vec![0u32; a + 1];
    let mut pool = Vec::new();
    let mut acc = 0u32;
    for (i, s) in strings.enumerate() {
        off[i] = acc;
        pool.extend_from_slice(s.as_bytes());
        acc += s.len() as u32;
    }
    off[a] = acc;
    (off, pool)
}

/// Write the address sidecar from collected records. Deduplicates by
/// (lowercased street, normalized number), keeping the first occurrence.
#[cfg(feature = "native")]
pub fn save<P: AsRef<Path>>(path: P, records: &[AddrRec]) -> std::io::Result<usize> {
    // Dedup by (street, number, city, postcode): collapses the same address
    // mapped twice (e.g. an addr node AND a building-way centroid) while keeping
    // genuinely distinct addresses that share a street+number across towns.
    let mut seen: HashSet<(String, i64, String, String, String)> = HashSet::new();
    let mut recs: Vec<&AddrRec> = Vec::with_capacity(records.len());
    for r in records {
        if r.housenumber.trim().is_empty() || r.street.trim().is_empty() {
            continue;
        }
        let (n, s) = norm_num(&r.housenumber);
        let key = (
            r.street.to_lowercase(),
            n,
            s,
            r.city.as_deref().unwrap_or("").to_lowercase(),
            r.postcode.as_deref().unwrap_or("").to_lowercase(),
        );
        if seen.insert(key) {
            recs.push(r);
        }
    }
    let a = recs.len();

    // Distinct street pool + per-address street id.
    let mut street_map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut street_pool: Vec<String> = Vec::new();
    let mut street_id = vec![0u32; a];
    for (i, r) in recs.iter().enumerate() {
        let id = *street_map.entry(r.street.clone()).or_insert_with(|| {
            let id = street_pool.len() as u32;
            street_pool.push(r.street.clone());
            id
        });
        street_id[i] = id;
    }
    let s = street_pool.len();

    // Forward CSR: addresses grouped by street, sorted by number.
    let mut by_street: Vec<Vec<u32>> = vec![Vec::new(); s];
    for (i, &sid) in street_id.iter().enumerate() {
        by_street[sid as usize].push(i as u32);
    }
    for g in by_street.iter_mut() {
        g.sort_by_key(|&ai| norm_num(&recs[ai as usize].housenumber));
    }
    let mut fwd_off = vec![0u32; s + 1];
    let mut fwd_idx = vec![0u32; a];
    let mut acc = 0u32;
    for (sid, g) in by_street.iter().enumerate() {
        fwd_off[sid] = acc;
        for &ai in g {
            fwd_idx[acc as usize] = ai;
            acc += 1;
        }
    }
    fwd_off[s] = acc;

    // Pools.
    let coords: Vec<(f32, f32)> = recs.iter().map(|r| (r.lat, r.lon)).collect();
    let (hn_off, hn_pool) = build_pool(recs.iter().map(|r| r.housenumber.as_str()), a);
    let mut st_off = vec![0u32; s + 1];
    let mut st_pool = Vec::new();
    let mut sacc = 0u32;
    for (i, name) in street_pool.iter().enumerate() {
        st_off[i] = sacc;
        st_pool.extend_from_slice(name.as_bytes());
        sacc += name.len() as u32;
    }
    st_off[s] = sacc;

    let has_city = recs.iter().any(|r| r.city.is_some());
    let has_pc = recs.iter().any(|r| r.postcode.is_some());
    let (city_off, city_pool) = if has_city {
        build_pool(recs.iter().map(|r| r.city.as_deref().unwrap_or("")), a)
    } else {
        (Vec::new(), Vec::new())
    };
    let (pc_off, pc_pool) = if has_pc {
        build_pool(recs.iter().map(|r| r.postcode.as_deref().unwrap_or("")), a)
    } else {
        (Vec::new(), Vec::new())
    };
    let mut flags = 0u64;
    if has_city {
        flags |= FLAG_CITY;
    }
    if has_pc {
        flags |= FLAG_PC;
    }

    let f = OpenOptions::new().write(true).create(true).truncate(true).open(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);
    w.write_all(MAGIC)?;
    w.write_all(&(a as u64).to_le_bytes())?;
    w.write_all(&(s as u64).to_le_bytes())?;
    w.write_all(&(hn_pool.len() as u64).to_le_bytes())?;
    w.write_all(&(st_pool.len() as u64).to_le_bytes())?;
    w.write_all(&(city_pool.len() as u64).to_le_bytes())?;
    w.write_all(&(pc_pool.len() as u64).to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;
    // Fixed-width arrays first (keeps every typed array naturally aligned)…
    w.write_all(slice_u8(&coords))?;
    w.write_all(slice_u8(&hn_off))?;
    w.write_all(slice_u8(&street_id))?;
    w.write_all(slice_u8(&st_off))?;
    w.write_all(slice_u8(&fwd_off))?;
    w.write_all(slice_u8(&fwd_idx))?;
    if has_city {
        w.write_all(slice_u8(&city_off))?;
    }
    if has_pc {
        w.write_all(slice_u8(&pc_off))?;
    }
    // …then the variable-length byte pools.
    w.write_all(&hn_pool)?;
    w.write_all(&st_pool)?;
    if has_city {
        w.write_all(&city_pool)?;
    }
    if has_pc {
        w.write_all(&pc_pool)?;
    }
    w.flush()?;
    Ok(a)
}

#[cfg(feature = "native")]
#[inline]
fn slice_u8<T>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    fn rec(lat: f32, lon: f32, hn: &str, st: &str, city: Option<&str>) -> AddrRec {
        AddrRec {
            lat,
            lon,
            housenumber: hn.to_string(),
            street: st.to_string(),
            city: city.map(|c| c.to_string()),
            postcode: None,
        }
    }

    #[test]
    fn round_trip_forward_reverse_nearest_and_multicity() {
        let recs = vec![
            rec(59.9100, 10.7400, "1", "Storgata", Some("Oslo")),
            rec(59.9110, 10.7410, "3", "Storgata", Some("Oslo")),
            rec(59.9120, 10.7420, "5", "Storgata", Some("Oslo")), // 4 missing on purpose
            rec(59.9130, 10.7430, "42B", "Karl Johans gate", Some("Oslo")),
            rec(63.4300, 10.3900, "1", "Storgata", Some("Trondheim")), // same name, other town
        ];
        let p = std::env::temp_dir().join("mpee_addr_test.addr");
        let n = save(&p, &recs).unwrap();
        assert_eq!(n, 5);
        let idx = AddressIndex::load_mmap(&p).unwrap();

        // Forward exact.
        let h = idx.forward("Storgata", "3", None).unwrap();
        assert_eq!(h.housenumber, "3");
        assert!(!h.approximate);
        assert_eq!(h.city.as_deref(), Some("Oslo"));

        // Nearest-on-street: 4 missing → returns 3 or 5, flagged approximate.
        let h = idx.forward("Storgata", "4", None).unwrap();
        assert!(h.approximate);
        assert!(h.housenumber == "3" || h.housenumber == "5");

        // Suffix number.
        let h = idx.forward("Karl Johans gate", "42B", None).unwrap();
        assert_eq!(h.housenumber, "42B");
        assert!(!h.approximate);

        // Multi-city disambiguation: "Storgata 1" near Trondheim → the Trondheim one.
        let h = idx.forward("Storgata", "1", Some((63.43, 10.39))).unwrap();
        assert_eq!(h.city.as_deref(), Some("Trondheim"));
        let h = idx.forward("Storgata", "1", Some((59.91, 10.74))).unwrap();
        assert_eq!(h.city.as_deref(), Some("Oslo"));

        // Reverse: nearest point to a coord near Storgata 5 (Oslo).
        let h = idx.reverse(59.9121, 10.7421).unwrap();
        assert_eq!(h.street, "Storgata");
        assert_eq!(h.housenumber, "5");

        let _ = std::fs::remove_file(&p);
    }
}
