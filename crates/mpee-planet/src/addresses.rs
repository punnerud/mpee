//! Pass 5 — offline geocoding, both directions.
//!
//! Roughly half a billion OSM objects carry `addr:*` on a planet, so the
//! layout matters as much as it does for the graph. The trick that makes it
//! cheap is that address *strings* are enormously repetitive — "42" occurs
//! millions of times, "Main Street" hundreds of thousands — while address
//! *points* are unique. So every string is interned into a global pool and
//! each address becomes five fixed-width words:
//!
//! ```text
//! coord (i32,i32) · housenumber u32 · street u32 · place u32
//! ```
//!
//! Two orderings are then laid over the same records:
//!
//!   * **spatial** (grid cell) — the storage order, which serves reverse
//!     lookup and gives page locality;
//!   * **(street, place, number)** — a permutation array, which serves forward
//!     lookup by binary search, with no text index to build or keep in RAM.

use crate::bitmap::BitMap;
use crate::build::Paths;
use crate::graph::cell_of;
use crate::mmapvec;
use crate::varint::{get_bytes, get_i, get_u};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Lowercase, trim, and collapse internal whitespace. Diacritics are kept —
/// "Bogstadveien" and "Bøgata" are different streets, and folding æøå would
/// merge Norwegian names that users expect to stay apart.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.trim().chars() {
        if c.is_whitespace() {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        for l in c.to_lowercase() {
            out.push(l);
        }
    }
    out
}

/// Leading integer of a house number, for ordering and nearest-number search.
/// "42B" → 42, "12-14" → 12, "C" → `u32::MAX` (sorts last).
pub fn hn_number(s: &str) -> u32 {
    let d: String = s.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
    d.parse().unwrap_or(u32::MAX)
}

/// A string pool with dedup, written as `pool` + `off`.
struct Pool {
    map: HashMap<Vec<u8>, u32>,
    off: Vec<u64>,
    bytes: Vec<u8>,
}

impl Pool {
    fn new() -> Pool {
        Pool { map: HashMap::with_capacity(1 << 16), off: vec![0], bytes: Vec::new() }
    }
    fn intern(&mut self, s: &[u8]) -> u32 {
        if let Some(&i) = self.map.get(s) {
            return i;
        }
        let i = (self.off.len() - 1) as u32;
        self.bytes.extend_from_slice(s);
        self.off.push(self.bytes.len() as u64);
        self.map.insert(s.to_vec(), i);
        i
    }
    fn len(&self) -> usize {
        self.off.len() - 1
    }
    fn get(&self, i: u32) -> &[u8] {
        &self.bytes[self.off[i as usize] as usize..self.off[i as usize + 1] as usize]
    }
    fn write(&self, paths: &Paths, name: &str) -> io::Result<()> {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f(&format!("{name}.pool")))?);
        w.write_all(&self.bytes)?;
        w.flush()?;
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f(&format!("{name}.off")))?);
        for &o in &self.off {
            w.write_all(&o.to_le_bytes())?;
        }
        w.flush()
    }
}

/// `repr(C)` is load-bearing: this struct is written field by field and read
/// back by reinterpreting a mapping, so the declared order has to be the
/// stored order. Rust's default representation makes no such promise.
/// A numbered endpoint of an `addr:interpolation` way: where it is, and which
/// pooled strings describe it.
type InterpPoint = (i32, i32, u32, u32, u32);

#[derive(Clone, Copy)]
#[repr(C)]
struct Rec {
    lat: i32,
    lon: i32,
    hn: u32,
    street: u32,
    place: u32,
}

pub struct AddrStats {
    pub points: u64,
    pub streets: u64,
    pub housenumbers: u64,
    pub places: u64,
    pub groups: u64,
}

pub fn pass5(paths: &Paths, need: &BitMap) -> io::Result<AddrStats> {
    let t0 = std::time::Instant::now();
    let mut streets = Pool::new();
    let mut hns = Pool::new();
    let mut cities = Pool::new();
    let mut pcs = Pool::new();
    let mut places: HashMap<(u32, u32), u32> = HashMap::new();
    let mut place_list: Vec<(u32, u32)> = Vec::new();
    // Records are spilled to a file rather than kept on the heap. This is not
    // about the 4 GB they occupy — it is about *what kind* of memory it is: a
    // heap `Vec` is anonymous and can only be reclaimed by swapping it out,
    // whereas a file-backed mapping can be dropped and re-read. During this
    // pass the pages we actually need resident are the 16.5 GB coordinate
    // table being randomly probed, and heap records were evicting it.
    let raw_path = paths.f("addr.raw");
    let mut raw_w = BufWriter::with_capacity(1 << 22, File::create(&raw_path)?);
    let mut n_recs: u64 = 0;
    macro_rules! emit {
        ($r:expr) => {{
            let r: Rec = $r;
            raw_w.write_all(&r.lat.to_le_bytes())?;
            raw_w.write_all(&r.lon.to_le_bytes())?;
            raw_w.write_all(&r.hn.to_le_bytes())?;
            raw_w.write_all(&r.street.to_le_bytes())?;
            raw_w.write_all(&r.place.to_le_bytes())?;
            n_recs += 1;
        }};
    }

    let place_id = |cities: &mut Pool,
                        pcs: &mut Pool,
                        places: &mut HashMap<(u32, u32), u32>,
                        place_list: &mut Vec<(u32, u32)>,
                        city: &[u8],
                        pc: &[u8]|
     -> u32 {
        let c = if city.is_empty() { u32::MAX } else { cities.intern(city) };
        let p = if pc.is_empty() { u32::MAX } else { pcs.intern(pc) };
        *places.entry((c, p)).or_insert_with(|| {
            place_list.push((c, p));
            (place_list.len() - 1) as u32
        })
    };

    // ---- address nodes ---------------------------------------------------
    // Mapped, not read: on a planet these two streams are 3.3 GB and 4.2 GB,
    // and they are consumed once, front to back. Reading them into `Vec`s
    // would cost 7.5 GB of resident memory that the sort stages need instead.
    let an_map = mmapvec::open(&paths.f("addrnodes.bin"))?;
    let an: &[u8] = &an_map[..];
    let mut p = 0usize;
    // Interpolation ways reference numbered nodes by id; remember those we see.
    let interp_ids = interpolation_refs(paths)?;
    let mut interp_pts: HashMap<u64, InterpPoint> = HashMap::new();
    while p < an.len() {
        let id = get_u(an, &mut p);
        let lat = i32::from_le_bytes(an[p..p + 4].try_into().unwrap());
        p += 4;
        let lon = i32::from_le_bytes(an[p..p + 4].try_into().unwrap());
        p += 4;
        let hn = get_bytes(an, &mut p);
        let st = get_bytes(an, &mut p);
        let ci = get_bytes(an, &mut p);
        let pc = get_bytes(an, &mut p);
        if st.is_empty() || hn.is_empty() {
            continue;
        }
        let s = streets.intern(st);
        let h = hns.intern(hn);
        let pl = place_id(&mut cities, &mut pcs, &mut places, &mut place_list, ci, pc);
        if interp_ids.contains(&id) {
            interp_pts.insert(id, (lat, lon, h, s, pl));
        }
        emit!(Rec { lat, lon, hn: h, street: s, place: pl });
    }
    drop(an_map);
    eprintln!(
        "[pass5] {} address nodes ({:.1} s)",
        n_recs,
        t0.elapsed().as_secs_f64()
    );

    // ---- address ways: building centroids and interpolation --------------
    let coords_map = mmapvec::open(&paths.f("coords.bin"))?;
    // Warm the coordinate table before probing it.
    //
    // Every outline node is a random probe into 16.5 GB that pass 2 wrote an
    // hour ago and that nothing has touched since, so cold it costs a disk
    // read each — measured at 8 % CPU utilisation, i.e. entirely I/O-bound.
    // Pulling the file in sequentially first turns those probes into cache
    // hits: the read runs at streaming speed, and on a 36 GB machine the
    // table then stays resident for the whole pass.
    let tw = std::time::Instant::now();
    {
        let _ = coords_map.advise(memmap2::Advice::WillNeed);
        // Touch one byte per 16 KB page so the pages are actually resident,
        // not merely hinted; `advise` is advisory and may be a no-op.
        let raw: &[u8] = &coords_map[..];
        let mut acc = 0u64;
        let mut i = 0usize;
        while i < raw.len() {
            acc = acc.wrapping_add(raw[i] as u64);
            i += 16384;
        }
        std::hint::black_box(acc);
    }
    eprintln!(
        "[pass5] coordinate table warmed ({:.2} GB in {:.1} s)",
        coords_map.len() as f64 / 1e9,
        tw.elapsed().as_secs_f64()
    );
    let coords: &[(i32, i32)] = unsafe { mmapvec::as_slice(&coords_map[..]) };
    let aw_map = mmapvec::open(&paths.f("addrways.bin"))?;
    let aw: &[u8] = &aw_map[..];
    let mut p = 0usize;
    let mut n_centroid = 0u64;
    let mut n_interp = 0u64;
    while p < aw.len() {
        let kind = aw[p];
        p += 1;
        let hn = get_bytes(aw, &mut p).to_vec();
        let st = get_bytes(aw, &mut p).to_vec();
        let ci = get_bytes(aw, &mut p).to_vec();
        let pc = get_bytes(aw, &mut p).to_vec();
        let rule = get_bytes(aw, &mut p).to_vec();
        let n = get_u(aw, &mut p) as usize;
        let mut refs = Vec::with_capacity(n);
        let mut acc = 0i64;
        for _ in 0..n {
            acc += get_i(aw, &mut p);
            refs.push(acc as u64);
        }
        if kind == 1 {
            if st.is_empty() || hn.is_empty() {
                continue;
            }
            // Centroid of the resolved outline. Each ref costs a random probe
            // into the 16.5 GB coordinate table, so sample the outline instead
            // of walking it: eight evenly spaced corners put the centre of a
            // building well inside a metre, far below the accuracy this
            // dataset promises, and it bounds the probes a 400-node factory
            // perimeter can demand.
            const MAX_OUTLINE: usize = 8;
            let stride = refs.len().div_ceil(MAX_OUTLINE).max(1);
            let (mut sla, mut slo, mut k) = (0i64, 0i64, 0i64);
            let mut last = u64::MAX;
            for &r in refs.iter().step_by(stride) {
                if r == last || !need.get(r) {
                    continue;
                }
                last = r;
                let c = coords[need.rank1(r) as usize];
                if c.0 == 0 && c.1 == 0 {
                    continue;
                }
                sla += c.0 as i64;
                slo += c.1 as i64;
                k += 1;
            }
            if k == 0 {
                continue;
            }
            let s = streets.intern(&st);
            let h = hns.intern(&hn);
            let pl = place_id(&mut cities, &mut pcs, &mut places, &mut place_list, &ci, &pc);
            emit!(Rec {
                lat: (sla / k) as i32,
                lon: (slo / k) as i32,
                hn: h,
                street: s,
                place: pl,
            });
            n_centroid += 1;
        } else {
            // addr:interpolation — synthesise the numbers between two
            // numbered endpoints, spaced along the way.
            let rule = String::from_utf8_lossy(&rule).to_lowercase();
            let step_tag: Option<u32> = rule.parse().ok();
            let pts: Vec<(u64, InterpPoint)> = refs
                .iter()
                .filter_map(|r| interp_pts.get(r).map(|v| (*r, *v)))
                .collect();
            for w in pts.windows(2) {
                let (a, b) = (w[0].1, w[1].1);
                let na = hn_number(std::str::from_utf8(hns.get(a.2)).unwrap_or(""));
                let nb = hn_number(std::str::from_utf8(hns.get(b.2)).unwrap_or(""));
                if na == u32::MAX || nb == u32::MAX || nb <= na || nb - na > 2000 {
                    continue;
                }
                let step = match rule.as_str() {
                    "all" => 1,
                    "even" | "odd" => 2,
                    _ => step_tag.filter(|&s| s > 0).unwrap_or(1),
                };
                let mut k = na;
                while k <= nb {
                    let ok = match rule.as_str() {
                        "even" => k.is_multiple_of(2),
                        "odd" => !k.is_multiple_of(2),
                        _ => true,
                    };
                    if ok && k != na && k != nb {
                        let f = (k - na) as f64 / (nb - na) as f64;
                        let h = hns.intern(k.to_string().as_bytes());
                        emit!(Rec {
                            lat: (a.0 as f64 + (b.0 - a.0) as f64 * f).round() as i32,
                            lon: (a.1 as f64 + (b.1 - a.1) as f64 * f).round() as i32,
                            hn: h,
                            street: a.3,
                            place: a.4,
                        });
                        n_interp += 1;
                    }
                    k += step;
                }
            }
        }
    }
    drop(aw_map);
    raw_w.flush()?;
    drop(raw_w);
    eprintln!(
        "[pass5] + {n_centroid} building centroids, + {n_interp} interpolated — {n_recs} total ({:.1} s)",
        t0.elapsed().as_secs_f64()
    );
    let recs_map = mmapvec::open(&raw_path)?;
    let recs: &[Rec] = unsafe { mmapvec::as_slice(&recs_map[..]) };

    // ---- spatial order ---------------------------------------------------
    let t = std::time::Instant::now();
    let mut order: Vec<(u64, u32)> = (0..n_recs as u32)
        .into_par_iter()
        .map(|i| (cell_of(recs[i as usize].lat, recs[i as usize].lon), i))
        .collect();
    order.par_sort_unstable();

    let mut w_c = BufWriter::with_capacity(1 << 22, File::create(paths.f("addr.coord"))?);
    let mut w_h = BufWriter::with_capacity(1 << 22, File::create(paths.f("addr.hn"))?);
    let mut w_s = BufWriter::with_capacity(1 << 22, File::create(paths.f("addr.street"))?);
    let mut w_p = BufWriter::with_capacity(1 << 22, File::create(paths.f("addr.place"))?);
    let mut w_ck = BufWriter::with_capacity(1 << 20, File::create(paths.f("acell.key"))?);
    let mut w_co = BufWriter::with_capacity(1 << 20, File::create(paths.f("acell.off"))?);
    // `slot` maps an original record index to its position in spatial order,
    // so the forward permutation below can be expressed in final indices.
    let mut slot: Vec<u32> = vec![0; n_recs as usize];
    let mut prev_key = u64::MAX;
    for (pos, &(key, idx)) in order.iter().enumerate() {
        let r = recs[idx as usize];
        slot[idx as usize] = pos as u32;
        w_c.write_all(&r.lat.to_le_bytes())?;
        w_c.write_all(&r.lon.to_le_bytes())?;
        w_h.write_all(&r.hn.to_le_bytes())?;
        w_s.write_all(&r.street.to_le_bytes())?;
        w_p.write_all(&r.place.to_le_bytes())?;
        if key != prev_key {
            w_ck.write_all(&key.to_le_bytes())?;
            w_co.write_all(&(pos as u32).to_le_bytes())?;
            prev_key = key;
        }
    }
    w_co.write_all(&(order.len() as u32).to_le_bytes())?;
    for w in [
        &mut w_c as &mut dyn Write,
        &mut w_h,
        &mut w_s,
        &mut w_p,
        &mut w_ck,
        &mut w_co,
    ] {
        w.flush()?;
    }
    drop(order);
    eprintln!("[pass5] spatial order written ({:.1} s)", t.elapsed().as_secs_f64());

    // ---- forward index: (street, place, number) --------------------------
    let t = std::time::Instant::now();
    let hn_num: Vec<u32> = (0..hns.len() as u32)
        .map(|i| hn_number(std::str::from_utf8(hns.get(i)).unwrap_or("")))
        .collect();
    let mut fwd: Vec<(u32, u32, u32, u32)> = (0..n_recs as u32)
        .into_par_iter()
        .map(|i| {
            let r = recs[i as usize];
            (r.street, r.place, hn_num[r.hn as usize], slot[i as usize])
        })
        .collect();
    fwd.par_sort_unstable();
    let mut w_fi = BufWriter::with_capacity(1 << 22, File::create(paths.f("fwd.idx"))?);
    let mut w_fg = BufWriter::with_capacity(1 << 22, File::create(paths.f("fwd.group"))?);
    let mut groups = 0u64;
    let mut prev = (u32::MAX, u32::MAX);
    for (pos, &(s, pl, _, idx)) in fwd.iter().enumerate() {
        if (s, pl) != prev {
            w_fg.write_all(&s.to_le_bytes())?;
            w_fg.write_all(&pl.to_le_bytes())?;
            w_fg.write_all(&(pos as u32).to_le_bytes())?;
            groups += 1;
            prev = (s, pl);
        }
        w_fi.write_all(&idx.to_le_bytes())?;
    }
    // Sentinel group so the last one has an end offset.
    w_fg.write_all(&u32::MAX.to_le_bytes())?;
    w_fg.write_all(&u32::MAX.to_le_bytes())?;
    w_fg.write_all(&(fwd.len() as u32).to_le_bytes())?;
    w_fi.flush()?;
    w_fg.flush()?;
    let n_points = n_recs;
    drop(fwd);
    drop(recs_map);
    let _ = std::fs::remove_file(&raw_path);

    // ---- pools and the street name ordering ------------------------------
    streets.write(paths, "street")?;
    hns.write(paths, "hn")?;
    {
        // Numeric value of each distinct house number, so a forward lookup
        // compares integers instead of re-parsing strings at every probe.
        let mut w = BufWriter::with_capacity(1 << 20, File::create(paths.f("hn.num"))?);
        for &n in &hn_num {
            w.write_all(&n.to_le_bytes())?;
        }
        w.flush()?;
    }
    cities.write(paths, "city")?;
    pcs.write(paths, "pc")?;
    {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f("place.tab"))?);
        for &(c, p) in &place_list {
            w.write_all(&c.to_le_bytes())?;
            w.write_all(&p.to_le_bytes())?;
        }
        w.flush()?;
    }
    {
        // Street ids ordered by normalised name — the binary-search key for
        // forward lookup. Normalising once here beats normalising 24 times
        // per query.
        let mut ids: Vec<u32> = (0..streets.len() as u32).collect();
        let norm: Vec<String> = (0..streets.len() as u32)
            .into_par_iter()
            .map(|i| normalize(&String::from_utf8_lossy(streets.get(i))))
            .collect();
        ids.par_sort_unstable_by(|&a, &b| {
            norm[a as usize].cmp(&norm[b as usize]).then(a.cmp(&b))
        });
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f("street.sorted"))?);
        for &i in &ids {
            w.write_all(&i.to_le_bytes())?;
        }
        w.flush()?;
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f("street.norm"))?);
        let mut off = BufWriter::with_capacity(1 << 22, File::create(paths.f("street.normoff"))?);
        let mut acc = 0u64;
        off.write_all(&acc.to_le_bytes())?;
        for &i in &ids {
            let s = norm[i as usize].as_bytes();
            w.write_all(s)?;
            acc += s.len() as u64;
            off.write_all(&acc.to_le_bytes())?;
        }
        w.flush()?;
        off.flush()?;
    }
    eprintln!("[pass5] forward index + pools written ({:.1} s)", t.elapsed().as_secs_f64());

    let st = AddrStats {
        points: n_points,
        streets: streets.len() as u64,
        housenumbers: hns.len() as u64,
        places: place_list.len() as u64,
        groups,
    };
    eprintln!(
        "[pass5] {} address points, {} distinct streets, {} distinct house numbers, {} places, {} (street,place) groups — {:.1} s",
        st.points, st.streets, st.housenumbers, st.places, st.groups,
        t0.elapsed().as_secs_f64()
    );
    Ok(st)
}

/// Node ids referenced by `addr:interpolation` ways, so their house numbers
/// can be captured while streaming the address nodes.
fn interpolation_refs(paths: &Paths) -> io::Result<std::collections::HashSet<u64>> {
    let map = mmapvec::open(&paths.f("addrways.bin"))?;
    let aw: &[u8] = &map[..];
    let mut out = std::collections::HashSet::new();
    let mut p = 0usize;
    while p < aw.len() {
        let kind = aw[p];
        p += 1;
        for _ in 0..5 {
            get_bytes(aw, &mut p);
        }
        let n = get_u(aw, &mut p) as usize;
        let mut acc = 0i64;
        for _ in 0..n {
            acc += get_i(aw, &mut p);
            if kind == 2 {
                out.insert(acc as u64);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalisation_keeps_norwegian_letters() {
        assert_eq!(normalize("  Karl  Johans   gate "), "karl johans gate");
        assert_eq!(normalize("Bøgata"), "bøgata");
        assert_eq!(normalize("ÅSVEIEN"), "åsveien");
        assert_ne!(normalize("Bogata"), normalize("Bøgata"));
    }

    #[test]
    fn house_numbers_order_by_their_leading_integer() {
        assert_eq!(hn_number("42"), 42);
        assert_eq!(hn_number("42B"), 42);
        assert_eq!(hn_number("12-14"), 12);
        assert_eq!(hn_number(" 7 "), 7);
        assert_eq!(hn_number("C"), u32::MAX);
    }

    #[test]
    fn pool_dedups_and_roundtrips() {
        let mut p = Pool::new();
        let a = p.intern(b"Main Street");
        let b = p.intern(b"42");
        let c = p.intern(b"Main Street");
        assert_eq!(a, c, "repeat interning returns the same id");
        assert_ne!(a, b);
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(a), b"Main Street");
        assert_eq!(p.get(b), b"42");
    }
}
