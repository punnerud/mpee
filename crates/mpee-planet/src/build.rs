//! Out-of-core planet build, pass 1 and pass 2.
//!
//! The reason this is not "just parse the PBF" is memory. MPEE's existing
//! loader puts every OSM node in a `HashMap<i64,(f32,f32)>`; on a planet that
//! is ~9.6 billion entries, several hundred GB. The way around it is to never
//! key anything by OSM id:
//!
//!   **Pass 1** reads *only the way half* of the file (found by binary search
//!   over the blob directory) and records, in bitmaps over the raw id space,
//!   which nodes a drivable way touches and which of those are junctions — a
//!   node is a junction exactly when a second way touches it, which
//!   `fetch_or` reports for free. Ways are re-emitted to a compact varint
//!   stream so pass 3 never has to inflate the PBF again.
//!
//!   **Pass 2** reads only the node half and writes coordinates into a *dense*
//!   file indexed by `rank1(id)`. Nodes arrive in ascending id order, so this
//!   is a sequential fill, and every later lookup is O(1) with no hashing.
//!
//! Peak RAM is the three bitmaps (~2 GB each at the current id ceiling), not
//! the data.

use crate::bitmap::BitMap;
use crate::mmapvec;
use crate::parallel::{scan_blobs, Scatter};
use crate::pbf::{self, BlobDesc, BlobKind, Block};
use crate::profile::{self, Class, OneWay, WayAttr};
use crate::varint::*;
use std::io;
use std::path::{Path, PathBuf};

pub struct Paths {
    pub work: PathBuf,
}

impl Paths {
    pub fn new(work: &Path) -> Self {
        std::fs::create_dir_all(work).ok();
        Paths { work: work.to_path_buf() }
    }
    pub fn f(&self, name: &str) -> PathBuf {
        self.work.join(name)
    }

    /// The directory itself, for the few readers that open by directory rather
    /// than by artifact name.
    pub fn dir(&self) -> &std::path::Path {
        &self.work
    }
}

// ------------------------------------------------------------ attr packing

pub const A_CLASS: u32 = 0x0000_000F;
pub const A_ONEWAY_SH: u32 = 4;
pub const A_TOLL: u32 = 1 << 6;
pub const A_BRIDGE: u32 = 1 << 7;
pub const A_TUNNEL: u32 = 1 << 8;
pub const A_ROUNDABOUT: u32 = 1 << 9;
pub const A_KMH_SH: u32 = 16;

/// The on-disk form of one routable way.
///
/// Shared between the build and the update, and that sharing is the point: an
/// update decides whether a way has changed by hashing this record and
/// comparing it with the one stored at build time, so the two must produce
/// byte-identical output for identical input or every way looks changed.
pub fn encode_way(b: &mut Vec<u8>, id: u64, attr: u32, name: &[u8], rf: &[u8], refs: &[i64]) {
    put_u(b, id);
    put_u(b, attr as u64);
    put_bytes(b, name);
    put_bytes(b, rf);
    put_u(b, refs.len() as u64);
    let mut prev = 0i64;
    for &r in refs {
        put_i(b, r - prev);
        prev = r;
    }
}

/// A way's identity, split into the two halves that behave differently under
/// an update.
///
/// The attribute half — packed class, direction, speed, plus name and ref — is
/// what a table's numbers are computed *from*. Change it and the segments stay
/// exactly where they are; only their cost changes, and that can be patched in
/// place.
///
/// The topology half is the node list. Change it and the way splits into
/// different segments at different junctions, with different ids, which no
/// amount of recomputation can patch: the graph's shape has moved.
///
/// Keeping them apart is what tells an update which of the two it is looking
/// at. A single hash over the whole record says only "something changed", and
/// would send a corrected speed limit down the same expensive path as a new
/// motorway.
/// What `way.hash` and `way.topo` mean, on disk.
///
/// Revision 1 was a single FNV over the whole record after the id — one hash
/// that mixed a way's cost with its shape, so an update could not tell a
/// renamed street from a re-drawn one. Revision 2 is the split below: the
/// attribute half in `way.hash`, the node list in `way.topo`.
///
/// The split happened in the code before it happened on disk, and for a while
/// `contract` wrote one thing while `diff` compared another. Nothing noticed,
/// because every dataset predated the change and both sides were reading the
/// older format. A rebuild would have made every way look changed. Hence a
/// stamp: an index whose revision is not this one is refused, not reinterpreted.
pub const WAY_INDEX_REVISION: u32 = 2;

/// The file recording which revision a way index was written in.
pub fn way_fmt_file() -> &'static str {
    "way.fmt"
}

/// Write the way index's revision stamp. Last, after the index itself.
pub fn write_way_stamp(path: &std::path::Path, ways: u64) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(20);
    b.extend_from_slice(b"MPEEWAY\0");
    b.extend_from_slice(&WAY_INDEX_REVISION.to_le_bytes());
    b.extend_from_slice(&ways.to_le_bytes());
    std::fs::write(path, &b)
}

/// Check a way index is in the revision this build reads.
///
/// A missing stamp means an index written before stamps existed, which is
/// revision 1 — the one whose hash mixed cost and shape. Saying so is the whole
/// point: the alternative is a diff that reports every way as changed and an
/// update that recomputes the planet while reporting success.
pub fn check_way_stamp(dir: &std::path::Path) -> Result<(), String> {
    let p = dir.join(way_fmt_file());
    let Ok(b) = std::fs::read(&p) else {
        return Err(format!(
            "{}: no way-index stamp, so it was written before revision {} — its \
             way.hash mixes a way's cost with its node list and cannot be compared \
             against one that separates them. Re-run `contract` to rebuild the index.",
            dir.display(),
            WAY_INDEX_REVISION
        ));
    };
    if b.len() < 20 || &b[..8] != b"MPEEWAY\0" {
        return Err(format!("{}: not a way-index stamp", p.display()));
    }
    let rev = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
    if rev != WAY_INDEX_REVISION {
        return Err(format!(
            "{}: way index is revision {rev}, this build reads revision {} — \
             re-run `contract`",
            p.display(),
            WAY_INDEX_REVISION
        ));
    }
    Ok(())
}

pub fn way_hashes(rec: &[u8]) -> (u64, u64, u64) {
    fn read_u(b: &[u8], p: &mut usize) -> u64 {
        let (mut v, mut sh) = (0u64, 0u32);
        while *p < b.len() {
            let c = b[*p];
            *p += 1;
            v |= ((c & 0x7f) as u64) << sh;
            if c & 0x80 == 0 {
                break;
            }
            sh += 7;
        }
        v
    }
    fn fnv(b: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    }
    let mut p = 0usize;
    let id = read_u(rec, &mut p);
    let a0 = p;
    read_u(rec, &mut p); // attr
    let n = read_u(rec, &mut p) as usize;
    p += n; // name
    let n = read_u(rec, &mut p) as usize;
    p += n; // ref
    let a1 = p;
    (id, fnv(&rec[a0..a1]), fnv(&rec[a1..]))
}

pub fn pack_attr(a: &WayAttr) -> u32 {
    let ow = match a.oneway {
        OneWay::No => 0u32,
        OneWay::Forward => 1,
        OneWay::Backward => 2,
    };
    (a.class as u32)
        | (ow << A_ONEWAY_SH)
        | if a.toll { A_TOLL } else { 0 }
        | if a.bridge { A_BRIDGE } else { 0 }
        | if a.tunnel { A_TUNNEL } else { 0 }
        | if a.roundabout { A_ROUNDABOUT } else { 0 }
        | ((a.kmh as u32) << A_KMH_SH)
}

#[inline]
pub fn attr_class(a: u32) -> u8 {
    (a & A_CLASS) as u8
}
#[inline]
pub fn attr_oneway(a: u32) -> u8 {
    ((a >> A_ONEWAY_SH) & 3) as u8
}
#[inline]
pub fn attr_kmh(a: u32) -> u16 {
    (a >> A_KMH_SH) as u16
}

pub fn class_of(c: u8) -> Class {
    // Safe: the builder only ever writes discriminants 0..=15.
    match c {
        0 => Class::Motorway,
        1 => Class::MotorwayLink,
        2 => Class::Trunk,
        3 => Class::TrunkLink,
        4 => Class::Primary,
        5 => Class::PrimaryLink,
        6 => Class::Secondary,
        7 => Class::SecondaryLink,
        8 => Class::Tertiary,
        9 => Class::TertiaryLink,
        10 => Class::Unclassified,
        11 => Class::Residential,
        12 => Class::LivingStreet,
        13 => Class::Service,
        14 => Class::Ferry,
        _ => Class::Other,
    }
}

// --------------------------------------------------------- blob directory

pub struct Dir {
    pub blobs: Vec<BlobDesc>,
    pub first_way: usize,
    pub max_node_id: u64,
}

pub fn open_dir(pbf: &Path, paths: &Paths) -> io::Result<Dir> {
    let cache = paths.f("blobs.idx");
    // The cache is a list of byte offsets into one particular file. Reusing it
    // for a different file reads whatever happens to sit at those offsets,
    // which surfaces as a protobuf wire-type panic somewhere unrelated — so
    // the file it was built for is recorded and checked. Length is enough:
    // two PBFs of exactly the same size have the same blob boundaries only if
    // they are the same file, and if that ever failed the check below would
    // still catch it when the offsets stopped landing on blob headers.
    let stamp = cache.with_extension("idx.for");
    let want = std::fs::metadata(pbf)?.len();
    let fresh = std::fs::read(&stamp)
        .ok()
        .and_then(|b| b.get(..8).map(|x| u64::from_le_bytes(x.try_into().unwrap())))
        == Some(want);
    let blobs = if cache.exists() && fresh {
        let raw = std::fs::read(&cache)?;
        let n = raw.len() / 16;
        (0..n)
            .map(|i| {
                let o = u64::from_le_bytes(raw[i * 16..i * 16 + 8].try_into().unwrap());
                let l = u32::from_le_bytes(raw[i * 16 + 8..i * 16 + 12].try_into().unwrap());
                let k = raw[i * 16 + 12];
                BlobDesc {
                    offset: o,
                    len: l,
                    kind: if k == 0 { BlobKind::Header } else { BlobKind::Data },
                }
            })
            .collect()
    } else {
        eprintln!("[dir] indexing blobs …");
        let t = std::time::Instant::now();
        let b = pbf::index_blobs(pbf)?;
        let mut raw = Vec::with_capacity(b.len() * 16);
        for d in &b {
            raw.extend_from_slice(&d.offset.to_le_bytes());
            raw.extend_from_slice(&d.len.to_le_bytes());
            raw.push(if d.kind == BlobKind::Header { 0 } else { 1 });
            raw.extend_from_slice(&[0u8; 3]);
        }
        std::fs::write(&cache, &raw)?;
        std::fs::write(&stamp, want.to_le_bytes())?;
        eprintln!("[dir] {} blobs in {:.1} s", b.len(), t.elapsed().as_secs_f64());
        b
    };
    let first_way = pbf::find_first_way_blob(pbf, &blobs)?;
    let max_node_id = last_node_id(pbf, &blobs, first_way)?;
    eprintln!(
        "[dir] {} blobs, ways start at blob {}, max node id {}",
        blobs.len(),
        first_way,
        max_node_id
    );
    Ok(Dir { blobs, first_way, max_node_id })
}

/// Largest node id in the file. Nodes are sorted, so it lives in the last
/// node-bearing blob — one inflation instead of a full scan.
fn last_node_id(pbf: &Path, blobs: &[BlobDesc], first_way: usize) -> io::Result<u64> {
    let f = std::fs::File::open(pbf)?;
    let mut inf = pbf::Inflater::default();
    let mut i = first_way;
    while i > 0 {
        i -= 1;
        if blobs[i].kind != BlobKind::Data {
            continue;
        }
        inf.load(&f, &blobs[i])?;
        let blk = Block::parse(&inf.out);
        let mut max = 0i64;
        blk.for_each_node(|n| {
            if n.id > max {
                max = n.id;
            }
        });
        if max > 0 {
            return Ok(max as u64);
        }
    }
    Ok(0)
}

// ------------------------------------------------------------------ pass 1

pub struct P1 {
    /// Nodes touched by a drivable way (drives junction detection).
    pub road: BitMap,
    /// Nodes whose coordinates we must keep (road nodes ∪ address-way nodes).
    pub need: BitMap,
    /// Junctions and dead ends — these become graph vertices.
    pub junc: BitMap,
}

/// Pull the address tags out of a way's tag list.
struct AddrTags<'a> {
    hn: Option<&'a [u8]>,
    street: Option<&'a [u8]>,
    city: Option<&'a [u8]>,
    postcode: Option<&'a [u8]>,
    interpolation: Option<&'a [u8]>,
}

fn addr_tags<'a>(tags: &[(&'a [u8], &'a [u8])]) -> AddrTags<'a> {
    let mut a =
        AddrTags { hn: None, street: None, city: None, postcode: None, interpolation: None };
    for &(k, v) in tags {
        match k {
            b"addr:housenumber" => a.hn = Some(v),
            b"addr:street" => a.street = Some(v),
            b"addr:city" | b"addr:place" => {
                if a.city.is_none() {
                    a.city = Some(v)
                }
            }
            b"addr:postcode" => a.postcode = Some(v),
            b"addr:interpolation" => a.interpolation = Some(v),
            _ => {}
        }
    }
    a
}

pub fn pass1(pbf: &Path, paths: &Paths, dir: &Dir) -> io::Result<P1> {
    let n_bits = dir.max_node_id + 1;
    eprintln!("[pass1] bitmaps over {} ids ({:.2} GB each)", n_bits, n_bits as f64 / 8e9);
    let st = P1 {
        road: BitMap::new(n_bits),
        need: BitMap::new(n_bits),
        junc: BitMap::new(n_bits),
    };
    let outs = vec![paths.f("ways.bin"), paths.f("addrways.bin")];
    scan_blobs(
        pbf,
        &dir.blobs,
        dir.first_way..dir.blobs.len(),
        64,
        &st,
        &outs,
        "pass1/ways",
        |blk, st, bufs| {
            let mut tags: Vec<(&[u8], &[u8])> = Vec::with_capacity(24);
            let mut refs: Vec<i64> = Vec::with_capacity(64);
            blk.for_each_way(|w| {
                tags.clear();
                tags.extend(w.tags(blk));
                let road = profile::classify(tags.iter().copied(), |_| profile::NO_STR);
                let at = addr_tags(&tags);
                let is_addr_way = (at.hn.is_some() && at.street.is_some())
                    || at.interpolation.is_some();
                if road.is_none() && !is_addr_way {
                    return;
                }
                refs.clear();
                refs.extend(w.refs());
                if let Some(a) = road {
                    if refs.len() >= 2 {
                        // Endpoints are always graph vertices; interior nodes
                        // become one the moment a second way touches them.
                        for (i, &r) in refs.iter().enumerate() {
                            let id = r as u64;
                            st.need.set_quiet(id);
                            let repeat = st.road.set(id);
                            if repeat || i == 0 || i == refs.len() - 1 {
                                st.junc.set_quiet(id);
                            }
                        }
                        let name = tags.iter().find(|(k, _)| *k == b"name").map(|(_, v)| *v);
                        let rf = tags.iter().find(|(k, _)| *k == b"ref").map(|(_, v)| *v);
                        encode_way(
                            &mut bufs[0],
                            w.id as u64,
                            pack_attr(&a),
                            name.unwrap_or(b""),
                            rf.unwrap_or(b""),
                            &refs,
                        );
                    }
                }
                if is_addr_way && !refs.is_empty() {
                    for &r in refs.iter() {
                        st.need.set_quiet(r as u64);
                    }
                    let b = &mut bufs[1];
                    let kind: u8 = if at.interpolation.is_some() { 2 } else { 1 };
                    b.push(kind);
                    put_bytes(b, at.hn.unwrap_or(b""));
                    put_bytes(b, at.street.unwrap_or(b""));
                    put_bytes(b, at.city.unwrap_or(b""));
                    put_bytes(b, at.postcode.unwrap_or(b""));
                    put_bytes(b, at.interpolation.unwrap_or(b""));
                    put_u(b, refs.len() as u64);
                    let mut prev = 0i64;
                    for &r in refs.iter() {
                        put_i(b, r - prev);
                        prev = r;
                    }
                }
            });
        },
    )?;
    Ok(st)
}

// ------------------------------------------------------------------ pass 2

/// Nodes that must become graph vertices even without a second way: toll
/// gantries, barriers and the like.
fn node_is_forced_vertex(kv: &[u32], blk: &Block) -> bool {
    let mut i = 0;
    while i + 1 < kv.len() && kv[i] != 0 {
        let (k, v) = (blk.s(kv[i]), blk.s(kv[i + 1]));
        let hit = matches!(
            (k, v),
            (b"barrier", b"toll_booth")
                | (b"barrier", b"lift_gate")
                | (b"barrier", b"gate")
                | (b"highway", b"toll_gantry")
                | (b"amenity", b"toll_booth")
                | (b"barrier", b"border_control")
        );
        if hit {
            return true;
        }
        i += 2;
    }
    false
}

fn node_is_toll(kv: &[u32], blk: &Block) -> bool {
    let mut i = 0;
    while i + 1 < kv.len() && kv[i] != 0 {
        let (k, v) = (blk.s(kv[i]), blk.s(kv[i + 1]));
        if matches!(
            (k, v),
            (b"barrier", b"toll_booth") | (b"highway", b"toll_gantry") | (b"amenity", b"toll_booth")
        ) {
            return true;
        }
        i += 2;
    }
    false
}

pub struct P2 {
    pub coords_path: PathBuf,
    pub n_coords: u64,
}

/// Fill the dense coordinate table and harvest address / toll nodes.
pub fn pass2(pbf: &Path, paths: &Paths, dir: &Dir, st: &mut P1) -> io::Result<P2> {
    st.need.build_rank();
    let total = st.need.total();
    eprintln!(
        "[pass2] {} nodes needed ({:.2} GB of coordinates)",
        total,
        total as f64 * 8.0 / 1e9
    );
    let coords_path = paths.f("coords.bin");
    let mut map = mmapvec::create::<(i32, i32)>(&coords_path, total as usize)?;
    let scat: Scatter<(i32, i32)> = {
        let s = unsafe { mmapvec::as_mut_slice::<(i32, i32)>(&mut map[..]) };
        Scatter(s.as_mut_ptr(), s.len())
    };
    let shared = (&st.need, &st.junc, scat);
    let outs = vec![paths.f("addrnodes.bin"), paths.f("tollnodes.bin")];
    scan_blobs(
        pbf,
        &dir.blobs,
        0..dir.first_way,
        64,
        &shared,
        &outs,
        "pass2/nodes",
        |blk, (need, junc, scat), bufs| {
            blk.for_each_node(|n| {
                let id = n.id as u64;
                let wanted = need.get(id);
                if wanted {
                    // Unique slot per node id — disjoint across threads.
                    unsafe { scat.put(need.rank1(id) as usize, (n.lat_e7, n.lon_e7)) };
                    if node_is_forced_vertex(n.kv, blk) {
                        junc.set_quiet(id);
                    }
                }
                if node_is_toll(n.kv, blk) {
                    let b = &mut bufs[1];
                    put_u(b, id);
                    b.extend_from_slice(&n.lat_e7.to_le_bytes());
                    b.extend_from_slice(&n.lon_e7.to_le_bytes());
                    let mut i = 0;
                    let mut cnt = 0u64;
                    while i + 1 < n.kv.len() && n.kv[i] != 0 {
                        cnt += 1;
                        i += 2;
                    }
                    put_u(b, cnt);
                    let mut i = 0;
                    while i + 1 < n.kv.len() && n.kv[i] != 0 {
                        put_bytes(b, blk.s(n.kv[i]));
                        put_bytes(b, blk.s(n.kv[i + 1]));
                        i += 2;
                    }
                }
                // Address points carried by a standalone node.
                let mut hn: Option<&[u8]> = None;
                let mut street: Option<&[u8]> = None;
                let mut city: Option<&[u8]> = None;
                let mut pc: Option<&[u8]> = None;
                let mut i = 0;
                while i + 1 < n.kv.len() && n.kv[i] != 0 {
                    match blk.s(n.kv[i]) {
                        b"addr:housenumber" => hn = Some(blk.s(n.kv[i + 1])),
                        b"addr:street" => street = Some(blk.s(n.kv[i + 1])),
                        b"addr:city" | b"addr:place" => {
                            if city.is_none() {
                                city = Some(blk.s(n.kv[i + 1]))
                            }
                        }
                        b"addr:postcode" => pc = Some(blk.s(n.kv[i + 1])),
                        _ => {}
                    }
                    i += 2;
                }
                if hn.is_some() || street.is_some() {
                    let b = &mut bufs[0];
                    put_u(b, id);
                    b.extend_from_slice(&n.lat_e7.to_le_bytes());
                    b.extend_from_slice(&n.lon_e7.to_le_bytes());
                    put_bytes(b, hn.unwrap_or(b""));
                    put_bytes(b, street.unwrap_or(b""));
                    put_bytes(b, city.unwrap_or(b""));
                    put_bytes(b, pc.unwrap_or(b""));
                }
            });
        },
    )?;
    map.flush()?;
    Ok(P2 { coords_path, n_coords: total })
}

/// Fold the road bitmap into `need` so the dense coordinate index covers both
/// road nodes and address-way nodes, then report the vertex count.
pub fn finish_pass1(st: &mut P1) -> (u64, u64, u64) {
    let road = st.road.count_ones();
    let need = st.need.count_ones();
    let junc = st.junc.count_ones();
    eprintln!(
        "[pass1] road nodes {road}, coords needed {need}, junctions {junc} ({:.1} % of road nodes)",
        junc as f64 / road.max(1) as f64 * 100.0
    );
    (road, need, junc)
}
