//! The finished dataset: a set of memory-mapped arrays, opened in microseconds.
//!
//! Nothing is parsed or decompressed at open time. Every file is a dense array
//! of fixed-width records, so the whole planet becomes live the moment the
//! mappings exist, and the OS pages in only what a query actually touches.

use crate::mmapvec;
use memmap2::Mmap;
use std::io;
use std::path::Path;

pub struct Dataset {
    _maps: Vec<Mmap>,
    /// CSR: out-edges of `u` are `head[u]..head[u+1]`.
    pub head: &'static [u32],
    /// Legacy full-width targets; empty on a packed dataset.
    pub to: &'static [u32],
    to_d16: &'static [i16],
    to_exc: &'static [(u32, u32)],
    /// Segment id, with bit 31 set when the edge runs against the geometry.
    pub eseg: &'static [u32],
    /// Transposed CSR, for the backward half of a bidirectional search.
    pub rhead: &'static [u32],
    pub rto: &'static [u32],
    rto_d16: &'static [i16],
    rto_exc: &'static [(u32, u32)],
    pub rseg: &'static [u32],
    pub vcoord: &'static [(i32, i32)],
    /// Each vertex as a point in space, metres from the Earth's centre on a
    /// sphere of the polar radius.
    ///
    /// Optional, and — measured — not worth generating.
    ///
    /// It was built to take the trigonometry out of the A* potential, which on
    /// the planet ran once per settled vertex, 200 000 times a query. It does
    /// that. It buys nothing: Lisboa-Warszawa runs in 797 ms with this table
    /// and 799 ms deriving the same numbers from `vcoord`, for 3.24 GB.
    ///
    /// What actually cost the time was recomputing the *target's* position
    /// inside the loop — a value constant for the whole query, evaluated four
    /// times per settled vertex. Hoisting it out took A* from 37 % slower than
    /// Dijkstra to 12 % faster. The table is kept because the fallback costs
    /// one perfectly-predicted branch and the artifact is optional, not because
    /// it earns its size.
    pub vxyz: &'static [[f32; 3]],
    pub seg_len: &'static [u32],
    /// Legacy full-width attribute words. Kept so a dataset built before the
    /// dictionary still opens; `attr()` prefers the packed form.
    pub seg_attr: &'static [u32],
    seg_attr16: &'static [u16],
    attr_dict: &'static [u32],
    pub seg_name: &'static [u32],
    pub seg_u: &'static [u32],
    pub seg_v: &'static [u32],
    pub geom_pts: &'static [(i32, i32)],
    pub geom_off: &'static [u32],
    pub snap_cell: &'static [u64],
    pub snap_off: &'static [u32],
    pub snap_seg: &'static [u32],
    /// Coarse overview index (1° cells, major roads only) for continental zoom.
    pub big_cell: &'static [u64],
    pub big_off: &'static [u32],
    pub big_seg: &'static [u32],
    pub name_pool: &'static [u8],
    pub name_off: &'static [u64],
}

/// Map `name` and reinterpret it as `&[T]`.
///
/// The returned slice is tied to a mapping that lives in `keep` for as long as
/// the `Dataset` does; moving the `Mmap` value never moves the mapping itself,
/// so the pointer stays valid.
unsafe fn map_as<T>(dir: &Path, name: &str, keep: &mut Vec<Mmap>) -> io::Result<&'static [T]> {
    let m = mmapvec::open(&dir.join(name))?;
    let s: &[T] = mmapvec::as_slice(&m[..]);
    let out = std::slice::from_raw_parts(s.as_ptr(), s.len());
    keep.push(m);
    Ok(out)
}

impl Dataset {
    /// The mappings behind every array, for putting the dataset under a
    /// resident-cache budget. See [`crate::cachecap`].
    pub fn maps(&self) -> &[memmap2::Mmap] {
        &self._maps
    }

    pub fn open(dir: &Path) -> io::Result<Dataset> {
        let mut keep = Vec::new();
        unsafe {
            Ok(Dataset {
                head: map_as(dir, "csr.head", &mut keep)?,
                to: map_as(dir, "csr.to", &mut keep).unwrap_or(&[]),
                to_d16: map_as(dir, "to.d16", &mut keep).unwrap_or(&[]),
                to_exc: map_as(dir, "to.exc", &mut keep).unwrap_or(&[]),
                eseg: map_as(dir, "csr.seg", &mut keep)?,
                rhead: map_as(dir, "csr.rhead", &mut keep)?,
                rto: map_as(dir, "csr.rto", &mut keep).unwrap_or(&[]),
                rto_d16: map_as(dir, "rto.d16", &mut keep).unwrap_or(&[]),
                rto_exc: map_as(dir, "rto.exc", &mut keep).unwrap_or(&[]),
                rseg: map_as(dir, "csr.rseg", &mut keep)?,
                vcoord: map_as(dir, "vcoord.bin", &mut keep)?,
                vxyz: map_as(dir, "vxyz.bin", &mut keep).unwrap_or(&[]),
                seg_len: map_as(dir, "seg.len", &mut keep)?,
                seg_attr: map_as(dir, "seg.attr", &mut keep).unwrap_or(&[]),
                seg_attr16: map_as(dir, "seg.attr16", &mut keep).unwrap_or(&[]),
                attr_dict: map_as(dir, "attr.dict", &mut keep).unwrap_or(&[]),
                seg_name: map_as(dir, "seg.name", &mut keep)?,
                seg_u: map_as(dir, "seg.u", &mut keep)?,
                seg_v: map_as(dir, "seg.v", &mut keep)?,
                geom_pts: map_as(dir, "geom.pts", &mut keep)?,
                geom_off: map_as(dir, "geom.off", &mut keep)?,
                snap_cell: map_as(dir, "snap.cell", &mut keep)?,
                snap_off: map_as(dir, "snap.off", &mut keep)?,
                snap_seg: map_as(dir, "snap.seg", &mut keep)?,
                big_cell: map_as(dir, "big.cell", &mut keep)?,
                big_off: map_as(dir, "big.off", &mut keep)?,
                big_seg: map_as(dir, "big.seg", &mut keep)?,
                name_pool: map_as(dir, "name.pool", &mut keep)?,
                name_off: map_as(dir, "name.off", &mut keep)?,
                _maps: keep,
            })
        }
    }

    /// Target of forward edge `k`, whose source is `u`.
    ///
    /// The delta is stored, not the target: Hilbert renumbering makes it fit
    /// in 16 bits for 99.77 % of edges, which halves the largest array in the
    /// dataset. Decoding is one add; the rest take a binary search through a
    /// list small enough to stay resident.
    #[inline(always)]
    pub fn target(&self, u: u32, k: usize) -> u32 {
        Self::unpack(self.to_d16, self.to_exc, self.to, u, k)
    }

    /// Target of backward edge `k` — i.e. the vertex that reaches `u`.
    #[inline(always)]
    pub fn rtarget(&self, u: u32, k: usize) -> u32 {
        Self::unpack(self.rto_d16, self.rto_exc, self.rto, u, k)
    }

    #[inline(always)]
    fn unpack(d16: &[i16], exc: &[(u32, u32)], raw: &[u32], u: u32, k: usize) -> u32 {
        if d16.is_empty() {
            return raw[k]; // dataset built before packing
        }
        let d = d16[k];
        if d != i16::MIN {
            (u as i64 + d as i64) as u32
        } else {
            match exc.binary_search_by_key(&(k as u32), |e| e.0) {
                Ok(i) => exc[i].1,
                // Cannot happen on a consistent dataset; returning the source
                // keeps the search well-formed rather than panicking in a
                // server thread.
                Err(_) => u,
            }
        }
    }

    /// Packed attribute word for a segment.
    ///
    /// Half the bytes of the raw array, and the table it indexes is a few
    /// kilobytes — so on a machine that cannot cache the dataset this is one
    /// fewer page to fault in per edge, at the cost of an L1 hit.
    #[inline(always)]
    pub fn attr(&self, sid: usize) -> u32 {
        if self.attr_dict.is_empty() {
            self.seg_attr[sid]
        } else {
            self.attr_dict[self.seg_attr16[sid] as usize]
        }
    }

    #[inline]
    /// The fastest speed any segment claims, in km/h.
    ///
    /// A geometric lower bound on remaining travel time divides by this, so it
    /// has to be the true maximum or the bound stops being a bound. Read from
    /// the attribute dictionary, which is a few thousand entries.
    pub fn max_kmh(&self) -> u16 {
        let it: Box<dyn Iterator<Item = u32> + '_> = if self.attr_dict.is_empty() {
            Box::new(self.seg_attr.iter().copied())
        } else {
            Box::new(self.attr_dict.iter().copied())
        };
        it.map(crate::build::attr_kmh).max().unwrap_or(1).max(1)
    }

    pub fn n_vertices(&self) -> usize {
        self.head.len().saturating_sub(1)
    }
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.to.len()
    }
    #[inline]
    pub fn n_segments(&self) -> usize {
        self.seg_len.len()
    }

    /// Full drawn polyline of a segment, from `seg_u` to `seg_v`.
    pub fn seg_geometry(&self, sid: usize) -> Vec<(i32, i32)> {
        let mut v = Vec::new();
        v.push(self.vcoord[self.seg_u[sid] as usize]);
        v.extend_from_slice(
            &self.geom_pts[self.geom_off[sid] as usize..self.geom_off[sid + 1] as usize],
        );
        v.push(self.vcoord[self.seg_v[sid] as usize]);
        v
    }

    pub fn street_name(&self, sid: usize) -> Option<&str> {
        let n = *self.seg_name.get(sid)? as usize;
        if n == u32::MAX as usize {
            return None;
        }
        let (a, b) = (self.name_off[n] as usize, self.name_off[n + 1] as usize);
        std::str::from_utf8(&self.name_pool[a..b]).ok()
    }

    /// Segment ids registered in one snap-grid cell.
    pub fn cell_segments(&self, key: u64) -> &[u32] {
        match self.snap_cell.binary_search(&key) {
            Ok(i) => &self.snap_seg[self.snap_off[i] as usize..self.snap_off[i + 1] as usize],
            Err(_) => &[],
        }
    }

    /// Major-road segments in one 1° overview cell.
    pub fn big_cell_segments(&self, key: u64) -> &[u32] {
        match self.big_cell.binary_search(&key) {
            Ok(i) => &self.big_seg[self.big_off[i] as usize..self.big_off[i + 1] as usize],
            Err(_) => &[],
        }
    }
}

/// Files a running server maps. Everything else in a work directory is a
/// build intermediate and can be deleted once the build finishes.
pub const RUNTIME_ARTIFACTS: &[&str] = &[
    "csr.head", "csr.to", "to.d16", "to.exc",
    "csr.seg", "csr.rhead", "csr.rto", "rto.d16", "rto.exc", "csr.rseg",
    "vcoord.bin", "seg.len", "seg.attr", "seg.attr16", "attr.dict",
    "seg.name", "seg.u", "seg.v",
    "geom.pts", "geom.off", "snap.cell", "snap.off", "snap.seg",
    "big.cell", "big.off", "big.seg", "name.pool", "name.off",
    "addr.coord", "addr.hn", "addr.street", "addr.place",
    "acell.key", "acell.off", "fwd.idx", "fwd.group", "hn.num",
    "street.pool", "street.off", "street.sorted", "street.norm",
    "street.normoff", "city.pool", "city.off", "pc.pool", "pc.off",
    "hn.pool", "hn.off", "place.tab", "toll.vertex", "toll.meta",
    "toll.tariff.tsv", "toll.mpedb",
    // The way index: which OSM way each segment came from, and what that way
    // looked like. Only an update needs it, so it is optional.
    "way.id", "way.hash", "way.topo", "way.head", "way.seg",
    // Which revision those hashes are in. Optional only so a dataset built
    // before stamps existed still verifies; an update refuses without it.
    "way.fmt",
    // Cartesian vertex positions. Derivable from `vcoord`, so optional; only
    // the A* potential reads it, and only to avoid trigonometry.
    "vxyz.bin",
    // The overlay is optional: a dataset routes correctly without it, just
    // by touching more pages.
    "cell.of", "cell.bnd", "cell.bhead", "ov.head", "ov.mat", "ov.width", "ov.bounds",
];

/// The six arrays one rung of the overlay ladder above level 0 lives in.
const LEVEL_SUFFIXES: &[&str] = &[
    ".of", ".bnd", ".bhead", ".head", ".mat", ".width", ".vhead", ".vlist", ".lazy", ".have",
    ".use",
];

/// What every rung needs, whichever way it stores its values.
const LEVEL_SHAPE: &[&str] = &[".of", ".bnd", ".bhead", ".head"];
/// A rung has its values one way or the other.
const LEVEL_EAGER: &[&str] = &[".mat", ".width"];
const LEVEL_LAZY: &[&str] = &[".vhead", ".vlist", ".lazy", ".have"];

/// Rungs above level 0 are named `l2.*`, `l3.*` and so on. They are optional —
/// without them a query still answers exactly, it just walks every boundary
/// vertex on the way — so they are matched by shape rather than listed.
pub fn is_level_artifact(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('l') else { return false };
    let Some((n, suffix)) = rest.split_once('.') else { return false };
    n.parse::<u32>().map(|n| (2..=16).contains(&n)).unwrap_or(false)
        && LEVEL_SUFFIXES.iter().any(|s| *s == format!(".{suffix}"))
}

pub fn is_runtime_artifact(name: &str) -> bool {
    RUNTIME_ARTIFACTS.contains(&name) || is_level_artifact(name)
}

/// Bytes a file actually occupies. The tariff database is sparse — it reserves
/// 512 MB and fills 14 — so apparent length would overstate the dataset.
fn allocated(p: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(p).map(|m| (m.blocks() * 512).min(m.len().max(m.blocks() * 512))).unwrap_or(0)
}

/// Counts a dataset implies about itself.
///
/// Every array is a dense table of fixed-width records, so the sizes are not
/// metadata to be trusted — they are arithmetic. That is what lets a dataset
/// built before the catalogue existed be adopted with *measured* statistics
/// rather than numbers copied out of a log.
pub struct Counts {
    pub vertices: u64,
    pub edges: u64,
    pub segments: u64,
    pub addresses: u64,
    pub geometry_points: u64,
    pub street_names: u64,
}

pub fn counts(dir: &Path) -> Option<Counts> {
    let g = |f: &str| len_of(dir, f).unwrap_or(0);
    let head = g("csr.head") / 4;
    if head == 0 {
        return None;
    }
    Some(Counts {
        vertices: head - 1,
        edges: g("csr.to") / 4,
        segments: g("seg.len") / 4,
        addresses: g("addr.coord") / 8,
        geometry_points: g("geom.pts") / 8,
        street_names: (g("name.off") / 8).saturating_sub(1),
    })
}

/// A structural problem found by [`verify`].
pub struct Problem {
    pub file: String,
    pub detail: String,
}

fn len_of(dir: &Path, f: &str) -> Option<u64> {
    std::fs::metadata(dir.join(f)).ok().map(|m| m.len())
}

/// Check that a dataset is internally consistent.
///
/// Every array in the dataset is a dense table of fixed-width records, and
/// their lengths are related: `csr.to` must hold exactly one `u32` per edge,
/// `geom.off` one per segment plus a terminator, `addr.coord` one coordinate
/// pair per address. That makes the dataset *self-describing* — a build torn
/// halfway through a pass produces lengths that cannot all be true at once,
/// and this finds it without consulting the catalogue or rereading a byte of
/// content.
pub fn verify(dir: &Path) -> Vec<Problem> {
    let mut out: Vec<Problem> = Vec::new();
    fn problem(out: &mut Vec<Problem>, file: &str, detail: String) {
        out.push(Problem { file: file.into(), detail });
    }

    // Some arrays exist in one of two forms — a packed one and the full-width
    // one it replaced — and a dataset carries whichever its build produced.
    // Requiring both would reject every older dataset; requiring neither would
    // let a half-built one pass. So each pair is checked as "at least one".
    const EITHER: &[(&str, &str)] = &[
        ("csr.to", "to.d16"),
        ("csr.rto", "rto.d16"),
        ("seg.attr", "seg.attr16"),
    ];
    let in_pair = |f: &str| EITHER.iter().any(|(a, b)| *a == f || *b == f);
    for f in RUNTIME_ARTIFACTS {
        // The toll layer is genuinely optional; `to.exc` may legitimately be
        // empty (no edge needed one) and an empty file is indistinguishable
        // from a missing one for our purposes.
        // `attr.dict` belongs to the dictionary form and is required only
        // when that form is present.
        let optional = f.starts_with("toll.")
            || f.starts_with("way.")
            // Derived entirely from `vcoord.bin`, so a dataset without it is
            // complete — the A* potential falls back to the trigonometry.
            || *f == "vxyz.bin"
            || *f == "way.fmt"
            || f.starts_with("cell.")
            || f.starts_with("ov.")
            || is_level_artifact(f)
            || f.ends_with(".exc")
            || in_pair(f)
            || (*f == "attr.dict" && len_of(dir, "seg.attr16").is_none());
        if len_of(dir, f).is_none() && !optional {
            problem(&mut out, f, "missing".into());
        }
    }
    // Each rung above level 0 is optional, but only as a whole, and the
    // ladder must have no gaps. A partial rung would open — the reader
    // tolerates a whole level being absent — and then index a table that is
    // not there; a gap would let a query climb to a level it cannot descend.
    // A rung stores its values either eagerly or lazily, so "whole" means the
    // shape plus one complete set of values.
    {
        let mut ended = false;
        for n in 2..=16u32 {
            let has = |sfx: &[&str]| -> (usize, Vec<String>) {
                let f: Vec<String> = sfx.iter().map(|x| format!("l{n}{x}")).collect();
                let missing: Vec<String> =
                    f.iter().filter(|x| len_of(dir, x).is_none()).cloned().collect();
                (f.len() - missing.len(), missing)
            };
            let (shape_n, shape_missing) = has(LEVEL_SHAPE);
            let (eager_n, _) = has(LEVEL_EAGER);
            let (lazy_n, _) = has(LEVEL_LAZY);
            if shape_n == 0 && eager_n == 0 && lazy_n == 0 {
                ended = true;
                continue;
            }
            if ended {
                problem(
                    &mut out,
                    &format!("l{n}.of"),
                    format!(
                        "level {n} is present but a level below it is not — a gap \
                         lets a query climb to a level it cannot descend"
                    ),
                );
            }
            for f in shape_missing {
                problem(&mut out, &f, format!("missing from level {n}'s shape"));
            }
            let eager = eager_n == LEVEL_EAGER.len();
            let lazy = lazy_n == LEVEL_LAZY.len();
            if !eager && !lazy {
                problem(
                    &mut out,
                    &format!("l{n}.mat"),
                    format!(
                        "level {n} has neither a complete eager table ({eager_n} of {}) \
                         nor a complete lazy one ({lazy_n} of {}) — a partial level \
                         would index a table that is not there",
                        LEVEL_EAGER.len(),
                        LEVEL_LAZY.len()
                    ),
                );
            }
        }
    }
    for (raw, packed) in EITHER {
        if len_of(dir, raw).is_none() && len_of(dir, packed).is_none() {
            problem(
                &mut out,
                packed,
                format!("missing, and so is {raw} — the dataset has neither form"),
            );
        }
    }
    if !out.is_empty() {
        return out;
    }

    let g = |f: &str| len_of(dir, f).unwrap_or(0);
    // Vertices and edges are implied by the CSR header arrays.
    let v = g("csr.head") / 4;
    if v == 0 {
        problem(&mut out, "csr.head", "empty".into());
        return out;
    }
    let v = v - 1; // head has n+1 entries
    // Edge count: from the packed deltas when present, else the raw array.
    let e = if len_of(dir, "to.d16").is_some() { g("to.d16") / 2 } else { g("csr.to") / 4 };
    let s = g("seg.len") / 4;
    let a = g("addr.coord") / 8;

    let mut expect: Vec<(&str, u64)> = vec![
        ("csr.rhead", (v + 1) * 4),
        ("vcoord.bin", v * 8),
        ("csr.seg", e * 4),
        ("csr.rseg", e * 4),
        ("geom.off", (s + 1) * 4),
    ];
    if len_of(dir, "to.d16").is_some() {
        expect.push(("to.d16", e * 2));
    }
    for f in ["csr.to", "csr.rto"] {
        if len_of(dir, f).is_some() {
            expect.push((f, e * 4));
        }
    }
    expect.extend(["seg.name", "seg.u", "seg.v"].map(|f| (f, s * 4)));
    if len_of(dir, "seg.attr").is_some() {
        expect.push(("seg.attr", s * 4));
    }
    // The dictionary index is one u16 per segment. Its absence is allowed —
    // datasets built before it exists still open — but a wrong length is not.
    if len_of(dir, "seg.attr16").is_some() {
        expect.push(("seg.attr16", s * 2));
        if len_of(dir, "attr.dict").is_none() {
            problem(&mut out, "attr.dict", "missing, but seg.attr16 indexes it".into());
        }
    }
    if len_of(dir, "rto.d16").is_some() {
        expect.push(("rto.d16", e * 2));
    }
    expect.extend(["addr.hn", "addr.street", "addr.place", "fwd.idx"].map(|f| (f, a * 4)));
    for (f, want) in expect {
        let got = g(f);
        if got != want {
            problem(
                &mut out,
                f,
                format!("is {got} bytes, but the rest of the dataset implies {want}"),
            );
        }
    }

    // Cross-checks that need one value read rather than a length.
    if let Some(last) = tail_u32(dir, "geom.off") {
        let want = last as u64 * 8;
        if g("geom.pts") != want {
            out.push(Problem {
                file: "geom.pts".into(),
                detail: format!(
                    "holds {} points but geom.off ends at {last} — the geometry pass did not finish",
                    g("geom.pts") / 8
                ),
            });
        }
    }
    // CSR-over-cells indexes: one more offset than keys.
    for (k, o) in [("snap.cell", "snap.off"), ("big.cell", "big.off"), ("acell.key", "acell.off")] {
        let (keys, offs) = (g(k) / 8, g(o) / 4);
        if offs != keys + 1 {
            out.push(Problem {
                file: o.into(),
                detail: format!("{offs} offsets for {keys} cells (expected {})", keys + 1),
            });
        }
    }
    // String pools: the last offset is the pool length.
    for (off, pool) in [
        ("name.off", "name.pool"),
        ("street.off", "street.pool"),
        ("hn.off", "hn.pool"),
        ("city.off", "city.pool"),
        ("pc.off", "pc.pool"),
        ("street.normoff", "street.norm"),
    ] {
        if let Some(last) = tail_u64(dir, off) {
            if g(pool) != last {
                out.push(Problem {
                    file: pool.into(),
                    detail: format!("is {} bytes but {off} ends at {last}", g(pool)),
                });
            }
        }
    }
    out
}

fn tail_u32(dir: &Path, f: &str) -> Option<u32> {
    use std::io::{Read, Seek, SeekFrom};
    let mut fh = std::fs::File::open(dir.join(f)).ok()?;
    fh.seek(SeekFrom::End(-4)).ok()?;
    let mut b = [0u8; 4];
    fh.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn tail_u64(dir: &Path, f: &str) -> Option<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut fh = std::fs::File::open(dir.join(f)).ok()?;
    fh.seek(SeekFrom::End(-8)).ok()?;
    let mut b = [0u8; 8];
    fh.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

/// Size breakdown of a built dataset, split into what a query needs and what
/// is only build scaffolding.
pub fn report_sizes(dir: &Path) -> std::io::Result<()> {
    const RUNTIME: &[&str] = RUNTIME_ARTIFACTS;
    let group = |f: &str| -> &'static str {
        if f.starts_with("csr.") || f == "vcoord.bin" {
            "graph"
        } else if f.starts_with("seg.") {
            "segments"
        } else if f.starts_with("geom.") {
            "geometry (drawn line)"
        } else if f.starts_with("snap.") || f.starts_with("big.") {
            "spatial index"
        } else if f.starts_with("name.") {
            "street names"
        } else if f.starts_with("toll.") {
            "toll layer (mpedb)"
        } else if f.starts_with("ov.") || f.starts_with("cell.") {
            "overlay (regions)"
        } else {
            "addresses + geocoding"
        }
    };
    let allocated = |p: std::path::PathBuf| -> u64 { allocated(&p) };
    let mut by_group: std::collections::BTreeMap<&str, u64> = Default::default();
    let mut runtime_total = 0u64;
    let mut rows: Vec<(String, u64)> = Vec::new();
    for f in RUNTIME {
        let n = allocated(dir.join(f));
        runtime_total += n;
        *by_group.entry(group(f)).or_default() += n;
        rows.push((f.to_string(), n));
    }
    let mut scratch = 0u64;
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        if !RUNTIME.contains(&name.as_str()) {
            scratch += allocated(e.path());
        }
    }
    let gb = |v: u64| v as f64 / 1e9;
    println!("Runtime dataset — what an offline install actually needs\n");
    for (g, n) in &by_group {
        println!("  {:<26} {:>8.2} GB", g, gb(*n));
    }
    println!("  {:<26} {:>8.2} GB", "TOTAL", gb(runtime_total));
    println!("\n  build scratch (deletable) {:>8.2} GB", gb(scratch));
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    println!("\nLargest files:");
    for (f, n) in rows.iter().take(12) {
        println!("  {:<20} {:>8.2} GB", f, gb(*n));
    }
    Ok(())
}
