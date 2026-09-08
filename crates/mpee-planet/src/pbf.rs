//! Minimal, fast OSM PBF reader built for planet-scale streaming.
//!
//! Why not the `osmpbf` crate: at 94 GB we need three things it does not give
//! us cheaply — (1) a *blob directory* so a pass can skip the node half of the
//! file without inflating it, (2) `pread`-based parallelism so every core
//! inflates its own blob with no shared cursor, and (3) allocation-free way
//! iteration (a `Vec<i64>` per way is 1.6 billion allocations on the planet).
//!
//! The format is small enough to parse directly:
//!
//! ```text
//! repeat:  [u32 BE: len(BlobHeader)] [BlobHeader] [Blob: BlobHeader.datasize bytes]
//! BlobHeader { 1: string type, 3: int32 datasize }
//! Blob       { 1: bytes raw, 2: int32 raw_size, 3: bytes zlib_data }
//! ```
//!
//! `tests/pbf_conformance.rs` cross-checks every number this module produces
//! against the `osmpbf` crate on a real extract, so the hand-rolled protobuf
//! is held to a reference implementation rather than to inspection.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::Path;

// ---------------------------------------------------------------- protobuf

#[inline(always)]
pub fn varint(b: &[u8], p: &mut usize) -> u64 {
    let mut x = 0u64;
    let mut s = 0u32;
    loop {
        let c = b[*p];
        *p += 1;
        x |= ((c & 0x7f) as u64) << s;
        if c < 0x80 {
            return x;
        }
        s += 7;
        if s >= 64 {
            return x;
        }
    }
}

#[inline(always)]
pub fn zigzag(x: u64) -> i64 {
    ((x >> 1) as i64) ^ -((x & 1) as i64)
}

#[inline(always)]
fn skip_field(b: &[u8], p: &mut usize, wire: u64) {
    match wire {
        0 => {
            varint(b, p);
        }
        1 => *p += 8,
        2 => {
            let l = varint(b, p) as usize;
            *p += l;
        }
        5 => *p += 4,
        _ => panic!("unsupported protobuf wire type {wire}"),
    }
}

/// Iterator over a packed repeated varint field.
pub struct PackedVarint<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Iterator for PackedVarint<'a> {
    type Item = u64;
    #[inline(always)]
    fn next(&mut self) -> Option<u64> {
        if self.p >= self.b.len() {
            return None;
        }
        Some(varint(self.b, &mut self.p))
    }
}
impl<'a> PackedVarint<'a> {
    #[inline]
    pub fn new(b: &'a [u8]) -> Self {
        PackedVarint { b, p: 0 }
    }
}

/// Iterator over a packed repeated *delta-coded sint64* field (ids, refs, coords).
pub struct DeltaSint<'a> {
    inner: PackedVarint<'a>,
    acc: i64,
}
impl<'a> Iterator for DeltaSint<'a> {
    type Item = i64;
    #[inline(always)]
    fn next(&mut self) -> Option<i64> {
        let v = self.inner.next()?;
        self.acc += zigzag(v);
        Some(self.acc)
    }
}
impl<'a> DeltaSint<'a> {
    #[inline]
    pub fn new(b: &'a [u8]) -> Self {
        DeltaSint { inner: PackedVarint::new(b), acc: 0 }
    }
}

// ------------------------------------------------------------- blob index

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlobKind {
    Header,
    Data,
}

#[derive(Clone, Copy, Debug)]
pub struct BlobDesc {
    /// Byte offset of the Blob payload itself (past the BlobHeader).
    pub offset: u64,
    pub len: u32,
    pub kind: BlobKind,
}

/// Walk the file reading only BlobHeaders, seeking over every payload.
/// No decompression, so this is I/O-bound and finishes a planet in seconds.
pub fn index_blobs(path: &Path) -> io::Result<Vec<BlobDesc>> {
    let f = File::open(path)?;
    let total = f.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut out = Vec::with_capacity(1 << 20);
    let mut pos: u64 = 0;
    let mut hdr = vec![0u8; 64 * 1024];
    loop {
        if pos >= total {
            break;
        }
        let mut len4 = [0u8; 4];
        match r.read_exact(&mut len4) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let hlen = u32::from_be_bytes(len4) as usize;
        if hlen == 0 || hlen > hdr.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("implausible BlobHeader length {hlen} at offset {pos}"),
            ));
        }
        r.read_exact(&mut hdr[..hlen])?;
        let (kind, datasize) = parse_blob_header(&hdr[..hlen])?;
        let payload_off = pos + 4 + hlen as u64;
        out.push(BlobDesc { offset: payload_off, len: datasize, kind });
        r.seek_relative(datasize as i64)?;
        pos = payload_off + datasize as u64;
    }
    Ok(out)
}

fn parse_blob_header(b: &[u8]) -> io::Result<(BlobKind, u32)> {
    let mut p = 0usize;
    let mut kind = BlobKind::Data;
    let mut datasize = 0u32;
    while p < b.len() {
        let k = varint(b, &mut p);
        match (k >> 3, k & 7) {
            (1, 2) => {
                let l = varint(b, &mut p) as usize;
                kind = if &b[p..p + l] == b"OSMHeader" { BlobKind::Header } else { BlobKind::Data };
                p += l;
            }
            (3, 0) => datasize = varint(b, &mut p) as u32,
            (_, w) => skip_field(b, &mut p, w),
        }
    }
    if datasize == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "BlobHeader without datasize"));
    }
    Ok((kind, datasize))
}

/// Per-thread scratch so a parallel scan allocates once, not once per blob.
pub struct Inflater {
    dec: libdeflater::Decompressor,
    pub raw: Vec<u8>,
    pub out: Vec<u8>,
}

impl Default for Inflater {
    fn default() -> Self {
        Inflater {
            dec: libdeflater::Decompressor::new(),
            raw: Vec::with_capacity(1 << 21),
            out: Vec::with_capacity(1 << 25),
        }
    }
}

impl Inflater {
    /// Read one blob via `pread` (no shared cursor) and inflate into `self.out`.
    pub fn load(&mut self, f: &File, d: &BlobDesc) -> io::Result<()> {
        self.raw.resize(d.len as usize, 0);
        f.read_exact_at(&mut self.raw, d.offset)?;
        // Blob { 1: bytes raw, 2: int32 raw_size, 3: bytes zlib_data }
        let b = std::mem::take(&mut self.raw);
        let mut p = 0usize;
        let mut raw_size = 0usize;
        let mut zlib: Option<(usize, usize)> = None;
        let mut plain: Option<(usize, usize)> = None;
        while p < b.len() {
            let k = varint(&b, &mut p);
            match (k >> 3, k & 7) {
                (1, 2) => {
                    let l = varint(&b, &mut p) as usize;
                    plain = Some((p, l));
                    p += l;
                }
                (2, 0) => raw_size = varint(&b, &mut p) as usize,
                (3, 2) => {
                    let l = varint(&b, &mut p) as usize;
                    zlib = Some((p, l));
                    p += l;
                }
                (_, w) => skip_field(&b, &mut p, w),
            }
        }
        let r = if let Some((o, l)) = zlib {
            self.out.clear();
            self.out.resize(raw_size, 0);
            self.dec
                .zlib_decompress(&b[o..o + l], &mut self.out)
                .map(|n| {
                    self.out.truncate(n);
                })
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("zlib: {e:?}")))
        } else if let Some((o, l)) = plain {
            self.out.clear();
            self.out.extend_from_slice(&b[o..o + l]);
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "blob uses an unsupported compression (only raw and zlib are handled)",
            ))
        };
        self.raw = b;
        r
    }
}

// ---------------------------------------------------------- PrimitiveBlock

pub struct Block<'a> {
    pub strings: Vec<&'a [u8]>,
    pub granularity: i64,
    pub lat_offset: i64,
    pub lon_offset: i64,
    groups: Vec<&'a [u8]>,
}

/// A node from a `DenseNodes` group.
pub struct DenseNode<'a> {
    pub id: i64,
    pub lat_e7: i32,
    pub lon_e7: i32,
    /// Interleaved key/value string-table indices, terminated by 0.
    pub kv: &'a [u32],
}

pub struct Way<'a> {
    pub id: i64,
    keys: &'a [u8],
    vals: &'a [u8],
    refs: &'a [u8],
}

impl<'a> Way<'a> {
    /// Delta-decoded node references, in way order.
    #[inline]
    pub fn refs(&self) -> DeltaSint<'a> {
        DeltaSint::new(self.refs)
    }
    /// `(key, value)` pairs resolved against the block string table.
    #[inline]
    pub fn tags(&self, blk: &'a Block<'a>) -> impl Iterator<Item = (&'a [u8], &'a [u8])> + 'a {
        let mut kk = PackedVarint::new(self.keys);
        let mut vv = PackedVarint::new(self.vals);
        std::iter::from_fn(move || {
            let k = kk.next()?;
            let v = vv.next()?;
            Some((blk.strings[k as usize], blk.strings[v as usize]))
        })
    }
}

impl<'a> Block<'a> {
    pub fn parse(b: &'a [u8]) -> Block<'a> {
        let mut p = 0usize;
        let mut strings: Vec<&[u8]> = Vec::new();
        let mut groups: Vec<&[u8]> = Vec::new();
        let mut granularity = 100i64;
        let mut lat_offset = 0i64;
        let mut lon_offset = 0i64;
        while p < b.len() {
            let k = varint(b, &mut p);
            match (k >> 3, k & 7) {
                (1, 2) => {
                    let l = varint(b, &mut p) as usize;
                    let st = &b[p..p + l];
                    p += l;
                    let mut q = 0usize;
                    while q < st.len() {
                        let kk = varint(st, &mut q);
                        if kk >> 3 == 1 && kk & 7 == 2 {
                            let sl = varint(st, &mut q) as usize;
                            strings.push(&st[q..q + sl]);
                            q += sl;
                        } else {
                            skip_field(st, &mut q, kk & 7);
                        }
                    }
                }
                (2, 2) => {
                    let l = varint(b, &mut p) as usize;
                    groups.push(&b[p..p + l]);
                    p += l;
                }
                (17, 0) => granularity = varint(b, &mut p) as i64,
                (19, 0) => lat_offset = zigzag(varint(b, &mut p)),
                (20, 0) => lon_offset = zigzag(varint(b, &mut p)),
                (_, w) => skip_field(b, &mut p, w),
            }
        }
        Block { strings, granularity, lat_offset, lon_offset, groups }
    }

    /// True when this block carries at least one way. Used to binary-search
    /// the node→way boundary of a sorted PBF without inflating the whole file.
    pub fn has_ways(&self) -> bool {
        self.groups.iter().any(|g| {
            let mut p = 0usize;
            while p < g.len() {
                let k = varint(g, &mut p);
                if k >> 3 == 3 && k & 7 == 2 {
                    return true;
                }
                skip_field(g, &mut p, k & 7);
            }
            false
        })
    }

    #[inline]
    fn to_e7(&self, v: i64, off: i64) -> i32 {
        // lat = 1e-9 * (offset + granularity * value)  →  e7 = that * 1e7
        let nano = off + self.granularity * v;
        (nano / 100) as i32
    }

    /// Visit every node in the block (both `DenseNodes` and plain `Node`).
    pub fn for_each_node<F: FnMut(DenseNode)>(&self, mut f: F) {
        let mut kvbuf: Vec<u32> = Vec::with_capacity(32);
        for g in &self.groups {
            let mut p = 0usize;
            while p < g.len() {
                let k = varint(g, &mut p);
                match (k >> 3, k & 7) {
                    (2, 2) => {
                        let l = varint(g, &mut p) as usize;
                        self.dense_nodes(&g[p..p + l], &mut kvbuf, &mut f);
                        p += l;
                    }
                    (1, 2) => {
                        let l = varint(g, &mut p) as usize;
                        self.plain_node(&g[p..p + l], &mut kvbuf, &mut f);
                        p += l;
                    }
                    (_, w) => skip_field(g, &mut p, w),
                }
            }
        }
    }

    fn plain_node<F: FnMut(DenseNode)>(&self, b: &[u8], kvbuf: &mut Vec<u32>, f: &mut F) {
        let mut p = 0usize;
        let (mut id, mut lat, mut lon) = (0i64, 0i64, 0i64);
        let (mut keys, mut vals): (&[u8], &[u8]) = (&[], &[]);
        while p < b.len() {
            let k = varint(b, &mut p);
            match (k >> 3, k & 7) {
                (1, 0) => id = zigzag(varint(b, &mut p)),
                (2, 2) => {
                    let l = varint(b, &mut p) as usize;
                    keys = &b[p..p + l];
                    p += l;
                }
                (3, 2) => {
                    let l = varint(b, &mut p) as usize;
                    vals = &b[p..p + l];
                    p += l;
                }
                (8, 0) => lat = zigzag(varint(b, &mut p)),
                (9, 0) => lon = zigzag(varint(b, &mut p)),
                (_, w) => skip_field(b, &mut p, w),
            }
        }
        kvbuf.clear();
        let mut kk = PackedVarint::new(keys);
        let mut vv = PackedVarint::new(vals);
        while let (Some(a), Some(c)) = (kk.next(), vv.next()) {
            kvbuf.push(a as u32);
            kvbuf.push(c as u32);
        }
        kvbuf.push(0);
        f(DenseNode {
            id,
            lat_e7: self.to_e7(lat, self.lat_offset),
            lon_e7: self.to_e7(lon, self.lon_offset),
            kv: kvbuf,
        });
    }

    fn dense_nodes<F: FnMut(DenseNode)>(&self, b: &[u8], kvbuf: &mut Vec<u32>, f: &mut F) {
        let mut p = 0usize;
        let (mut ids, mut lats, mut lons, mut kvs): (&[u8], &[u8], &[u8], &[u8]) =
            (&[], &[], &[], &[]);
        while p < b.len() {
            let k = varint(b, &mut p);
            match (k >> 3, k & 7) {
                (1, 2) => {
                    let l = varint(b, &mut p) as usize;
                    ids = &b[p..p + l];
                    p += l;
                }
                (8, 2) => {
                    let l = varint(b, &mut p) as usize;
                    lats = &b[p..p + l];
                    p += l;
                }
                (9, 2) => {
                    let l = varint(b, &mut p) as usize;
                    lons = &b[p..p + l];
                    p += l;
                }
                (10, 2) => {
                    let l = varint(b, &mut p) as usize;
                    kvs = &b[p..p + l];
                    p += l;
                }
                (_, w) => skip_field(b, &mut p, w),
            }
        }
        let mut idi = DeltaSint::new(ids);
        let mut lai = DeltaSint::new(lats);
        let mut loi = DeltaSint::new(lons);
        let mut kvp = 0usize;
        while let (Some(id), Some(la), Some(lo)) = (idi.next(), lai.next(), loi.next()) {
            kvbuf.clear();
            // keys_vals is one flat stream: k,v,k,v,...,0 per node.
            if !kvs.is_empty() {
                loop {
                    if kvp >= kvs.len() {
                        break;
                    }
                    let kx = varint(kvs, &mut kvp) as u32;
                    if kx == 0 {
                        break;
                    }
                    let vx = varint(kvs, &mut kvp) as u32;
                    kvbuf.push(kx);
                    kvbuf.push(vx);
                }
            }
            kvbuf.push(0);
            f(DenseNode {
                id,
                lat_e7: self.to_e7(la, self.lat_offset),
                lon_e7: self.to_e7(lo, self.lon_offset),
                kv: kvbuf,
            });
        }
    }

    /// Visit every way in the block.
    pub fn for_each_way<F: FnMut(Way<'a>)>(&self, mut f: F) {
        for g in &self.groups {
            let mut p = 0usize;
            while p < g.len() {
                let k = varint(g, &mut p);
                match (k >> 3, k & 7) {
                    (3, 2) => {
                        let l = varint(g, &mut p) as usize;
                        f(parse_way(&g[p..p + l]));
                        p += l;
                    }
                    (_, w) => skip_field(g, &mut p, w),
                }
            }
        }
    }

    #[inline]
    pub fn s(&self, i: u32) -> &'a [u8] {
        self.strings[i as usize]
    }
}

fn parse_way(b: &[u8]) -> Way<'_> {
    let mut p = 0usize;
    let mut id = 0i64;
    let (mut keys, mut vals, mut refs): (&[u8], &[u8], &[u8]) = (&[], &[], &[]);
    while p < b.len() {
        let k = varint(b, &mut p);
        match (k >> 3, k & 7) {
            (1, 0) => id = varint(b, &mut p) as i64,
            (2, 2) => {
                let l = varint(b, &mut p) as usize;
                keys = &b[p..p + l];
                p += l;
            }
            (3, 2) => {
                let l = varint(b, &mut p) as usize;
                vals = &b[p..p + l];
                p += l;
            }
            (8, 2) => {
                let l = varint(b, &mut p) as usize;
                refs = &b[p..p + l];
                p += l;
            }
            (_, w) => skip_field(b, &mut p, w),
        }
    }
    Way { id, keys, vals, refs }
}

/// Index of the first data blob that contains ways. A sorted PBF stores all
/// nodes before all ways, so a binary search over the blob directory finds the
/// boundary in ~20 inflations instead of reading 60 GB of node blocks.
pub fn find_first_way_blob(path: &Path, blobs: &[BlobDesc]) -> io::Result<usize> {
    let f = File::open(path)?;
    let mut inf = Inflater::default();
    let data: Vec<usize> = (0..blobs.len()).filter(|&i| blobs[i].kind == BlobKind::Data).collect();
    let (mut lo, mut hi) = (0usize, data.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        inf.load(&f, &blobs[data[mid]])?;
        if Block::parse(&inf.out).has_ways() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(if lo >= data.len() { blobs.len() } else { data[lo] })
}

/// Total uncompressed size the file will expand to — used only for progress.
pub fn file_len(path: &Path) -> io::Result<u64> {
    let mut f = File::open(path)?;
    let n = f.seek(SeekFrom::End(0))?;
    Ok(n)
}
