//! Measure how compressible each array is, before writing a codec for any of
//! them.
//!
//! matcodec's method is: predict from a cheap model, store the exact residual,
//! zigzag-varint it, then deflate — and it degrades gracefully to plain
//! deflate when the structure it assumes is absent. That last property is the
//! tell: the design was measured, not assumed. So this measures ours.
//!
//! Every candidate here is tested under the predictor its *construction*
//! suggests, which is where the structure would come from if it exists:
//!
//! * `csr.to` — vertices were renumbered along a Hilbert curve, so a vertex's
//!   neighbours should be numerically near it. Predict the source id.
//! * `geom.pts` — roads are locally straight, so the next shape point should
//!   sit near the linear extrapolation of the previous two. That is the same
//!   rank-1 idea as matcodec's gateway base, one dimension down.
//! * `vcoord`, `addr.coord` — both are stored in spatial order, so the
//!   previous record predicts the next.
//! * `seg.len`, `seg.attr` — small magnitudes and repeated bit patterns; no
//!   predictor, just an entropy check.
//!
//! Nothing is compressed here. The point is to find out what *would* pay
//! before committing to a format, and to be able to say so with numbers.

use crate::mmapvec;
use std::io::Write;
use std::path::Path;

/// Blocks are the unit of random access: a query decompresses one, so they
/// must stay small. 4096 records ≈ 16 KB raw for a u32 array — one or two
/// pages in, one block out.
pub const BLOCK: usize = 4096;

pub struct Measured {
    pub name: String,
    pub raw_bytes: u64,
    pub varint_bytes: u64,
    pub deflate_bytes: u64,
    pub predicted_deflate_bytes: u64,
    pub blocks: u64,
}

impl Measured {
    fn ratio(&self, v: u64) -> f64 {
        if v == 0 {
            0.0
        } else {
            self.raw_bytes as f64 / v as f64
        }
    }
    pub fn report(&self) {
        println!(
            "  {:<14} {:>7.2} GB   varint {:>4.2}x   deflate {:>4.2}x   predict+varint+deflate {:>4.2}x",
            self.name,
            self.raw_bytes as f64 / 1e9,
            self.ratio(self.varint_bytes),
            self.ratio(self.deflate_bytes),
            self.ratio(self.predicted_deflate_bytes),
        );
    }
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn varint_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

fn deflate_len(raw: &[u8]) -> usize {
    let mut e =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    e.write_all(raw).expect("deflate");
    e.finish().expect("finish").len()
}

/// Sample `n` blocks spread evenly through `len` records, so a 6 GB array is
/// characterised in seconds without reading all of it.
fn sample_starts(len: usize, n: usize) -> Vec<usize> {
    if len <= BLOCK {
        return vec![0];
    }
    let nblocks = len / BLOCK;
    let take = n.min(nblocks);
    (0..take).map(|i| (i * nblocks / take) * BLOCK).collect()
}

/// Measure one array of fixed-width records under a caller-supplied predictor.
///
/// `residual` receives a block and returns the per-record residuals the
/// predictor leaves behind; returning the values unchanged measures the
/// no-predictor case.
fn measure<T: Copy>(
    name: &str,
    path: &Path,
    samples: usize,
    residual: impl Fn(&[T], usize) -> Vec<i64>,
) -> std::io::Result<Option<Measured>> {
    let Ok(map) = mmapvec::open(path) else { return Ok(None) };
    let data: &[T] = unsafe { mmapvec::as_slice(&map[..]) };
    if data.is_empty() {
        return Ok(None);
    }
    let starts = sample_starts(data.len(), samples);
    let (mut raw, mut vi, mut de, mut pd) = (0u64, 0u64, 0u64, 0u64);
    for s in &starts {
        let end = (s + BLOCK).min(data.len());
        let block = &data[*s..end];
        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(block.as_ptr() as *const u8, std::mem::size_of_val(block))
        };
        raw += raw_bytes.len() as u64;
        de += deflate_len(raw_bytes) as u64;

        let res = residual(block, *s);
        let mut buf = Vec::with_capacity(res.len() * 2);
        for &r in &res {
            put_varint(&mut buf, zigzag(r));
        }
        vi += buf.len() as u64;
        pd += deflate_len(&buf) as u64;
    }
    // Scale the sample to the whole array.
    let total_raw = std::mem::size_of_val(data) as u64;
    let scale = |v: u64| if raw == 0 { 0 } else { (v as f64 * total_raw as f64 / raw as f64) as u64 };
    Ok(Some(Measured {
        name: name.into(),
        raw_bytes: total_raw,
        varint_bytes: scale(vi),
        deflate_bytes: scale(de),
        predicted_deflate_bytes: scale(pd),
        blocks: (data.len() / BLOCK) as u64,
    }))
}

/// Is decompression cheaper than the read it replaces?
///
/// That is the whole question, and the answer is a property of *this machine*:
/// a fast NVMe with 23 µs random reads leaves very little room for a codec to
/// pay for itself, while a slower disk or a dataset that does not fit in RAM
/// changes the arithmetic completely. So it is measured, not assumed — with
/// the decompressor we would actually ship, not a stand-in.
pub fn bench_codec(dir: &Path) -> std::io::Result<()> {
    let map = mmapvec::open(&dir.join("csr.to"))?;
    let raw: &[u8] = &map[..];
    let block = BLOCK * 4;
    let n = 400.min(raw.len() / block);
    let blocks: Vec<&[u8]> = raw.chunks_exact(block).take(n).collect();

    let mut comp = Vec::with_capacity(n);
    let mut c = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
    for b in &blocks {
        let mut out = vec![0u8; c.zlib_compress_bound(b.len())];
        let k = c.zlib_compress(b, &mut out).expect("compress");
        out.truncate(k);
        comp.push(out);
    }
    let ratio = (n * block) as f64 / comp.iter().map(|c| c.len()).sum::<usize>() as f64;

    let mut d = libdeflater::Decompressor::new();
    let mut out = vec![0u8; block];
    // Warm the branch predictor and the allocator before timing.
    for cb in comp.iter().take(8) {
        let _ = d.zlib_decompress(cb, &mut out);
    }
    let t = std::time::Instant::now();
    let reps = 20;
    for _ in 0..reps {
        for cb in &comp {
            d.zlib_decompress(cb, &mut out).expect("decompress");
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let mb = (n * block * reps) as f64 / 1e6;
    let per_page = 16384.0 / (mb / secs * 1e6) * 1e6;
    println!(
        "libdeflate on csr.to: {:.0} MB/s of output, ratio {ratio:.2}x, {per_page:.0} us per 16 KB",
        mb / secs
    );
    Ok(())
}

/// Characterise a built dataset.
pub fn analyse(dir: &Path, samples: usize) -> std::io::Result<Vec<Measured>> {
    let mut out = Vec::new();
    let mut push = |m: Option<Measured>| {
        if let Some(m) = m {
            m.report();
            out.push(m);
        }
    };
    let _ = varint_len(0);

    // csr.to — the target of each edge. Hilbert renumbering should make a
    // neighbour numerically close to its source, but the source is only known
    // via csr.head, so approximate it with the previous target in the block:
    // adjacency lists are consecutive, so this is the same locality.
    push(measure::<u32>("csr.to", &dir.join("csr.to"), samples, |b, _| {
        let mut prev = 0i64;
        b.iter()
            .map(|&v| {
                let d = v as i64 - prev;
                prev = v as i64;
                d
            })
            .collect()
    })?);
    push(measure::<u32>("csr.rto", &dir.join("csr.rto"), samples, |b, _| {
        let mut prev = 0i64;
        b.iter()
            .map(|&v| {
                let d = v as i64 - prev;
                prev = v as i64;
                d
            })
            .collect()
    })?);
    // csr.seg — segment ids, emitted in CSR order.
    push(measure::<u32>("csr.seg", &dir.join("csr.seg"), samples, |b, _| {
        let mut prev = 0i64;
        b.iter()
            .map(|&v| {
                let d = (v & 0x7fff_ffff) as i64 - prev;
                prev = (v & 0x7fff_ffff) as i64;
                d
            })
            .collect()
    })?);
    // geom.pts — linear extrapolation of the previous two points; roads are
    // locally straight, so this should leave near-zero residuals.
    push(measure::<(i32, i32)>("geom.pts", &dir.join("geom.pts"), samples, |b, _| {
        let mut r = Vec::with_capacity(b.len() * 2);
        for i in 0..b.len() {
            let (pl, plo) = match i {
                0 => (0i64, 0i64),
                1 => (b[0].0 as i64, b[0].1 as i64),
                _ => (
                    2 * b[i - 1].0 as i64 - b[i - 2].0 as i64,
                    2 * b[i - 1].1 as i64 - b[i - 2].1 as i64,
                ),
            };
            r.push(b[i].0 as i64 - pl);
            r.push(b[i].1 as i64 - plo);
        }
        r
    })?);
    // vcoord — Hilbert order, so the previous vertex predicts the next.
    push(measure::<(i32, i32)>("vcoord.bin", &dir.join("vcoord.bin"), samples, |b, _| {
        let mut r = Vec::with_capacity(b.len() * 2);
        let (mut pa, mut pb) = (0i64, 0i64);
        for &(a, c) in b {
            r.push(a as i64 - pa);
            r.push(c as i64 - pb);
            pa = a as i64;
            pb = c as i64;
        }
        r
    })?);
    // addr.coord — cell order, same argument.
    push(measure::<(i32, i32)>("addr.coord", &dir.join("addr.coord"), samples, |b, _| {
        let mut r = Vec::with_capacity(b.len() * 2);
        let (mut pa, mut pb) = (0i64, 0i64);
        for &(a, c) in b {
            r.push(a as i64 - pa);
            r.push(c as i64 - pb);
            pa = a as i64;
            pb = c as i64;
        }
        r
    })?);
    // snap.seg — segment ids grouped by cell, ascending inside a cell.
    push(measure::<u32>("snap.seg", &dir.join("snap.seg"), samples, |b, _| {
        let mut prev = 0i64;
        b.iter()
            .map(|&v| {
                let d = v as i64 - prev;
                prev = v as i64;
                d
            })
            .collect()
    })?);
    // seg.len — centimetres, mostly small; no predictor, just magnitude.
    push(measure::<u32>("seg.len", &dir.join("seg.len"), samples, |b, _| {
        b.iter().map(|&v| v as i64).collect()
    })?);
    // seg.attr — packed bit fields, highly repetitive.
    push(measure::<u32>("seg.attr", &dir.join("seg.attr"), samples, |b, _| {
        b.iter().map(|&v| v as i64).collect()
    })?);
    push(measure::<u32>("addr.street", &dir.join("addr.street"), samples, |b, _| {
        let mut prev = 0i64;
        b.iter()
            .map(|&v| {
                let d = v as i64 - prev;
                prev = v as i64;
                d
            })
            .collect()
    })?);
    Ok(out)
}
