//! Binary row-streamed many-to-many matrix format (Variant A).
//!
//! Layout (little-endian, Apple Silicon native):
//!
//! ```text
//! Header (32 bytes, fixed):
//!   [0..8]   magic = b"RTBL0001"
//!   [8..10]  version u16 = 1
//!   [10..12] flags u16   (bit 0: dual-channel duration+distance interleaved)
//!   [12..16] cell_dtype u32  (0=f32, 1=u16, 2=u32)
//!   [16..20] n_src u32
//!   [20..24] n_dst u32
//!   [24..28] scale_exp i32  (encoded = real × 10^(-scale_exp);
//!                            ignored when dtype = f32)
//!   [28..32] reserved u32 = 0
//!
//! Per row, written in stream order (which may be FF-permuted):
//!   [0..4]   row_index u32   (original src index in caller's input order)
//!   [4..8]   row_byte_len u32 (= n_dst × dtype_size × channel_count;
//!                              lets the reader skip without parsing cells)
//!   [8..8+row_byte_len] cell data
//! ```
//!
//! Cell layout when `dual-channel` is set: each cell is two values
//! interleaved `[duration, distance]`. So per-cell byte size is
//! `dtype_size × 2`.
//!
//! Sentinel for unreachable: `f32::INFINITY` for f32, `u16::MAX` for u16,
//! `u32::MAX` for u32.

use std::io::{self, Write};

pub const MAGIC: &[u8; 8] = b"RTBL0001";
pub const VERSION: u16 = 1;

/// 16-byte CRC trailer marker (only present when FLAG_CRC32_FOOTER is set).
pub const CRC_MAGIC: &[u8; 8] = b"RTBLCRC1";

// Flag bits in the 16-bit `flags` header field.
pub const FLAG_DUAL_CHANNEL: u16 = 1 << 0;
/// Each row body is followed by zero bytes so that
/// `(8 + row_byte_len + pad)` is a multiple of 64 → next row's header starts
/// on a cache-line boundary. Makes mmap-cast reads cleaner for SIMD consumers.
pub const FLAG_PAD_64: u16 = 1 << 1;
/// File ends with a 16-byte trailer: `b"RTBLCRC1"` (8) + `u32 crc32` (4) +
/// `u32 reserved` (4). CRC is computed over every byte from file start up
/// to (but not including) the trailer itself.
pub const FLAG_CRC32_FOOTER: u16 = 1 << 2;
/// Variant B (symmetric upper-triangle) layout: no per-row records, just
/// `N*(N-1)/2` cells (i<j) packed flat after the header. `SymmetricBinaryWriter`
/// emits this layout.
pub const FLAG_SYMMETRIC_UT: u16 = 1 << 3;

/// Standard IEEE 802.3 CRC32 (same poly as zlib/gzip).
/// Pre-computed forward table; ~0.3 GB/s on a single core.
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut j = 0;
        while j < 8 {
            c = if c & 1 != 0 {
                0xEDB88320 ^ (c >> 1)
            } else {
                c >> 1
            };
            j += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

#[inline]
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in data {
        c = CRC32_TABLE[((c as u8) ^ b) as usize] ^ (c >> 8);
    }
    !c
}

#[inline]
fn pad_to_64(n: usize) -> usize {
    let r = n % 64;
    if r == 0 { 0 } else { 64 - r }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CellDtype {
    F32 = 0,
    U16 = 1,
    U32 = 2,
}

impl CellDtype {
    #[inline]
    pub fn size(self) -> usize {
        match self {
            CellDtype::F32 => 4,
            CellDtype::U16 => 2,
            CellDtype::U32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WriterConfig {
    pub n_src: u32,
    pub n_dst: u32,
    pub dual_channel: bool,
    pub cell_dtype: CellDtype,
    /// Encoded value = `real × 10^(-scale_exp)`. Ignored when dtype = f32.
    /// Example: scale_exp = -1 → 0.1-unit precision (u16 cap = 6553.5).
    /// Example: scale_exp =  0 → 1-unit precision  (u16 cap = 65535).
    pub scale_exp: i32,
    /// Pad each row body so the next row's header lands on a 64-byte
    /// boundary (cache line). Useful for mmap+SIMD consumers.
    pub pad_64: bool,
    /// Emit a CRC32 trailer (16 bytes) at the end of the file. Reader can
    /// detect corruption (important on iOS where files may be truncated
    /// when the app is killed).
    pub crc32_footer: bool,
}

impl WriterConfig {
    /// Minimal sane default: dual channel, f32, no padding, no CRC trailer.
    pub fn simple(n_src: u32, n_dst: u32, dual_channel: bool) -> Self {
        Self {
            n_src,
            n_dst,
            dual_channel,
            cell_dtype: CellDtype::F32,
            scale_exp: 0,
            pad_64: false,
            crc32_footer: false,
        }
    }
}

/// Streamed writer: header is emitted up front, then `write_row` produces
/// one record per source. Wraps any `Write` (file, socket, Vec<u8>).
///
/// **Must call `finish()`** to emit the CRC32 trailer when
/// `cfg.crc32_footer == true`. Dropping the writer without calling
/// `finish()` leaves a file without a trailer (and the reader will reject
/// it if it expects one).
pub struct BinaryTableWriter<W: Write> {
    out: W,
    cfg: WriterConfig,
    channel_count: u32,
    scale_factor: f64,    // 10^(-scale_exp), precomputed for hot loop
    crc: u32,             // running CRC32 over every byte written
    bytes_written: u64,   // total bytes (for sanity / debug)
}

impl<W: Write> BinaryTableWriter<W> {
    pub fn new(out: W, cfg: WriterConfig) -> io::Result<Self> {
        let channel_count = if cfg.dual_channel { 2 } else { 1 };
        let scale_factor = 10f64.powi(-cfg.scale_exp);
        let mut w = Self {
            out,
            cfg,
            channel_count,
            scale_factor,
            crc: 0,
            bytes_written: 0,
        };
        let mut flags = 0u16;
        if cfg.dual_channel {
            flags |= FLAG_DUAL_CHANNEL;
        }
        if cfg.pad_64 {
            flags |= FLAG_PAD_64;
        }
        if cfg.crc32_footer {
            flags |= FLAG_CRC32_FOOTER;
        }
        let mut hdr = [0u8; 32];
        hdr[0..8].copy_from_slice(MAGIC);
        hdr[8..10].copy_from_slice(&VERSION.to_le_bytes());
        hdr[10..12].copy_from_slice(&flags.to_le_bytes());
        hdr[12..16].copy_from_slice(&(cfg.cell_dtype as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&cfg.n_src.to_le_bytes());
        hdr[20..24].copy_from_slice(&cfg.n_dst.to_le_bytes());
        hdr[24..28].copy_from_slice(&cfg.scale_exp.to_le_bytes());
        // hdr[28..32] zero
        w.write_bytes(&hdr)?;
        Ok(w)
    }

    /// Internal write helper: updates running CRC + byte count, then writes.
    #[inline]
    fn write_bytes(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.cfg.crc32_footer {
            self.crc = crc32_update(self.crc, buf);
        }
        self.bytes_written += buf.len() as u64;
        self.out.write_all(buf)
    }

    /// Write a single row. `dur` must have length `n_dst`. If `dual_channel`,
    /// `dist` is `Some(&[..])` with the same length; otherwise pass `None`.
    pub fn write_row(
        &mut self,
        row_index: u32,
        dur: &[f32],
        dist: Option<&[f32]>,
    ) -> io::Result<()> {
        let n_dst = self.cfg.n_dst as usize;
        assert_eq!(dur.len(), n_dst, "dur length must equal n_dst");
        if self.cfg.dual_channel {
            let d = dist.expect("dual_channel writer requires Some(dist)");
            assert_eq!(d.len(), n_dst, "dist length must equal n_dst");
        }
        let dtype_size = self.cfg.cell_dtype.size();
        let row_byte_len = (n_dst as u32) * (dtype_size as u32) * self.channel_count;
        self.write_bytes(&row_index.to_le_bytes())?;
        self.write_bytes(&row_byte_len.to_le_bytes())?;

        match self.cfg.cell_dtype {
            CellDtype::F32 => self.write_cells_f32(dur, dist)?,
            CellDtype::U16 => self.write_cells_u16(dur, dist)?,
            CellDtype::U32 => self.write_cells_u32(dur, dist)?,
        }
        // Optional 64-byte padding so next row's header lands on a cache line.
        if self.cfg.pad_64 {
            let total = 8 + row_byte_len as usize; // row header + body
            let pad = pad_to_64(total);
            if pad > 0 {
                let zeros = [0u8; 64];
                self.write_bytes(&zeros[..pad])?;
            }
        }
        Ok(())
    }

    /// Finish the stream: flush the inner writer and emit the optional
    /// CRC32 trailer. Returns the CRC32 value (0 if trailer disabled).
    /// Always call this after the last row — droppening without
    /// `finish()` skips the trailer.
    pub fn finish(mut self) -> io::Result<u32> {
        let crc = self.crc;
        if self.cfg.crc32_footer {
            // Build trailer locally; do NOT update self.crc with its own bytes.
            let mut trailer = [0u8; 16];
            trailer[0..8].copy_from_slice(CRC_MAGIC);
            trailer[8..12].copy_from_slice(&crc.to_le_bytes());
            // trailer[12..16] zero (reserved for future use)
            self.out.write_all(&trailer)?;
        }
        self.out.flush()?;
        Ok(crc)
    }

    fn write_cells_f32(&mut self, dur: &[f32], dist: Option<&[f32]>) -> io::Result<()> {
        // Hot loop: copy via stack buffer of ~4 KB to amortise write syscalls.
        const BUF_CELLS: usize = 256;
        let mut buf = [0u8; BUF_CELLS * 4 * 2];
        let mut i = 0;
        let n = dur.len();
        while i < n {
            let end = (i + BUF_CELLS).min(n);
            let mut p = 0;
            if let Some(d) = dist {
                for k in i..end {
                    buf[p..p + 4].copy_from_slice(&dur[k].to_le_bytes());
                    buf[p + 4..p + 8].copy_from_slice(&d[k].to_le_bytes());
                    p += 8;
                }
            } else {
                for k in i..end {
                    buf[p..p + 4].copy_from_slice(&dur[k].to_le_bytes());
                    p += 4;
                }
            }
            self.write_bytes(&buf[..p])?;
            i = end;
        }
        Ok(())
    }

    fn write_cells_u16(&mut self, dur: &[f32], dist: Option<&[f32]>) -> io::Result<()> {
        let scale = self.scale_factor;
        let enc = |v: f32| -> u16 {
            if !v.is_finite() {
                return u16::MAX;
            }
            let scaled = (v as f64) * scale;
            if scaled <= 0.0 {
                0
            } else if scaled >= (u16::MAX - 1) as f64 {
                u16::MAX - 1
            } else {
                scaled.round() as u16
            }
        };
        const BUF_CELLS: usize = 512;
        let mut buf = [0u8; BUF_CELLS * 2 * 2];
        let mut i = 0;
        let n = dur.len();
        while i < n {
            let end = (i + BUF_CELLS).min(n);
            let mut p = 0;
            if let Some(d) = dist {
                for k in i..end {
                    buf[p..p + 2].copy_from_slice(&enc(dur[k]).to_le_bytes());
                    buf[p + 2..p + 4].copy_from_slice(&enc(d[k]).to_le_bytes());
                    p += 4;
                }
            } else {
                for k in i..end {
                    buf[p..p + 2].copy_from_slice(&enc(dur[k]).to_le_bytes());
                    p += 2;
                }
            }
            self.write_bytes(&buf[..p])?;
            i = end;
        }
        Ok(())
    }

    fn write_cells_u32(&mut self, dur: &[f32], dist: Option<&[f32]>) -> io::Result<()> {
        let scale = self.scale_factor;
        let enc = |v: f32| -> u32 {
            if !v.is_finite() {
                return u32::MAX;
            }
            let scaled = (v as f64) * scale;
            if scaled <= 0.0 {
                0
            } else if scaled >= (u32::MAX - 1) as f64 {
                u32::MAX - 1
            } else {
                scaled.round() as u32
            }
        };
        const BUF_CELLS: usize = 256;
        let mut buf = [0u8; BUF_CELLS * 4 * 2];
        let mut i = 0;
        let n = dur.len();
        while i < n {
            let end = (i + BUF_CELLS).min(n);
            let mut p = 0;
            if let Some(d) = dist {
                for k in i..end {
                    buf[p..p + 4].copy_from_slice(&enc(dur[k]).to_le_bytes());
                    buf[p + 4..p + 8].copy_from_slice(&enc(d[k]).to_le_bytes());
                    p += 8;
                }
            } else {
                for k in i..end {
                    buf[p..p + 4].copy_from_slice(&enc(dur[k]).to_le_bytes());
                    p += 4;
                }
            }
            self.write_bytes(&buf[..p])?;
            i = end;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }

    pub fn into_inner(self) -> W {
        self.out
    }
}

/// Reader for verification / consumers that want a Rust-side parser.
///
/// Reads the header lazily, then `read_row` returns the next row's
/// `(row_index, dur, dist)`. Streams forward; cannot seek backwards.
pub struct BinaryTableReader<R: io::Read> {
    input: R,
    pub n_src: u32,
    pub n_dst: u32,
    pub dual_channel: bool,
    pub cell_dtype: CellDtype,
    pub scale_exp: i32,
    pub pad_64: bool,
    pub crc32_footer: bool,
    pub symmetric_ut: bool,
    rows_read: u32,
    inv_scale: f64, // 10^scale_exp, used to decode
}

impl<R: io::Read> BinaryTableReader<R> {
    pub fn new(mut input: R) -> io::Result<Self> {
        let mut hdr = [0u8; 32];
        input.read_exact(&mut hdr)?;
        if &hdr[0..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = u16::from_le_bytes(hdr[8..10].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version {version}"),
            ));
        }
        let flags = u16::from_le_bytes(hdr[10..12].try_into().unwrap());
        let dual_channel = flags & FLAG_DUAL_CHANNEL != 0;
        let pad_64 = flags & FLAG_PAD_64 != 0;
        let crc32_footer = flags & FLAG_CRC32_FOOTER != 0;
        let symmetric_ut = flags & FLAG_SYMMETRIC_UT != 0;
        let cell_dtype_u32 = u32::from_le_bytes(hdr[12..16].try_into().unwrap());
        let cell_dtype = match cell_dtype_u32 {
            0 => CellDtype::F32,
            1 => CellDtype::U16,
            2 => CellDtype::U32,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown cell_dtype {other}"),
                ))
            }
        };
        let n_src = u32::from_le_bytes(hdr[16..20].try_into().unwrap());
        let n_dst = u32::from_le_bytes(hdr[20..24].try_into().unwrap());
        let scale_exp = i32::from_le_bytes(hdr[24..28].try_into().unwrap());
        let inv_scale = 10f64.powi(scale_exp);
        Ok(Self {
            input,
            n_src,
            n_dst,
            dual_channel,
            cell_dtype,
            scale_exp,
            pad_64,
            crc32_footer,
            symmetric_ut,
            rows_read: 0,
            inv_scale,
        })
    }

    /// Returns Ok(None) on EOF, Ok(Some((row_index, dur, dist))) otherwise.
    pub fn read_row(&mut self) -> io::Result<Option<(u32, Vec<f32>, Option<Vec<f32>>)>> {
        if self.symmetric_ut {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "symmetric UT file: use SymmetricBinaryReader instead",
            ));
        }
        if self.rows_read >= self.n_src {
            return Ok(None);
        }
        let mut hdr = [0u8; 8];
        match self.input.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let row_index = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        let row_byte_len = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
        let mut body = vec![0u8; row_byte_len];
        self.input.read_exact(&mut body)?;
        let (dur, dist) = self.decode_row(&body);
        // Skip 64-byte padding after the row body, if enabled.
        if self.pad_64 {
            let pad = pad_to_64(8 + row_byte_len);
            if pad > 0 {
                let mut scratch = [0u8; 64];
                self.input.read_exact(&mut scratch[..pad])?;
            }
        }
        self.rows_read += 1;
        Ok(Some((row_index, dur, dist)))
    }

    /// After all rows have been read, optionally consume + validate the
    /// 16-byte CRC trailer. Returns the trailer's CRC value. Caller is
    /// responsible for re-computing the CRC over file bytes if it wants to
    /// verify integrity (the reader does not buffer the file).
    pub fn read_trailer(&mut self) -> io::Result<Option<u32>> {
        if !self.crc32_footer {
            return Ok(None);
        }
        let mut trailer = [0u8; 16];
        self.input.read_exact(&mut trailer)?;
        if &trailer[0..8] != CRC_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad CRC trailer magic",
            ));
        }
        let crc = u32::from_le_bytes(trailer[8..12].try_into().unwrap());
        Ok(Some(crc))
    }

    fn decode_row(&self, body: &[u8]) -> (Vec<f32>, Option<Vec<f32>>) {
        let n_dst = self.n_dst as usize;
        let dtype_size = self.cell_dtype.size();
        let stride = if self.dual_channel { 2 } else { 1 };
        assert_eq!(body.len(), n_dst * dtype_size * stride);
        let mut dur = Vec::with_capacity(n_dst);
        let mut dist = if self.dual_channel {
            Some(Vec::with_capacity(n_dst))
        } else {
            None
        };
        let cell_pair_size = dtype_size * stride;
        for k in 0..n_dst {
            let off = k * cell_pair_size;
            let (d_dur, d_dist) = match self.cell_dtype {
                CellDtype::F32 => {
                    let dv = f32::from_le_bytes(body[off..off + 4].try_into().unwrap());
                    let dd = if self.dual_channel {
                        Some(f32::from_le_bytes(
                            body[off + 4..off + 8].try_into().unwrap(),
                        ))
                    } else {
                        None
                    };
                    (dv, dd)
                }
                CellDtype::U16 => {
                    let raw = u16::from_le_bytes(body[off..off + 2].try_into().unwrap());
                    let dv = if raw == u16::MAX {
                        f32::INFINITY
                    } else {
                        (raw as f64 * self.inv_scale) as f32
                    };
                    let dd = if self.dual_channel {
                        let r2 = u16::from_le_bytes(
                            body[off + 2..off + 4].try_into().unwrap(),
                        );
                        Some(if r2 == u16::MAX {
                            f32::INFINITY
                        } else {
                            (r2 as f64 * self.inv_scale) as f32
                        })
                    } else {
                        None
                    };
                    (dv, dd)
                }
                CellDtype::U32 => {
                    let raw = u32::from_le_bytes(body[off..off + 4].try_into().unwrap());
                    let dv = if raw == u32::MAX {
                        f32::INFINITY
                    } else {
                        (raw as f64 * self.inv_scale) as f32
                    };
                    let dd = if self.dual_channel {
                        let r2 = u32::from_le_bytes(
                            body[off + 4..off + 8].try_into().unwrap(),
                        );
                        Some(if r2 == u32::MAX {
                            f32::INFINITY
                        } else {
                            (r2 as f64 * self.inv_scale) as f32
                        })
                    } else {
                        None
                    };
                    (dv, dd)
                }
            };
            dur.push(d_dur);
            if let (Some(dist), Some(dd)) = (dist.as_mut(), d_dist) {
                dist.push(dd);
            }
        }
        (dur, dist)
    }
}

// ─── Variant B: symmetric upper-triangle layout ────────────────────────────
//
// For a symmetric matrix, only N*(N-1)/2 cells need to be stored. The file
// uses the same 32-byte header as Variant A but with `FLAG_SYMMETRIC_UT`
// set; after the header, cells are packed flat in the canonical (i<j) scan
// order:
//
//   (0,1) (0,2) ... (0,N-1)
//   (1,2) (1,3) ... (1,N-1)
//   ...
//   (N-2,N-1)
//
// Index formula (i < j):
//   idx = i*(2N - i - 1) / 2 + (j - i - 1)
//
// Optional 64-byte padding (FLAG_PAD_64) pads the whole UT block at the end
// up to a 64-byte multiple — there are no per-row records to align.

/// Flat-index of cell (i, j) in the upper-triangle layout. Requires i < j.
#[inline]
pub fn ut_index(n: u32, i: u32, j: u32) -> u64 {
    debug_assert!(i < j, "ut_index requires i < j");
    debug_assert!(j < n, "j out of range");
    let i = i as u64;
    let j = j as u64;
    let n = n as u64;
    i * (2 * n - i - 1) / 2 + (j - i - 1)
}

pub struct SymmetricBinaryWriter<W: Write> {
    out: W,
    n: u32,
    cfg: WriterConfig,
    channel_count: u32,
    scale_factor: f64,
    expected_cells: u64,
    cells_written: u64,
    crc: u32,
    bytes_written: u64,
}

impl<W: Write> SymmetricBinaryWriter<W> {
    /// Construct a writer for an N×N symmetric matrix. `cfg.n_src` and
    /// `cfg.n_dst` must both equal `n`.
    pub fn new(out: W, cfg: WriterConfig) -> io::Result<Self> {
        assert_eq!(
            cfg.n_src, cfg.n_dst,
            "symmetric writer requires n_src == n_dst"
        );
        let n = cfg.n_src;
        let channel_count = if cfg.dual_channel { 2 } else { 1 };
        let scale_factor = 10f64.powi(-cfg.scale_exp);
        let expected_cells = (n as u64) * (n as u64 - 1) / 2;
        let mut w = Self {
            out,
            n,
            cfg,
            channel_count,
            scale_factor,
            expected_cells,
            cells_written: 0,
            crc: 0,
            bytes_written: 0,
        };
        let mut flags = FLAG_SYMMETRIC_UT;
        if cfg.dual_channel {
            flags |= FLAG_DUAL_CHANNEL;
        }
        if cfg.pad_64 {
            flags |= FLAG_PAD_64;
        }
        if cfg.crc32_footer {
            flags |= FLAG_CRC32_FOOTER;
        }
        let mut hdr = [0u8; 32];
        hdr[0..8].copy_from_slice(MAGIC);
        hdr[8..10].copy_from_slice(&VERSION.to_le_bytes());
        hdr[10..12].copy_from_slice(&flags.to_le_bytes());
        hdr[12..16].copy_from_slice(&(cfg.cell_dtype as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&n.to_le_bytes());
        hdr[20..24].copy_from_slice(&n.to_le_bytes());
        hdr[24..28].copy_from_slice(&cfg.scale_exp.to_le_bytes());
        w.write_bytes(&hdr)?;
        Ok(w)
    }

    #[inline]
    fn write_bytes(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.cfg.crc32_footer {
            self.crc = crc32_update(self.crc, buf);
        }
        self.bytes_written += buf.len() as u64;
        self.out.write_all(buf)
    }

    /// Convenience: given a *full* duration/distance row for source `i`
    /// (length `n`), emit the upper-triangle portion (`j` in `i+1..n`).
    /// Must be called with `i = 0, 1, ..., n-2` in order.
    pub fn write_ut_from_full_row(
        &mut self,
        i: u32,
        full_dur: &[f32],
        full_dist: Option<&[f32]>,
    ) -> io::Result<()> {
        assert_eq!(full_dur.len(), self.n as usize);
        if self.cfg.dual_channel {
            let d = full_dist.expect("dual_channel writer requires Some(full_dist)");
            assert_eq!(d.len(), self.n as usize);
        }
        let start = i as usize + 1;
        let dur_slice = &full_dur[start..];
        let dist_slice = full_dist.map(|d| &d[start..]);
        self.write_cells_slice(dur_slice, dist_slice)
    }

    /// Lower-level: append `dur.len()` consecutive cells in scan order.
    /// Caller is responsible for emitting cells in (i<j) order.
    pub fn write_cells_slice(
        &mut self,
        dur: &[f32],
        dist: Option<&[f32]>,
    ) -> io::Result<()> {
        if let Some(d) = dist {
            assert_eq!(d.len(), dur.len());
        }
        let n_cells = dur.len() as u64;
        if n_cells == 0 {
            return Ok(());
        }
        // Re-use the Variant A cell encoders by writing to a small scratch.
        match self.cfg.cell_dtype {
            CellDtype::F32 => {
                const BUF_CELLS: usize = 256;
                let mut buf = [0u8; BUF_CELLS * 4 * 2];
                let mut i = 0;
                let n = dur.len();
                while i < n {
                    let end = (i + BUF_CELLS).min(n);
                    let mut p = 0;
                    if let Some(d) = dist {
                        for k in i..end {
                            buf[p..p + 4].copy_from_slice(&dur[k].to_le_bytes());
                            buf[p + 4..p + 8].copy_from_slice(&d[k].to_le_bytes());
                            p += 8;
                        }
                    } else {
                        for k in i..end {
                            buf[p..p + 4].copy_from_slice(&dur[k].to_le_bytes());
                            p += 4;
                        }
                    }
                    self.write_bytes(&buf[..p])?;
                    i = end;
                }
            }
            CellDtype::U16 => {
                let scale = self.scale_factor;
                let enc = |v: f32| -> u16 {
                    if !v.is_finite() {
                        return u16::MAX;
                    }
                    let s = (v as f64) * scale;
                    if s <= 0.0 {
                        0
                    } else if s >= (u16::MAX - 1) as f64 {
                        u16::MAX - 1
                    } else {
                        s.round() as u16
                    }
                };
                const BUF_CELLS: usize = 512;
                let mut buf = [0u8; BUF_CELLS * 2 * 2];
                let mut i = 0;
                let n = dur.len();
                while i < n {
                    let end = (i + BUF_CELLS).min(n);
                    let mut p = 0;
                    if let Some(d) = dist {
                        for k in i..end {
                            buf[p..p + 2].copy_from_slice(&enc(dur[k]).to_le_bytes());
                            buf[p + 2..p + 4].copy_from_slice(&enc(d[k]).to_le_bytes());
                            p += 4;
                        }
                    } else {
                        for k in i..end {
                            buf[p..p + 2].copy_from_slice(&enc(dur[k]).to_le_bytes());
                            p += 2;
                        }
                    }
                    self.write_bytes(&buf[..p])?;
                    i = end;
                }
            }
            CellDtype::U32 => {
                let scale = self.scale_factor;
                let enc = |v: f32| -> u32 {
                    if !v.is_finite() {
                        return u32::MAX;
                    }
                    let s = (v as f64) * scale;
                    if s <= 0.0 {
                        0
                    } else if s >= (u32::MAX - 1) as f64 {
                        u32::MAX - 1
                    } else {
                        s.round() as u32
                    }
                };
                const BUF_CELLS: usize = 256;
                let mut buf = [0u8; BUF_CELLS * 4 * 2];
                let mut i = 0;
                let n = dur.len();
                while i < n {
                    let end = (i + BUF_CELLS).min(n);
                    let mut p = 0;
                    if let Some(d) = dist {
                        for k in i..end {
                            buf[p..p + 4].copy_from_slice(&enc(dur[k]).to_le_bytes());
                            buf[p + 4..p + 8].copy_from_slice(&enc(d[k]).to_le_bytes());
                            p += 8;
                        }
                    } else {
                        for k in i..end {
                            buf[p..p + 4].copy_from_slice(&enc(dur[k]).to_le_bytes());
                            p += 4;
                        }
                    }
                    self.write_bytes(&buf[..p])?;
                    i = end;
                }
            }
        }
        self.cells_written += n_cells;
        Ok(())
    }

    /// Pad the UT block to a 64-byte boundary (if FLAG_PAD_64) and emit the
    /// optional CRC32 trailer. Returns the CRC value.
    pub fn finish(mut self) -> io::Result<u32> {
        // Allow short-writes (some symmetric problems may legitimately skip
        // unreachable cells), but warn if the count is wrong in debug.
        debug_assert!(
            self.cells_written == self.expected_cells,
            "wrote {} cells, expected {}",
            self.cells_written,
            self.expected_cells
        );
        if self.cfg.pad_64 {
            // Align the entire body (post-header) — header is 32 bytes, so
            // align (bytes_written - 32) up.
            let body = self.bytes_written - 32;
            let pad = pad_to_64(body as usize);
            if pad > 0 {
                let zeros = [0u8; 64];
                self.write_bytes(&zeros[..pad])?;
            }
        }
        let crc = self.crc;
        if self.cfg.crc32_footer {
            let mut trailer = [0u8; 16];
            trailer[0..8].copy_from_slice(CRC_MAGIC);
            trailer[8..12].copy_from_slice(&crc.to_le_bytes());
            self.out.write_all(&trailer)?;
        }
        self.out.flush()?;
        Ok(crc)
    }
}

pub struct SymmetricBinaryReader<R: io::Read> {
    input: R,
    pub n: u32,
    pub dual_channel: bool,
    pub cell_dtype: CellDtype,
    pub scale_exp: i32,
    pub pad_64: bool,
    pub crc32_footer: bool,
    expected_cells: u64,
    cells_read: u64,
    inv_scale: f64,
}

impl<R: io::Read> SymmetricBinaryReader<R> {
    pub fn new(mut input: R) -> io::Result<Self> {
        let mut hdr = [0u8; 32];
        input.read_exact(&mut hdr)?;
        if &hdr[0..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = u16::from_le_bytes(hdr[8..10].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version {version}"),
            ));
        }
        let flags = u16::from_le_bytes(hdr[10..12].try_into().unwrap());
        if flags & FLAG_SYMMETRIC_UT == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file is not symmetric-UT — use BinaryTableReader instead",
            ));
        }
        let dual_channel = flags & FLAG_DUAL_CHANNEL != 0;
        let pad_64 = flags & FLAG_PAD_64 != 0;
        let crc32_footer = flags & FLAG_CRC32_FOOTER != 0;
        let cell_dtype_u32 = u32::from_le_bytes(hdr[12..16].try_into().unwrap());
        let cell_dtype = match cell_dtype_u32 {
            0 => CellDtype::F32,
            1 => CellDtype::U16,
            2 => CellDtype::U32,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown cell_dtype {other}"),
                ))
            }
        };
        let n_src = u32::from_le_bytes(hdr[16..20].try_into().unwrap());
        let n_dst = u32::from_le_bytes(hdr[20..24].try_into().unwrap());
        if n_src != n_dst {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "symmetric file: n_src != n_dst",
            ));
        }
        let n = n_src;
        let scale_exp = i32::from_le_bytes(hdr[24..28].try_into().unwrap());
        let inv_scale = 10f64.powi(scale_exp);
        let expected_cells = (n as u64) * (n as u64 - 1) / 2;
        Ok(Self {
            input,
            n,
            dual_channel,
            cell_dtype,
            scale_exp,
            pad_64,
            crc32_footer,
            expected_cells,
            cells_read: 0,
            inv_scale,
        })
    }

    /// Returns Ok(None) after `expected_cells` cells, otherwise the next
    /// `(duration, distance)` in scan order.
    pub fn read_cell(&mut self) -> io::Result<Option<(f32, Option<f32>)>> {
        if self.cells_read >= self.expected_cells {
            return Ok(None);
        }
        let dt = self.cell_dtype;
        let read_one = |input: &mut R, inv_scale: f64| -> io::Result<f32> {
            match dt {
                CellDtype::F32 => {
                    let mut b = [0u8; 4];
                    input.read_exact(&mut b)?;
                    Ok(f32::from_le_bytes(b))
                }
                CellDtype::U16 => {
                    let mut b = [0u8; 2];
                    input.read_exact(&mut b)?;
                    let raw = u16::from_le_bytes(b);
                    Ok(if raw == u16::MAX {
                        f32::INFINITY
                    } else {
                        (raw as f64 * inv_scale) as f32
                    })
                }
                CellDtype::U32 => {
                    let mut b = [0u8; 4];
                    input.read_exact(&mut b)?;
                    let raw = u32::from_le_bytes(b);
                    Ok(if raw == u32::MAX {
                        f32::INFINITY
                    } else {
                        (raw as f64 * inv_scale) as f32
                    })
                }
            }
        };
        let dur = read_one(&mut self.input, self.inv_scale)?;
        let dist = if self.dual_channel {
            Some(read_one(&mut self.input, self.inv_scale)?)
        } else {
            None
        };
        self.cells_read += 1;
        Ok(Some((dur, dist)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn roundtrip(cfg: WriterConfig, dur: &[f32], dist: Option<&[f32]>) -> (Vec<f32>, Option<Vec<f32>>) {
        let mut buf = Vec::new();
        {
            let mut w = BinaryTableWriter::new(&mut buf, cfg).unwrap();
            w.write_row(42, dur, dist).unwrap();
            w.flush().unwrap();
        }
        let mut r = BinaryTableReader::new(Cursor::new(&buf)).unwrap();
        let (idx, d, di) = r.read_row().unwrap().unwrap();
        assert_eq!(idx, 42);
        (d, di)
    }

    #[test]
    fn f32_dual_channel_roundtrip() {
        let cfg = WriterConfig {
            n_src: 1,
            n_dst: 3,
            dual_channel: true,
            cell_dtype: CellDtype::F32,
            scale_exp: 0,
            pad_64: false,
            crc32_footer: false,
        };
        let dur = vec![1.0, 2.5, f32::INFINITY];
        let dist = vec![10.0, 20.5, f32::INFINITY];
        let (d, di) = roundtrip(cfg, &dur, Some(&dist));
        assert_eq!(d, dur);
        assert_eq!(di.unwrap(), dist);
    }

    #[test]
    fn u16_scale_dist_roundtrip() {
        // scale_exp = 0 → 1 m precision, max ~65 km. London-friendly.
        let cfg = WriterConfig {
            n_src: 1,
            n_dst: 4,
            dual_channel: true,
            cell_dtype: CellDtype::U16,
            scale_exp: 0,
            pad_64: false,
            crc32_footer: false,
        };
        let dur = vec![1.0, 2.0, 60000.0, f32::INFINITY];
        let dist = vec![100.0, 24000.0, 50000.0, f32::INFINITY];
        let (d, di) = roundtrip(cfg, &dur, Some(&dist));
        assert!((d[0] - 1.0).abs() <= 1.0);
        assert!((d[1] - 2.0).abs() <= 1.0);
        assert!((d[2] - 60000.0).abs() <= 1.0);
        assert!(d[3].is_infinite());
        let di = di.unwrap();
        assert!((di[0] - 100.0).abs() <= 1.0);
        assert!((di[1] - 24000.0).abs() <= 1.0);
        assert!((di[2] - 50000.0).abs() <= 1.0);
        assert!(di[3].is_infinite());
    }

    #[test]
    fn u16_scale_dist_high_precision_capped() {
        // scale_exp = -1 → 0.1 m precision, but caps at 6553.4 m.
        // 24,000 m and 50,000 m should clamp to ~6553 m.
        let cfg = WriterConfig {
            n_src: 1,
            n_dst: 3,
            dual_channel: false,
            cell_dtype: CellDtype::U16,
            scale_exp: -1,
            pad_64: false,
            crc32_footer: false,
        };
        let dur = vec![100.0, 6553.0, 50000.0]; // last should clamp to ~6553.4
        let (d, _) = roundtrip(cfg, &dur, None);
        assert!((d[0] - 100.0).abs() < 0.2);
        assert!((d[1] - 6553.0).abs() < 0.2);
        assert!(d[2] > 6500.0 && d[2] <= 6553.5);
    }

    #[test]
    fn pad_64_aligns_rows() {
        // 3 rows, 5 dsts × dual f32 = 40 bytes body, +8 row header = 48 →
        // pad 16 bytes to next 64-byte boundary.
        let cfg = WriterConfig {
            n_src: 3,
            n_dst: 5,
            dual_channel: true,
            cell_dtype: CellDtype::F32,
            scale_exp: 0,
            pad_64: true,
            crc32_footer: false,
        };
        let dur = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dist = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut buf = Vec::new();
        {
            let mut w = BinaryTableWriter::new(&mut buf, cfg).unwrap();
            for i in 0..3 {
                w.write_row(i as u32, &dur, Some(&dist)).unwrap();
            }
            w.finish().unwrap();
        }
        // 32 (header) + 3 × 64 (row + pad) = 224 bytes total.
        assert_eq!(buf.len(), 32 + 3 * 64);
        let mut r = BinaryTableReader::new(Cursor::new(&buf)).unwrap();
        assert!(r.pad_64);
        for i in 0..3 {
            let (idx, d, di) = r.read_row().unwrap().unwrap();
            assert_eq!(idx, i as u32);
            assert_eq!(d, dur);
            assert_eq!(di.unwrap(), dist);
        }
        assert!(r.read_row().unwrap().is_none());
    }

    #[test]
    fn crc32_footer_roundtrip() {
        let cfg = WriterConfig {
            n_src: 2,
            n_dst: 3,
            dual_channel: false,
            cell_dtype: CellDtype::F32,
            scale_exp: 0,
            pad_64: false,
            crc32_footer: true,
        };
        let dur = vec![1.0, 2.0, 3.0];
        let mut buf = Vec::new();
        let crc_written = {
            let mut w = BinaryTableWriter::new(&mut buf, cfg).unwrap();
            w.write_row(0, &dur, None).unwrap();
            w.write_row(1, &dur, None).unwrap();
            w.finish().unwrap()
        };
        assert_ne!(crc_written, 0, "crc must be non-zero for non-trivial input");
        // 32 (header) + 2 × (8 + 12) + 16 (trailer) = 88.
        assert_eq!(buf.len(), 32 + 2 * 20 + 16);
        // Trailer magic at end-16:
        assert_eq!(&buf[buf.len() - 16..buf.len() - 8], CRC_MAGIC);
        let trailer_crc = u32::from_le_bytes(
            buf[buf.len() - 8..buf.len() - 4].try_into().unwrap(),
        );
        assert_eq!(trailer_crc, crc_written);

        // Reader can decode rows and trailer.
        let mut r = BinaryTableReader::new(Cursor::new(&buf)).unwrap();
        assert!(r.crc32_footer);
        let _ = r.read_row().unwrap().unwrap();
        let _ = r.read_row().unwrap().unwrap();
        let crc_read = r.read_trailer().unwrap().unwrap();
        assert_eq!(crc_read, crc_written);
    }

    #[test]
    fn crc32_detects_corruption() {
        let cfg = WriterConfig {
            n_src: 1,
            n_dst: 4,
            dual_channel: false,
            cell_dtype: CellDtype::F32,
            scale_exp: 0,
            pad_64: false,
            crc32_footer: true,
        };
        let dur = vec![1.0, 2.0, 3.0, 4.0];
        let mut buf = Vec::new();
        let crc_clean = {
            let mut w = BinaryTableWriter::new(&mut buf, cfg).unwrap();
            w.write_row(0, &dur, None).unwrap();
            w.finish().unwrap()
        };
        // Flip a byte in the row body and recompute CRC over the original;
        // the consumer would compute the CRC over the on-disk bytes and
        // notice the mismatch.
        let body_start = 32 + 8; // header + row header
        buf[body_start] ^= 0xFF;
        let mut crc_corrupt = 0u32;
        crc_corrupt = crc32_update(crc_corrupt, &buf[..buf.len() - 16]);
        assert_ne!(crc_corrupt, crc_clean);
    }

    #[test]
    fn ut_index_formula() {
        // For N=5: indices for (0,1), (0,2), (0,3), (0,4),
        //                       (1,2), (1,3), (1,4),
        //                       (2,3), (2,4),
        //                       (3,4) → 0..10.
        let expected = [
            (0, 1, 0),
            (0, 2, 1),
            (0, 3, 2),
            (0, 4, 3),
            (1, 2, 4),
            (1, 3, 5),
            (1, 4, 6),
            (2, 3, 7),
            (2, 4, 8),
            (3, 4, 9),
        ];
        for &(i, j, idx) in &expected {
            assert_eq!(ut_index(5, i, j), idx, "({i},{j})");
        }
    }

    #[test]
    fn symmetric_writer_reader_roundtrip() {
        // 4×4 symmetric matrix, 6 UT cells.
        let n = 4u32;
        let cfg = WriterConfig {
            n_src: n,
            n_dst: n,
            dual_channel: true,
            cell_dtype: CellDtype::F32,
            scale_exp: 0,
            pad_64: false,
            crc32_footer: true,
        };
        // Construct full 4×4 dur/dist matrices that are symmetric.
        let dur_full: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0,
            1.0, 0.0, 4.0, 5.0,
            2.0, 4.0, 0.0, 6.0,
            3.0, 5.0, 6.0, 0.0,
        ];
        let dist_full: Vec<f32> = vec![
            0.0, 10.0, 20.0, 30.0,
            10.0, 0.0, 40.0, 50.0,
            20.0, 40.0, 0.0, 60.0,
            30.0, 50.0, 60.0, 0.0,
        ];
        let mut buf = Vec::new();
        {
            let mut w = SymmetricBinaryWriter::new(&mut buf, cfg).unwrap();
            for i in 0..n - 1 {
                let row_dur = &dur_full[(i * n) as usize..((i + 1) * n) as usize];
                let row_dist = &dist_full[(i * n) as usize..((i + 1) * n) as usize];
                w.write_ut_from_full_row(i, row_dur, Some(row_dist)).unwrap();
            }
            w.finish().unwrap();
        }

        let mut r = SymmetricBinaryReader::new(Cursor::new(&buf)).unwrap();
        assert_eq!(r.n, 4);
        assert!(r.dual_channel);
        let expected_pairs = [
            (1.0, 10.0),
            (2.0, 20.0),
            (3.0, 30.0),
            (4.0, 40.0),
            (5.0, 50.0),
            (6.0, 60.0),
        ];
        for &(want_dur, want_dist) in &expected_pairs {
            let (dur, dist) = r.read_cell().unwrap().unwrap();
            assert_eq!(dur, want_dur);
            assert_eq!(dist.unwrap(), want_dist);
        }
        assert!(r.read_cell().unwrap().is_none());
    }
}
