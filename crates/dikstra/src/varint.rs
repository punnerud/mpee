//! Unsigned LEB128 (varint) and zig-zag (signed varint) codecs.
//!
//! For graph-structured payloads where most values are small after
//! delta-encoding (consecutive neighbour IDs, edge weights with little
//! variation across a row), a 7-bits-per-byte varint typically compresses
//! 4× compared to raw `u32` and is decoded with a tiny hot loop.
//!
//! Typical use: serialise a sparse graph section in the route blob:
//!
//! ```text
//! [u32]    n_edges
//! repeat n_edges times:
//!     varint  from_id
//!     varint  to_id
//!     varint  weight   (signed delta, zig-zag encoded)
//! ```
//!
//! Encode `u64` (or signed `i64` via [`zigzag`]) into 1..10 bytes. Decode
//! returns `(value, bytes_consumed)`.

use std::io::{self, Read, Write};

/// Maximum bytes needed to encode `u64` (10 × 7 = 70 bits ≥ 64).
pub const MAX_VARINT_BYTES: usize = 10;

#[inline]
pub fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
pub fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// Encode `value` into `buf`; returns the number of bytes written.
/// `buf` must be at least [`MAX_VARINT_BYTES`] long.
pub fn encode_u64_buf(mut value: u64, buf: &mut [u8]) -> usize {
    let mut n = 0;
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
            buf[n] = byte;
            n += 1;
        } else {
            buf[n] = byte;
            n += 1;
            return n;
        }
    }
}

/// Decode a varint from `buf` starting at `start`. Returns
/// `(value, bytes_consumed)`. Errors if the encoded value exceeds 64 bits.
pub fn decode_u64_buf(buf: &[u8], start: usize) -> io::Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    let mut i = start;
    loop {
        if i >= buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "varint truncated",
            ));
        }
        let byte = buf[i];
        i += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, i - start));
        }
        shift += 7;
        if shift >= 70 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint too long",
            ));
        }
    }
}

/// Write a varint to a `Write`. Returns bytes written.
pub fn write_u64<W: Write>(out: &mut W, value: u64) -> io::Result<usize> {
    let mut buf = [0u8; MAX_VARINT_BYTES];
    let n = encode_u64_buf(value, &mut buf);
    out.write_all(&buf[..n])?;
    Ok(n)
}

/// Write a zig-zag-encoded signed varint.
pub fn write_i64<W: Write>(out: &mut W, value: i64) -> io::Result<usize> {
    write_u64(out, zigzag(value))
}

/// Read a varint from a `Read`. Reads byte-by-byte; fine for buffered
/// readers but not the most efficient on raw file descriptors.
pub fn read_u64<R: Read>(input: &mut R) -> io::Result<u64> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    let mut byte = [0u8; 1];
    loop {
        input.read_exact(&mut byte)?;
        value |= ((byte[0] & 0x7F) as u64) << shift;
        if byte[0] & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 70 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint too long",
            ));
        }
    }
}

pub fn read_i64<R: Read>(input: &mut R) -> io::Result<i64> {
    Ok(unzigzag(read_u64(input)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn small_values_one_byte() {
        for v in 0u64..128 {
            let mut buf = [0u8; MAX_VARINT_BYTES];
            assert_eq!(encode_u64_buf(v, &mut buf), 1);
            let (got, n) = decode_u64_buf(&buf, 0).unwrap();
            assert_eq!(got, v);
            assert_eq!(n, 1);
        }
    }

    #[test]
    fn boundary_values() {
        let cases: &[(u64, usize)] = &[
            (0, 1),
            (127, 1),
            (128, 2),
            (16383, 2),
            (16384, 3),
            (u64::MAX, MAX_VARINT_BYTES),
        ];
        for &(v, expected_bytes) in cases {
            let mut buf = [0u8; MAX_VARINT_BYTES];
            let n = encode_u64_buf(v, &mut buf);
            assert_eq!(n, expected_bytes, "v={v}");
            let (got, m) = decode_u64_buf(&buf, 0).unwrap();
            assert_eq!(got, v);
            assert_eq!(m, expected_bytes);
        }
    }

    #[test]
    fn zigzag_signed_roundtrip() {
        let cases: &[i64] = &[0, -1, 1, -2, 2, -64, 64, i64::MIN, i64::MAX];
        for &v in cases {
            assert_eq!(unzigzag(zigzag(v)), v, "v={v}");
        }
        // Zig-zag should keep small magnitudes in few bytes.
        let mut buf = [0u8; MAX_VARINT_BYTES];
        let n = encode_u64_buf(zigzag(-1), &mut buf);
        assert_eq!(n, 1); // -1 → zz=1 → 1 byte
        let n = encode_u64_buf(zigzag(-128), &mut buf);
        assert_eq!(n, 2);
    }

    #[test]
    fn write_read_roundtrip_via_io() {
        let mut buf = Vec::new();
        write_u64(&mut buf, 42).unwrap();
        write_u64(&mut buf, 1_000_000).unwrap();
        write_i64(&mut buf, -7).unwrap();
        let mut cur = Cursor::new(&buf);
        assert_eq!(read_u64(&mut cur).unwrap(), 42);
        assert_eq!(read_u64(&mut cur).unwrap(), 1_000_000);
        assert_eq!(read_i64(&mut cur).unwrap(), -7);
    }

    #[test]
    fn delta_compression_demo() {
        // Encode a sorted sequence as gaps — the producer-side compression
        // pattern for sparse-graph neighbour lists.
        let values: Vec<u64> = vec![10, 12, 13, 50, 51, 51, 200, 300_000];
        let mut buf = Vec::new();
        let mut prev = 0u64;
        for &v in &values {
            write_u64(&mut buf, v - prev).unwrap();
            prev = v;
        }
        // 8 deltas should fit in well under 16 bytes for this input.
        assert!(buf.len() < 16, "got {} bytes", buf.len());

        let mut cur = Cursor::new(&buf);
        let mut decoded = Vec::with_capacity(values.len());
        let mut running = 0u64;
        for _ in 0..values.len() {
            let gap = read_u64(&mut cur).unwrap();
            running += gap;
            decoded.push(running);
        }
        assert_eq!(decoded, values);
    }
}
