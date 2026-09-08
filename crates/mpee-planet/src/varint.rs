//! LEB128 varints for the intermediate build streams.
//!
//! The pass-1 way stream is ~6 GB on a planet even after varint packing; fixed
//! 64-bit fields would triple that and every byte here is read back twice.

#[inline]
pub fn put_u(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

#[inline]
pub fn put_i(buf: &mut Vec<u8>, v: i64) {
    put_u(buf, ((v << 1) ^ (v >> 63)) as u64);
}

#[inline]
pub fn get_u(b: &[u8], p: &mut usize) -> u64 {
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
    }
}

#[inline]
pub fn get_i(b: &[u8], p: &mut usize) -> i64 {
    let u = get_u(b, p);
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

#[inline]
pub fn put_bytes(buf: &mut Vec<u8>, s: &[u8]) {
    put_u(buf, s.len() as u64);
    buf.extend_from_slice(s);
}

#[inline]
pub fn get_bytes<'a>(b: &'a [u8], p: &mut usize) -> &'a [u8] {
    let l = get_u(b, p) as usize;
    let s = &b[*p..*p + l];
    *p += l;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_across_the_size_boundaries() {
        let vals: [u64; 9] = [0, 1, 127, 128, 300, 16_383, 16_384, u32::MAX as u64, u64::MAX];
        let mut b = Vec::new();
        for &v in &vals {
            put_u(&mut b, v);
        }
        let mut p = 0;
        for &v in &vals {
            assert_eq!(get_u(&b, &mut p), v);
        }
        assert_eq!(p, b.len());
    }

    #[test]
    fn signed_roundtrips_including_extremes() {
        let vals: [i64; 8] = [0, -1, 1, -64, 64, i32::MIN as i64, i64::MIN, i64::MAX];
        let mut b = Vec::new();
        for &v in &vals {
            put_i(&mut b, v);
        }
        let mut p = 0;
        for &v in &vals {
            assert_eq!(get_i(&b, &mut p), v);
        }
    }

    #[test]
    fn small_negative_deltas_stay_one_byte() {
        // Way refs delta-code to small signed numbers; that is the whole point.
        let mut b = Vec::new();
        put_i(&mut b, -3);
        assert_eq!(b.len(), 1);
    }
}
