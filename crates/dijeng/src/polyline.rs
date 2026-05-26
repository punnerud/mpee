//! Google "encoded polyline" algorithm for compact route geometry.
//!
//! This is the format OSRM and Google Directions use for `geometry` in
//! their HTTP responses. It encodes a sequence of (lat, lon) into ASCII,
//! with delta-encoding so straight roads compress well.
//!
//! Two precisions in common use:
//!   * **5** (Google default, OSRM's `geometries=polyline`): 1e-5 deg ≈ 1.1 m
//!   * **6** (OSRM's `geometries=polyline6`, Mapbox): 1e-6 deg ≈ 0.11 m

/// Encode a coordinate sequence into a Google polyline string.
/// `precision` is the number of decimal digits preserved (typically 5 or 6).
pub fn encode(coords: &[(f32, f32)], precision: u32) -> String {
    let mul = 10f64.powi(precision as i32);
    let mut out = String::with_capacity(coords.len() * 6);
    let mut prev_lat: i64 = 0;
    let mut prev_lon: i64 = 0;
    for &(lat, lon) in coords {
        let lat_i = (lat as f64 * mul).round() as i64;
        let lon_i = (lon as f64 * mul).round() as i64;
        encode_signed(&mut out, lat_i - prev_lat);
        encode_signed(&mut out, lon_i - prev_lon);
        prev_lat = lat_i;
        prev_lon = lon_i;
    }
    out
}

fn encode_signed(out: &mut String, v: i64) {
    // ZigZag (left-shift, complement if negative).
    let mut v: i64 = if v < 0 { !(v << 1) } else { v << 1 };
    while v >= 0x20 {
        let c = (((v & 0x1f) | 0x20) as u8 + 63) as char;
        out.push(c);
        v >>= 5;
    }
    out.push(((v as u8) + 63) as char);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vector from Google's documentation.
    /// `(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)` →
    /// `_p~iF~ps|U_ulLnnqC_mqNvxq`@`
    #[test]
    fn google_reference_vector() {
        let coords = vec![(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)];
        let encoded = encode(&coords, 5);
        assert_eq!(encoded, "_p~iF~ps|U_ulLnnqC_mqNvxq`@");
    }

    #[test]
    fn empty_input() {
        assert_eq!(encode(&[], 5), "");
    }

    #[test]
    fn single_point() {
        let encoded = encode(&[(38.5, -120.2)], 5);
        // Just the deltas from (0,0) to (38.5, -120.2).
        assert_eq!(encoded, "_p~iF~ps|U");
    }
}
