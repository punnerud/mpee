//! Elevation: SRTM `.hgt` tile reader + grade-aware edge costing.
//!
//! `Dem` reads a directory of SRTM height tiles (`N51W001.hgt` — the format
//! NASA/USGS, Copernicus GLO-30 conversions and viewfinderpanoramas all ship):
//! a square grid of big-endian i16 metres, 3601² (SRTM1, ~30 m) or 1201²
//! (SRTM3, ~90 m) samples per 1°×1° tile, row-major from the NORTH-WEST
//! corner, voids = -32768. No external dependencies.
//!
//! At cache-build time (`--dem <dir>` / `Router.build(dem=...)`):
//!   1. every node gets a bilinear-sampled elevation → `.elev` sidecar
//!      (magic + n + f32 metres; NaN where no tile covers the node),
//!   2. edge travel times get a grade factor — climbing slows you down much
//!      more than descending speeds you up (asymmetry is what makes hilly
//!      routing differ per direction).
//!
//! Grade model (documented, deliberately simple):
//!   * uphill   (g > 0): time × (1 + K_UP · g)        — K_UP = 8
//!   * downhill (g < 0): time × max(0.75, 1 + K_DOWN·g) — K_DOWN = 2
//! where g = Δelev / horizontal_distance, clamped to ±0.30. For cars the
//! effect is mild and disabled by default; bicycle/foot builds apply it
//! automatically when a DEM is given (a 5 % climb costs a cyclist ~40 %
//! extra time — in line with measured cycling power models).

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

const VOID: i16 = -32768;

/// One loaded 1°×1° tile.
struct Tile {
    /// Samples per row/column (1201 or 3601).
    n: usize,
    /// Big-endian i16 decoded to native, row-major from the NW corner.
    data: Vec<i16>,
}

/// A directory of `.hgt` tiles, loaded lazily and cached.
pub struct Dem {
    dir: PathBuf,
    tiles: std::sync::Mutex<HashMap<(i32, i32), Option<std::sync::Arc<Tile>>>>,
}

impl Dem {
    pub fn open(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into(), tiles: std::sync::Mutex::new(HashMap::new()) }
    }

    fn tile(&self, lat_floor: i32, lon_floor: i32) -> Option<std::sync::Arc<Tile>> {
        let key = (lat_floor, lon_floor);
        if let Some(t) = self.tiles.lock().unwrap().get(&key) {
            return t.clone();
        }
        let name = format!(
            "{}{:02}{}{:03}.hgt",
            if lat_floor >= 0 { "N" } else { "S" },
            lat_floor.abs(),
            if lon_floor >= 0 { "E" } else { "W" },
            lon_floor.abs(),
        );
        let loaded = load_hgt(&self.dir.join(&name)).map(std::sync::Arc::new);
        self.tiles.lock().unwrap().insert(key, loaded.clone());
        loaded
    }

    /// Bilinear elevation sample (metres). `None` outside coverage or on voids.
    pub fn sample(&self, lat: f64, lon: f64) -> Option<f32> {
        let lat_f = lat.floor();
        let lon_f = lon.floor();
        let tile = self.tile(lat_f as i32, lon_f as i32)?;
        let n = tile.n;
        // Fractional position inside the tile. Row 0 is the NORTH edge.
        let fx = (lon - lon_f) * (n - 1) as f64;
        let fy = (1.0 - (lat - lat_f)) * (n - 1) as f64;
        let x0 = (fx.floor() as usize).min(n - 2);
        let y0 = (fy.floor() as usize).min(n - 2);
        let dx = (fx - x0 as f64) as f32;
        let dy = (fy - y0 as f64) as f32;
        let at = |x: usize, y: usize| -> Option<f32> {
            let v = tile.data[y * n + x];
            (v != VOID).then_some(v as f32)
        };
        let (a, b, c, d) = (at(x0, y0)?, at(x0 + 1, y0)?, at(x0, y0 + 1)?, at(x0 + 1, y0 + 1)?);
        Some(a * (1.0 - dx) * (1.0 - dy) + b * dx * (1.0 - dy) + c * (1.0 - dx) * dy + d * dx * dy)
    }
}

fn load_hgt(path: &Path) -> Option<Tile> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).ok()?;
    let samples = bytes.len() / 2;
    let n = match samples {
        12_967_201 => 3601, // SRTM1
        1_442_401 => 1201,  // SRTM3
        _ => {
            // Any square grid is accepted (lets tests ship tiny tiles).
            let r = (samples as f64).sqrt() as usize;
            if r * r != samples || r < 2 {
                return None;
            }
            r
        }
    };
    let mut data = Vec::with_capacity(samples);
    for ch in bytes.chunks_exact(2) {
        data.push(i16::from_be_bytes([ch[0], ch[1]]));
    }
    Some(Tile { n, data })
}

/// Grade-adjusted travel-time factor for an edge climbing `delta_m` metres
/// over `dist_m` horizontal metres. ≥ 0.75, 1.0 on flat/unknown.
pub fn grade_time_factor(delta_m: f32, dist_m: f32) -> f32 {
    if !(dist_m > 1.0) || !delta_m.is_finite() {
        return 1.0;
    }
    let g = (delta_m / dist_m).clamp(-0.30, 0.30);
    if g > 0.0 {
        1.0 + 8.0 * g
    } else {
        (1.0 + 2.0 * g).max(0.75)
    }
}

// ── .elev sidecar ────────────────────────────────────────────────────────────

const ELEV_MAGIC: &[u8; 8] = b"SSSPEL1A";

/// Save per-node elevations (`NaN` = unknown) next to the other caches.
pub fn save_elev(path: &Path, elev: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(ELEV_MAGIC)?;
    f.write_all(&(elev.len() as u64).to_le_bytes())?;
    for v in elev {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

pub fn load_elev(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let err = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < 16 || &bytes[..8] != ELEV_MAGIC {
        return Err(err("bad .elev magic"));
    }
    let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    if bytes.len() != 16 + n * 4 {
        return Err(err("bad .elev length"));
    }
    Ok(bytes[16..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a tiny synthetic 3×3 tile: elevation = 100·row + 10·col (from NW).
    fn write_tile(dir: &Path, name: &str) {
        let mut bytes = Vec::new();
        for row in 0..3i16 {
            for col in 0..3i16 {
                bytes.extend_from_slice(&(100 * row + 10 * col).to_be_bytes());
            }
        }
        std::fs::write(dir.join(name), bytes).unwrap();
    }

    #[test]
    fn sample_bilinear_and_voids() {
        let dir = std::env::temp_dir().join("dijeng_dem_test");
        std::fs::create_dir_all(&dir).unwrap();
        write_tile(&dir, "N51W001.hgt");
        let dem = Dem::open(&dir);
        // NW corner of tile N51W001 = (lat 52, lon -1): row 0, col 0 → 0 m.
        let nw = dem.sample(51.9999999, -0.9999999).unwrap();
        assert!(nw.abs() < 1.0, "nw={nw}");
        // SE corner → row 2, col 2 → 220 m.
        let se = dem.sample(51.0000001, -0.0000001).unwrap();
        assert!((se - 220.0).abs() < 1.0, "se={se}");
        // Tile centre → row 1, col 1 → 110 m.
        let mid = dem.sample(51.5, -0.5).unwrap();
        assert!((mid - 110.0).abs() < 1.0, "mid={mid}");
        // Outside coverage → None.
        assert!(dem.sample(40.5, -0.5).is_none());
    }

    #[test]
    fn grade_factors() {
        assert!((grade_time_factor(0.0, 100.0) - 1.0).abs() < 1e-6);
        // 5% climb → 1.4×.
        assert!((grade_time_factor(5.0, 100.0) - 1.4).abs() < 1e-6);
        // 5% descent → 0.9×.
        assert!((grade_time_factor(-5.0, 100.0) - 0.9).abs() < 1e-6);
        // Steep descent floors at 0.75.
        assert!((grade_time_factor(-30.0, 100.0) - 0.75).abs() < 1e-6);
        // Degenerate distances are neutral.
        assert!((grade_time_factor(10.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn elev_sidecar_roundtrip() {
        let path = std::env::temp_dir().join("dijeng_test.elev");
        let data = vec![1.5f32, f32::NAN, -3.0];
        save_elev(&path, &data).unwrap();
        let back = load_elev(&path).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back[0], 1.5);
        assert!(back[1].is_nan());
        assert_eq!(back[2], -3.0);
    }
}
