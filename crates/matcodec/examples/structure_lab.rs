//! Quick experiments for two candidate MTZT improvements — measurement only,
//! no format change:
//!
//!   cargo run --release -p matcodec --example structure_lab -- varint  [matrix.json|gateway:N]
//!   cargo run --release -p matcodec --example structure_lab -- pyramid [matrix.json|gateway:N]
//!
//! `varint`: polyline-style encodings (zigzag + varint, optional cell-grouped
//! delta) of the residual frames and the resident tables, vs today's plain
//! deflate-over-raw-i32.
//!
//! `pyramid`: a 2-level "road hierarchy" index — each point keeps only its k
//! nearest hubs out of H, plus a dense H×H hub matrix; base(i,j) =
//! min over (a in hubs(i), b in hubs(j)) of d(i,a) + d(a,b) + d(b,j).
//! Compared against today's flat n×L table at equal and at much larger
//! per-point memory.

use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write;

fn deflate_len(raw: &[u8]) -> usize {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::best());
    e.write_all(raw).unwrap();
    e.finish().unwrap().len()
}

fn i32s_le(v: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

fn varint_push(out: &mut Vec<u8>, mut x: u32) {
    loop {
        let b = (x & 0x7f) as u8;
        x >>= 7;
        if x == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

fn zz_varint(vals: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        varint_push(&mut out, zigzag(v));
    }
    out
}

// ---------------------------------------------------------------- worlds
fn load_json_matrix(path: &str) -> (Vec<i32>, usize) {
    let txt = std::fs::read_to_string(path).expect("read matrix json");
    let body = txt.trim().trim_start_matches("{\"durations\":").trim_end_matches('}');
    let mut rows: Vec<Vec<i32>> = Vec::new();
    for row in body.split("],") {
        let cells: Vec<i32> = row
            .matches(|c: char| c.is_ascii_digit() || c == '-' || c == ',')
            .collect::<String>()
            .split(',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().expect("cell"))
            .collect();
        if !cells.is_empty() {
            rows.push(cells);
        }
    }
    let n = rows.len();
    let mut d = vec![0i32; n * n];
    for (i, r) in rows.iter().enumerate() {
        d[i * n..i * n + n].copy_from_slice(r);
    }
    (d, n)
}

fn gateway_world(n: usize, regions: usize) -> (Vec<i32>, usize) {
    let per = n / regions;
    let n = per * regions;
    let mut s: u64 = 0xC0FFEE;
    let mut rnd = |range: i64| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as i64) % range
    };
    let mut pts = vec![(0i64, 0i64); n];
    for r in 0..regions {
        let c = ((r % 4) as i64 * 100_000, (r / 4) as i64 * 100_000);
        for i in 0..per {
            pts[r * per + i] = (c.0 + rnd(4000), c.1 + rnd(4000));
        }
    }
    let l1 = |a: (i64, i64), b: (i64, i64)| (a.0 - b.0).abs() + (a.1 - b.1).abs();
    let gw = |r: usize, k: usize| r * per + k;
    let road = 30_000i64;
    let mut d = vec![0i32; n * n];
    for i in 0..n {
        for j in 0..n {
            let (ri, rj) = (i / per, j / per);
            let v = if ri == rj {
                l1(pts[i], pts[j])
            } else {
                (0..3)
                    .map(|k| l1(pts[i], pts[gw(ri, k)]) + road + l1(pts[gw(rj, k)], pts[j]))
                    .min()
                    .unwrap()
            };
            d[i * n + j] = v as i32;
        }
    }
    (d, n)
}

fn load_world(arg: &str) -> (Vec<i32>, usize) {
    if let Some(k) = arg.strip_prefix("gateway:") {
        gateway_world(k.parse().expect("gateway:N"), 8)
    } else {
        load_json_matrix(arg)
    }
}

// ------------------------------------------------------------ shared model
struct Bridge {
    lm: Vec<usize>,
    l: usize,
    dil: Vec<i32>,  // n×l
    dlj: Vec<i32>,  // l×n
    resid: Vec<i32>, // n×n
}

fn bridge_model(d: &[i32], n: usize, l: usize) -> Bridge {
    let lm = matcodec::pick_landmarks(d, n, l);
    let l = lm.len();
    let mut dil = vec![0i32; n * l];
    let mut dlj = vec![0i32; l * n];
    for (a, &la) in lm.iter().enumerate() {
        for i in 0..n {
            dil[i * l + a] = d[i * n + la];
        }
        dlj[a * n..a * n + n].copy_from_slice(&d[la * n..la * n + n]);
    }
    let mut resid = vec![0i32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut base = i64::MAX;
            for a in 0..l {
                let v = dil[i * l + a] as i64 + dlj[a * n + j] as i64;
                if v < base {
                    base = v;
                }
            }
            resid[i * n + j] = (d[i * n + j] as i64 - base) as i32;
        }
    }
    Bridge { lm, l, dil, dlj, resid }
}

/// Voronoi cell of each column (same rule as matcodec's assign_cells).
fn cells(dlj: &[i32], n: usize, l: usize) -> Vec<u8> {
    let mut c = vec![0u8; n];
    for j in 0..n {
        let mut best = i64::MAX;
        for a in 0..l {
            if (dlj[a * n + j] as i64) < best {
                best = dlj[a * n + j] as i64;
                c[j] = a as u8;
            }
        }
    }
    c
}

// ------------------------------------------------------------------ varint
fn varint_experiment(d: &[i32], n: usize) {
    let l = 32usize.min(n - 1);
    let b = bridge_model(d, n, l);
    let cell_of = cells(&b.dlj, n, b.l);
    // column order grouped by cell (spatial-ish order without storing a perm
    // would not be possible — this measures the *potential* of reordering)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&j| (cell_of[j], b.dlj[(cell_of[j] as usize) * n + j]));

    println!("== varint lab  n={n} L={} ==", b.l);

    // --- residual frames (the bulk of the blob) ---
    let mut cur = 0usize; // today: deflate(raw i32) per row
    let mut vz = 0usize; // zigzag varint, no deflate
    let mut vzd = 0usize; // zigzag varint + deflate
    let mut grp_d = 0usize; // cell-grouped column order, delta in group, varint + deflate
    for i in 0..n {
        let row = &b.resid[i * n..i * n + n];
        cur += deflate_len(&i32s_le(row)) + 4;
        let v = zz_varint(row);
        vzd += deflate_len(&v) + 4;
        vz += v.len() + 4;
        let mut grouped = Vec::with_capacity(n);
        let mut prev = 0i32;
        let mut prev_cell = 255u8;
        for &j in &order {
            if cell_of[j] != prev_cell {
                prev = 0;
                prev_cell = cell_of[j];
            }
            grouped.push(row[j].wrapping_sub(prev));
            prev = row[j];
        }
        grp_d += deflate_len(&zz_varint(&grouped)) + 4;
    }
    println!("  resid frames:  deflate(raw)={:.2} MB   zzvarint={:.2} MB   zzvarint+deflate={:.2} MB   cellgroup-delta+vz+deflate={:.2} MB",
        cur as f64 / 1e6, vz as f64 / 1e6, vzd as f64 / 1e6, grp_d as f64 / 1e6);

    // --- resident tables ---
    let dil_cur = deflate_len(&i32s_le(&b.dil));
    let dil_vzd = deflate_len(&zz_varint(&b.dil));
    // delta over landmarks per point after sorting each point's landmark dists?
    // order must be reconstructible — use delta vs previous POINT (same landmark):
    let mut dil_dp = vec![0i32; n * b.l];
    for i in 0..n {
        for a in 0..b.l {
            let prev = if i == 0 { 0 } else { b.dil[(i - 1) * b.l + a] };
            dil_dp[i * b.l + a] = b.dil[i * b.l + a].wrapping_sub(prev);
        }
    }
    let dil_dpd = deflate_len(&zz_varint(&dil_dp));
    println!("  dil (n×L):     deflate(raw)={:.0} KB   zzvarint+deflate={:.0} KB   point-delta+vz+deflate={:.0} KB",
        dil_cur as f64 / 1e3, dil_vzd as f64 / 1e3, dil_dpd as f64 / 1e3);

    let dlj_cur = deflate_len(&i32s_le(&b.dlj));
    // delta along the cell-grouped column order per landmark row
    let mut dlj_g = vec![0i32; b.l * n];
    for a in 0..b.l {
        let mut prev = 0i32;
        let mut prev_cell = 255u8;
        for (idx, &j) in order.iter().enumerate() {
            if cell_of[j] != prev_cell {
                prev = 0;
                prev_cell = cell_of[j];
            }
            dlj_g[a * n + idx] = b.dlj[a * n + j].wrapping_sub(prev);
            prev = b.dlj[a * n + j];
        }
    }
    let dlj_gd = deflate_len(&zz_varint(&dlj_g));
    println!("  dlj (L×n):     deflate(raw)={:.0} KB   cellgroup-delta+vz+deflate={:.0} KB",
        dlj_cur as f64 / 1e3, dlj_gd as f64 / 1e3);

    let total_cur = cur + dil_cur + dlj_cur;
    let total_new = grp_d.min(vzd) + dil_dpd.min(dil_vzd) + dlj_gd;
    println!("  TOTAL:         today {:.2} MB  ->  best-variant {:.2} MB  ({:.1}% mindre)",
        total_cur as f64 / 1e6, total_new as f64 / 1e6,
        100.0 * (1.0 - total_new as f64 / total_cur as f64));
}

// ----------------------------------------------------------------- pyramid
/// 2-level hub index: H hubs (pivot-mined), each point keeps its k nearest
/// out-hubs (by d(i,h)) and in-hubs (by d(h,i)); dense H×H hub matrix.
fn pyramid_stats(d: &[i32], n: usize, h_count: usize, k: usize) -> (f64, [f64; 4], usize) {
    let hubs = matcodec::pick_landmarks(d, n, h_count);
    let h = hubs.len();
    // per-point k nearest out/in hubs
    let mut out_h = vec![0u8; n * k]; // hub indices
    let mut out_d = vec![0i32; n * k];
    let mut in_h = vec![0u8; n * k];
    let mut in_d = vec![0i32; n * k];
    for i in 0..n {
        let mut by_out: Vec<(i32, usize)> =
            hubs.iter().enumerate().map(|(a, &ha)| (d[i * n + ha], a)).collect();
        by_out.sort();
        let mut by_in: Vec<(i32, usize)> =
            hubs.iter().enumerate().map(|(a, &ha)| (d[ha * n + i], a)).collect();
        by_in.sort();
        for q in 0..k {
            out_h[i * k + q] = by_out[q].1 as u8;
            out_d[i * k + q] = by_out[q].0;
            in_h[i * k + q] = by_in[q].1 as u8;
            in_d[i * k + q] = by_in[q].0;
        }
    }
    let mut dhh = vec![0i32; h * h];
    for (a, &ha) in hubs.iter().enumerate() {
        for (bb, &hb) in hubs.iter().enumerate() {
            dhh[a * h + bb] = d[ha * n + hb];
        }
    }
    // residual stats over all cells
    let mut exact = 0usize;
    let mut within = [0usize; 4]; // 2,5,15,60
    let tols = [2i64, 5, 15, 60];
    for i in 0..n {
        for j in 0..n {
            let mut base = i64::MAX;
            for q in 0..k {
                let a = out_h[i * k + q] as usize;
                let da = out_d[i * k + q] as i64;
                for r in 0..k {
                    let bb = in_h[j * k + r] as usize;
                    let v = da + dhh[a * h + bb] as i64 + in_d[j * k + r] as i64;
                    if v < base {
                        base = v;
                    }
                }
            }
            let resid = (d[i * n + j] as i64 - base).abs();
            if resid == 0 {
                exact += 1;
            }
            for (t, &tol) in tols.iter().enumerate() {
                if resid <= tol {
                    within[t] += 1;
                }
            }
        }
    }
    let total = (n * n) as f64;
    // resident bytes: per point 2×k×(4+1) + dense hub matrix
    let bytes = n * 2 * k * 5 + h * h * 4;
    (
        exact as f64 / total,
        [
            within[0] as f64 / total,
            within[1] as f64 / total,
            within[2] as f64 / total,
            within[3] as f64 / total,
        ],
        bytes,
    )
}

/// Flat landmark base (today's model) — share stats + resident bytes.
fn flat_stats(d: &[i32], n: usize, l: usize) -> (f64, [f64; 4], usize) {
    let b = bridge_model(d, n, l);
    let mut exact = 0usize;
    let mut within = [0usize; 4];
    let tols = [2i32, 5, 15, 60];
    for &r in &b.resid {
        let r = r.abs();
        if r == 0 {
            exact += 1;
        }
        for (t, &tol) in tols.iter().enumerate() {
            if r <= tol {
                within[t] += 1;
            }
        }
    }
    let total = (n * n) as f64;
    let bytes = 2 * b.l * n * 4 + b.l * n + 2 * n; // dlj + dil + blockmax + cell_of/rowmax
    (
        exact as f64 / total,
        [
            within[0] as f64 / total,
            within[1] as f64 / total,
            within[2] as f64 / total,
            within[3] as f64 / total,
        ],
        bytes,
    )
}

fn pyramid_experiment(d: &[i32], n: usize) {
    println!("== pyramid lab  n={n}  (andel celler eksakt / innen 2/5/15/60) ==");
    for (label, ex, w, bytes) in [
        ("flat  L=8           ", flat_stats(d, n, 8)),
        ("flat  L=32          ", flat_stats(d, n, 32)),
        ("flat  L=64          ", flat_stats(d, n, 64)),
        ("pyr   H=64  k=2     ", pyramid_stats(d, n, 64, 2)),
        ("pyr   H=64  k=4     ", pyramid_stats(d, n, 64, 4)),
        ("pyr   H=128 k=4     ", pyramid_stats(d, n, 128, 4)),
        ("pyr   H=128 k=8     ", pyramid_stats(d, n, 128, 8)),
    ]
    .map(|(s, (a, b, c))| (s, a, b, c))
    {
        println!(
            "  {label} resident {:>6.0} KB   exact {:>5.1}%   tol {:>5.1}/{:>5.1}/{:>5.1}/{:>5.1}%",
            bytes as f64 / 1e3,
            100.0 * ex,
            100.0 * w[0],
            100.0 * w[1],
            100.0 * w[2],
            100.0 * w[3]
        );
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "varint".into());
    let world = std::env::args().nth(2).unwrap_or_else(|| "gateway:3000".into());
    let (d, n) = load_world(&world);
    match mode.as_str() {
        "varint" => varint_experiment(&d, n),
        "pyramid" => pyramid_experiment(&d, n),
        other => eprintln!("unknown mode {other} — use varint|pyramid"),
    }
}
