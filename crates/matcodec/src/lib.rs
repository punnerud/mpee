//! Lossless structural codec for travel-time / distance matrices.
//!
//! Road networks between separated regions are connected by few "gateways"
//! (bridges, passes, single highways), so a cross-region block
//! `D[a][b] = d(a, gw) + HWY + d(gw, b)` is **additive rank-1** (one gateway) or
//! min-plus rank-k (k gateways). We exploit that:
//!
//! 1. Cluster the points into regions (k-medoids on the symmetrised matrix).
//! 2. Diagonal (intra-region) blocks: store dense (local detail), deflated.
//! 3. Off-diagonal (cross-region) blocks: store a rank-1 base
//!    `base[p][q] = col0[p] + row0[q] - c00` plus an **exact residual**, deflated.
//!
//! Lossless by construction — the residual restores anything the base misses, so
//! `decompress(compress(D)) == D` byte for byte. The win comes from the residual
//! having near-zero entropy when real bottleneck structure exists (it collapses to
//! ±1 integer-rounding noise). No structure ⇒ residual is the full block ⇒ it
//! degrades gracefully to roughly plain deflate. Measured: ~6.4× on a real OSRM
//! road matrix across 8 Norwegian cities, ~10× on single-gateway synthetic worlds,
//! ~1.8× on structureless uniform points.
//!
//! The streamed `MTZT` container additionally doubles as an **in-memory query
//! index**: the per-point landmark distances (the "which of the few roads out
//! of each region do I use" table) plus a per-(row, landmark-cell)
//! max-|residual| byte stay resident in [`MtzReader`], so blocks the min-plus
//! base reproduces exactly are answered in O(L) with zero decompression, and
//! every cell gets O(L) lower/upper bounds ([`MtzReader::cell_bounds`]) or
//! tolerance-bounded values ([`MtzReader::cell_within`]) for solver pruning.
//! Landmarks are picked by greedy pivot mining ([`pick_landmarks`]), which
//! converges on the actual gateways when they exist.

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::{self, Read, Write};

const MAGIC: &[u8; 4] = b"MTZ2";
/// Legacy streamed container (decode only): framed per-row with the landmark
/// distances embedded in each frame, so even one cell needs a frame inflated.
const MAGIC_STREAM: &[u8; 4] = b"MTZS";
/// Streamed container v2: frames hold only the residual row; the per-point
/// landmark distances (`dil`, the "gateway index") and a per-(row,
/// landmark-cell) max-|residual| byte table live in a footer that
/// [`MtzReader`] keeps resident. Blocks whose residual is all-zero — the
/// cross-region blocks when real bottleneck structure exists — are then
/// answered in O(L) straight from the index, no inflate.
/// Peak encode memory is the L landmark rows + the n×L index + one working row.
const MAGIC_STREAM2: &[u8; 4] = b"MTZT";
/// Streamed container v3: the flat n×L landmark table is replaced by
/// **directional path labels** — per point, the k hubs most often on its
/// shortest paths (mined from sampled pairs), out- and in-side separately,
/// plus a dense H×H hub matrix. Residual frames are zigzag-varint coded
/// (optionally cell-grouped delta, chosen per blob) before deflate.
/// Measured: beats the MTZT base at a fraction of the resident memory and
/// compresses 26–36 % smaller (see docs/matcodec-gateway-index.md).
const MAGIC_STREAM3: &[u8; 4] = b"MTZU";
const METHOD_CLUSTER: u8 = 0;
const METHOD_BRIDGE: u8 = 1;
/// Landmark counts the bridge model sweeps (best-of, capped to < n).
const BRIDGE_LS: [usize; 4] = [8, 16, 32, 64];
/// A cell at/above this is treated as "unreachable" (routing/snapping failure).
pub const UNREACHABLE: i32 = 1_000_000_000;
/// Triangle violations up to this magnitude are integer-rounding noise (a true
/// rounded metric violates by ≤1), not data errors, so they are not flagged.
const TRIANGLE_TOL: i64 = 2;

// ---------------------------------------------------------------- byte helpers
fn deflate(raw: &[u8]) -> Vec<u8> {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::best());
    e.write_all(raw).expect("deflate");
    e.finish().expect("deflate finish")
}

fn inflate(comp: &[u8]) -> io::Result<Vec<u8>> {
    let mut d = ZlibDecoder::new(comp);
    let mut out = Vec::new();
    d.read_to_end(&mut out)?;
    Ok(out)
}

fn i32s_to_le(v: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn le_to_i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn put_blob(out: &mut Vec<u8>, raw: &[u8]) {
    let comp = deflate(raw);
    out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
    out.extend_from_slice(&comp);
}

// ------------------------------------------------------------- varint coding
/// Zigzag + LEB128 varint: near-zero residuals cost 1 byte instead of 4
/// *before* deflate — measured 26–36 % smaller blobs than deflate-over-raw.
fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

fn unzigzag(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

fn zz_varint(vals: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let mut x = zigzag(v);
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
    out
}

fn zz_varint_decode(bytes: &[u8], expect: usize) -> io::Result<Vec<i32>> {
    let mut out = Vec::with_capacity(expect);
    let mut x: u32 = 0;
    let mut shift = 0u32;
    for &b in bytes {
        x |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            out.push(unzigzag(x));
            x = 0;
            shift = 0;
        } else {
            shift += 7;
            if shift > 28 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "varint overflow"));
            }
        }
    }
    if shift != 0 || out.len() != expect {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "varint stream length"));
    }
    Ok(out)
}

/// Length-prefixed deflate of the zigzag-varint coding of `vals`.
fn put_blob_vz(out: &mut Vec<u8>, vals: &[i32]) {
    put_blob(out, &zz_varint(vals));
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn u32(&mut self) -> io::Result<u32> {
        if self.pos + 4 > self.b.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "u32"));
        }
        let v = u32::from_le_bytes([
            self.b[self.pos],
            self.b[self.pos + 1],
            self.b[self.pos + 2],
            self.b[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }
    fn i32(&mut self) -> io::Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn blob_raw(&mut self) -> io::Result<Vec<u8>> {
        let len = self.u32()? as usize;
        if self.pos + len > self.b.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "blob"));
        }
        let raw = inflate(&self.b[self.pos..self.pos + len])?;
        self.pos += len;
        Ok(raw)
    }
    fn blob(&mut self) -> io::Result<Vec<i32>> {
        Ok(le_to_i32s(&self.blob_raw()?))
    }
    /// Inverse of [`put_blob_vz`].
    fn blob_vz(&mut self, expect: usize) -> io::Result<Vec<i32>> {
        let raw = self.blob_raw()?;
        zz_varint_decode(&raw, expect)
    }
}

// ---------------------------------------------------------------- clustering
/// k-medoids on the symmetrised matrix. Deterministic (fixed-seed LCG), so the
/// same matrix always yields the same partition (and the partition is stored, so
/// the decoder never reruns this).
fn kmedoids(d: &[i32], n: usize, k: usize, iters: usize) -> Vec<u32> {
    if k <= 1 || n == 0 {
        return vec![0u32; n];
    }
    let k = k.min(n);
    // symmetrised distances, i64 to avoid overflow on sums
    let sym = |i: usize, j: usize| -> i64 {
        (d[i * n + j] as i64 + d[j * n + i] as i64) / 2
    };
    // LCG for a reproducible distinct-medoid init
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut medoids: Vec<usize> = Vec::with_capacity(k);
    while medoids.len() < k {
        let cand = next() % n;
        if !medoids.contains(&cand) {
            medoids.push(cand);
        }
    }
    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        // assign each point to nearest medoid
        for i in 0..n {
            let mut best = i64::MAX;
            let mut bc = 0u32;
            for (c, &m) in medoids.iter().enumerate() {
                let dv = sym(i, m);
                if dv < best {
                    best = dv;
                    bc = c as u32;
                }
            }
            assign[i] = bc;
        }
        // update medoid of each cluster = member minimising sum of dist to members
        let mut changed = false;
        for c in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| assign[i] == c as u32).collect();
            if members.is_empty() {
                continue;
            }
            let mut best_cost = i64::MAX;
            let mut best_m = medoids[c];
            for &cand in &members {
                let mut cost = 0i64;
                for &x in &members {
                    cost += sym(cand, x);
                }
                if cost < best_cost {
                    best_cost = cost;
                    best_m = cand;
                }
            }
            if best_m != medoids[c] {
                medoids[c] = best_m;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // final assignment
    for i in 0..n {
        let mut best = i64::MAX;
        let mut bc = 0u32;
        for (c, &m) in medoids.iter().enumerate() {
            let dv = sym(i, m);
            if dv < best {
                best = dv;
                bc = c as u32;
            }
        }
        assign[i] = bc;
    }
    assign
}

/// Non-empty groups (member index lists) derived from a label vector, in label
/// order. Both encoder and decoder derive groups this way ⇒ identical layout.
fn groups_from(assign: &[u32], k: usize) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for label in 0..k as u32 {
        let g: Vec<usize> = (0..assign.len())
            .filter(|&i| assign[i] == label)
            .collect();
        if !g.is_empty() {
            groups.push(g);
        }
    }
    groups
}

/// A reasonable default cluster count for an N×N matrix.
pub fn default_k(n: usize) -> usize {
    (n / 40).clamp(2, 64)
}

/// Pick `l` landmarks for streaming compression / the bridge model by **pivot
/// mining**: greedily choose the points that minimise the min-plus residual
/// over a deterministic sample of (i,j) pairs. On gateway-structured data this
/// converges on the actual gateways (the "3 roads" joining regions), which is
/// what makes whole blocks index-exact; on structureless data it degrades to a
/// k-medians-like spread. O(l·n·S) with S = sample size.
pub fn pick_landmarks(d: &[i32], n: usize, l: usize) -> Vec<usize> {
    let l = l.min(n);
    if n < 3 || l == 0 {
        return (0..l.max(1).min(n)).collect();
    }
    let s_count = (n * 4).clamp(512, 4096);
    let mut state: u64 = 0xA5A5_5A5A_DEAD_BEEF;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut pairs = Vec::with_capacity(s_count);
    while pairs.len() < s_count {
        let i = next() % n;
        let j = next() % n;
        if i != j {
            pairs.push((i, j));
        }
    }
    // greedy facility location: each new landmark x minimises
    // Σ_s min(cur[s], d(i,x)+d(x,j)) over the sampled pairs
    let mut cur = vec![i64::MAX; s_count];
    let mut chosen: Vec<usize> = Vec::with_capacity(l);
    let mut used = vec![false; n];
    for _ in 0..l {
        let mut best_x = usize::MAX;
        let mut best_cost = i64::MAX;
        for x in 0..n {
            if used[x] {
                continue;
            }
            let mut cost = 0i64;
            for (s, &(i, j)) in pairs.iter().enumerate() {
                let via = d[i * n + x] as i64 + d[x * n + j] as i64;
                cost += cur[s].min(via);
            }
            if cost < best_cost {
                best_cost = cost;
                best_x = x;
            }
        }
        if best_x == usize::MAX {
            break;
        }
        used[best_x] = true;
        chosen.push(best_x);
        for (s, &(i, j)) in pairs.iter().enumerate() {
            let via = d[i * n + best_x] as i64 + d[best_x * n + j] as i64;
            cur[s] = cur[s].min(via);
        }
    }
    chosen
}

// ---------------------------------------------------------------- landmarks
/// Farthest-point sampling: the "max-distance" points make the best bridges /
/// landmarks. Start at an endpoint of the global widest pair, then greedily add
/// the point farthest from everything chosen so far.
fn farthest_landmarks(d: &[i32], n: usize, l_count: usize) -> Vec<usize> {
    let l_count = l_count.min(n);
    let sym = |i: usize, j: usize| -> i64 { (d[i * n + j] as i64 + d[j * n + i] as i64) / 2 };
    // i0 = a point in the globally widest pair (max row-max)
    let mut i0 = 0usize;
    let mut best = i64::MIN;
    for i in 0..n {
        let mut rm = i64::MIN;
        for j in 0..n {
            rm = rm.max(sym(i, j));
        }
        if rm > best {
            best = rm;
            i0 = i;
        }
    }
    let mut chosen = vec![i0];
    let mut mind: Vec<i64> = (0..n).map(|i| sym(i, i0)).collect();
    while chosen.len() < l_count {
        let mut nxt = 0usize;
        let mut bv = i64::MIN;
        for i in 0..n {
            if mind[i] > bv {
                bv = mind[i];
                nxt = i;
            }
        }
        chosen.push(nxt);
        for i in 0..n {
            let s = sym(i, nxt);
            if s < mind[i] {
                mind[i] = s;
            }
        }
    }
    chosen
}

// ---------------------------------------------------------------- encode
/// Cluster model payload: `[k u32][assign blob][per-block ...]`.
fn encode_cluster(d: &[i32], n: usize, k: usize) -> Vec<u8> {
    let assign = kmedoids(d, n, k, 8);
    let groups = groups_from(&assign, k);

    let mut out = Vec::new();
    out.extend_from_slice(&(k as u32).to_le_bytes());
    put_blob(&mut out, &i32s_to_le(&assign.iter().map(|&x| x as i32).collect::<Vec<_>>()));

    for (gi_idx, gi) in groups.iter().enumerate() {
        for (gj_idx, gj) in groups.iter().enumerate() {
            let (ri, cj) = (gi.len(), gj.len());
            // local block, row-major
            let mut block = vec![0i32; ri * cj];
            for (p, &gp) in gi.iter().enumerate() {
                for (q, &gq) in gj.iter().enumerate() {
                    block[p * cj + q] = d[gp * n + gq];
                }
            }
            if gi_idx == gj_idx {
                // diagonal: dense
                put_blob(&mut out, &i32s_to_le(&block));
            } else {
                // off-diagonal: rank-1 base + exact residual
                let c00 = block[0];
                let mut col0 = vec![0i32; ri];
                for p in 0..ri {
                    col0[p] = block[p * cj]; // B[p][0]
                }
                let row0 = block[0..cj].to_vec(); // B[0][q]
                let mut resid = vec![0i32; ri * cj];
                for p in 0..ri {
                    for q in 0..cj {
                        let base = col0[p] + row0[q] - c00;
                        resid[p * cj + q] = block[p * cj + q] - base;
                    }
                }
                put_blob(&mut out, &i32s_to_le(&col0));
                put_blob(&mut out, &i32s_to_le(&row0));
                out.extend_from_slice(&c00.to_le_bytes());
                put_blob(&mut out, &i32s_to_le(&resid));
            }
        }
    }
    out
}

/// Bridge model payload: `[lm blob][Dil blob][Dlj blob][resid blob]`, where
/// `base(i,j) = min_l d(i,l)+d(l,j)` over the landmarks and `resid = D - base`
/// (≤ 0, exact). L is recovered from the landmark blob length.
fn encode_bridge(d: &[i32], n: usize, l_count: usize) -> Vec<u8> {
    let lm = farthest_landmarks(d, n, l_count);
    let l = lm.len();
    // Dil[i*l+a] = d(i, lm[a]) ; Dlj[a*n+j] = d(lm[a], j)
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
    let mut out = Vec::new();
    put_blob(&mut out, &i32s_to_le(&lm.iter().map(|&x| x as i32).collect::<Vec<_>>()));
    put_blob(&mut out, &i32s_to_le(&dil));
    put_blob(&mut out, &i32s_to_le(&dlj));
    put_blob(&mut out, &i32s_to_le(&resid));
    out
}

/// Compress a row-major `n×n` matrix into the `.mtz` byte stream. Tries the
/// cluster model (with `k` regions, see [`default_k`]) and the bridge model
/// (sweeping a few landmark counts) and keeps whichever is smaller — always
/// lossless. The chosen model is tagged in a single header byte.
pub fn compress(d: &[i32], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(d.len(), n * n, "matrix must be n*n");
    let dbg = std::env::var("MATCODEC_DEBUG").is_ok();
    let mut best_method = METHOD_CLUSTER;
    let mut best_payload = encode_cluster(d, n, k);
    if dbg {
        eprintln!("  cluster(k={k}): {} bytes", best_payload.len());
    }
    for &l in BRIDGE_LS.iter() {
        if l >= n {
            continue;
        }
        let p = encode_bridge(d, n, l);
        if dbg {
            eprintln!("  bridge(L={l}): {} bytes", p.len());
        }
        if p.len() < best_payload.len() {
            best_payload = p;
            best_method = METHOD_BRIDGE;
        }
    }
    let mut out = Vec::with_capacity(best_payload.len() + 16);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.push(best_method);
    out.extend_from_slice(&best_payload);
    out
}

// ---------------------------------------------------------------- validation
/// Anomalies found in a matrix. The metric checks (`triangle`, `asymmetric`)
/// double as a *safety gate*: when they fire, triangle-inequality shortcuts
/// (ALT pruning, the bridge predictor) are unsafe and callers should fall back
/// to value-driven methods. The data checks (`negative`, `self_nonzero`,
/// `unreachable`) usually signal a real input bug.
#[derive(Debug, Default, Clone)]
pub struct ValidationReport {
    pub rows_seen: usize,
    pub negative: usize,
    pub self_nonzero: usize,
    pub unreachable: usize,
    pub asymmetric: usize,
    pub triangle_violation: usize,
    /// Capped human-readable examples (with offending indices) for debugging.
    pub examples: Vec<String>,
}

impl ValidationReport {
    fn note(&mut self, field: u8, msg: impl FnOnce() -> String) {
        match field {
            0 => self.negative += 1,
            1 => self.self_nonzero += 1,
            2 => self.unreachable += 1,
            3 => self.asymmetric += 1,
            _ => self.triangle_violation += 1,
        }
        if self.examples.len() < 24 {
            self.examples.push(msg());
        }
    }
    /// True when no anomaly that would invalidate triangle-inequality shortcuts
    /// was seen — i.e. the matrix looks like a usable symmetric metric.
    pub fn metric_ok(&self) -> bool {
        self.triangle_violation == 0 && self.asymmetric == 0 && self.negative == 0
    }
    /// True when a hard data error (not just a non-metric property) was seen.
    pub fn has_hard_error(&self) -> bool {
        self.negative > 0 || self.self_nonzero > 0 || self.unreachable > 0
    }
    /// One-line-per-finding warnings, ready for stderr.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        let add = |w: &mut Vec<String>, n: usize, what: &str| {
            if n > 0 {
                w.push(format!("  ! {n} {what}"));
            }
        };
        add(&mut w, self.negative, "negative cell(s)");
        add(&mut w, self.self_nonzero, "non-zero diagonal cell(s) d(i,i)!=0");
        add(&mut w, self.unreachable, "unreachable cell(s) (>= UNREACHABLE)");
        add(&mut w, self.asymmetric, "asymmetric landmark sample(s) d(i,j)!=d(j,i)");
        add(&mut w, self.triangle_violation, "triangle-inequality violation(s)");
        w
    }
}

/// Validate one already-fetched row `i` against the resident landmark rows
/// (`dlj`, shape `L×n`, with landmark ids `lm`). Used by both the in-RAM
/// validator and the streaming compressor, so detection is identical. Returns
/// the bridge residual row (so the streamer can encode it without recomputing).
fn validate_row(
    rep: &mut ValidationReport,
    i: usize,
    row: &[i32],
    n: usize,
    lm: &[usize],
    dlj: &[i32],
) -> Vec<i32> {
    rep.rows_seen += 1;
    let l = lm.len();
    let dil: Vec<i32> = lm.iter().map(|&a| row[a]).collect();
    let mut resid = vec![0i32; n];
    for j in 0..n {
        let dij = row[j];
        if dij < 0 {
            rep.note(0, || format!("negative d({i},{j})={dij}"));
        } else if i != j && dij >= UNREACHABLE {
            rep.note(2, || format!("unreachable d({i},{j})={dij}"));
        }
        if i == j && dij != 0 {
            rep.note(1, || format!("d({i},{i})={dij} (expected 0)"));
        }
        let mut base = i64::MAX;
        for a in 0..l {
            let v = dil[a] as i64 + dlj[a * n + j] as i64;
            if v < base {
                base = v;
            }
        }
        let r = dij as i64 - base;
        // base >= d(i,j) under the triangle inequality ⇒ a positive residual is
        // a triangle violation (through every landmark). Free metric check.
        // Small positive residual is integer-rounding noise, not a data error.
        if r > TRIANGLE_TOL {
            rep.note(4, || format!("triangle: d({i},{j})={dij} > min via landmarks by {r}"));
        }
        resid[j] = r as i32;
    }
    // asymmetry sampled against landmark columns: d(i,lm[a]) vs d(lm[a],i)
    for a in 0..l {
        if lm[a] != i && dil[a] != dlj[a * n + i] {
            let (x, y) = (dil[a], dlj[a * n + i]);
            rep.note(3, || format!("asymmetric d({i},{})={x} vs d({},{i})={y}", lm[a], lm[a]));
        }
    }
    resid
}

/// Validate a full in-RAM matrix (uses farthest-point landmarks for the metric
/// checks). Cheap: O(L·n²).
pub fn validate(d: &[i32], n: usize) -> ValidationReport {
    let l = BRIDGE_LS.last().copied().unwrap_or(32).min(n);
    let lm = farthest_landmarks(d, n, l);
    let mut dlj = vec![0i32; lm.len() * n];
    for (a, &la) in lm.iter().enumerate() {
        dlj[a * n..a * n + n].copy_from_slice(&d[la * n..la * n + n]);
    }
    let mut rep = ValidationReport::default();
    for i in 0..n {
        validate_row(&mut rep, i, &d[i * n..i * n + n], n, &lm, &dlj);
    }
    rep
}

// ---------------------------------------------------------------- streaming
/// Voronoi cell of each column: the landmark nearest to `j` (by landmark→j
/// distance, ties to the lowest index). Encoder and reader derive this from
/// the same resident `dlj`, so the block layout never has to be stored.
fn assign_cells(dlj: &[i32], n: usize, l: usize) -> Vec<u8> {
    let mut cell_of = vec![0u8; n];
    for j in 0..n {
        let mut best = i64::MAX;
        for a in 0..l {
            let v = dlj[a * n + j] as i64;
            if v < best {
                best = v;
                cell_of[j] = a as u8;
            }
        }
    }
    cell_of
}

/// A source of matrix rows in `0..n` order. Implement this over dijeng's chunked
/// many-to-many (or any generator) to compress a matrix that never fully
/// materialises in RAM.
pub trait RowSource {
    fn n(&self) -> usize;
    /// Row `i` (row-major, length `n`). May be called for landmark rows before
    /// the main 0..n sweep, so random access by index is required.
    fn row(&mut self, i: usize) -> Vec<i32>;
}

/// A trivial in-RAM [`RowSource`] over a dense slice (for tests / the CLI).
pub struct SliceRows<'a> {
    pub d: &'a [i32],
    pub n: usize,
}
impl RowSource for SliceRows<'_> {
    fn n(&self) -> usize {
        self.n
    }
    fn row(&mut self, i: usize) -> Vec<i32> {
        self.d[i * self.n..i * self.n + self.n].to_vec()
    }
}

/// Stream-compress a matrix to `out` using the bridge model, validating each row
/// as it passes. Peak memory is `L×n` (resident landmark rows) + `n×L` (the
/// gateway index accumulated for the footer) plus a working row — independent
/// of the full `n²`. Writes the `MTZT` framed container, which
/// [`decompress_rows`] decodes row-by-row and [`MtzReader`] random-accesses.
/// Returns the validation report gathered during the single forward pass.
pub fn compress_stream<S: RowSource, W: Write>(
    src: &mut S,
    lm: &[usize],
    out: &mut W,
) -> io::Result<ValidationReport> {
    let n = src.n();
    let l = lm.len();
    // resident landmark rows
    let mut dlj = vec![0i32; l * n];
    for (a, &la) in lm.iter().enumerate() {
        let r = src.row(la);
        dlj[a * n..a * n + n].copy_from_slice(&r);
    }
    out.write_all(MAGIC_STREAM2)?;
    out.write_all(&(n as u32).to_le_bytes())?;
    out.write_all(&(l as u32).to_le_bytes())?;
    for &id in lm {
        out.write_all(&(id as u32).to_le_bytes())?;
    }
    let dljc = deflate(&i32s_to_le(&dlj));
    out.write_all(&(dljc.len() as u32).to_le_bytes())?;
    out.write_all(&dljc)?;

    assert!(l <= 255, "MTZT block index supports at most 255 landmarks");
    let cell_of = assign_cells(&dlj, n, l);

    let mut rep = ValidationReport::default();
    // gateway index + per-(row, landmark-cell) residual bound, for the footer
    let mut dil = vec![0i32; n * l];
    let mut blockmax = vec![0u8; n * l];
    for i in 0..n {
        let row = src.row(i);
        let resid = validate_row(&mut rep, i, &row, n, lm, &dlj);
        for (a, &la) in lm.iter().enumerate() {
            dil[i * l + a] = row[la];
        }
        for j in 0..n {
            let b = &mut blockmax[i * l + cell_of[j] as usize];
            if row[j] >= UNREACHABLE {
                // The index fast path answers `base >= UNREACHABLE` cells as
                // exactly UNREACHABLE; that is verified here cell by cell, and
                // any cell it would get wrong poisons the block (255).
                let base = row[j] as i64 - resid[j] as i64;
                if !(row[j] == UNREACHABLE && base >= UNREACHABLE as i64) {
                    *b = 255;
                }
                continue;
            }
            let m = (resid[j] as i64).unsigned_abs().min(255) as u8;
            if m > *b {
                *b = m;
            }
        }
        let fc = deflate(&i32s_to_le(&resid));
        out.write_all(&(fc.len() as u32).to_le_bytes())?;
        out.write_all(&fc)?;
    }
    // footer: kept resident by MtzReader so blocks with all-zero residual are
    // answered from the index alone, without touching any frame
    let dilc = deflate(&i32s_to_le(&dil));
    out.write_all(&(dilc.len() as u32).to_le_bytes())?;
    out.write_all(&dilc)?;
    let bmc = deflate(&blockmax);
    out.write_all(&(bmc.len() as u32).to_le_bytes())?;
    out.write_all(&bmc)?;
    Ok(rep)
}

/// Decode the `MTZT` container: header + resident tables, then resid-only
/// frames. `dil` lives in the footer (after the frames), so the frames are
/// skip-walked once to reach it before any row is reconstructed.
fn decode_stream2<F: FnMut(usize, &[i32])>(bytes: &[u8], emit: &mut F) -> io::Result<()> {
    let mut cur = Cursor { b: bytes, pos: 4 };
    let n = cur.u32()? as usize;
    let l = cur.u32()? as usize;
    for _ in 0..l {
        cur.u32()?; // landmark ids: not needed to reconstruct values
    }
    let dlj = cur.blob()?;
    if dlj.len() != l * n {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "stream dlj size"));
    }
    let mut frame_off = Vec::with_capacity(n);
    for _ in 0..n {
        frame_off.push(cur.pos);
        let len = cur.u32()? as usize;
        cur.pos += len;
        if cur.pos > bytes.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "frame overrun"));
        }
    }
    let dil = cur.blob()?;
    if dil.len() != n * l {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "stream dil size"));
    }
    let mut row = vec![0i32; n];
    for i in 0..n {
        let mut fc = Cursor { b: bytes, pos: frame_off[i] };
        let resid = fc.blob()?;
        if resid.len() != n {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "stream frame size"));
        }
        for j in 0..n {
            let mut base = i64::MAX;
            for a in 0..l {
                let v = dil[i * l + a] as i64 + dlj[a * n + j] as i64;
                if v < base {
                    base = v;
                }
            }
            row[j] = (base + resid[j] as i64) as i32;
        }
        emit(i, &row);
    }
    Ok(())
}

// --------------------------------------------------------- MTZU (path labels)
/// Tuning for the path-label (MTZU) encoder.
pub struct HubOpts {
    /// Global hub count H (chosen from the candidate rows by greedy
    /// facility location over sampled pairs).
    pub hubs: usize,
    /// Per-point label width k (out- and in-side each keep k hubs).
    pub k: usize,
    /// Candidate/sample row count fetched up front (also the in-credit
    /// mining sample). Bounded by n.
    pub candidates: usize,
    /// Destinations sampled per row for out-label mining.
    pub samples_per_row: usize,
}

impl Default for HubOpts {
    /// `candidates = 0` means auto: `clamp(n/4, 256, 1024)` — hub quality is
    /// driven almost entirely by how many candidate rows the miner sees
    /// (measured on real London: tol-5s block share 6 % at 192 candidates,
    /// 32 % at 1024).
    fn default() -> Self {
        Self { hubs: 128, k: 16, candidates: 0, samples_per_row: 128 }
    }
}

/// Deterministic column order for the grouped-delta frame coding: columns
/// sorted by (cell, j). Encoder and decoder derive it from the same `cell_of`.
fn group_order(cell_of: &[u8]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..cell_of.len() as u32).collect();
    order.sort_by_key(|&j| (cell_of[j as usize], j));
    order
}

/// Frame codings (header flag): plain zigzag-varint vs cell-grouped delta
/// zigzag-varint, both deflated. Chosen once per stream on the first rows.
const FRAME_PLAIN: u8 = 0;
const FRAME_GROUPED: u8 = 1;

fn frame_encode(resid: &[i32], order: &[u32], grouped: bool) -> Vec<u8> {
    if !grouped {
        return deflate(&zz_varint(resid));
    }
    // sequential delta chain along the cell-grouped column order — neighbours
    // in the same cell correlate, so the deltas collapse toward zero
    let mut vals = Vec::with_capacity(resid.len());
    let mut prev = 0i32;
    for &j in order {
        let v = resid[j as usize];
        vals.push(v.wrapping_sub(prev));
        prev = v;
    }
    deflate(&zz_varint(&vals))
}

fn frame_decode(raw: &[u8], order: &[u32], grouped: bool, n: usize) -> io::Result<Vec<i32>> {
    let vals = zz_varint_decode(&inflate(raw)?, n)?;
    if !grouped {
        return Ok(vals);
    }
    let mut resid = vec![0i32; n];
    let mut prev = 0i32;
    for (idx, &j) in order.iter().enumerate() {
        let v = vals[idx].wrapping_add(prev);
        resid[j as usize] = v;
        prev = v;
    }
    Ok(resid)
}

/// The MTZU base: `min over (a in out(i), b in in(j)) of
/// out_d(i,a) + dhh(a,b) + in_d(b,j)` — evaluated for a whole row with the
/// min-plus row trick (k×H + n×k instead of n×k²).
struct HubModel {
    h: usize,
    k: usize,
    dhh: Vec<i32>,    // h×h
    out_h: Vec<u8>,   // n×k
    out_d: Vec<i32>,  // n×k
    in_h: Vec<u8>,    // n×k
    in_d: Vec<i32>,   // n×k
}

impl HubModel {
    /// `m[b] = min over a in out(i) of out_d(i,a) + dhh(a,b)` — the row's
    /// reach-to-hub vector.
    fn row_reach(&self, i: usize) -> Vec<i64> {
        let mut m = vec![i64::MAX; self.h];
        for q in 0..self.k {
            let a = self.out_h[i * self.k + q] as usize;
            let da = self.out_d[i * self.k + q] as i64;
            for b in 0..self.h {
                let v = da + self.dhh[a * self.h + b] as i64;
                if v < m[b] {
                    m[b] = v;
                }
            }
        }
        m
    }
    fn base_from_reach(&self, m: &[i64], j: usize) -> i64 {
        let mut base = i64::MAX;
        for r in 0..self.k {
            let b = self.in_h[j * self.k + r] as usize;
            let v = m[b] + self.in_d[j * self.k + r] as i64;
            if v < base {
                base = v;
            }
        }
        base
    }
    fn base_cell(&self, i: usize, j: usize) -> i64 {
        let mut base = i64::MAX;
        for q in 0..self.k {
            let a = self.out_h[i * self.k + q] as usize;
            let da = self.out_d[i * self.k + q] as i64;
            for r in 0..self.k {
                let b = self.in_h[j * self.k + r] as usize;
                let v = da + self.dhh[a * self.h + b] as i64 + self.in_d[j * self.k + r] as i64;
                if v < base {
                    base = v;
                }
            }
        }
        base
    }
}

/// Stream-compress through the **path-label model** into the `MTZU`
/// container. Fetches `candidates` rows up front (RowSource random access),
/// picks H hubs by greedy facility location over pairs sampled from those
/// rows, mines in-labels from them, then streams every row once: out-labels
/// from the row itself, residual vs the hub base, varint-coded frame.
/// Peak memory: the candidate rows (C×n) + the labels — never n².
pub fn compress_stream_hub<S: RowSource, W: Write>(
    src: &mut S,
    opts: &HubOpts,
    out: &mut W,
) -> io::Result<ValidationReport> {
    let n = src.n();
    let c_auto = (n / 4).clamp(256, 1024);
    let c_count = if opts.candidates == 0 { c_auto } else { opts.candidates }.min(n).max(2);
    let h_count = opts.hubs.min(c_count).min(255).max(1);
    let k = opts.k.min(h_count).max(1);

    // deterministic candidate rows (distinct)
    let mut state: u64 = 0xDA7A_BA5E_0F1C_E5;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut cand: Vec<usize> = Vec::with_capacity(c_count);
    let mut seen = vec![false; n];
    while cand.len() < c_count {
        let i = next() % n;
        if !seen[i] {
            seen[i] = true;
            cand.push(i);
        }
    }
    let mut cand_rows = vec![0i32; c_count * n];
    for (ci, &i) in cand.iter().enumerate() {
        let r = src.row(i);
        cand_rows[ci * n..ci * n + n].copy_from_slice(&r);
    }

    // hub selection: greedy facility location over sampled (cand_i, j) pairs;
    // via-cost of candidate a for pair (i,j) = d(i,a) + d(a,j) where
    // d(i,a) = cand_rows[i][cand[a]] and d(a,j) = cand_rows[a][j].
    let s_count = (c_count * 24).min(16384);
    let mut pairs = Vec::with_capacity(s_count);
    while pairs.len() < s_count {
        let ci = next() % c_count;
        let j = next() % n;
        if cand[ci] != j {
            pairs.push((ci, j));
        }
    }
    let mut cur = vec![i64::MAX; s_count];
    let mut used = vec![false; c_count];
    let mut hubs: Vec<usize> = Vec::with_capacity(h_count); // candidate indices
    for _ in 0..h_count {
        let mut best_a = usize::MAX;
        let mut best_cost = i64::MAX;
        for a in 0..c_count {
            if used[a] {
                continue;
            }
            let mut cost = 0i64;
            for (s, &(ci, j)) in pairs.iter().enumerate() {
                let via = cand_rows[ci * n + cand[a]] as i64 + cand_rows[a * n + j] as i64;
                cost += cur[s].min(via);
            }
            if cost < best_cost {
                best_cost = cost;
                best_a = a;
            }
        }
        if best_a == usize::MAX {
            break;
        }
        used[best_a] = true;
        hubs.push(best_a);
        for (s, &(ci, j)) in pairs.iter().enumerate() {
            let via = cand_rows[ci * n + cand[best_a]] as i64 + cand_rows[best_a * n + j] as i64;
            cur[s] = cur[s].min(via);
        }
    }
    let h = hubs.len();
    let hub_ids: Vec<usize> = hubs.iter().map(|&a| cand[a]).collect();
    // hub rows (d(hub, j) for all j) — a view into cand_rows
    let hub_row = |a: usize| -> &[i32] { &cand_rows[hubs[a] * n..hubs[a] * n + n] };

    let mut dhh = vec![0i32; h * h];
    for a in 0..h {
        for b in 0..h {
            dhh[a * h + b] = hub_row(a)[hub_ids[b]];
        }
    }
    // cell of column j = nearest hub by hub→j distance
    let mut cell_of = vec![0u8; n];
    for j in 0..n {
        let mut best = i64::MAX;
        for a in 0..h {
            if (hub_row(a)[j] as i64) < best {
                best = hub_row(a)[j] as i64;
                cell_of[j] = a as u8;
            }
        }
    }
    let order = group_order(&cell_of);

    // in-labels: credit the best via hub of (cand_i, j) to in(j); fill with
    // nearest-by-in-distance
    let mut credit_in = vec![0u32; n * h];
    for ci in 0..c_count {
        let crow = &cand_rows[ci * n..ci * n + n];
        for j in 0..n {
            let mut best = i64::MAX;
            let mut ba = 0usize;
            for a in 0..h {
                let v = crow[hub_ids[a]] as i64 + hub_row(a)[j] as i64;
                if v < best {
                    best = v;
                    ba = a;
                }
            }
            credit_in[j * h + ba] += 1;
        }
    }
    let mut in_h = vec![0u8; n * k];
    let mut in_d = vec![0i32; n * k];
    {
        let mut idx: Vec<usize> = Vec::with_capacity(h);
        for j in 0..n {
            idx.clear();
            idx.extend(0..h);
            idx.sort_by_key(|&a| {
                (std::cmp::Reverse(credit_in[j * h + a]), hub_row(a)[j])
            });
            for q in 0..k {
                in_h[j * k + q] = idx[q] as u8;
                in_d[j * k + q] = hub_row(idx[q])[j];
            }
        }
    }

    // header
    out.write_all(MAGIC_STREAM3)?;
    out.write_all(&(n as u32).to_le_bytes())?;
    out.write_all(&(h as u32).to_le_bytes())?;
    out.write_all(&(k as u32).to_le_bytes())?;
    for &id in &hub_ids {
        out.write_all(&(id as u32).to_le_bytes())?;
    }
    let mut head = Vec::new();
    put_blob_vz(&mut head, &dhh);
    put_blob(&mut head, &cell_of);
    put_blob(&mut head, &in_h);
    put_blob_vz(&mut head, &in_d);
    out.write_all(&head)?;

    // landmark-style rows for the validation checks: reuse the hub rows
    let lm_val: Vec<usize> = hub_ids.clone();
    let mut dlj_val = vec![0i32; h * n];
    for a in 0..h {
        dlj_val[a * n..a * n + n].copy_from_slice(hub_row(a));
    }

    // main pass
    let mut rep = ValidationReport::default();
    let mut out_h = vec![0u8; n * k];
    let mut out_d = vec![0i32; n * k];
    let mut blockmax = vec![0u8; n * h];
    let mut frame_mode = u8::MAX; // decided on the first row (try both)
    let mut model = HubModel { h, k, dhh, out_h: Vec::new(), out_d: Vec::new(), in_h, in_d };
    let mut credit_out = vec![0u32; h];
    let mut idx: Vec<usize> = Vec::with_capacity(h);
    for i in 0..n {
        let row = src.row(i);
        validate_row(&mut rep, i, &row, n, &lm_val, &dlj_val);
        // out-label mining for this row
        for c in credit_out.iter_mut() {
            *c = 0;
        }
        for _ in 0..opts.samples_per_row {
            let j = next() % n;
            if j == i {
                continue;
            }
            let mut best = i64::MAX;
            let mut ba = 0usize;
            for a in 0..h {
                let v = row[hub_ids[a]] as i64 + dlj_val[a * n + j] as i64;
                if v < best {
                    best = v;
                    ba = a;
                }
            }
            credit_out[ba] += 1;
        }
        idx.clear();
        idx.extend(0..h);
        idx.sort_by_key(|&a| (std::cmp::Reverse(credit_out[a]), row[hub_ids[a]]));
        for q in 0..k {
            out_h[i * k + q] = idx[q] as u8;
            out_d[i * k + q] = row[hub_ids[idx[q]]];
        }
        // residual vs the hub base (row trick)
        model.out_h = std::mem::take(&mut out_h);
        model.out_d = std::mem::take(&mut out_d);
        let m = model.row_reach(i);
        out_h = std::mem::take(&mut model.out_h);
        out_d = std::mem::take(&mut model.out_d);
        let mut resid = vec![0i32; n];
        for j in 0..n {
            let base = model.base_from_reach(&m, j);
            resid[j] = (row[j] as i64 - base) as i32;
            let bm = &mut blockmax[i * h + cell_of[j] as usize];
            if row[j] >= UNREACHABLE {
                if !(row[j] == UNREACHABLE && base >= UNREACHABLE as i64) {
                    *bm = 255;
                }
                continue;
            }
            let mag = (resid[j] as i64).unsigned_abs().min(255) as u8;
            if mag > *bm {
                *bm = mag;
            }
        }
        if frame_mode == u8::MAX {
            // pick the coding that wins on this first row
            let plain = frame_encode(&resid, &order, false).len();
            let grouped = frame_encode(&resid, &order, true).len();
            frame_mode = if grouped < plain { FRAME_GROUPED } else { FRAME_PLAIN };
            out.write_all(&[frame_mode])?;
        }
        let fc = frame_encode(&resid, &order, frame_mode == FRAME_GROUPED);
        out.write_all(&(fc.len() as u32).to_le_bytes())?;
        out.write_all(&fc)?;
    }
    // footer
    let mut foot = Vec::new();
    put_blob(&mut foot, &out_h);
    put_blob_vz(&mut foot, &out_d);
    put_blob(&mut foot, &blockmax);
    out.write_all(&foot)?;
    Ok(rep)
}

/// Decode the `MTZU` container row by row.
fn decode_stream3<F: FnMut(usize, &[i32])>(bytes: &[u8], emit: &mut F) -> io::Result<()> {
    let mut cur = Cursor { b: bytes, pos: 4 };
    let n = cur.u32()? as usize;
    let h = cur.u32()? as usize;
    let k = cur.u32()? as usize;
    for _ in 0..h {
        cur.u32()?;
    }
    let dhh = cur.blob_vz(h * h)?;
    let cell_of = cur.blob_raw()?;
    let in_h = cur.blob_raw()?;
    let in_d = cur.blob_vz(n * k)?;
    if cell_of.len() != n || in_h.len() != n * k {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "mtzu header sizes"));
    }
    if cur.pos >= bytes.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "frame mode"));
    }
    let frame_mode = bytes[cur.pos];
    cur.pos += 1;
    let mut frame_off = Vec::with_capacity(n);
    for _ in 0..n {
        frame_off.push(cur.pos);
        let len = cur.u32()? as usize;
        cur.pos += len;
        if cur.pos > bytes.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "frame overrun"));
        }
    }
    let out_h = cur.blob_raw()?;
    let out_d = cur.blob_vz(n * k)?;
    let _blockmax = cur.blob_raw()?;
    if out_h.len() != n * k {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "mtzu footer sizes"));
    }
    let order = group_order(&cell_of);
    let model = HubModel { h, k, dhh, out_h, out_d, in_h, in_d };
    let mut row = vec![0i32; n];
    for i in 0..n {
        let mut fc = Cursor { b: bytes, pos: frame_off[i] };
        let raw = {
            let len = fc.u32()? as usize;
            &bytes[fc.pos..fc.pos + len]
        };
        let resid = frame_decode(raw, &order, frame_mode == FRAME_GROUPED, n)?;
        let m = model.row_reach(i);
        for j in 0..n {
            row[j] = (model.base_from_reach(&m, j) + resid[j] as i64) as i32;
        }
        emit(i, &row);
    }
    Ok(())
}

fn decode_stream<F: FnMut(usize, &[i32])>(bytes: &[u8], emit: &mut F) -> io::Result<()> {
    let mut cur = Cursor { b: bytes, pos: 4 };
    let n = cur.u32()? as usize;
    let l = cur.u32()? as usize;
    let mut lm = vec![0usize; l];
    for a in 0..l {
        lm[a] = cur.u32()? as usize;
    }
    let dlj = cur.blob()?;
    if dlj.len() != l * n {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "stream dlj size"));
    }
    let mut row = vec![0i32; n];
    for i in 0..n {
        let frame = cur.blob()?;
        if frame.len() != l + n {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "stream frame size"));
        }
        let (dil, resid) = frame.split_at(l);
        for j in 0..n {
            let mut base = i64::MAX;
            for a in 0..l {
                let v = dil[a] as i64 + dlj[a * n + j] as i64;
                if v < base {
                    base = v;
                }
            }
            row[j] = (base + resid[j] as i64) as i32;
        }
        emit(i, &row);
    }
    Ok(())
}

// ------------------------------------------------------- compressed random access
/// Tiny capacity-bounded LRU of inflated residual rows (approximate, tick-based).
struct RowCache {
    cap: usize,
    map: std::collections::HashMap<usize, (Vec<i32>, u64)>,
    tick: u64,
    pub hits: u64,
    pub misses: u64,
}
impl RowCache {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            map: std::collections::HashMap::new(),
            tick: 0,
            hits: 0,
            misses: 0,
        }
    }
    /// Refresh row `i` and borrow it without cloning (cell lookups).
    fn touch(&mut self, i: usize) -> Option<&[i32]> {
        self.tick += 1;
        let t = self.tick;
        if let Some(e) = self.map.get_mut(&i) {
            e.1 = t;
            self.hits += 1;
            Some(&e.0)
        } else {
            self.misses += 1;
            None
        }
    }
    fn get(&mut self, i: usize) -> Option<Vec<i32>> {
        self.touch(i).map(|r| r.to_vec())
    }
    fn put(&mut self, i: usize, row: Vec<i32>) {
        self.tick += 1;
        let t = self.tick;
        if self.map.len() >= self.cap && !self.map.contains_key(&i) {
            if let Some((&k, _)) = self.map.iter().min_by_key(|(_, (_, u))| *u) {
                self.map.remove(&k);
            }
        }
        self.map.insert(i, (row, t));
    }
}

/// Random-access reader over an `MTZT` (streamed) blob: keeps the compressed
/// bytes + the resident landmark rows (`dlj`, `L×n`), the gateway index
/// (`dil`, `n×L`) and the per-row max-|residual| bytes, indexes each row's
/// frame, and reconstructs rows on demand, caching hot rows in an LRU.
///
/// The resident index makes two things O(L) with **zero decompression**:
/// - exact `cell(i,j)` whenever the (row, landmark-cell-of-`j`) block has an
///   all-zero residual — under real gateway structure that covers the
///   cross-region blocks, i.e. most random lookups, and
/// - `cell_bounds(i,j)` — bridge upper bound + ALT lower bound, tightened by
///   the block's max-|residual| — for every cell, usable to prune before
///   paying for an exact lookup.
///
/// Peak resident memory is `2·L×n` ints + `L×n + 2n` bytes + the cache, never
/// the full `n²`. The compressed blob itself can be in RAM or memory-mapped.
enum BaseModel {
    /// MTZT: flat n×L landmark table.
    Landmark { l: usize, lm: Vec<usize>, dlj: Vec<i32>, dil: Vec<i32> },
    /// MTZU: directional path labels + dense hub matrix.
    Hub { ids: Vec<usize>, model: HubModel, order: Vec<u32>, frame_mode: u8 },
}

pub struct MtzReader {
    bytes: Vec<u8>,
    n: usize,
    base: BaseModel,
    /// number of cells per row in `blockmax` (L or H)
    cells: usize,
    /// max |residual| per (row, cell), saturated at 255
    blockmax: Vec<u8>,
    /// which cell each column belongs to
    cell_of: Vec<u8>,
    /// derived: max |residual| over each whole row
    rowmax: Vec<u8>,
    frame_off: Vec<usize>,
    cache: RowCache,
    use_index: bool,
}

impl MtzReader {
    /// Open an `MTZT` or `MTZU` blob with an LRU holding up to `cache_rows`
    /// inflated residual rows. Only the stream formats support cheap per-row
    /// random access; legacy `MTZS` blobs must be recompressed (full decode
    /// still works via [`decompress_rows`]).
    pub fn open(bytes: Vec<u8>, cache_rows: usize) -> io::Result<Self> {
        if bytes.len() >= 12 && &bytes[0..4] == MAGIC_STREAM3 {
            return Self::open_hub(bytes, cache_rows);
        }
        if bytes.len() < 12 || &bytes[0..4] != MAGIC_STREAM2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MtzReader requires the MTZT/MTZU (--stream) formats; recompress legacy MTZS blobs",
            ));
        }
        let (n, l, lm, dlj, dil, blockmax, frame_off);
        {
            let mut cur = Cursor { b: &bytes, pos: 4 };
            n = cur.u32()? as usize;
            l = cur.u32()? as usize;
            let mut lmv = vec![0usize; l];
            for a in 0..l {
                lmv[a] = cur.u32()? as usize;
            }
            let dljv = cur.blob()?;
            if dljv.len() != l * n {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "dlj size"));
            }
            let mut offs = Vec::with_capacity(n);
            for _ in 0..n {
                offs.push(cur.pos);
                let len = cur.u32()? as usize;
                cur.pos += len;
                if cur.pos > bytes.len() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "frame overrun"));
                }
            }
            let dilv = cur.blob()?;
            if dilv.len() != n * l {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "dil size"));
            }
            let bmv = cur.blob_raw()?;
            if bmv.len() != n * l {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "blockmax size"));
            }
            lm = lmv;
            dlj = dljv;
            dil = dilv;
            blockmax = bmv;
            frame_off = offs;
        }
        let cell_of = assign_cells(&dlj, n, l);
        let rowmax: Vec<u8> = (0..n)
            .map(|i| blockmax[i * l..i * l + l].iter().copied().max().unwrap_or(0))
            .collect();
        Ok(Self {
            bytes,
            n,
            base: BaseModel::Landmark { l, lm, dlj, dil },
            cells: l,
            blockmax,
            cell_of,
            rowmax,
            frame_off,
            cache: RowCache::new(cache_rows),
            use_index: true,
        })
    }

    fn open_hub(bytes: Vec<u8>, cache_rows: usize) -> io::Result<Self> {
        let (n, h, k, ids, model, order, frame_mode, cell_of, blockmax, frame_off);
        {
            let mut cur = Cursor { b: &bytes, pos: 4 };
            n = cur.u32()? as usize;
            h = cur.u32()? as usize;
            k = cur.u32()? as usize;
            let mut idv = vec![0usize; h];
            for a in 0..h {
                idv[a] = cur.u32()? as usize;
            }
            let dhh = cur.blob_vz(h * h)?;
            let cov = cur.blob_raw()?;
            let in_h = cur.blob_raw()?;
            let in_d = cur.blob_vz(n * k)?;
            if cov.len() != n || in_h.len() != n * k {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "mtzu header sizes"));
            }
            if cur.pos >= bytes.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "frame mode"));
            }
            let fm = bytes[cur.pos];
            cur.pos += 1;
            let mut offs = Vec::with_capacity(n);
            for _ in 0..n {
                offs.push(cur.pos);
                let len = cur.u32()? as usize;
                cur.pos += len;
                if cur.pos > bytes.len() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "frame overrun"));
                }
            }
            let out_h = cur.blob_raw()?;
            let out_d = cur.blob_vz(n * k)?;
            let bmv = cur.blob_raw()?;
            if out_h.len() != n * k || bmv.len() != n * h {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "mtzu footer sizes"));
            }
            ids = idv;
            order = group_order(&cov);
            model = HubModel { h, k, dhh, out_h, out_d, in_h, in_d };
            frame_mode = fm;
            cell_of = cov;
            blockmax = bmv;
            frame_off = offs;
        }
        let rowmax: Vec<u8> = (0..n)
            .map(|i| blockmax[i * h..i * h + h].iter().copied().max().unwrap_or(0))
            .collect();
        Ok(Self {
            bytes,
            n,
            base: BaseModel::Hub { ids, model, order, frame_mode },
            cells: h,
            blockmax,
            cell_of,
            rowmax,
            frame_off,
            cache: RowCache::new(cache_rows),
            use_index: true,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }
    /// The global anchor points: landmarks (MTZT) or hubs (MTZU).
    pub fn landmarks(&self) -> &[usize] {
        match &self.base {
            BaseModel::Landmark { lm, .. } => lm,
            BaseModel::Hub { ids, .. } => ids,
        }
    }
    /// (cache hits, misses) so far.
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.cache.hits, self.cache.misses)
    }
    /// Bytes held resident for the index (excluding the compressed blob and
    /// the row cache).
    pub fn resident_bytes(&self) -> usize {
        let base = match &self.base {
            BaseModel::Landmark { l, .. } => 2 * l * self.n * 4,
            BaseModel::Hub { model, .. } => {
                self.n * 2 * model.k * 5 + model.h * model.h * 4 + self.n * 4 // labels + dhh + order
            }
        };
        base + self.blockmax.len() + self.cell_of.len() + self.rowmax.len()
    }
    /// Disable the O(L) index fast path (every lookup goes through frame
    /// inflation). For benchmarking/verification only.
    pub fn set_index_fast_path(&mut self, on: bool) {
        self.use_index = on;
    }
    /// Number of rows answerable from the resident index alone (residual 0).
    pub fn exact_index_rows(&self) -> usize {
        self.rowmax.iter().filter(|&&m| m == 0).count()
    }
    /// Fraction of (row, landmark-cell) blocks answerable from the index alone.
    pub fn exact_index_block_share(&self) -> f64 {
        if self.blockmax.is_empty() {
            return 0.0;
        }
        self.blockmax.iter().filter(|&&m| m == 0).count() as f64 / self.blockmax.len() as f64
    }

    /// The model base — O(L) (landmarks) or O(k²) (path labels), straight
    /// from the resident index.
    fn base_cell(&self, i: usize, j: usize) -> i64 {
        match &self.base {
            BaseModel::Landmark { l, dlj, dil, .. } => {
                let mut base = i64::MAX;
                for a in 0..*l {
                    let v = dil[i * l + a] as i64 + dlj[a * self.n + j] as i64;
                    if v < base {
                        base = v;
                    }
                }
                base
            }
            BaseModel::Hub { model, .. } => model.base_cell(i, j),
        }
    }

    /// The value an index-exact block reports for `(i,j)`: the bridge base,
    /// with `base >= UNREACHABLE` collapsed to exactly UNREACHABLE (the
    /// encoder verified that collapse cell by cell before marking the block).
    fn index_cell(&self, i: usize, j: usize) -> i32 {
        let base = self.base_cell(i, j);
        if base >= UNREACHABLE as i64 {
            UNREACHABLE
        } else {
            base as i32
        }
    }

    /// `(lower, upper)` bounds on `d(i,j)` from the resident index alone —
    /// O(L), no decompression. Upper is the bridge base, lower is the directed
    /// ALT bound tightened by the block's max-|residual|; both widened by
    /// `TRIANGLE_TOL` to absorb integer-rounding noise. Only meaningful for
    /// matrices that validate as a metric ([`ValidationReport::metric_ok`]);
    /// when `lower == upper` the value is exact.
    pub fn cell_bounds(&self, i: usize, j: usize) -> (i32, i32) {
        let bm = self.blockmax[i * self.cells + self.cell_of[j] as usize];
        let base = self.base_cell(i, j);
        if bm == 0 {
            let v = self.index_cell(i, j);
            return (v, v);
        }
        if base >= UNREACHABLE as i64 {
            // unreachable territory is outside the metric contract — stay safe
            return (0, i32::MAX);
        }
        // residual ∈ [-bm, +min(bm, TRIANGLE_TOL)] (positive side is only noise)
        let up = base
            .saturating_add((bm as i64).min(TRIANGLE_TOL))
            .min(i32::MAX as i64) as i32;
        let mut lo = 0i64;
        if let BaseModel::Landmark { l, dlj, dil, .. } = &self.base {
            for a in 0..*l {
                // d(i,a) ≤ d(i,j)+d(j,a)  and  d(a,j) ≤ d(a,i)+d(i,j)
                let v1 = dil[i * l + a] as i64 - dil[j * l + a] as i64;
                let v2 = dlj[a * self.n + j] as i64 - dlj[a * self.n + i] as i64;
                lo = lo.max(v1).max(v2);
            }
            lo -= TRIANGLE_TOL;
        }
        if bm < 255 {
            lo = lo.max(base - bm as i64);
        }
        (lo.clamp(0, i32::MAX as i64) as i32, up)
    }

    /// `cell()` with a caller tolerance: when the (row, cell) block's
    /// max-|residual| is ≤ `tol`, return the index base — an
    /// overestimate by at most `tol` (and an underestimate by at most the
    /// integer-rounding noise, `TRIANGLE_TOL`) — without touching the
    /// compressed blob. Otherwise fall back to the exact path. Purely
    /// value-based (no metric assumption), so safe on asymmetric road
    /// matrices; `tol = 0` is exactly [`MtzReader::cell`]. Typical use: a VRP
    /// local search probing travel times where a few seconds of slack is
    /// irrelevant, with exact lookups reserved for accepted moves.
    pub fn cell_within(&mut self, i: usize, j: usize, tol: u8) -> io::Result<i32> {
        if self.use_index && self.blockmax[i * self.cells + self.cell_of[j] as usize] <= tol {
            return Ok(self.index_cell(i, j));
        }
        self.cell(i, j)
    }

    /// Share of (row, landmark-cell) blocks whose max-|residual| is ≤ `tol`,
    /// i.e. the fraction of lookups [`MtzReader::cell_within`] answers in O(L).
    pub fn index_share_within(&self, tol: u8) -> f64 {
        if self.blockmax.is_empty() {
            return 0.0;
        }
        self.blockmax.iter().filter(|&&m| m <= tol).count() as f64 / self.blockmax.len() as f64
    }

    /// The model base for a whole row `i` — landmark-outer loops (MTZT) or
    /// the reach-vector trick (MTZU) so big arrays are walked sequentially.
    fn base_row(&self, i: usize) -> Vec<i64> {
        match &self.base {
            BaseModel::Landmark { l, dlj, dil, .. } => {
                let mut base = vec![i64::MAX; self.n];
                for a in 0..*l {
                    let da = dil[i * l + a] as i64;
                    let drow = &dlj[a * self.n..(a + 1) * self.n];
                    for (b, &dj) in base.iter_mut().zip(drow) {
                        let v = da + dj as i64;
                        if v < *b {
                            *b = v;
                        }
                    }
                }
                base
            }
            BaseModel::Hub { model, .. } => {
                let m = model.row_reach(i);
                (0..self.n).map(|j| model.base_from_reach(&m, j)).collect()
            }
        }
    }

    /// Inflate row `i`'s residual frame (the cheap part — no base needed).
    fn load_resid(&self, i: usize) -> io::Result<Vec<i32>> {
        let mut cur = Cursor { b: &self.bytes, pos: self.frame_off[i] };
        match &self.base {
            BaseModel::Landmark { .. } => {
                let resid = cur.blob()?;
                if resid.len() != self.n {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "frame size"));
                }
                Ok(resid)
            }
            BaseModel::Hub { order, frame_mode, .. } => {
                let len = cur.u32()? as usize;
                if cur.pos + len > self.bytes.len() {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "frame"));
                }
                frame_decode(
                    &self.bytes[cur.pos..cur.pos + len],
                    order,
                    *frame_mode == FRAME_GROUPED,
                    self.n,
                )
            }
        }
    }

    /// Reconstruct row `i`. Skips the frame entirely when the row is
    /// index-exact; otherwise one frame inflate (residuals LRU-cached) plus an
    /// O(L·n) base sweep.
    pub fn row(&mut self, i: usize) -> io::Result<Vec<i32>> {
        if self.use_index && self.rowmax[i] == 0 {
            // residual all-zero: the bridge base IS the row, skip the frame
            return Ok(self
                .base_row(i)
                .iter()
                .map(|&b| if b >= UNREACHABLE as i64 { UNREACHABLE } else { b as i32 })
                .collect());
        }
        let base = self.base_row(i);
        let resid = match self.cache.get(i) {
            Some(r) => r,
            None => {
                let r = self.load_resid(i)?;
                self.cache.put(i, r.clone());
                r
            }
        };
        Ok(base
            .iter()
            .zip(&resid)
            .map(|(&b, &r)| (b + r as i64) as i32)
            .collect())
    }

    /// Single cell `d(i,j)`. O(L) straight from the resident index when the
    /// (row, landmark-cell-of-`j`) block is index-exact; otherwise O(L) + the
    /// cached residual (one frame inflate on a miss — never a full-row
    /// reconstruction).
    pub fn cell(&mut self, i: usize, j: usize) -> io::Result<i32> {
        if self.use_index && self.blockmax[i * self.cells + self.cell_of[j] as usize] == 0 {
            return Ok(self.index_cell(i, j));
        }
        let base = self.base_cell(i, j);
        if let Some(r) = self.cache.touch(i).map(|r| r[j]) {
            return Ok((base + r as i64) as i32);
        }
        let resid = self.load_resid(i)?;
        let v = (base + resid[j] as i64) as i32;
        self.cache.put(i, resid);
        Ok(v)
    }
}

// ---------------------------------------------------------------- decode
/// Decompress into a full row-major `n×n` matrix. Exact inverse of [`compress`].
pub fn decompress(bytes: &[u8]) -> io::Result<(Vec<i32>, usize)> {
    let mut full: Vec<i32> = Vec::new();
    let mut n = 0usize;
    decompress_rows(bytes, |row_idx, row| {
        if full.is_empty() {
            n = row.len();
            full = vec![0i32; n * n];
        }
        full[row_idx * n..row_idx * n + n].copy_from_slice(row);
    })?;
    Ok((full, n))
}

/// Streaming decode: reconstructs one **row-band** (the rows of one region) at a
/// time and invokes `emit(global_row_index, &row)` for each global row. Peak
/// memory is one band (≈ `n²/k`), not the full `n²` — so this scales to matrices
/// far larger than RAM would hold densely.
pub fn decompress_rows<F: FnMut(usize, &[i32])>(bytes: &[u8], mut emit: F) -> io::Result<()> {
    if bytes.len() < 9 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated"));
    }
    if &bytes[0..4] == MAGIC_STREAM3 {
        return decode_stream3(bytes, &mut emit);
    }
    if &bytes[0..4] == MAGIC_STREAM2 {
        return decode_stream2(bytes, &mut emit);
    }
    if &bytes[0..4] == MAGIC_STREAM {
        return decode_stream(bytes, &mut emit);
    }
    if &bytes[0..4] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let mut cur = Cursor { b: bytes, pos: 4 };
    let n = cur.u32()? as usize;
    let method = bytes[cur.pos];
    cur.pos += 1;
    match method {
        METHOD_BRIDGE => decode_bridge(&mut cur, n, &mut emit),
        METHOD_CLUSTER => decode_cluster(&mut cur, n, &mut emit),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "unknown method")),
    }
}

fn decode_cluster<F: FnMut(usize, &[i32])>(
    cur: &mut Cursor,
    n: usize,
    emit: &mut F,
) -> io::Result<()> {
    let k = cur.u32()? as usize;
    let assign: Vec<u32> = cur.blob()?.iter().map(|&x| x as u32).collect();
    if assign.len() != n {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "assign len"));
    }
    let groups = groups_from(&assign, k);

    for (gi_idx, gi) in groups.iter().enumerate() {
        let ri = gi.len();
        // band: ri rows × n cols (global column order)
        let mut band = vec![0i32; ri * n];
        for (gj_idx, gj) in groups.iter().enumerate() {
            let cj = gj.len();
            if gi_idx == gj_idx {
                let block = cur.blob()?; // ri*cj dense
                if block.len() != ri * cj {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "diag size"));
                }
                for p in 0..ri {
                    for q in 0..cj {
                        band[p * n + gj[q]] = block[p * cj + q];
                    }
                }
            } else {
                let col0 = cur.blob()?;
                let row0 = cur.blob()?;
                let c00 = cur.i32()?;
                let resid = cur.blob()?;
                if col0.len() != ri || row0.len() != cj || resid.len() != ri * cj {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "offdiag size"));
                }
                for p in 0..ri {
                    for q in 0..cj {
                        let v = col0[p] + row0[q] - c00 + resid[p * cj + q];
                        band[p * n + gj[q]] = v;
                    }
                }
            }
        }
        for p in 0..ri {
            emit(gi[p], &band[p * n..p * n + n]);
        }
    }
    Ok(())
}

fn decode_bridge<F: FnMut(usize, &[i32])>(
    cur: &mut Cursor,
    n: usize,
    emit: &mut F,
) -> io::Result<()> {
    let lm = cur.blob()?;
    let l = lm.len();
    let dil = cur.blob()?; // n*l
    let dlj = cur.blob()?; // l*n
    let resid = cur.blob()?; // n*n
    if l == 0 || dil.len() != n * l || dlj.len() != l * n || resid.len() != n * n {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bridge size"));
    }
    let mut row = vec![0i32; n];
    for i in 0..n {
        for j in 0..n {
            let mut base = i64::MAX;
            for a in 0..l {
                let v = dil[i * l + a] as i64 + dlj[a * n + j] as i64;
                if v < base {
                    base = v;
                }
            }
            row[j] = (base + resid[i * n + j] as i64) as i32;
        }
        emit(i, &row);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_roundtrip(d: &[i32], n: usize, k: usize) -> (usize, usize) {
        let comp = compress(d, n, k);
        let (back, n2) = decompress(&comp).unwrap();
        assert_eq!(n2, n);
        assert_eq!(&back, d, "lossless roundtrip failed");
        (n * n * 4, comp.len())
    }

    #[test]
    fn roundtrip_single_gateway_world() {
        // 4 regions of 25 points; cross-region = additive rank-1 (one gateway).
        let regions = 4;
        let per = 25;
        let n = regions * per;
        let mut pts = vec![(0.0f64, 0.0f64); n];
        let centers = [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)];
        let mut s: u64 = 1;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f64 / (1u64 << 31) as f64) * 40.0 - 20.0
        };
        for r in 0..regions {
            for i in 0..per {
                pts[r * per + i] = (centers[r].0 + rnd(), centers[r].1 + rnd());
            }
        }
        let gw: Vec<usize> = (0..regions).map(|r| r * per).collect(); // one gateway/region
        let eu = |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        let mut d = vec![0i32; n * n];
        for i in 0..n {
            for j in 0..n {
                let (ri, rj) = (i / per, j / per);
                let v = if ri == rj {
                    eu(pts[i], pts[j])
                } else {
                    eu(pts[i], pts[gw[ri]]) + 5000.0 + eu(pts[gw[rj]], pts[j])
                };
                d[i * n + j] = v.round() as i32;
            }
        }
        let (raw, comp) = check_roundtrip(&d, n, default_k(n));
        // structured world must beat plain storage clearly
        assert!(comp * 3 < raw, "expected >3x, got raw={} comp={}", raw, comp);
    }

    #[test]
    fn roundtrip_smooth_euclidean_exercises_bridge() {
        // a smooth 2-D Euclidean metric — the bridge/landmark model tends to win
        // here; either way the roundtrip must be exact.
        let n = 120;
        let mut s: u64 = 42;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as f64 / (1u64 << 31) as f64 * 1000.0
        };
        let pts: Vec<(f64, f64)> = (0..n).map(|_| (rnd(), rnd())).collect();
        let mut d = vec![0i32; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] =
                    (((pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2)).sqrt())
                        .round() as i32;
            }
        }
        let comp = compress(&d, n, default_k(n));
        let (back, n2) = decompress(&comp).unwrap();
        assert_eq!(n2, n);
        assert_eq!(back, d, "lossless roundtrip failed (smooth)");
    }

    #[test]
    fn roundtrip_trivial_and_edge() {
        // single cluster path
        let d = vec![0, 3, 5, 3, 0, 2, 5, 2, 0];
        check_roundtrip(&d, 3, 1);
        check_roundtrip(&d, 3, 2);
        // 1x1
        check_roundtrip(&[0], 1, 1);
        // asymmetric (one-way streets)
        let d2 = vec![0, 10, 7, 0];
        check_roundtrip(&d2, 2, 2);
    }

    fn euclid_matrix(n: usize, seed: u64) -> Vec<i32> {
        let mut s = seed;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as f64 / (1u64 << 31) as f64 * 1000.0
        };
        let pts: Vec<(f64, f64)> = (0..n).map(|_| (rnd(), rnd())).collect();
        let mut d = vec![0i32; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] =
                    (((pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2)).sqrt())
                        .round() as i32;
            }
        }
        d
    }

    #[test]
    fn streaming_roundtrip_lossless() {
        let n = 100;
        let d = euclid_matrix(n, 7);
        let lm = farthest_landmarks(&d, n, 16);
        let mut src = SliceRows { d: &d, n };
        let mut buf = Vec::new();
        let rep = compress_stream(&mut src, &lm, &mut buf).unwrap();
        assert_eq!(rep.rows_seen, n);
        assert!(rep.metric_ok(), "clean euclidean should validate as metric");
        let (back, n2) = decompress(&buf).unwrap();
        assert_eq!(n2, n);
        assert_eq!(back, d, "streaming roundtrip must be lossless");
    }

    #[test]
    fn validation_flags_injected_anomalies() {
        let n = 60;
        let mut d = euclid_matrix(n, 11);
        d[5 * n + 9] = -3; // negative
        d[7 * n + 7] = 42; // non-zero diagonal
        d[3 * n + 4] = UNREACHABLE + 1; // unreachable
        // triangle violation: make a direct edge longer than a 2-hop path
        d[2 * n + 50] = d[2 * n + 1] + d[1 * n + 50] + 1000;
        let rep = validate(&d, n);
        assert!(rep.negative >= 1, "negative not caught");
        assert!(rep.self_nonzero >= 1, "self-distance not caught");
        assert!(rep.unreachable >= 1, "unreachable not caught");
        assert!(rep.triangle_violation >= 1, "triangle violation not caught");
        assert!(rep.has_hard_error());
        assert!(!rep.metric_ok());
    }

    #[test]
    fn random_access_reader_matches_original() {
        let n = 80;
        let d = euclid_matrix(n, 5);
        let lm = farthest_landmarks(&d, n, 16);
        let mut src = SliceRows { d: &d, n };
        let mut buf = Vec::new();
        compress_stream(&mut src, &lm, &mut buf).unwrap();
        let mut rd = MtzReader::open(buf, 8).unwrap(); // tiny cache to force eviction
        assert_eq!(rd.n(), n);
        // probe scattered cells, including repeats (exercise the cache)
        let probes = [(0, 0), (3, 70), (70, 3), (3, 70), (79, 79), (50, 12), (3, 70)];
        for &(i, j) in &probes {
            assert_eq!(rd.cell(i, j).unwrap(), d[i * n + j], "cell({i},{j}) mismatch");
        }
        // full sweep equals original
        for i in 0..n {
            assert_eq!(rd.row(i).unwrap(), d[i * n..i * n + n].to_vec());
        }
        // frame path (fast path disabled) must agree cell-for-cell and hit the cache
        rd.set_index_fast_path(false);
        for &(i, j) in &probes {
            assert_eq!(rd.cell(i, j).unwrap(), d[i * n + j], "frame cell({i},{j}) mismatch");
        }
        for i in 0..n {
            assert_eq!(rd.row(i).unwrap(), d[i * n..i * n + n].to_vec());
        }
        let (hits, misses) = rd.cache_stats();
        assert!(hits > 0, "cache never hit on repeated probes");
        assert!(misses > 0);
    }

    /// Exact integer gateway world (L1 metric, no rounding noise): regions
    /// joined pairwise by 3 distinct roads, each road k running gateway
    /// `gw[ra][k] ↔ gw[rb][k]`. Cross-region distances are min-plus exact
    /// through the gateway points, so with pivot-mined landmarks the
    /// cross-region blocks must be answerable from the resident index alone.
    fn l1_gateway_world(regions: usize, per: usize, seed: u64) -> (Vec<i32>, usize) {
        let n = regions * per;
        let mut s = seed;
        let mut rnd = |range: i64| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
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
        let gw = |r: usize, k: usize| r * per + k; // 3 gateways = first 3 points
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

    #[test]
    fn reader_index_fast_path_and_bounds() {
        let (mut d, n) = l1_gateway_world(4, 30, 9);
        // a dead point (snapping failure): unreachable from/to everywhere.
        // Its base goes >= UNREACHABLE too, so it must NOT poison its blocks.
        let dead = 47;
        for x in 0..n {
            if x != dead {
                d[x * n + dead] = UNREACHABLE;
                d[dead * n + x] = UNREACHABLE;
            }
        }
        let lm = pick_landmarks(&d, n, 16);
        let mut src = SliceRows { d: &d, n };
        let mut buf = Vec::new();
        let rep = compress_stream(&mut src, &lm, &mut buf).unwrap();
        assert!(rep.metric_ok(), "L1 gateway world should be a clean metric");
        let mut rd = MtzReader::open(buf, 4).unwrap();
        // every cell exact via cell(), regardless of path; bounds bracket all
        // metric cells (the dead point is a hard data error, outside the
        // bounds contract — but cell() must stay lossless even there)
        for i in 0..n {
            for j in 0..n {
                assert_eq!(rd.cell(i, j).unwrap(), d[i * n + j], "cell({i},{j})");
                if i == dead || j == dead {
                    continue;
                }
                let (lo, up) = rd.cell_bounds(i, j);
                assert!(
                    lo <= d[i * n + j] && d[i * n + j] <= up,
                    "bounds ({lo},{up}) miss d({i},{j})={}",
                    d[i * n + j]
                );
            }
        }
        // pivot-mined landmarks must capture the gateways: the bulk of the
        // (row, cell) blocks — all cross-region ones — become index-exact
        assert!(
            rd.exact_index_block_share() > 0.5,
            "expected most blocks index-exact, got {:.2}",
            rd.exact_index_block_share()
        );
        // landmark rows are always index-exact (their residual is identically 0)
        assert!(rd.exact_index_rows() >= lm.len(), "no index-exact rows found");
    }

    /// Legacy MTZS layout (dil embedded per frame) replicated byte-for-byte so
    /// blobs written by the previous version stay decodable.
    fn encode_stream_v1(d: &[i32], n: usize, lm: &[usize]) -> Vec<u8> {
        let l = lm.len();
        let mut dlj = vec![0i32; l * n];
        for (a, &la) in lm.iter().enumerate() {
            dlj[a * n..a * n + n].copy_from_slice(&d[la * n..la * n + n]);
        }
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC_STREAM);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&(l as u32).to_le_bytes());
        for &id in lm {
            out.extend_from_slice(&(id as u32).to_le_bytes());
        }
        put_blob(&mut out, &i32s_to_le(&dlj));
        for i in 0..n {
            let dil: Vec<i32> = lm.iter().map(|&a| d[i * n + a]).collect();
            let mut resid = vec![0i32; n];
            for j in 0..n {
                let mut base = i64::MAX;
                for a in 0..l {
                    let v = dil[a] as i64 + dlj[a * n + j] as i64;
                    if v < base {
                        base = v;
                    }
                }
                resid[j] = (d[i * n + j] as i64 - base) as i32;
            }
            let mut frame = i32s_to_le(&dil);
            frame.extend_from_slice(&i32s_to_le(&resid));
            put_blob(&mut out, &frame);
        }
        out
    }

    fn compress_hub_buf(d: &[i32], n: usize, opts: &HubOpts) -> (Vec<u8>, ValidationReport) {
        let mut src = SliceRows { d, n };
        let mut buf = Vec::new();
        let rep = compress_stream_hub(&mut src, opts, &mut buf).unwrap();
        (buf, rep)
    }

    #[test]
    fn mtzu_roundtrip_lossless() {
        // smooth euclidean, asymmetric tweak, and the gateway world — all must
        // roundtrip byte-for-byte through the path-label container
        let n = 90;
        let mut d = euclid_matrix(n, 21);
        for q in 1..7 {
            d[0 * n + q] += 31; // directed cells
        }
        let (buf, _) = compress_hub_buf(&d, n, &HubOpts::default());
        let (back, n2) = decompress(&buf).unwrap();
        assert_eq!(n2, n);
        assert_eq!(back, d, "MTZU euclid+asym roundtrip failed");

        let (g, gn) = l1_gateway_world(4, 25, 5);
        let (buf, rep) = compress_hub_buf(&g, gn, &HubOpts::default());
        assert!(rep.metric_ok());
        let (back, n2) = decompress(&buf).unwrap();
        assert_eq!(n2, gn);
        assert_eq!(back, g, "MTZU gateway roundtrip failed");

        // tiny edge cases
        let t = vec![0, 3, 5, 3, 0, 2, 5, 2, 0];
        let (buf, _) = compress_hub_buf(&t, 3, &HubOpts { hubs: 2, k: 2, candidates: 3, samples_per_row: 8 });
        let (back, n2) = decompress(&buf).unwrap();
        assert_eq!((back, n2), (t, 3));
    }

    #[test]
    fn mtzu_reader_exact_and_bounds() {
        let (mut d, n) = l1_gateway_world(4, 30, 13);
        let dead = 71;
        for x in 0..n {
            if x != dead {
                d[x * n + dead] = UNREACHABLE;
                d[dead * n + x] = UNREACHABLE;
            }
        }
        let (buf, _) = compress_hub_buf(&d, n, &HubOpts::default());
        let mut rd = MtzReader::open(buf, 4).unwrap();
        assert_eq!(rd.n(), n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(rd.cell(i, j).unwrap(), d[i * n + j], "MTZU cell({i},{j})");
                if i == dead || j == dead {
                    continue;
                }
                let (lo, up) = rd.cell_bounds(i, j);
                assert!(
                    lo <= d[i * n + j] && d[i * n + j] <= up,
                    "MTZU bounds ({lo},{up}) miss d({i},{j})={}",
                    d[i * n + j]
                );
                let w = rd.cell_within(i, j, 5).unwrap();
                assert!(
                    (w - d[i * n + j]).abs() <= 5,
                    "within(5) gave {w} for true {} at ({i},{j})",
                    d[i * n + j]
                );
            }
        }
        // path-labels must make a healthy share of blocks index-exact here
        assert!(
            rd.exact_index_block_share() > 0.3,
            "expected index-exact blocks, got {:.2}",
            rd.exact_index_block_share()
        );
        // frame path agrees with the index path
        rd.set_index_fast_path(false);
        for i in (0..n).step_by(7) {
            assert_eq!(rd.row(i).unwrap(), d[i * n..i * n + n].to_vec());
        }
    }

    #[test]
    fn mtzu_cell_within_tolerance_holds() {
        let n = 80;
        let d = euclid_matrix(n, 33);
        let (buf, _) = compress_hub_buf(&d, n, &HubOpts { hubs: 32, k: 6, candidates: 48, samples_per_row: 64 });
        let mut rd = MtzReader::open(buf, 8).unwrap();
        for i in 0..n {
            for j in 0..n {
                let v = rd.cell_within(i, j, 7).unwrap();
                let truth = d[i * n + j];
                assert!(
                    v >= truth - TRIANGLE_TOL as i32 && v <= truth + 7,
                    "within(7) gave {v} for true {truth} at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn legacy_mtzs_still_decodes() {
        let n = 60;
        let d = euclid_matrix(n, 13);
        let lm = farthest_landmarks(&d, n, 8);
        let v1 = encode_stream_v1(&d, n, &lm);
        let (back, n2) = decompress(&v1).unwrap();
        assert_eq!(n2, n);
        assert_eq!(back, d, "legacy MTZS roundtrip failed");
        // random access requires the new format
        assert!(MtzReader::open(v1, 4).is_err());
    }

    #[test]
    fn validation_flags_asymmetry() {
        let n = 40;
        let mut d = euclid_matrix(n, 3);
        // make it directed at a few cells
        for k in 1..6 {
            d[0 * n + k] += 50;
        }
        let rep = validate(&d, n);
        assert!(rep.asymmetric >= 1, "asymmetry not caught");
        assert!(!rep.metric_ok());
    }
}
