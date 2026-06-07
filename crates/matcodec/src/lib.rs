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

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::{self, Read, Write};

const MAGIC: &[u8; 4] = b"MTZ2";
/// Streamed container: framed per-row, bridge model only. Peak memory is the L
/// landmark rows plus one working row — never the full n². Fed by any
/// [`RowSource`] (e.g. dijeng's chunked many-to-many) and decoded row-by-row.
const MAGIC_STREAM: &[u8; 4] = b"MTZS";
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
    fn blob(&mut self) -> io::Result<Vec<i32>> {
        let len = self.u32()? as usize;
        if self.pos + len > self.b.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "blob"));
        }
        let raw = inflate(&self.b[self.pos..self.pos + len])?;
        self.pos += len;
        Ok(le_to_i32s(&raw))
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

/// Pick `l` landmarks by farthest-point sampling (for streaming compression /
/// the bridge model). Public wrapper over the internal sampler.
pub fn pick_landmarks(d: &[i32], n: usize, l: usize) -> Vec<usize> {
    farthest_landmarks(d, n, l)
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
/// as it passes. Peak memory is `L×n` (resident landmark rows) plus a working
/// row — independent of the full matrix size. Writes the `MTZS` framed
/// container, which [`decompress_rows`] decodes row-by-row. Returns the
/// validation report gathered during the single forward pass.
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
    out.write_all(MAGIC_STREAM)?;
    out.write_all(&(n as u32).to_le_bytes())?;
    out.write_all(&(l as u32).to_le_bytes())?;
    for &id in lm {
        out.write_all(&(id as u32).to_le_bytes())?;
    }
    let dljc = deflate(&i32s_to_le(&dlj));
    out.write_all(&(dljc.len() as u32).to_le_bytes())?;
    out.write_all(&dljc)?;

    let mut rep = ValidationReport::default();
    for i in 0..n {
        let row = src.row(i);
        let resid = validate_row(&mut rep, i, &row, n, lm, &dlj);
        let dil: Vec<i32> = lm.iter().map(|&a| row[a]).collect();
        let mut frame = i32s_to_le(&dil);
        frame.extend_from_slice(&i32s_to_le(&resid));
        let fc = deflate(&frame);
        out.write_all(&(fc.len() as u32).to_le_bytes())?;
        out.write_all(&fc)?;
    }
    Ok(rep)
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
/// Tiny capacity-bounded LRU of reconstructed rows (approximate, tick-based).
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
    fn get(&mut self, i: usize) -> Option<Vec<i32>> {
        self.tick += 1;
        let t = self.tick;
        if let Some(e) = self.map.get_mut(&i) {
            e.1 = t;
            self.hits += 1;
            Some(e.0.clone())
        } else {
            self.misses += 1;
            None
        }
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

/// Random-access reader over an `MTZS` (streamed) blob: keeps the compressed
/// bytes + the resident `L×n` landmark rows, indexes each row's frame, and
/// reconstructs a row on demand (inflate one small frame), caching hot rows in
/// an LRU. This turns the compressed matrix into a true in-RAM random-access
/// store — peak resident memory is `L×n` + the cache, never the full `n²`. The
/// compressed blob itself can be held in RAM or memory-mapped from disk.
pub struct MtzReader {
    bytes: Vec<u8>,
    n: usize,
    l: usize,
    lm: Vec<usize>,
    dlj: Vec<i32>,
    frame_off: Vec<usize>,
    cache: RowCache,
}

impl MtzReader {
    /// Open an `MTZS` blob with an LRU holding up to `cache_rows` reconstructed
    /// rows. Only the stream format supports cheap per-row random access.
    pub fn open(bytes: Vec<u8>, cache_rows: usize) -> io::Result<Self> {
        if bytes.len() < 12 || &bytes[0..4] != MAGIC_STREAM {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MtzReader requires the MTZS (--stream) format",
            ));
        }
        let (n, l, lm, dlj, frame_off);
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
            lm = lmv;
            dlj = dljv;
            frame_off = offs;
        }
        Ok(Self {
            bytes,
            n,
            l,
            lm,
            dlj,
            frame_off,
            cache: RowCache::new(cache_rows),
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn landmarks(&self) -> &[usize] {
        &self.lm
    }
    /// (cache hits, misses) so far.
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.cache.hits, self.cache.misses)
    }

    /// Reconstruct row `i` (cached). Inflates one frame on a miss.
    pub fn row(&mut self, i: usize) -> io::Result<Vec<i32>> {
        if let Some(r) = self.cache.get(i) {
            return Ok(r);
        }
        let off = self.frame_off[i];
        let len = u32::from_le_bytes([
            self.bytes[off],
            self.bytes[off + 1],
            self.bytes[off + 2],
            self.bytes[off + 3],
        ]) as usize;
        let raw = inflate(&self.bytes[off + 4..off + 4 + len])?;
        let frame = le_to_i32s(&raw);
        if frame.len() != self.l + self.n {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "frame size"));
        }
        let (dil, resid) = frame.split_at(self.l);
        let mut row = vec![0i32; self.n];
        for j in 0..self.n {
            let mut base = i64::MAX;
            for a in 0..self.l {
                let v = dil[a] as i64 + self.dlj[a * self.n + j] as i64;
                if v < base {
                    base = v;
                }
            }
            row[j] = (base + resid[j] as i64) as i32;
        }
        self.cache.put(i, row.clone());
        Ok(row)
    }

    /// Single cell `d(i,j)` via the row cache.
    pub fn cell(&mut self, i: usize, j: usize) -> io::Result<i32> {
        Ok(self.row(i)?[j])
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
        let (hits, misses) = rd.cache_stats();
        assert!(hits > 0, "cache never hit on repeated probes");
        assert!(misses > 0);
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
