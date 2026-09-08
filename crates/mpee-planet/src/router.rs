//! Snapping and shortest-path search over the planet dataset.
//!
//! The accuracy contract this implements is the one the whole format was
//! designed around:
//!
//!   * **Endpoints are approximate.** A query coordinate is snapped to the
//!     nearest point on the nearest *segment* — not the nearest junction,
//!     which could be a kilometre away. The snap is exact to the geometry we
//!     stored, so the error is the simplification tolerance (~1 m), not the
//!     junction spacing.
//!   * **Everything between them is exact.** Interior segments contribute
//!     `seg_len` verbatim: a `u64` sum of centimetre integers, measured on the
//!     original full-resolution OSM geometry with the WGS84 metric. No float
//!     accumulation, no dependence on the drawn line.
//!
//! Only the two partial end segments are prorated, and only those carry the
//! ~1 m ambiguity.

use crate::build::{attr_class, attr_kmh, attr_oneway, class_of};
use crate::dataset::Dataset;
use crate::geo;
use crate::overrides::{Effect, Overrides};
use crate::graph::{cell_of, CELL_E7};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub const SEG_MASK: u32 = 0x7fff_ffff;
pub const DIR_BIT: u32 = 0x8000_0000;

/// Where a query coordinate landed on the network.
#[derive(Clone, Debug)]
pub struct Snap {
    pub seg: u32,
    pub u: u32,
    pub v: u32,
    /// Fraction of the segment's length from `u` to the snapped point.
    pub t: f64,
    pub lat_e7: i32,
    pub lon_e7: i32,
    /// Ground distance from the query coordinate to the road, in metres.
    pub off_m: f64,
}

/// Traversal time of a segment in tenths of a second, or `None` when an
/// override closes it.
///
/// The `Option<&Overrides>` is the whole cost of the feature in the hot loop:
/// a null check that predicts perfectly, and — when overrides exist — one bit
/// test against a presence bitmap. The sorted table behind it is only touched
/// for the handful of segments that bit marks.
#[inline(always)]
fn edge_ds(ds: &Dataset, ov: Option<&Overrides>, sid: usize) -> Option<u32> {
    let len = ds.seg_len[sid];
    let kmh = attr_kmh(ds.attr(sid));
    if let Some(o) = ov {
        if o.touches(sid as u32) {
            return match o.effect(sid as u32) {
                Some(Effect::Closed) => None,
                Some(Effect::Speed(k)) => Some(dur_ds(len, k)),
                // A penalty must never round to zero, or a discouraged road
                // becomes a free one.
                Some(Effect::Penalty(f)) => Some(((dur_ds(len, kmh) as f32 * f) as u32).max(1)),
                None => Some(dur_ds(len, kmh)),
            };
        }
    }
    Some(dur_ds(len, kmh))
}

#[inline]
fn dur_ds(len_cm: u32, kmh: u16) -> u32 {
    // len_cm/100 m ÷ (kmh/3.6 m/s), in tenths of a second.
    ((len_cm as u64 * 36) / (kmh.max(1) as u64 * 100)).min(u32::MAX as u64) as u32
}

/// Metres-per-1e7-degree at a latitude, for the local planar projection used
/// when measuring how far a query point is from a road.
#[inline]
fn scale(lat_e7: i32) -> (f64, f64) {
    let lat = geo::e7_to_deg(lat_e7);
    let mlat = 111_132.92 - 559.82 * (2.0 * lat.to_radians()).cos();
    let mlon = 111_412.84 * lat.to_radians().cos() - 93.5 * (3.0 * lat.to_radians()).cos();
    (mlat * 1e-7, mlon.abs().max(1e-6) * 1e-7)
}

impl Dataset {
    /// Nearest point on the road network within `max_m` metres.
    pub fn snap(&self, lat_e7: i32, lon_e7: i32, max_m: f64) -> Option<Snap> {
        let (sy, sx) = scale(lat_e7);
        let mut best: Option<(f64, Snap)> = None;
        let cell_m = CELL_E7 as f64 * sy;
        let mut ring = 0i64;
        loop {
            let mut looked = false;
            for dy in -ring..=ring {
                for dx in -ring..=ring {
                    if dy.abs() != ring && dx.abs() != ring {
                        continue;
                    }
                    looked = true;
                    let key = cell_of(
                        lat_e7 + (dy * CELL_E7 as i64) as i32,
                        lon_e7 + (dx * CELL_E7 as i64) as i32,
                    );
                    for &sid in self.cell_segments(key) {
                        if let Some(c) = self.project(sid as usize, lat_e7, lon_e7, sy, sx) {
                            if best.as_ref().is_none_or(|(b, _)| c.0 < *b) {
                                best = Some(c);
                            }
                        }
                    }
                }
            }
            // Stop once the next ring cannot hold anything closer.
            if let Some((b, _)) = &best {
                if b.sqrt() <= ring as f64 * cell_m {
                    break;
                }
            }
            ring += 1;
            if ring as f64 * cell_m > max_m * 2.0 + cell_m || !looked && ring > 4 {
                break;
            }
        }
        best.filter(|(d2, _)| d2.sqrt() <= max_m).map(|(d2, mut s)| {
            s.off_m = d2.sqrt();
            s
        })
    }

    /// Closest point on one segment, as (squared metres, snap).
    fn project(
        &self,
        sid: usize,
        qlat: i32,
        qlon: i32,
        sy: f64,
        sx: f64,
    ) -> Option<(f64, Snap)> {
        let pts = self.seg_geometry(sid);
        if pts.len() < 2 {
            return None;
        }
        let qx = qlon as f64 * sx;
        let qy = qlat as f64 * sy;
        let mut best_d2 = f64::INFINITY;
        let mut best_run = 0.0f64;
        let mut best_pt = pts[0];
        let mut run = 0.0f64;
        for w in pts.windows(2) {
            let (ax, ay) = (w[0].1 as f64 * sx, w[0].0 as f64 * sy);
            let (bx, by) = (w[1].1 as f64 * sx, w[1].0 as f64 * sy);
            let (dx, dy) = (bx - ax, by - ay);
            let l2 = dx * dx + dy * dy;
            let t = if l2 <= f64::EPSILON {
                0.0
            } else {
                (((qx - ax) * dx + (qy - ay) * dy) / l2).clamp(0.0, 1.0)
            };
            let (px, py) = (ax + t * dx, ay + t * dy);
            let d2 = (qx - px) * (qx - px) + (qy - py) * (qy - py);
            let seglen = l2.sqrt();
            if d2 < best_d2 {
                best_d2 = d2;
                best_run = run + t * seglen;
                best_pt = (
                    (w[0].0 as f64 + t * (w[1].0 - w[0].0) as f64).round() as i32,
                    (w[0].1 as f64 + t * (w[1].1 - w[0].1) as f64).round() as i32,
                );
            }
            run += seglen;
        }
        let t = if run > 0.0 { (best_run / run).clamp(0.0, 1.0) } else { 0.0 };
        Some((
            best_d2,
            Snap {
                seg: sid as u32,
                u: self.seg_u[sid],
                v: self.seg_v[sid],
                t,
                lat_e7: best_pt.0,
                lon_e7: best_pt.1,
                off_m: 0.0,
            },
        ))
    }
}

/// One traversed segment of the answer.
#[derive(Clone, Copy, Debug)]
pub struct Leg {
    pub seg: u32,
    /// True when the segment is traversed against its stored geometry.
    pub reverse: bool,
    /// Centimetres actually travelled on this segment (partial at the ends).
    pub len_cm: u64,
}

pub struct Route {
    /// Exact travelled distance: an integer sum of centimetres.
    pub dist_cm: u64,
    pub dur_ds: u64,
    pub legs: Vec<Leg>,
    /// Vertices passed through, in order. A toll gate is a vertex, so this is
    /// what the toll layer is matched against.
    pub vertices: Vec<u32>,
    pub from: Snap,
    pub to: Snap,
}

impl Route {
    pub fn dist_m(&self) -> f64 {
        self.dist_cm as f64 / 100.0
    }
    pub fn dur_s(&self) -> f64 {
        self.dur_ds as f64 / 10.0
    }
}

/// Reusable search buffers.
///
/// At planet scale these arrays are the memory footprint of the whole engine:
/// 270 M vertices × 4 bytes × two directions is 2.2 GB *per array*. A
/// generation-stamp design would need a third pair and cost another 2.2 GB, so
/// instead each search records the vertices it touched and resets exactly
/// those afterwards — a query settles a few million vertices, not 270 million,
/// so the reset is far cheaper than the memory it saves.
pub struct Router {
    dist: [Vec<u32>; 2],
    parent: [Vec<u32>; 2],
    touched: [Vec<u32>; 2],
    pub settled: u64,
}

const NONE: u32 = u32::MAX;
const INF: u32 = u32::MAX;

impl Router {
    pub fn new(n: usize) -> Router {
        Router {
            dist: [vec![INF; n], vec![INF; n]],
            parent: [vec![NONE; n], vec![NONE; n]],
            touched: [Vec::new(), Vec::new()],
            settled: 0,
        }
    }

    /// Return the buffers to their all-INF state by undoing only what the last
    /// query wrote.
    fn reset(&mut self) {
        for s in 0..2 {
            for &v in &self.touched[s] {
                self.dist[s][v as usize] = INF;
                self.parent[s][v as usize] = NONE;
            }
            self.touched[s].clear();
        }
    }

    #[inline]
    fn d(&self, s: usize, v: u32) -> u32 {
        self.dist[s][v as usize]
    }

    #[inline]
    fn relax(&mut self, s: usize, v: u32, nd: u32, par: u32) -> bool {
        let i = v as usize;
        if self.dist[s][i] > nd {
            if self.dist[s][i] == INF {
                self.touched[s].push(v);
            }
            self.dist[s][i] = nd;
            self.parent[s][i] = par;
            true
        } else {
            false
        }
    }

    pub fn route(&mut self, ds: &Dataset, from: &Snap, to: &Snap) -> Option<Route> {
        self.route_with(ds, from, to, None)
    }

    /// As [`Router::route`], honouring local overrides.
    pub fn route_with(
        &mut self,
        ds: &Dataset,
        from: &Snap,
        to: &Snap,
        ov: Option<&Overrides>,
    ) -> Option<Route> {
        self.reset();
        self.settled = 0;
        // An override with no live rules is the same as none at all, and
        // skipping it keeps the branch out of the loop entirely.
        let ov = ov.filter(|o| !o.is_empty());

        // A route that never leaves the segment it started on.
        if let Some(r) = same_segment(ds, from, to, ov) {
            return Some(r);
        }

        let mut hf: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        let mut hb: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        // Entry costs: the part of the start segment still to be driven.
        for (v, cm) in seg_exits(ds, from, ov) {
            let d = dur_ds(cm as u32, eff_kmh(ds, ov, from.seg));
            if self.relax(0, v, d, NONE) {
                hf.push(Reverse((d, v)));
            }
        }
        for (v, cm) in seg_entries(ds, to, ov) {
            let d = dur_ds(cm as u32, eff_kmh(ds, ov, to.seg));
            if self.relax(1, v, d, NONE) {
                hb.push(Reverse((d, v)));
            }
        }
        if hf.is_empty() || hb.is_empty() {
            return None;
        }

        let mut best = u32::MAX;
        let mut meet = NONE;
        loop {
            let tf = hf.peek().map(|r| r.0 .0).unwrap_or(u32::MAX);
            let tb = hb.peek().map(|r| r.0 .0).unwrap_or(u32::MAX);
            if tf == u32::MAX && tb == u32::MAX {
                break;
            }
            // The classic termination bound: no path can beat `best` once the
            // two frontiers together already cost that much.
            if tf.saturating_add(tb) >= best {
                break;
            }
            let side = if tf <= tb { 0 } else { 1 };
            let Reverse((d, u)) = if side == 0 { hf.pop().unwrap() } else { hb.pop().unwrap() };
            if d > self.d(side, u) {
                continue;
            }
            self.settled += 1;
            let fwd = side == 0;
            let (head, segs) = if fwd { (ds.head, ds.eseg) } else { (ds.rhead, ds.rseg) };
            let (s, e) = (head[u as usize] as usize, head[u as usize + 1] as usize);
            // The index addresses several parallel arrays, not just this one.
            #[allow(clippy::needless_range_loop)]
            for k in s..e {
                // Targets are stored as deltas from the source, so they are
                // decoded here rather than read.
                let v = if fwd { ds.target(u, k) } else { ds.rtarget(u, k) };
                let sid = (segs[k] & SEG_MASK) as usize;
                let Some(w) = edge_ds(ds, ov, sid) else { continue };
                let nd = d.saturating_add(w);
                if nd >= best {
                    continue;
                }
                if self.relax(side, v, nd, u) {
                    let heap = if side == 0 { &mut hf } else { &mut hb };
                    heap.push(Reverse((nd, v)));
                    let other = self.d(1 - side, v);
                    if other != u32::MAX && nd.saturating_add(other) < best {
                        best = nd + other;
                        meet = v;
                    }
                }
            }
        }
        if meet == NONE {
            return None;
        }
        let r = self.build_route(ds, from, to, meet, ov);
        Some(r)
    }

    fn build_route(
        &self,
        ds: &Dataset,
        from: &Snap,
        to: &Snap,
        meet: u32,
        ov: Option<&Overrides>,
    ) -> Route {
        // Forward half: walk parents back to the entry vertex.
        let mut fwd: Vec<u32> = Vec::new();
        let mut v = meet;
        loop {
            fwd.push(v);
            let p = self.parent[0][v as usize];
            if p == NONE || self.dist[0][v as usize] == INF {
                break;
            }
            v = p;
        }
        fwd.reverse();
        // Backward half: parents already point forwards along the route.
        let mut bwd: Vec<u32> = Vec::new();
        let mut v = meet;
        loop {
            let p = self.parent[1][v as usize];
            if p == NONE || self.dist[1][v as usize] == INF {
                break;
            }
            bwd.push(p);
            v = p;
        }
        let mut path = fwd;
        path.extend(bwd);

        let mut legs: Vec<Leg> = Vec::new();
        let mut dist_cm: u64 = 0;
        let mut dur: u64 = 0;

        // Leading partial segment.
        let entry = path[0];
        if let Some((cm, rev)) = partial_from(ds, from, entry) {
            legs.push(Leg { seg: from.seg, reverse: rev, len_cm: cm });
            dist_cm += cm;
            dur += dur_ds(cm as u32, eff_kmh(ds, ov, from.seg)) as u64;
        }
        // Interior: every whole segment contributes its exact stored length.
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (s, e) = (ds.head[a as usize] as usize, ds.head[a as usize + 1] as usize);
            let mut pick: Option<(usize, u32)> = None;
            for k in s..e {
                if ds.target(a, k) != b {
                    continue;
                }
                let sid = (ds.eseg[k] & SEG_MASK) as usize;
                // A closed segment was never relaxed, so it cannot be the edge
                // the search actually used.
                if edge_ds(ds, ov, sid).is_none() {
                    continue;
                }
                // Pick by *duration*, not length: the search minimised time,
                // so choosing the shorter of two parallel edges can report a
                // journey slower than the one actually found. The distance was
                // always right; the duration was not.
                let Some(w) = edge_ds(ds, ov, sid) else { continue };
                if pick.is_none_or(|(_, bw)| w < bw) {
                    pick = Some((k, w));
                }
            }
            if let Some((k, w)) = pick {
                let sid = (ds.eseg[k] & SEG_MASK) as usize;
                let l = ds.seg_len[sid] as u64;
                legs.push(Leg {
                    seg: sid as u32,
                    reverse: ds.eseg[k] & DIR_BIT != 0,
                    len_cm: l,
                });
                dist_cm += l;
                dur += w as u64;
            }
        }
        // Trailing partial segment.
        let exit = *path.last().unwrap();
        if let Some((cm, rev)) = partial_to(ds, to, exit) {
            legs.push(Leg { seg: to.seg, reverse: rev, len_cm: cm });
            dist_cm += cm;
            dur += dur_ds(cm as u32, eff_kmh(ds, ov, to.seg)) as u64;
        }
        Route { dist_cm, dur_ds: dur, legs, vertices: path, from: from.clone(), to: to.clone() }
    }
}

/// Speed to use on a partially-driven end segment, after overrides.
#[inline]
fn eff_kmh(ds: &Dataset, ov: Option<&Overrides>, sid: u32) -> u16 {
    let base = attr_kmh(ds.attr(sid as usize));
    match ov.filter(|o| o.touches(sid)).and_then(|o| o.effect(sid)) {
        Some(Effect::Speed(k)) => k,
        Some(Effect::Penalty(f)) if f > 0.0 => ((base as f32 / f).round() as u16).max(1),
        _ => base,
    }
}

/// Vertices reachable by finishing the segment the trip starts on, with the
/// centimetres that costs.
pub(crate) fn seg_exits(ds: &Dataset, s: &Snap, ov: Option<&Overrides>) -> Vec<(u32, u64)> {
    let sid = s.seg as usize;
    // Starting on a closed road: there is nowhere to go from it.
    if edge_ds(ds, ov, sid).is_none() {
        return Vec::new();
    }
    let len = ds.seg_len[sid] as f64;
    let ow = attr_oneway(ds.attr(sid));
    let mut v = Vec::with_capacity(2);
    if ow == 0 || ow == 1 {
        v.push((s.v, ((1.0 - s.t) * len).round() as u64));
    }
    if ow == 0 || ow == 2 {
        v.push((s.u, (s.t * len).round() as u64));
    }
    v
}

/// Vertices from which the destination point can be reached along its segment.
pub(crate) fn seg_entries(ds: &Dataset, s: &Snap, ov: Option<&Overrides>) -> Vec<(u32, u64)> {
    let sid = s.seg as usize;
    if edge_ds(ds, ov, sid).is_none() {
        return Vec::new();
    }
    let len = ds.seg_len[sid] as f64;
    let ow = attr_oneway(ds.attr(sid));
    let mut v = Vec::with_capacity(2);
    if ow == 0 || ow == 1 {
        v.push((s.u, (s.t * len).round() as u64));
    }
    if ow == 0 || ow == 2 {
        v.push((s.v, ((1.0 - s.t) * len).round() as u64));
    }
    v
}

fn partial_from(ds: &Dataset, s: &Snap, entry: u32) -> Option<(u64, bool)> {
    let len = ds.seg_len[s.seg as usize] as f64;
    if entry == s.v {
        Some((((1.0 - s.t) * len).round() as u64, false))
    } else if entry == s.u {
        Some(((s.t * len).round() as u64, true))
    } else {
        None
    }
}

fn partial_to(ds: &Dataset, s: &Snap, exit: u32) -> Option<(u64, bool)> {
    let len = ds.seg_len[s.seg as usize] as f64;
    if exit == s.u {
        Some(((s.t * len).round() as u64, false))
    } else if exit == s.v {
        Some((((1.0 - s.t) * len).round() as u64, true))
    } else {
        None
    }
}

/// Both endpoints on the same segment: no search needed, and the answer is a
/// straight proration of that segment's exact length.
fn same_segment(ds: &Dataset, a: &Snap, b: &Snap, ov: Option<&Overrides>) -> Option<Route> {
    if a.seg != b.seg {
        return None;
    }
    let sid = a.seg as usize;
    edge_ds(ds, ov, sid)?;
    let ow = attr_oneway(ds.attr(sid));
    let forward = b.t >= a.t;
    let legal = match ow {
        0 => true,
        1 => forward,
        2 => !forward,
        _ => false,
    };
    if !legal {
        return None;
    }
    let len = ds.seg_len[sid] as f64;
    let cm = ((b.t - a.t).abs() * len).round() as u64;
    Some(Route {
        dist_cm: cm,
        dur_ds: dur_ds(cm as u32, eff_kmh(ds, ov, a.seg)) as u64,
        legs: vec![Leg { seg: a.seg, reverse: !forward, len_cm: cm }],
        vertices: Vec::new(),
        from: a.clone(),
        to: b.clone(),
    })
}

/// Full drawn polyline of a route, in degrees.
pub fn route_geometry(ds: &Dataset, r: &Route) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    let push = |out: &mut Vec<(f64, f64)>, p: (i32, i32)| {
        let q = (geo::e7_to_deg(p.0), geo::e7_to_deg(p.1));
        if out.last() != Some(&q) {
            out.push(q);
        }
    };
    push(&mut out, (r.from.lat_e7, r.from.lon_e7));
    for (i, leg) in r.legs.iter().enumerate() {
        let mut pts = ds.seg_geometry(leg.seg as usize);
        if leg.reverse {
            pts.reverse();
        }
        // The first and last legs are entered mid-segment; keep only the part
        // beyond the snapped point.
        let first = i == 0;
        let last = i == r.legs.len() - 1;
        for (j, p) in pts.iter().enumerate() {
            if first && j == 0 {
                continue;
            }
            if last && j == pts.len() - 1 {
                continue;
            }
            push(&mut out, *p);
        }
    }
    push(&mut out, (r.to.lat_e7, r.to.lon_e7));
    out
}

/// Turn-by-turn-ish summary: consecutive legs sharing a street name are merged.
pub fn route_steps(ds: &Dataset, r: &Route) -> Vec<(String, u64, u8)> {
    let mut out: Vec<(String, u64, u8)> = Vec::new();
    for leg in &r.legs {
        let name = ds.street_name(leg.seg as usize).unwrap_or("").to_string();
        let cls = attr_class(ds.attr(leg.seg as usize));
        match out.last_mut() {
            Some(l) if l.0 == name && l.2 == cls => l.1 += leg.len_cm,
            _ => out.push((name, leg.len_cm, cls)),
        }
    }
    out
}

/// Toll gates the route drives through, in order, with the direction of
/// travel expressed as the vertices either side — enough for a tariff table to
/// charge one direction only.
pub struct TollHit {
    pub gate: usize,
    pub from_vertex: u32,
    pub to_vertex: u32,
    pub price: Option<(f64, String)>,
}

pub fn route_tolls(r: &Route, tolls: &crate::toll::TollIndex) -> Vec<TollHit> {
    let mut out = Vec::new();
    for (i, &v) in r.vertices.iter().enumerate() {
        // A gate only counts when the route passes *through* it: starting or
        // finishing at a booth is not a crossing.
        if i == 0 || i + 1 == r.vertices.len() {
            continue;
        }
        if let Some(g) = tolls.is_gate(v) {
            out.push(TollHit {
                gate: g,
                from_vertex: r.vertices[i - 1],
                to_vertex: r.vertices[i + 1],
                price: tolls.price(g),
            });
        }
    }
    out
}

pub fn class_name(c: u8) -> &'static str {
    match class_of(c) {
        crate::profile::Class::Motorway => "motorway",
        crate::profile::Class::MotorwayLink => "motorway_link",
        crate::profile::Class::Trunk => "trunk",
        crate::profile::Class::TrunkLink => "trunk_link",
        crate::profile::Class::Primary => "primary",
        crate::profile::Class::PrimaryLink => "primary_link",
        crate::profile::Class::Secondary => "secondary",
        crate::profile::Class::SecondaryLink => "secondary_link",
        crate::profile::Class::Tertiary => "tertiary",
        crate::profile::Class::TertiaryLink => "tertiary_link",
        crate::profile::Class::Unclassified => "unclassified",
        crate::profile::Class::Residential => "residential",
        crate::profile::Class::LivingStreet => "living_street",
        crate::profile::Class::Service => "service",
        crate::profile::Class::Ferry => "ferry",
        crate::profile::Class::Other => "road",
    }
}
