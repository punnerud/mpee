//! Local edits, applied without a rebuild.
//!
//! A road closes for roadworks, a speed limit is wrong, a bridge is out. The
//! dataset takes 55 minutes to rebuild and the graph it produces is an
//! immutable mapping, so neither waiting nor editing in place is an option.
//! Overrides are therefore a thin, mutable layer the router consults — stored
//! in mpedb, written by any process, and visible to a running server within
//! one request.
//!
//! # What an override is keyed on, and why it matters
//!
//! Not the segment id. Segment ids are **build-local**: the next rebuild
//! renumbers every one of them, so an override stored against `seg 41209883`
//! would silently start applying to a different road — the same class of
//! failure as a price column that parses to zero. It would be a bug that looks
//! like data.
//!
//! So an override is keyed on what survives a rebuild: **a position on the
//! ground**, optionally narrowed by street name. It is resolved to whatever
//! segments occupy that position in the dataset that is live *now*. Resolution
//! is reported, not assumed — an override that matches nothing says so.
//!
//! # Cost in the hot loop
//!
//! A planet route relaxes ~11 million edges, so "is this segment overridden?"
//! has to be answered in a bit test, not a lookup. Resolution therefore
//! produces a presence bitmap over segment ids (42 MB for the planet, one bit
//! each) and a small sorted table behind it. Edges that are not overridden —
//! which is essentially all of them — cost one `and`.

use crate::dataset::Dataset;
use crate::geo;
use crate::graph::{cell_of, CELL_E7};
use mpedb::{Config, Database, ExecResult, Value};
use std::io;
use std::path::Path;

const SCHEMA: &str = r#"
[[table]]
name = "road_override"
primary_key = ["override_id"]
  [[table.column]]
  name = "override_id"
  type = "int64"
  [[table.column]]
  name = "kind"
  type = "text"
  nullable = false
  indexed = true
  [[table.column]]
  name = "lat_e7"
  type = "int64"
  nullable = false
  [[table.column]]
  name = "lon_e7"
  type = "int64"
  nullable = false
  [[table.column]]
  name = "radius_m"
  type = "int64"
  nullable = false
  [[table.column]]
  name = "street"
  type = "text"
  [[table.column]]
  name = "speed_kmh"
  type = "int64"
  [[table.column]]
  name = "factor"
  type = "numeric"
  [[table.column]]
  name = "note"
  type = "text"
  [[table.column]]
  name = "author"
  type = "text"
  [[table.column]]
  name = "created"
  type = "timestamp"
  nullable = false
  [[table.column]]
  name = "expires"
  type = "timestamp"

[[table]]
name = "override_meta"
primary_key = ["key"]
  [[table.column]]
  name = "key"
  type = "text"
  [[table.column]]
  name = "value"
  type = "int64"
  nullable = false
"#;

pub const KIND_CLOSED: &str = "closed";
pub const KIND_SPEED: &str = "speed";
pub const KIND_PENALTY: &str = "penalty";

fn err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

pub fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

fn toml_path(p: &Path) -> String {
    let mut o = String::from("\"");
    for c in p.display().to_string().chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// What an override does to an edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Effect {
    /// Impassable — the edge is not relaxed at all.
    Closed,
    /// Drive it at this speed instead of the tagged one.
    Speed(u16),
    /// Multiply the traversal time. `> 1` discourages, `< 1` encourages.
    Penalty(f32),
}

/// One row as stored.
#[derive(Clone, Debug)]
pub struct Rule {
    pub id: i64,
    pub kind: String,
    pub lat_e7: i32,
    pub lon_e7: i32,
    pub radius_m: i64,
    pub street: Option<String>,
    pub speed_kmh: Option<u16>,
    pub factor: Option<f32>,
    pub note: String,
    pub author: String,
    pub created: i64,
    pub expires: Option<i64>,
}

impl Rule {
    fn effect(&self) -> Option<Effect> {
        match self.kind.as_str() {
            KIND_CLOSED => Some(Effect::Closed),
            KIND_SPEED => self.speed_kmh.map(Effect::Speed),
            KIND_PENALTY => self.factor.map(Effect::Penalty),
            _ => None,
        }
    }
    fn live_at(&self, now: i64) -> bool {
        self.expires.map(|e| e > now).unwrap_or(true)
    }
}

// ------------------------------------------------------------------ store

pub struct OverrideDb {
    db: Database,
}

impl OverrideDb {
    pub fn open(dir: &Path) -> io::Result<OverrideDb> {
        let toml = format!(
            "[database]\npath = {}\nsize_mb = 64\nmax_readers = 16\ndurability = \"commit\"\n{SCHEMA}",
            toml_path(&dir.join("overrides.mpedb"))
        );
        let db = Database::open_with_config(Config::from_toml_str(&toml).map_err(err)?)
            .map_err(err)?;
        Ok(OverrideDb { db })
    }


    fn rows(&self, sql: &str) -> io::Result<Vec<Vec<Value>>> {
        match self.db.query(sql, &[]).map_err(err)? {
            ExecResult::Rows { rows, .. } => Ok(rows),
            _ => Ok(Vec::new()),
        }
    }

    /// A counter bumped on every write.
    ///
    /// This is what lets a running server notice an edit made by another
    /// process: it is one point read per request, which is nothing beside a
    /// route, and it means an override takes effect on the next query rather
    /// than the next restart.
    pub fn generation(&self) -> i64 {
        self.rows("SELECT value FROM override_meta WHERE key = 'generation'")
            .ok()
            .and_then(|r| r.into_iter().next())
            .and_then(|r| match r.into_iter().next() {
                Some(Value::Int(i)) => Some(i),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Run a mutation and bump the generation **in one transaction**.
    ///
    /// Both halves matter. Reading inside the session means the writer lock
    /// covers the read too, which is what an id allocation needs: doing the
    /// `SELECT MAX(id)` outside and the `INSERT` inside is a read-then-write
    /// race, and under four concurrent writers it failed 27 times out of 40
    /// with a primary-key violation. mpedb refused the duplicates rather than
    /// accepting them — the bug was here, and the schema is what surfaced it.
    ///
    /// Bumping in the same transaction closes the other half: a committed edit
    /// that readers never learn about is an edit that silently does nothing
    /// until the next restart.
    fn with_write<T>(
        &self,
        f: impl FnOnce(&mut mpedb::WriteSession) -> io::Result<T>,
    ) -> io::Result<T> {
        let mut s = self.db.begin().map_err(err)?;
        let out = f(&mut s)?;
        // Increment in place: no value is read out and written back, so there
        // is nothing to lose to a concurrent writer.
        let bumped = matches!(
            s.query("UPDATE override_meta SET value = value + 1 WHERE key = 'generation'", &[])
                .map_err(err)?,
            ExecResult::Affected(n) if n > 0
        );
        if !bumped {
            s.query("INSERT INTO override_meta (key, value) VALUES ('generation', 1)", &[])
                .map_err(err)?;
        }
        s.commit().map_err(err)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &self,
        kind: &str,
        lat: f64,
        lon: f64,
        radius_m: i64,
        street: Option<&str>,
        speed_kmh: Option<u16>,
        factor: Option<f32>,
        note: &str,
        author: &str,
        expires_us: Option<i64>,
    ) -> io::Result<i64> {
        // Refuse a rule that cannot mean anything, rather than storing one that
        // silently does nothing.
        match kind {
            KIND_CLOSED => {}
            KIND_SPEED if speed_kmh.filter(|k| *k > 0).is_some() => {}
            KIND_PENALTY if factor.filter(|f| *f > 0.0).is_some() => {}
            KIND_SPEED => return Err(io::Error::other("a speed override needs --speed > 0")),
            KIND_PENALTY => return Err(io::Error::other("a penalty override needs --factor > 0")),
            other => return Err(io::Error::other(format!("unknown override kind {other:?}"))),
        }
        self.with_write(|s| {
            // Allocated under the writer lock, so two processes cannot pick
            // the same id.
            let id = match s
                .query("SELECT override_id FROM road_override ORDER BY override_id DESC", &[])
                .map_err(err)?
            {
                ExecResult::Rows { rows, .. } => match rows.first().and_then(|r| r.first()) {
                    Some(Value::Int(i)) => i + 1,
                    _ => 1,
                },
                _ => 1,
            };
            s.query(
                "INSERT INTO road_override (override_id, kind, lat_e7, lon_e7, radius_m, street, \
                 speed_kmh, factor, note, author, created, expires) \
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)",
            &[
                Value::Int(id),
                Value::Text(kind.into()),
                Value::Int((lat / geo::E7).round() as i64),
                Value::Int((lon / geo::E7).round() as i64),
                Value::Int(radius_m),
                street.map(|s| Value::Text(s.into())).unwrap_or(Value::Null),
                speed_kmh.map(|k| Value::Int(k as i64)).unwrap_or(Value::Null),
                factor.map(|f| Value::Numeric(format!("{f:.3}"))).unwrap_or(Value::Null),
                Value::Text(note.into()),
                Value::Text(author.into()),
                Value::Timestamp(now_us()),
                expires_us.map(Value::Timestamp).unwrap_or(Value::Null),
            ],
            )
            .map_err(err)?;
            Ok(id)
        })
    }

    pub fn remove(&self, id: i64) -> io::Result<bool> {
        self.with_write(|s| {
            let r = s
                .query("DELETE FROM road_override WHERE override_id = $1", &[Value::Int(id)])
                .map_err(err)?;
            Ok(matches!(r, ExecResult::Affected(n) if n > 0))
        })
    }

    pub fn list(&self) -> io::Result<Vec<Rule>> {
        let rows = self.rows(
            "SELECT override_id, kind, lat_e7, lon_e7, radius_m, street, speed_kmh, factor, \
             note, author, created, expires FROM road_override ORDER BY override_id",
        )?;
        let txt = |v: &Value| match v {
            Value::Text(s) => Some(s.clone()),
            _ => None,
        };
        let int = |v: &Value| match v {
            Value::Int(i) => Some(*i),
            Value::Timestamp(i) => Some(*i),
            _ => None,
        };
        Ok(rows
            .into_iter()
            .map(|r| Rule {
                id: int(&r[0]).unwrap_or(0),
                kind: txt(&r[1]).unwrap_or_default(),
                lat_e7: int(&r[2]).unwrap_or(0) as i32,
                lon_e7: int(&r[3]).unwrap_or(0) as i32,
                radius_m: int(&r[4]).unwrap_or(0),
                street: txt(&r[5]),
                speed_kmh: int(&r[6]).map(|v| v as u16),
                factor: match &r[7] {
                    Value::Numeric(s) => s.parse().ok(),
                    Value::Float(f) => Some(*f as f32),
                    _ => None,
                },
                note: txt(&r[8]).unwrap_or_default(),
                author: txt(&r[9]).unwrap_or_default(),
                created: int(&r[10]).unwrap_or(0),
                expires: int(&r[11]),
            })
            .collect())
    }

    pub fn query(&self, sql: &str) -> io::Result<ExecResult> {
        self.db.query(sql, &[]).map_err(err)
    }
}

// --------------------------------------------------------------- resolved

/// Overrides bound to the segment ids of the dataset that is live now.
pub struct Overrides {
    /// One bit per segment: "consult the table for this one".
    present: Vec<u64>,
    /// `(segment, effect)`, sorted by segment.
    entries: Vec<(u32, Effect)>,
    pub generation: i64,
    /// Per-rule resolution: how many segments each rule matched.
    pub matched: Vec<(i64, usize)>,
}

impl Overrides {
    pub fn empty() -> Overrides {
        Overrides { present: Vec::new(), entries: Vec::new(), generation: 0, matched: Vec::new() }
    }

    #[inline(always)]
    pub fn touches(&self, sid: u32) -> bool {
        let w = (sid >> 6) as usize;
        match self.present.get(w) {
            Some(x) => x & (1u64 << (sid & 63)) != 0,
            None => false,
        }
    }

    pub fn effect(&self, sid: u32) -> Option<Effect> {
        self.entries
            .binary_search_by_key(&sid, |e| e.0)
            .ok()
            .map(|i| self.entries[i].1)
    }

    /// The segments these overrides touch, ascending.
    pub fn segments(&self) -> impl Iterator<Item = u32> + '_ {
        self.entries.iter().map(|e| e.0)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bind rules to segments in `ds`.
    ///
    /// A rule names a place, not an id, so this is where it becomes concrete —
    /// and where a rule that matches nothing becomes visible instead of
    /// quietly doing nothing.
    pub fn resolve(ds: &Dataset, rules: &[Rule], generation: i64) -> Overrides {
        let now = now_us();
        let nseg = ds.n_segments();
        let mut present = vec![0u64; nseg.div_ceil(64)];
        let mut map: std::collections::HashMap<u32, Effect> = Default::default();
        let mut matched = Vec::new();
        for r in rules {
            let Some(eff) = r.effect() else { continue };
            if !r.live_at(now) {
                matched.push((r.id, 0));
                continue;
            }
            let segs = segments_near(ds, r.lat_e7, r.lon_e7, r.radius_m as f64, r.street.as_deref());
            matched.push((r.id, segs.len()));
            for s in segs {
                // Later rules win, so an edit can be corrected by a newer one
                // without deleting the old.
                map.insert(s, eff);
            }
        }
        let mut entries: Vec<(u32, Effect)> = map.into_iter().collect();
        entries.sort_unstable_by_key(|e| e.0);
        for (s, _) in &entries {
            present[(*s >> 6) as usize] |= 1u64 << (*s & 63);
        }
        Overrides { present, entries, generation, matched }
    }
}

/// Segments whose drawn line passes within `radius_m` of a point, optionally
/// filtered to one street name.
pub fn segments_near(
    ds: &Dataset,
    lat_e7: i32,
    lon_e7: i32,
    radius_m: f64,
    street: Option<&str>,
) -> Vec<u32> {
    let rings = ((radius_m / (CELL_E7 as f64 * 1.11e-2)).ceil() as i64).max(1);
    let mut out: Vec<u32> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let want = street.map(crate::addresses::normalize);
    for dy in -rings..=rings {
        for dx in -rings..=rings {
            let key = cell_of(
                lat_e7 + (dy * CELL_E7 as i64) as i32,
                lon_e7 + (dx * CELL_E7 as i64) as i32,
            );
            for &sid in ds.cell_segments(key) {
                if !seen.insert(sid) {
                    continue;
                }
                if let Some(w) = &want {
                    let name = ds.street_name(sid as usize).unwrap_or("");
                    if crate::addresses::normalize(name) != *w {
                        continue;
                    }
                }
                if seg_distance_m(ds, sid, lat_e7, lon_e7) <= radius_m {
                    out.push(sid);
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// Street names actually present within `radius_m` of a point.
///
/// A rule that matches nothing is usually a name that is not the one the data
/// carries — a motorway is identified by its `ref` ("E6") while the dataset
/// stores its `name`, which is often empty. Answering "here is what is there"
/// turns a dead end into the next command to type.
pub fn names_near(ds: &Dataset, lat_e7: i32, lon_e7: i32, radius_m: f64) -> Vec<(String, usize)> {
    let mut counts: std::collections::HashMap<String, usize> = Default::default();
    for sid in segments_near(ds, lat_e7, lon_e7, radius_m, None) {
        let n = ds.street_name(sid as usize).unwrap_or("");
        *counts.entry(if n.is_empty() { "(unnamed)".into() } else { n.to_string() }).or_default() +=
            1;
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// Perpendicular distance from a point to a segment's drawn line, in metres.
fn seg_distance_m(ds: &Dataset, sid: u32, lat_e7: i32, lon_e7: i32) -> f64 {
    let pts = ds.seg_geometry(sid as usize);
    if pts.len() < 2 {
        return f64::MAX;
    }
    let lat = geo::e7_to_deg(lat_e7);
    let sy = 111_132.0 * 1e-7;
    let sx = (111_320.0 * lat.to_radians().cos()).abs().max(1.0) * 1e-7;
    let (qx, qy) = (lon_e7 as f64 * sx, lat_e7 as f64 * sy);
    let mut best = f64::MAX;
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
        let (ex, ey) = (qx - (ax + t * dx), qy - (ay + t * dy));
        best = best.min((ex * ex + ey * ey).sqrt());
    }
    best
}
