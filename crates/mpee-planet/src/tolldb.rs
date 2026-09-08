//! The toll tariff layer, in mpedb.
//!
//! This is the one part of the dataset that is a *database* rather than a data
//! structure, and the split is deliberate:
//!
//!   * **Identity and membership stay in an array.** "Is vertex 41 209 883 a
//!     toll gate?" is asked once per vertex on a hot path that settles millions
//!     of them, and it is answered by a binary search over `toll.vertex`. A
//!     query there would be the same mistake as putting the graph in tables.
//!   * **Prices and rules live here.** They are small, they change, they arrive
//!     from several sources, and the questions asked of them are relational —
//!     vehicle class joined to rate kind joined to a time window joined to a
//!     scheme's hour rule. A flat file cannot answer those, and the previous
//!     `toll.tariff.tsv` did not: it collapsed everything to one number.
//!
//! What the TSV was throwing away, per gate, is exactly what NVDB publishes:
//! a rate for a small car *and* a large one, a rush-hour rate with its own two
//! daily windows, the direction of collection, and the scheme's hour rule
//! (inside one toll ring you pay once, not once per gantry). All of it is now
//! addressable.
//!
//! Money is `numeric`, not `float`. The old parser ended in
//! `.unwrap_or(0.0)`, so a typo in the price column became a silent free toll;
//! here a bad price is a typed refusal at insert time.

use mpedb::{Config, Database, ExecResult, Value};
use std::collections::HashMap;
use std::path::Path;

/// Schema. `check` constraints are load-bearing rather than decorative: they
/// are what turns a malformed tariff into an error instead of a wrong answer.
const SCHEMA: &str = r#"
[[table]]
name = "toll_gate"
primary_key = ["osm_id"]
  [[table.column]]
  name = "osm_id"
  type = "int64"
  [[table.column]]
  name = "vertex"
  type = "int64"
  indexed = true
  [[table.column]]
  name = "lat_e7"
  type = "int64"
  [[table.column]]
  name = "lon_e7"
  type = "int64"
  [[table.column]]
  name = "name"
  type = "text"
  [[table.column]]
  name = "operator"
  type = "text"
  [[table.column]]
  name = "direction"
  type = "text"
  [[table.column]]
  name = "osm_charge"
  type = "text"

[[table]]
name = "toll_scheme"
primary_key = ["scheme_id"]
  [[table.column]]
  name = "scheme_id"
  type = "int64"
  [[table.column]]
  name = "name"
  type = "text"
  nullable = false
  unique = true
  [[table.column]]
  name = "operator"
  type = "text"
  [[table.column]]
  name = "currency"
  type = "text"
  nullable = false
  [[table.column]]
  name = "hour_rule_min"
  type = "int64"
  nullable = false
  [[table.column]]
  name = "hour_rule_group"
  type = "int64"

[[table]]
name = "toll_rate"
primary_key = ["rate_id"]
  [[table.column]]
  name = "rate_id"
  type = "int64"
  [[table.column]]
  name = "osm_id"
  type = "int64"
  nullable = false
  indexed = true
  [[table.column]]
  name = "scheme_id"
  type = "int64"
  nullable = false
  indexed = true
  [[table.column]]
  name = "vehicle_class"
  type = "text"
  nullable = false
  [[table.column]]
  name = "rate_kind"
  type = "text"
  nullable = false
  [[table.column]]
  name = "price"
  type = "numeric"
  nullable = false
  [[table.column]]
  name = "currency"
  type = "text"
  nullable = false
  [[table.column]]
  name = "source"
  type = "text"
  nullable = false
  [[table.index]]
  columns = ["osm_id", "vehicle_class", "rate_kind"]
  unique = true

[[table]]
name = "toll_rush_window"
primary_key = ["osm_id", "slot"]
  [[table.column]]
  name = "osm_id"
  type = "int64"
  [[table.column]]
  name = "slot"
  type = "int64"
  [[table.column]]
  name = "from_time"
  type = "time"
  nullable = false
  [[table.column]]
  name = "to_time"
  type = "time"
  nullable = false
"#;

fn config_for(path: &Path, size_mb: u64) -> Result<Config, Box<dyn std::error::Error>> {
    let toml = format!(
        "[database]\npath = {}\nsize_mb = {size_mb}\nmax_readers = 16\ndurability = \"commit\"\n{SCHEMA}",
        toml_path(path)
    );
    Ok(Config::from_toml_str(&toml)?)
}

/// TOML basic strings treat backslashes as escapes, so a Windows path must not
/// be interpolated raw. Kept even though this build targets Unix, because the
/// failure it prevents is silent and confusing.
fn toml_path(p: &Path) -> String {
    let s = p.display().to_string();
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

pub const VEHICLE_CAR: &str = "car";
pub const VEHICLE_HGV: &str = "hgv";
pub const RATE_STANDARD: &str = "standard";
pub const RATE_RUSH: &str = "rush";

/// One gate's contribution to a quote.
#[derive(Debug, Clone)]
pub struct Line {
    pub osm_id: u64,
    pub gate: String,
    pub scheme: String,
    pub scheme_id: i64,
    pub price: f64,
    pub currency: String,
    pub rate_kind: String,
    /// False when an hour rule already covered this scheme upstream.
    pub charged: bool,
}

#[derive(Debug, Default)]
pub struct Quote {
    pub lines: Vec<Line>,
    /// Totals per currency, after hour rules.
    pub totals: Vec<(String, f64)>,
    /// Gates crossed for which no tariff is known.
    pub unpriced: usize,
}

pub struct TollDb {
    db: Database,
    q_rate: mpedb::PlanHash,
    q_rush: mpedb::PlanHash,
}

/// Microseconds since midnight, the unit mpedb's `time` type stores.
pub fn time_us(hour: u32, minute: u32) -> i64 {
    (hour as i64 * 3600 + minute as i64 * 60) * 1_000_000
}

impl TollDb {
    pub fn open(dir: &Path) -> Result<TollDb, Box<dyn std::error::Error>> {
        let db = Database::open_with_config(config_for(&dir.join("toll.mpedb"), 512)?)?;
        // The question the flat file could not answer: one gate, one vehicle
        // class, one rate kind — with the scheme's hour rule joined in, because
        // the total depends on it.
        let q_rate = db.prepare(
            "SELECT r.price, r.currency, r.rate_kind, r.source, \
                    s.scheme_id, s.name, s.hour_rule_min, g.name \
             FROM toll_rate r \
             JOIN toll_scheme s ON r.scheme_id = s.scheme_id \
             JOIN toll_gate g ON r.osm_id = g.osm_id \
             WHERE r.osm_id = $1 AND r.vehicle_class = $2 AND r.rate_kind = $3",
        )?;
        let q_rush = db.prepare(
            "SELECT slot FROM toll_rush_window \
             WHERE osm_id = $1 AND from_time <= $2 AND to_time >= $2",
        )?;
        Ok(TollDb { db, q_rate, q_rush })
    }

    /// Whether `at` (microseconds since midnight) falls in one of this gate's
    /// rush windows.
    fn in_rush(&self, osm_id: u64, at: i64) -> bool {
        matches!(
            self.db.execute(&self.q_rush, &[Value::Int(osm_id as i64), Value::Time(at)]),
            Ok(ExecResult::Rows { ref rows, .. }) if !rows.is_empty()
        )
    }

    fn lookup(
        &self,
        osm_id: u64,
        class: &str,
        kind: &str,
    ) -> Option<(f64, String, String, i64, String, i64, String)> {
        let rows = match self.db.execute(
            &self.q_rate,
            &[
                Value::Int(osm_id as i64),
                Value::Text(class.to_string()),
                Value::Text(kind.to_string()),
            ],
        ) {
            Ok(ExecResult::Rows { rows, .. }) => rows,
            _ => return None,
        };
        let r = rows.into_iter().next()?;
        let price = match &r[0] {
            Value::Numeric(s) => s.parse::<f64>().ok()?,
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            _ => return None,
        };
        let text = |v: &Value| match v {
            Value::Text(s) => s.clone(),
            _ => String::new(),
        };
        let int = |v: &Value| match v {
            Value::Int(i) => *i,
            _ => 0,
        };
        Some((
            price,
            text(&r[1]),
            text(&r[2]),
            int(&r[4]),
            text(&r[5]),
            int(&r[6]),
            text(&r[7]),
        ))
    }

    /// Price a list of gates crossed, in order.
    ///
    /// `at` is the time of day in microseconds since midnight; `None` prices at
    /// the standard rate regardless of when the trip runs.
    ///
    /// The hour rule is applied per scheme, and applied as the operators state
    /// it: inside one toll ring you pay **the dearest single passage**, once.
    /// Charging the first passage and then topping up to the dearest reaches
    /// the same total, but it misreports what happened — and a per-gate
    /// breakdown that says you paid twice is a breakdown a dispatcher cannot
    /// reconcile against an invoice.
    pub fn quote(&self, gates: &[u64], class: &str, at: Option<i64>) -> Quote {
        let mut q = Quote::default();
        // Collect first, decide afterwards: the dearest passage in a scheme is
        // not knowable until every passage has been seen.
        let mut hour_rule_scheme: HashMap<i64, bool> = HashMap::new();
        for &g in gates {
            let rush = at.map(|t| self.in_rush(g, t)).unwrap_or(false);
            // Fall back to the standard rate when a gate has no rush rate of
            // its own — most gates outside a city ring do not.
            let hit = if rush {
                self.lookup(g, class, RATE_RUSH).or_else(|| self.lookup(g, class, RATE_STANDARD))
            } else {
                self.lookup(g, class, RATE_STANDARD)
            };
            let Some((price, currency, kind, scheme_id, scheme, hour_rule, gate_name)) = hit else {
                q.unpriced += 1;
                continue;
            };
            hour_rule_scheme.insert(scheme_id, hour_rule > 0);
            q.lines.push(Line {
                osm_id: g,
                gate: gate_name,
                scheme,
                scheme_id,
                price,
                currency,
                rate_kind: kind,
                charged: true,
            });
        }
        // Within each hour-rule scheme keep exactly one charge: the dearest
        // passage, first occurrence on a tie.
        let mut dearest: HashMap<i64, usize> = HashMap::new();
        for (i, l) in q.lines.iter().enumerate() {
            if !hour_rule_scheme.get(&l.scheme_id).copied().unwrap_or(false) {
                continue;
            }
            match dearest.get(&l.scheme_id) {
                Some(&best) if q.lines[best].price >= l.price => {}
                _ => {
                    dearest.insert(l.scheme_id, i);
                }
            }
        }
        for (i, l) in q.lines.iter_mut().enumerate() {
            if hour_rule_scheme.get(&l.scheme_id).copied().unwrap_or(false) {
                l.charged = dearest.get(&l.scheme_id) == Some(&i);
            }
        }
        let mut totals: HashMap<String, f64> = HashMap::new();
        for l in q.lines.iter().filter(|l| l.charged) {
            *totals.entry(l.currency.clone()).or_default() += l.price;
        }
        q.totals = totals.into_iter().collect();
        q.totals.sort_by(|a, b| a.0.cmp(&b.0));
        q
    }

    /// Ad-hoc SQL, for the questions the arrays cannot answer.
    pub fn query(&self, sql: &str) -> Result<ExecResult, Box<dyn std::error::Error>> {
        Ok(self.db.query(sql, &[])?)
    }
}

// ----------------------------------------------------------------- ingest

use crate::toll::Gate;
use serde_json::Value as J;

pub struct IngestStats {
    pub gates: u64,
    pub schemes: u64,
    pub rates: u64,
    pub rush_windows: u64,
    pub matched_nvdb: u64,
    pub unmatched_nvdb: Vec<(String, String)>,
    pub from_osm_tag: u64,
}

/// NVDB coordinates land within a few metres of the road; the OSM gantry may
/// sit a little to the side. 300 m is loose enough to absorb that and tight
/// enough that two distinct stations cannot swap.
const MATCH_M: f64 = 300.0;

fn haversine(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let r = 6_371_000.0f64;
    let (p1, p2) = (a.to_radians(), c.to_radians());
    let (dp, dl) = ((c - a).to_radians(), (d - b).to_radians());
    let h = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * r * h.sqrt().asin()
}

/// "POINT Z (60.418 5.313 31.7)" and the spaceless "POINT(60.418 5.313)".
fn parse_point(wkt: &str) -> Option<(f64, f64)> {
    let open = wkt.find('(')?;
    let body = &wkt[open + 1..wkt.find(')')?];
    let mut it = body.split_whitespace();
    let a: f64 = it.next()?.parse().ok()?;
    let b: f64 = it.next()?.parse().ok()?;
    Some((a, b))
}

/// "06:30" → microseconds since midnight.
fn parse_hhmm(s: &str) -> Option<i64> {
    let (h, m) = s.trim().split_once(':')?;
    Some(time_us(h.parse().ok()?, m.parse().ok()?))
}

fn prop<'a>(o: &'a J, name: &str) -> Option<&'a J> {
    o.get("egenskaper")?
        .as_array()?
        .iter()
        .find(|p| p.get("navn").and_then(|n| n.as_str()) == Some(name))?
        .get("verdi")
}

fn prop_str(o: &J, name: &str) -> Option<String> {
    prop(o, name).and_then(|v| v.as_str()).map(str::to_string)
}
fn prop_f64(o: &J, name: &str) -> Option<f64> {
    prop(o, name).and_then(|v| v.as_f64())
}

/// Money as an exact decimal string — never a float. This is the column that
/// used to be parsed with `.unwrap_or(0.0)`, turning a typo into a free toll.
fn money(v: f64) -> String {
    format!("{v:.2}")
}

/// Build the tariff database from the gates pass 6 found and, when present,
/// NVDB's Norwegian toll-station export.
pub fn ingest(
    dir: &Path,
    gates: &[Gate],
    vertices: &[u32],
    nvdb_json: Option<&Path>,
) -> Result<IngestStats, Box<dyn std::error::Error>> {
    let path = dir.join("toll.mpedb");
    let _ = std::fs::remove_file(&path);
    let db = Database::open_with_config(config_for(&path, 512)?)?;

    let ins_gate = db.prepare(
        "INSERT INTO toll_gate (osm_id, vertex, lat_e7, lon_e7, name, operator, direction, osm_charge) \
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
    )?;
    let ins_scheme = db.prepare(
        "INSERT INTO toll_scheme (scheme_id, name, operator, currency, hour_rule_min, hour_rule_group) \
         VALUES ($1,$2,$3,$4,$5,$6)",
    )?;
    let ins_rate = db.prepare(
        "INSERT INTO toll_rate (rate_id, osm_id, scheme_id, vehicle_class, rate_kind, price, currency, source) \
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
    )?;
    let ins_rush = db.prepare(
        "INSERT INTO toll_rush_window (osm_id, slot, from_time, to_time) VALUES ($1,$2,$3,$4)",
    )?;

    let mut st = IngestStats {
        gates: 0,
        schemes: 0,
        rates: 0,
        rush_windows: 0,
        matched_nvdb: 0,
        unmatched_nvdb: Vec::new(),
        from_osm_tag: 0,
    };

    // ---- gates: identity, straight from the graph build ------------------
    {
        let mut s = db.begin()?;
        for (i, g) in gates.iter().enumerate() {
            let v = *vertices.get(i).unwrap_or(&u32::MAX);
            s.execute(
                &ins_gate,
                &[
                    Value::Int(g.osm_id as i64),
                    Value::Int(v as i64),
                    Value::Int(g.lat_e7 as i64),
                    Value::Int(g.lon_e7 as i64),
                    text_or_null(&g.name),
                    text_or_null(&g.operator),
                    text_or_null(&g.direction),
                    text_or_null(&g.charge),
                ],
            )?;
            st.gates += 1;
        }
        s.commit()?;
    }

    // Spatial buckets over the gates, so matching is a local scan.
    let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, g) in gates.iter().enumerate() {
        let k = ((g.lat_e7 / 1_000_000) as i64, (g.lon_e7 / 1_000_000) as i64);
        cells.entry(k).or_default().push(i);
    }

    let mut scheme_ids: HashMap<String, i64> = HashMap::new();
    let mut rate_id: i64 = 0;
    let mut used: std::collections::HashSet<u64> = Default::default();

    // ---- NVDB: the fields the flat file had to throw away ----------------
    if let Some(p) = nvdb_json {
        let doc: J = serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(p)?))?;
        let stations = doc.as_array().cloned().unwrap_or_default();
        let mut s = db.begin()?;
        for o in &stations {
            let name = prop_str(o, "Navn bomstasjon").unwrap_or_default();
            let Some(wkt) = o.get("geometri").and_then(|g| g.get("wkt")).and_then(|v| v.as_str())
            else {
                st.unmatched_nvdb.push((name, "no geometry".into()));
                continue;
            };
            let Some((lat, lon)) = parse_point(wkt) else {
                st.unmatched_nvdb.push((name, "unparseable geometry".into()));
                continue;
            };
            // Nearest unused OSM gate.
            let (mut best, mut bd) = (usize::MAX, f64::MAX);
            let (cy, cx) = ((lat * 10.0) as i64, (lon * 10.0) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    for &i in cells.get(&(cy + dy, cx + dx)).into_iter().flatten() {
                        if used.contains(&gates[i].osm_id) {
                            continue;
                        }
                        let d = haversine(
                            lat,
                            lon,
                            gates[i].lat_e7 as f64 * 1e-7,
                            gates[i].lon_e7 as f64 * 1e-7,
                        );
                        if d < bd {
                            bd = d;
                            best = i;
                        }
                    }
                }
            }
            if best == usize::MAX || bd > MATCH_M {
                st.unmatched_nvdb.push((name, format!("{bd:.0} m to nearest gate")));
                continue;
            }
            let osm_id = gates[best].osm_id;
            used.insert(osm_id);
            st.matched_nvdb += 1;

            let scheme_name = prop_str(o, "Navn bompengeanlegg")
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "(uspesifisert anlegg)".to_string());
            let hour_rule =
                prop(o, "Timesregel, varighet").and_then(|v| v.as_i64()).unwrap_or(0);
            let group = prop(o, "Timesregel, passeringsgruppe").and_then(|v| v.as_i64());
            let sid = if let Some(&id) = scheme_ids.get(&scheme_name) {
                id
            } else {
                let id = scheme_ids.len() as i64;
                s.execute(
                    &ins_scheme,
                    &[
                        Value::Int(id),
                        Value::Text(scheme_name.clone()),
                        text_or_null(&prop_str(o, "Bomstasjonstype").unwrap_or_default()),
                        Value::Text("NOK".into()),
                        Value::Int(hour_rule),
                        group.map(Value::Int).unwrap_or(Value::Null),
                    ],
                )?;
                scheme_ids.insert(scheme_name.clone(), id);
                st.schemes += 1;
                id
            };

            // Four rates per gate where NVDB has them: two vehicle classes ×
            // standard and rush. The old TSV kept one of these four.
            for (class, kind, field) in [
                (VEHICLE_CAR, RATE_STANDARD, "Takst liten bil"),
                (VEHICLE_CAR, RATE_RUSH, "Rushtidstakst liten bil"),
                (VEHICLE_HGV, RATE_STANDARD, "Takst stor bil"),
                (VEHICLE_HGV, RATE_RUSH, "Rushtidstakst stor bil"),
            ] {
                let Some(v) = prop_f64(o, field) else { continue };
                s.execute(
                    &ins_rate,
                    &[
                        Value::Int(rate_id),
                        Value::Int(osm_id as i64),
                        Value::Int(sid),
                        Value::Text(class.into()),
                        Value::Text(kind.into()),
                        Value::Numeric(money(v)),
                        Value::Text("NOK".into()),
                        Value::Text("nvdb".into()),
                    ],
                )?;
                rate_id += 1;
                st.rates += 1;
            }

            for (slot, from_f, to_f) in [
                (0i64, "Rushtid morgen, fra", "Rushtid morgen, til"),
                (1, "Rushtid ettermiddag, fra", "Rushtid ettermiddag, til"),
            ] {
                let (Some(a), Some(b)) = (
                    prop_str(o, from_f).as_deref().and_then(parse_hhmm),
                    prop_str(o, to_f).as_deref().and_then(parse_hhmm),
                ) else {
                    continue;
                };
                s.execute(
                    &ins_rush,
                    &[
                        Value::Int(osm_id as i64),
                        Value::Int(slot),
                        Value::Time(a),
                        Value::Time(b),
                    ],
                )?;
                st.rush_windows += 1;
            }
        }
        s.commit()?;
    }

    // ---- OSM `charge` tags, for gates NVDB does not cover ----------------
    {
        let mut s = db.begin()?;
        let osm_scheme = scheme_ids.len() as i64;
        let mut osm_scheme_written = false;
        for g in gates {
            if used.contains(&g.osm_id) || g.charge.is_empty() {
                continue;
            }
            let Some((v, cur)) = crate::toll::parse_charge(&g.charge) else { continue };
            if !osm_scheme_written {
                s.execute(
                    &ins_scheme,
                    &[
                        Value::Int(osm_scheme),
                        Value::Text("(OSM charge tag)".into()),
                        Value::Null,
                        Value::Text(cur.clone()),
                        Value::Int(0),
                        Value::Null,
                    ],
                )?;
                st.schemes += 1;
                osm_scheme_written = true;
            }
            s.execute(
                &ins_rate,
                &[
                    Value::Int(rate_id),
                    Value::Int(g.osm_id as i64),
                    Value::Int(osm_scheme),
                    Value::Text(VEHICLE_CAR.into()),
                    Value::Text(RATE_STANDARD.into()),
                    Value::Numeric(money(v)),
                    Value::Text(cur),
                    Value::Text("osm".into()),
                ],
            )?;
            rate_id += 1;
            st.rates += 1;
            st.from_osm_tag += 1;
        }
        s.commit()?;
    }

    db.analyze()?;
    Ok(st)
}

fn text_or_null(s: &str) -> Value {
    if s.is_empty() {
        Value::Null
    } else {
        Value::Text(s.to_string())
    }
}
