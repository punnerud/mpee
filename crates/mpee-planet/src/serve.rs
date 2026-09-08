//! A small HTTP server over the planet dataset.
//!
//! There is deliberately no map-tile dependency here. The viewer's background
//! *is* this dataset: `/api/roads` streams road geometry for the viewport
//! straight out of the same mmapped arrays the router uses. So the whole thing
//! — search, routing, and the map you draw the blue line on — runs from one
//! local directory with no network at all.

use crate::dataset::Dataset;
use crate::geocode::Geocoder;
use crate::graph::{big_cell_of, cell_of, BIG_E7, CELL_E7};
use crate::router::{self, Router};
use crate::toll::TollIndex;
use crate::tolldb::{self, TollDb};
use crate::catalog;
use crate::overrides::{OverrideDb, Overrides};
use crate::{build, geo};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

pub struct App {
    pub ds: Dataset,
    pub geo: Option<Geocoder>,
    pub tolls: TollIndex,
    /// The tariff layer. Absent when no `toll.mpedb` has been built, in which
    /// case pricing falls back to whatever OSM tagged.
    pub tariff: Option<TollDb>,
    /// Local edits. Stored at the *root* rather than in the build directory,
    /// because they are meant to outlive the dataset they were written
    /// against.
    ovdb: Option<OverrideDb>,
    ov: RwLock<Arc<Overrides>>,
    pub dir: PathBuf,
    /// Search buffers are ~2.6 GB each on a planet, so they are pooled rather
    /// than allocated per request.
    pool: Mutex<Vec<Router>>,
}

impl App {
    /// `root` is the catalogue root; the live dataset is resolved from `HEAD`.
    pub fn open(root: &std::path::Path) -> std::io::Result<App> {
        let dir = catalog::resolve(root);
        let ds = Dataset::open(&dir)?;
        let geo = Geocoder::open(&dir).ok();
        let tolls = TollIndex::open(&dir)?;
        let tariff = TollDb::open(&dir).ok();
        let ovdb = OverrideDb::open(root).ok();
        let ov = match (&ovdb, &ds) {
            (Some(db), ds) => {
                let rules = db.list().unwrap_or_default();
                Overrides::resolve(ds, &rules, db.generation())
            }
            _ => Overrides::empty(),
        };
        if !ov.is_empty() {
            eprintln!("[serve] {} segment(s) overridden at startup", ov.len());
        }
        Ok(App {
            ds,
            geo,
            tolls,
            tariff,
            ovdb,
            ov: RwLock::new(Arc::new(ov)),
            dir,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// The override set to route with, re-resolved when another process has
    /// written since we last looked.
    ///
    /// This is the cheap half of "applied without a restart": one point read
    /// of a counter per request, which is noise beside a route, and a rebuild
    /// only when it actually changed. Because mpedb's readers are never
    /// blocked by its writer, the edit lands in the next query rather than the
    /// next deployment.
    fn overrides(&self) -> Arc<Overrides> {
        let Some(db) = self.ovdb.as_ref() else {
            return self.ov.read().unwrap().clone();
        };
        let gen = db.generation();
        {
            let cur = self.ov.read().unwrap();
            if cur.generation == gen {
                return cur.clone();
            }
        }
        // Double-check under the write lock: several request threads can miss
        // the cache on the same edit, and resolving is a sweep over the snap
        // index — one of them should do it, not all of them.
        let mut w = self.ov.write().unwrap();
        if w.generation == gen {
            return w.clone();
        }
        let rules = db.list().unwrap_or_default();
        let fresh = Arc::new(Overrides::resolve(&self.ds, &rules, gen));
        eprintln!(
            "[serve] overrides reloaded at generation {gen} — {} segment(s)",
            fresh.len()
        );
        *w = fresh.clone();
        fresh
    }
    fn take_router(&self) -> Router {
        self.pool.lock().unwrap().pop().unwrap_or_else(|| Router::new(self.ds.n_vertices()))
    }
    fn give_router(&self, r: Router) {
        let mut p = self.pool.lock().unwrap();
        if p.len() < 4 {
            p.push(r);
        }
    }
}

pub fn run(app: Arc<App>, addr: &str) -> std::io::Result<()> {
    let l = TcpListener::bind(addr)?;
    eprintln!("[serve] http://{addr}  ({} vertices, {} segments)", app.ds.n_vertices(), app.ds.n_segments());
    for s in l.incoming() {
        let Ok(s) = s else { continue };
        let app = app.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(&app, s) {
                if e.kind() != std::io::ErrorKind::BrokenPipe {
                    eprintln!("[serve] {e}");
                }
            }
        });
    }
    Ok(())
}

fn handle(app: &App, mut s: TcpStream) -> std::io::Result<()> {
    let mut r = BufReader::new(s.try_clone()?);
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(());
    }
    let mut it = line.split_whitespace();
    let _method = it.next().unwrap_or("");
    let target = it.next().unwrap_or("/").to_string();
    // Drain headers.
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
    }
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.clone(), String::new()),
    };
    let q = Query::parse(&query);
    let t0 = std::time::Instant::now();
    let (ctype, body) = match path.as_str() {
        "/" | "/index.html" => ("text/html; charset=utf-8", index_html(app)),
        "/favicon.ico" => ("image/svg+xml", Vec::new()),
        "/api/route" => ("application/json", api_route(app, &q)),
        "/api/geocode" => ("application/json", api_geocode(app, &q)),
        "/api/reverse" => ("application/json", api_reverse(app, &q)),
        "/api/roads" => ("application/json", api_roads(app, &q)),
        "/api/info" => ("application/json", api_info(app)),
        "/api/overrides" => ("application/json", api_overrides(app)),
        _ => ("text/plain", b"not found".to_vec()),
    };
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nServer-Timing: app;dur={:.1}\r\nConnection: close\r\n\r\n",
        body.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    s.write_all(head.as_bytes())?;
    s.write_all(&body)?;
    s.flush()
}

/// The UI. Compiled in so a single binary plus a data directory is the whole
/// offline install, but overridable by dropping `index.html` next to the data.
fn index_html(app: &App) -> Vec<u8> {
    match std::fs::read(app.dir.join("index.html")) {
        Ok(v) => v,
        Err(_) => include_str!("../viewer/index.html").as_bytes().to_vec(),
    }
}

// --------------------------------------------------------------- query

struct Query(Vec<(String, String)>);

impl Query {
    fn parse(q: &str) -> Query {
        let mut v = Vec::new();
        for p in q.split('&') {
            if p.is_empty() {
                continue;
            }
            let (k, val) = p.split_once('=').unwrap_or((p, ""));
            v.push((urldecode(k), urldecode(val)));
        }
        Query(v)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.0.iter().find(|(a, _)| a == k).map(|(_, b)| b.as_str())
    }
    fn f64(&self, k: &str) -> Option<f64> {
        self.get(k)?.parse().ok()
    }
    fn ll(&self, k: &str) -> Option<(i32, i32)> {
        let s = self.get(k)?;
        let (a, b) = s.split_once(',')?;
        let la: f64 = a.trim().parse().ok()?;
        let lo: f64 = b.trim().parse().ok()?;
        Some(((la / geo::E7).round() as i32, (lo / geo::E7).round() as i32))
    }
}

fn urldecode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let h = u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16);
                match h {
                    Ok(v) => {
                        out.push(v);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(b[i]);
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

// ------------------------------------------------------------ endpoints

fn api_info(app: &App) -> Vec<u8> {
    format!(
        "{{\"vertices\":{},\"edges\":{},\"segments\":{},\"addresses\":{},\"streets\":{},\"toll_gates\":{}}}",
        app.ds.n_vertices(),
        app.ds.n_edges(),
        app.ds.n_segments(),
        app.geo.as_ref().map(|g| g.len()).unwrap_or(0),
        app.geo.as_ref().map(|g| g.street_count()).unwrap_or(0),
        app.tolls.vertex.len()
    )
    .into_bytes()
}

fn api_overrides(app: &App) -> Vec<u8> {
    let ov = app.overrides();
    let rules = app.ovdb.as_ref().and_then(|d| d.list().ok()).unwrap_or_default();
    let mut o = format!(
        "{{\"generation\":{},\"segments\":{},\"rules\":[",
        ov.generation,
        ov.len()
    );
    for (i, r) in rules.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        let matched = ov.matched.iter().find(|(id, _)| *id == r.id).map(|(_, n)| *n).unwrap_or(0);
        o.push_str(&format!(
            "{{\"id\":{},\"kind\":{},\"lat\":{:.6},\"lon\":{:.6},\"radius_m\":{},\"street\":{},\"speed_kmh\":{},\"factor\":{},\"note\":{},\"segments\":{matched}}}",
            r.id,
            jstr(&r.kind),
            r.lat_e7 as f64 * 1e-7,
            r.lon_e7 as f64 * 1e-7,
            r.radius_m,
            jstr(r.street.as_deref().unwrap_or("")),
            r.speed_kmh.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
            r.factor.map(|v| format!("{v}")).unwrap_or_else(|| "null".into()),
            jstr(&r.note)
        ));
    }
    o.push_str("]}");
    o.into_bytes()
}

fn api_route(app: &App, q: &Query) -> Vec<u8> {
    let (Some(a), Some(b)) = (q.ll("from"), q.ll("to")) else {
        return b"{\"error\":\"from and to are required as lat,lon\"}".to_vec();
    };
    let radius = q.f64("radius").unwrap_or(2000.0);
    let (Some(sa), Some(sb)) = (app.ds.snap(a.0, a.1, radius), app.ds.snap(b.0, b.1, radius)) else {
        return b"{\"error\":\"no road within range of one of the points\"}".to_vec();
    };
    let ov = app.overrides();
    let mut router = app.take_router();
    let t = std::time::Instant::now();
    let route = router.route_with(&app.ds, &sa, &sb, Some(&ov));
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let settled = router.settled;
    app.give_router(router);
    let Some(route) = route else {
        return if ov.is_empty() {
            b"{\"error\":\"no route\"}".to_vec()
        } else {
            // Say so plainly: an override that removes the only way through is
            // a very different situation from an unreachable destination.
            format!(
                "{{\"error\":\"no route\",\"overrides_active\":{},\"hint\":\"an override may have closed the only way through\"}}",
                ov.len()
            )
            .into_bytes()
        };
    };

    let mut o = String::with_capacity(1 << 16);
    o.push_str(&format!(
        "{{\"distance_m\":{:.2},\"duration_s\":{:.1},\"settled\":{},\"query_ms\":{:.1},\"snap_from_m\":{:.1},\"snap_to_m\":{:.1},",
        route.dist_m(),
        route.dur_s(),
        settled,
        ms,
        sa.off_m,
        sb.off_m
    ));
    // Geometry as [lat,lon] pairs, 6 decimals ≈ 11 cm.
    o.push_str("\"geometry\":[");
    for (i, (la, lo)) in router::route_geometry(&app.ds, &route).iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&format!("[{la:.6},{lo:.6}]"));
    }
    o.push_str("],\"steps\":[");
    for (i, (name, cm, cls)) in router::route_steps(&app.ds, &route).iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&format!(
            "{{\"name\":{},\"distance_m\":{:.1},\"class\":{}}}",
            jstr(name),
            *cm as f64 / 100.0,
            jstr(router::class_name(*cls))
        ));
    }
    o.push_str("],\"tolls\":[");
    let hits = router::route_tolls(&route, &app.tolls);
    let class = match q.get("class") {
        Some("hgv") => tolldb::VEHICLE_HGV,
        _ => tolldb::VEHICLE_CAR,
    };
    // `at=HH:MM` selects the rush rate where one applies.
    let at = q.get("at").and_then(|s| {
        let (h, m) = s.split_once(':')?;
        Some(tolldb::time_us(h.parse().ok()?, m.parse().ok()?))
    });
    let ids: Vec<u64> = hits.iter().map(|h| app.tolls.gates[h.gate].osm_id).collect();
    let quote = app.tariff.as_ref().map(|t| t.quote(&ids, class, at));

    for (i, h) in hits.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        let g = &app.tolls.gates[h.gate];
        // Prefer the tariff database's own name and price; fall back to the
        // OSM tag when no tariff is known for this gate.
        let line = quote
            .as_ref()
            .and_then(|q| q.lines.iter().find(|l| l.osm_id == g.osm_id));
        let name = match (line, g.name.is_empty()) {
            (Some(l), _) if !l.gate.is_empty() => l.gate.clone(),
            (_, false) => g.name.clone(),
            _ => String::new(),
        };
        let (price, currency, kind, scheme, charged) = match line {
            Some(l) => (
                format!("{:.2}", l.price),
                l.currency.clone(),
                l.rate_kind.clone(),
                l.scheme.clone(),
                l.charged,
            ),
            None => match h.price.as_ref() {
                Some((v, c)) => (format!("{v:.2}"), c.clone(), "osm".into(), String::new(), true),
                None => ("null".into(), String::new(), String::new(), String::new(), false),
            },
        };
        o.push_str(&format!(
            "{{\"name\":{},\"lat\":{:.6},\"lon\":{:.6},\"scheme\":{},\"price\":{},\"currency\":{},\"rate\":{},\"charged\":{}}}",
            jstr(&name),
            geo::e7_to_deg(g.lat_e7),
            geo::e7_to_deg(g.lon_e7),
            jstr(&scheme),
            price,
            jstr(&currency),
            jstr(&kind),
            charged
        ));
    }
    o.push_str(&format!("],\"toll_class\":{},", jstr(class)));
    o.push_str("\"toll_total\":[");
    match &quote {
        Some(qt) => {
            for (i, (cur, sum)) in qt.totals.iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                o.push_str(&format!("{{\"currency\":{},\"amount\":{sum:.2}}}", jstr(cur)));
            }
        }
        None => {
            let idx: Vec<usize> = hits.iter().map(|h| h.gate).collect();
            for (i, (cur, sum, _)) in app.tolls.total(&idx).iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                o.push_str(&format!("{{\"currency\":{},\"amount\":{sum:.2}}}", jstr(cur)));
            }
        }
    }
    o.push_str(&format!(
        "],\"toll_unpriced\":{},\"overrides_active\":{}",
        quote.as_ref().map(|q| q.unpriced).unwrap_or(0),
        ov.len()
    ));
    o.push('}');
    o.into_bytes()
}

/// Split "Karl Johans gate 22, Oslo" into street, number and city.
pub fn parse_address_query(q: &str) -> (String, String, Option<String>) {
    let (head, city) = match q.split_once(',') {
        Some((h, c)) => (h.trim(), Some(c.trim().to_string()).filter(|s| !s.is_empty())),
        None => (q.trim(), None),
    };
    // The house number is the last token that starts with a digit.
    let toks: Vec<&str> = head.split_whitespace().collect();
    if let Some(last) = toks.last() {
        if last.chars().next().is_some_and(|c| c.is_ascii_digit()) && toks.len() > 1 {
            return (toks[..toks.len() - 1].join(" "), last.to_string(), city);
        }
    }
    (head.to_string(), String::new(), city)
}

fn api_geocode(app: &App, q: &Query) -> Vec<u8> {
    let Some(g) = app.geo.as_ref() else {
        return b"{\"results\":[],\"error\":\"no address index\"}".to_vec();
    };
    let text = q.get("q").unwrap_or("");
    let limit = q.f64("limit").unwrap_or(8.0) as usize;
    let (street, number, city) = parse_address_query(text);
    let hits = g.forward(&street, &number, city.as_deref(), limit);
    let mut o = String::from("{\"results\":[");
    for (i, h) in hits.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&format!(
            "{{\"lat\":{:.7},\"lon\":{:.7},\"street\":{},\"housenumber\":{},\"city\":{},\"postcode\":{},\"approximate\":{}}}",
            h.lat,
            h.lon,
            jstr(&h.street),
            jstr(&h.housenumber),
            jstr(h.city.as_deref().unwrap_or("")),
            jstr(h.postcode.as_deref().unwrap_or("")),
            h.approximate
        ));
    }
    o.push_str("],\"suggest\":[");
    for (i, s) in g.suggest(&street, 8).iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&jstr(s));
    }
    o.push_str("]}");
    o.into_bytes()
}

fn api_reverse(app: &App, q: &Query) -> Vec<u8> {
    let (Some(lat), Some(lon)) = (q.f64("lat"), q.f64("lon")) else {
        return b"{\"error\":\"lat and lon are required\"}".to_vec();
    };
    let mut o = String::from("{");
    if let Some(g) = app.geo.as_ref() {
        if let Some(h) = g.reverse(lat, lon, q.f64("radius").unwrap_or(300.0)) {
            o.push_str(&format!(
                "\"address\":{{\"lat\":{:.7},\"lon\":{:.7},\"street\":{},\"housenumber\":{},\"city\":{},\"postcode\":{},\"distance_m\":{:.1}}},",
                h.lat, h.lon, jstr(&h.street), jstr(&h.housenumber),
                jstr(h.city.as_deref().unwrap_or("")),
                jstr(h.postcode.as_deref().unwrap_or("")),
                h.distance_m
            ));
        }
    }
    // The nearest road is useful even where no address is mapped.
    let e7 = ((lat / geo::E7).round() as i32, (lon / geo::E7).round() as i32);
    if let Some(s) = app.ds.snap(e7.0, e7.1, 500.0) {
        o.push_str(&format!(
            "\"road\":{{\"name\":{},\"lat\":{:.7},\"lon\":{:.7},\"distance_m\":{:.1}}},",
            jstr(app.ds.street_name(s.seg as usize).unwrap_or("")),
            geo::e7_to_deg(s.lat_e7),
            geo::e7_to_deg(s.lon_e7),
            s.off_m
        ));
    }
    if o.ends_with(',') {
        o.pop();
    }
    o.push('}');
    o.into_bytes()
}

/// Road geometry for a viewport — the offline basemap.
fn api_roads(app: &App, q: &Query) -> Vec<u8> {
    let Some(bbox) = q.get("bbox") else {
        return b"{\"roads\":[]}".to_vec();
    };
    let p: Vec<f64> = bbox.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if p.len() != 4 {
        return b"{\"roads\":[]}".to_vec();
    }
    let z = q.f64("z").unwrap_or(12.0);
    let (s, w, n, e) = (p[0], p[1], p[2], p[3]);
    let to7 = |v: f64| (v / geo::E7).round() as i32;
    let (s7, w7, n7, e7) = (to7(s), to7(w), to7(n), to7(e));

    // Which classes are worth drawing, and which index can supply them.
    let max_tier: u8 = match z as i32 {
        i32::MIN..=5 => 0,
        6..=8 => 1,
        9..=10 => 2,
        11..=12 => 3,
        _ => 4,
    };
    // The overview index only holds tier ≤ 1, so it can only serve the zooms
    // that draw nothing else. Choosing it on viewport *size* instead would
    // silently drop secondary roads from a wide mid-zoom window.
    let use_big = max_tier <= 1;

    // Budget the segments *per cell* rather than stopping when the total is
    // reached. Stopping would fill the budget from the south-west corner and
    // leave the rest of the viewport blank — at continental zoom you would see
    // one corner of the world. Thinning each cell instead keeps coverage
    // uniform and degrades detail evenly.
    let step = if use_big { BIG_E7 as i64 } else { CELL_E7 as i64 };
    let rows = ((n7 as i64 - s7 as i64) / step + 1).max(1) as usize;
    let cols = ((e7 as i64 - w7 as i64) / step + 1).max(1) as usize;
    // Zoomed far out every polyline collapses to two or three points, so more
    // segments cost little; zoomed in each one carries real geometry.
    let budget = if max_tier <= 1 { 150_000usize } else { 60_000 };
    let per_cell = (budget / rows.saturating_mul(cols).max(1)).max(1);

    let mut segs: Vec<u32> = Vec::with_capacity(budget.min(1 << 16));
    let mut kept: Vec<u32> = Vec::new();
    let mut y = (s7 as i64 / step) * step;
    while y <= n7 as i64 {
        let mut x = (w7 as i64 / step) * step;
        while x <= e7 as i64 {
            let cell = if use_big {
                app.ds.big_cell_segments(big_cell_of(y as i32, x as i32))
            } else {
                app.ds.cell_segments(cell_of(y as i32, x as i32))
            };
            // Drop the classes this zoom will not draw *before* budgeting, so
            // the quota is spent on roads that actually reach the canvas
            // instead of on residential streets we are about to discard.
            kept.clear();
            kept.extend(cell.iter().copied().filter(|&sid| {
                build::class_of(build::attr_class(app.ds.attr(sid as usize))).tier() <= max_tier
            }));
            if kept.len() <= per_cell {
                segs.extend_from_slice(&kept);
            } else {
                // Even stride through the cell: a representative sample of the
                // roads in it, not the first N.
                let stride = kept.len().div_ceil(per_cell);
                segs.extend(kept.iter().step_by(stride).copied());
            }
            x += step;
        }
        y += step;
    }
    segs.sort_unstable();
    segs.dedup();

    // Drop points that would land on the same pixel at this zoom.
    let px_deg = 360.0 / (256.0 * 2f64.powf(z));
    let tol = px_deg * 1.2;
    // The overview grid is 1° coarse, so one cell reaches far outside a
    // continental viewport. Keep a segment when its bounding box *overlaps*
    // the requested one — a "has a point inside" test would drop a motorway
    // that crosses the whole view without a vertex in it.
    let pad = (n - s).max(e - w) * 0.05 + 1e-4;
    let (cs, cw, cn, ce) = (s - pad, w - pad, n + pad, e + pad);

    let mut o = String::with_capacity(1 << 20);
    o.push_str("{\"roads\":[");
    let mut first = true;
    let mut drawn = 0usize;
    let mut pts_out: Vec<(f64, f64)> = Vec::with_capacity(64);
    for &sid in &segs {
        let cls = build::class_of(build::attr_class(app.ds.attr(sid as usize)));
        let pts = app.ds.seg_geometry(sid as usize);
        pts_out.clear();
        let mut last: Option<(f64, f64)> = None;
        let (mut blo_la, mut bhi_la) = (f64::MAX, f64::MIN);
        let (mut blo_lo, mut bhi_lo) = (f64::MAX, f64::MIN);
        for (i, pt) in pts.iter().enumerate() {
            let (la, lo) = (geo::e7_to_deg(pt.0), geo::e7_to_deg(pt.1));
            blo_la = blo_la.min(la);
            bhi_la = bhi_la.max(la);
            blo_lo = blo_lo.min(lo);
            bhi_lo = bhi_lo.max(lo);
            let keep = i == 0
                || i == pts.len() - 1
                || last.is_none_or(|(a, b)| (la - a).abs() > tol || (lo - b).abs() > tol);
            if keep {
                pts_out.push((la, lo));
                last = Some((la, lo));
            }
        }
        let overlaps = bhi_la >= cs && blo_la <= cn && bhi_lo >= cw && blo_lo <= ce;
        if !overlaps || pts_out.len() < 2 {
            continue;
        }
        if !first {
            o.push(',');
        }
        first = false;
        drawn += 1;
        o.push_str(&format!("{{\"c\":{},\"p\":[", cls as u8));
        for (i, (la, lo)) in pts_out.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&format!("{la:.5},{lo:.5}"));
        }
        o.push_str("]}");
    }
    o.push_str(&format!("],\"count\":{drawn},\"candidates\":{},\"zoom\":{z}}}", segs.len()));
    o.into_bytes()
}
