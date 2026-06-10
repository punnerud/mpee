//! Minimal OSRM-compatible HTTP server.
//!
//! Endpoints:
//!
//!   * `GET /route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}` — point-to-point
//!     route. Returns OSRM-shaped JSON: `{ code, routes: [{ distance, duration,
//!     geometry, ... }], waypoints }`.
//!   * `GET /health` — returns `ok` for liveness checks.
//!
//! Usage:
//!
//!   serve london              # binds 127.0.0.1:5003, loads London CH
//!   serve england 5005        # binds on a custom port
//!   serve <pp_path> <ch_path> [port]   # load specific caches
//!
//! No external dependencies — minimal HTTP/1.1 written by hand. The point
//! is to expose the CH stack behind a familiar API so it's drop-in for
//! anything pointed at OSRM.

use std::collections::HashMap;
use std::io::{BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use dijeng::cache_ch;
use dijeng::cache_pp;
use dijeng::osm_profile::Profile;
use dijeng::polyline;
use dijeng::routing::{RouteResponse, RoutingService};

/// Server state: which profiles are mounted, plus their RoutingServices.
struct ServerState {
    profiles: HashMap<&'static str, Arc<RoutingService>>,
    /// Aliases `Profile::from_name` recognises (driving→car, bike→bicycle, …).
    /// Resolved at request time so the URL `/route/v1/driving/...` works
    /// against the `car` profile.
    aliases: HashMap<&'static str, &'static str>,
}

impl ServerState {
    fn lookup(&self, profile_name: &str) -> Option<&Arc<RoutingService>> {
        if let Some(s) = self.profiles.get(profile_name) {
            return Some(s);
        }
        if let Some(canonical) = self.aliases.get(profile_name) {
            return self.profiles.get(canonical);
        }
        None
    }

    fn list_names(&self) -> Vec<&'static str> {
        let mut v: Vec<_> = self.profiles.keys().copied().collect();
        v.sort();
        v
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dataset, port) = match args.as_slice() {
        [] => ("london".to_string(), 5003),
        [d] => (d.clone(), 5003),
        [d, p] => (d.clone(), p.parse::<u16>().unwrap_or(5003)),
        _ => (args[0].clone(), args[1].parse::<u16>().unwrap_or(5003)),
    };

    let state = build_state(&dataset)?;
    if state.profiles.is_empty() {
        eprintln!(
            "[serve] no profile caches found for dataset '{dataset}'. Run \
             `bench_pp {dataset} <profile>` and `bench_ch {dataset} <profile>` first."
        );
        std::process::exit(1);
    }
    let state = Arc::new(state);

    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!(
        "[serve] listening on http://{addr}  profiles=[{}]",
        state.list_names().join(", ")
    );

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[serve] accept error: {e}");
                continue;
            }
        };
        let state = Arc::clone(&state);
        thread::spawn(move || {
            if let Err(e) = handle(stream, state.as_ref()) {
                eprintln!("[serve] client error: {e}");
            }
        });
    }
    Ok(())
}

fn build_state(dataset: &str) -> std::io::Result<ServerState> {
    let base = match dataset {
        "london" => "data/greater-london.osm.pbf",
        "england" => "data/england.osm.pbf",
        other => other,
    };

    let mut profiles: HashMap<&'static str, Arc<RoutingService>> = HashMap::new();
    let candidates: &[(Profile, &'static str)] = &[
        (Profile::Car, "car"),
        (Profile::Motorcycle, "motorcycle"),
        (Profile::Bicycle, "bicycle"),
        (Profile::Foot, "foot"),
    ];

    for (profile, name) in candidates {
        let suffix = if *profile == Profile::Car {
            String::new()
        } else {
            format!(".{}", profile.name())
        };
        let pp_path = format!("{base}{suffix}.pp");
        let ch_path = format!("{base}{suffix}.ch");
        if !std::path::Path::new(&pp_path).exists() || !std::path::Path::new(&ch_path).exists() {
            continue;
        }
        let t = Instant::now();
        let pp = match cache_pp::load_mmap(&pp_path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[serve] skip {name}: pp-cache {pp_path} error: {e}");
                continue;
            }
        };
        let ch = match cache_ch::load_mmap(&ch_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[serve] skip {name}: ch-cache {ch_path} error: {e}");
                continue;
            }
        };
        let svc = RoutingService::new(ch, pp.coords);
        println!(
            "[serve] mounted profile '{name}' ({:.0} ms, n={})",
            t.elapsed().as_secs_f64() * 1000.0,
            svc.node_count()
        );
        profiles.insert(*name, Arc::new(svc));
    }

    let aliases: HashMap<&'static str, &'static str> = [
        ("driving", "car"),
        ("moto", "motorcycle"),
        ("bike", "bicycle"),
        ("cycling", "bicycle"),
        ("walk", "foot"),
        ("walking", "foot"),
        ("pedestrian", "foot"),
    ]
    .into_iter()
    .collect();

    Ok(ServerState { profiles, aliases })
}

fn handle(mut stream: TcpStream, state: &ServerState) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    let request_line = read_request_line(&mut stream)?;

    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");
    if method != "GET" {
        return write_response(
            &mut stream,
            405,
            "Method Not Allowed",
            "application/json",
            r#"{"code":"InvalidUrl","message":"only GET supported"}"#,
        );
    }

    let (path_only, _query) = match path.find('?') {
        Some(i) => path.split_at(i),
        None => (path, ""),
    };

    if path_only == "/health" {
        return write_response(&mut stream, 200, "OK", "text/plain", "ok");
    }
    if path_only == "/profiles" {
        let names = state.list_names();
        let body = format!(
            r#"{{"profiles":[{}]}}"#,
            names
                .iter()
                .map(|n| format!("\"{}\"", n))
                .collect::<Vec<_>>()
                .join(",")
        );
        return write_response(&mut stream, 200, "OK", "application/json", &body);
    }

    if let Some(rest) = path_only.strip_prefix("/route/v1/") {
        let query = match path.find('?') {
            Some(i) => path[i + 1..].to_string(),
            None => String::new(),
        };
        return dispatch_with_profile(&mut stream, state, rest, move |stream, svc, rest| {
            handle_route(stream, svc, rest, &query)
        });
    }

    if let Some(rest) = path_only.strip_prefix("/table/v1/") {
        let query = match path.find('?') {
            Some(i) => path[i + 1..].to_string(),
            None => String::new(),
        };
        return dispatch_with_profile(&mut stream, state, rest, move |stream, svc, rest| {
            handle_table(stream, svc, rest, &query)
        });
    }

    if let Some(rest) = path_only.strip_prefix("/nearest/v1/") {
        let query = match path.find('?') {
            Some(i) => path[i + 1..].to_string(),
            None => String::new(),
        };
        return dispatch_with_profile(&mut stream, state, rest, move |stream, svc, rest| {
            handle_nearest(stream, svc, rest, &query)
        });
    }

    if let Some(rest) = path_only.strip_prefix("/trip/v1/") {
        let query = match path.find('?') {
            Some(i) => path[i + 1..].to_string(),
            None => String::new(),
        };
        return dispatch_with_profile(&mut stream, state, rest, move |stream, svc, rest| {
            handle_trip(stream, svc, rest, &query)
        });
    }

    if let Some(rest) = path_only.strip_prefix("/match/v1/") {
        let query = match path.find('?') {
            Some(i) => path[i + 1..].to_string(),
            None => String::new(),
        };
        return dispatch_with_profile(&mut stream, state, rest, move |stream, svc, rest| {
            handle_match(stream, svc, rest, &query)
        });
    }

    if let Some(rest) = path_only.strip_prefix("/isochrone/v1/") {
        let query = match path.find('?') {
            Some(i) => path[i + 1..].to_string(),
            None => String::new(),
        };
        return dispatch_with_profile(&mut stream, state, rest, move |stream, svc, rest| {
            handle_isochrone(stream, svc, rest, &query)
        });
    }

    write_response(
        &mut stream,
        404,
        "Not Found",
        "application/json",
        r#"{"code":"InvalidUrl","message":"no route matched"}"#,
    )
}

/// Pull the leading profile slug off `rest` and dispatch to the handler with
/// the matching `RoutingService`. Returns 404 if the profile is unknown.
fn dispatch_with_profile<F>(
    stream: &mut TcpStream,
    state: &ServerState,
    rest: &str,
    handler: F,
) -> std::io::Result<()>
where
    F: FnOnce(&mut TcpStream, &RoutingService, &str) -> std::io::Result<()>,
{
    let mut parts = rest.splitn(2, '/');
    let profile_name = parts.next().unwrap_or("");
    let tail = parts.next().unwrap_or("");
    let svc = match state.lookup(profile_name) {
        Some(s) => s,
        None => {
            let known = state.list_names().join(", ");
            let body = format!(
                r#"{{"code":"InvalidProfile","message":"unknown profile '{profile_name}', mounted: [{known}]"}}"#,
            );
            return write_response(stream, 400, "Bad Request", "application/json", &body);
        }
    };
    handler(stream, svc.as_ref(), tail)
}

/// Read until end-of-headers `\r\n\r\n`, return the request line (the bytes
/// before the first `\r\n`). We accept up to 4 MiB of request line + headers
/// so /table calls with thousands of coordinates over GET fit. We're a GET-
/// only server, so any body sent is ignored.
fn read_request_line(stream: &mut TcpStream) -> std::io::Result<String> {
    const LIMIT: usize = 4 * 1024 * 1024;
    let mut buf = Vec::with_capacity(8192);
    let mut chunk = [0u8; 8192];
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        // End-of-headers marker terminates header parsing.
        if find_subseq(&buf, b"\r\n\r\n").is_some() {
            break;
        }
        if buf.len() > LIMIT {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "request too long",
            ));
        }
    }
    let line_end = find_subseq(&buf, b"\r\n").unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..line_end]).into_owned())
}

fn find_subseq(buf: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || buf.len() < needle.len() {
        return None;
    }
    for i in 0..=buf.len() - needle.len() {
        if &buf[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}


fn handle_route(
    stream: &mut TcpStream,
    svc: &RoutingService,
    coords_part: &str,
    query: &str,
) -> std::io::Result<()> {
    let coords_part = coords_part.split('?').next().unwrap_or(coords_part);
    if coords_part.is_empty() {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"missing coordinates"}"#,
        );
    }
    let waypoints: Vec<(f32, f32)> = coords_part
        .split(';')
        .filter_map(|wp| {
            let mut it = wp.split(',');
            let lon = it.next()?.parse::<f32>().ok()?;
            let lat = it.next()?.parse::<f32>().ok()?;
            Some((lat, lon))
        })
        .collect();

    if waypoints.len() != 2 {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"v1 route requires exactly 2 waypoints"}"#,
        );
    }

    let (src_lat, src_lon) = waypoints[0];
    let (dst_lat, dst_lon) = waypoints[1];

    // OSRM-style `alternatives=true` (≤2 extras) or `alternatives=N` (≤5).
    let alts: usize = match query_param(query, "alternatives") {
        Some("true") => 2,
        Some(v) => v.parse::<usize>().unwrap_or(0).min(5),
        None => 0,
    };

    let t = Instant::now();
    if alts > 0 {
        let routes = svc.route_alternatives(src_lat, src_lon, dst_lat, dst_lon, alts, 0.25, 0.6);
        let elapsed_us = t.elapsed().as_micros();
        return match routes {
            Some(rs) if !rs.is_empty() => {
                let body = render_routes(&rs, elapsed_us);
                write_response(stream, 200, "OK", "application/json", &body)
            }
            _ => {
                let body = format!(
                    r#"{{"code":"NoRoute","message":"no path between waypoints","elapsed_us":{}}}"#,
                    t.elapsed().as_micros()
                );
                write_response(stream, 200, "OK", "application/json", &body)
            }
        };
    }
    let route = svc.route(src_lat, src_lon, dst_lat, dst_lon);
    let elapsed_us = t.elapsed().as_micros();

    match route {
        Some(r) => {
            let body = render_ok(&r, elapsed_us);
            write_response(stream, 200, "OK", "application/json", &body)
        }
        None => {
            let body = format!(
                r#"{{"code":"NoRoute","message":"no path between waypoints","elapsed_us":{}}}"#,
                elapsed_us
            );
            write_response(stream, 200, "OK", "application/json", &body)
        }
    }
}

fn handle_nearest(
    stream: &mut TcpStream,
    svc: &RoutingService,
    coords_part: &str,
    query: &str,
) -> std::io::Result<()> {
    // Strip any query string the dispatcher left on the coords part.
    let coords_part = coords_part.split('?').next().unwrap_or(coords_part);
    if coords_part.is_empty() {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"missing coordinate"}"#,
        );
    }
    let mut it = coords_part.split(',');
    let (Some(lon_s), Some(lat_s)) = (it.next(), it.next()) else {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"expected lon,lat"}"#,
        );
    };
    let (Ok(lon), Ok(lat)) = (lon_s.parse::<f32>(), lat_s.parse::<f32>()) else {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"unparseable coordinate"}"#,
        );
    };
    // OSRM-compatible `number=K` (default 1, capped to keep responses sane).
    let k: usize = query_param(query, "number")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .clamp(1, 100);
    let t = Instant::now();
    let hits = svc.nearest_nodes(lat, lon, k);
    let elapsed_us = t.elapsed().as_micros();
    let waypoints: Vec<String> = hits
        .iter()
        .map(|&(id, snap_lat, snap_lon)| {
            let dist = haversine_m(lat, lon, snap_lat, snap_lon);
            let name = svc.reverse(snap_lat, snap_lon).unwrap_or("");
            format!(
                r#"{{"location":[{slon},{slat}],"name":"{name}","distance":{d:.1},"node":{id}}}"#,
                slon = snap_lon,
                slat = snap_lat,
                d = dist,
            )
        })
        .collect();
    let body = format!(
        r#"{{"code":"Ok","waypoints":[{}],"elapsed_us":{e}}}"#,
        waypoints.join(","),
        e = elapsed_us,
    );
    write_response(stream, 200, "OK", "application/json", &body)
}

/// Map matching: `GET /match/v1/{profile}/{lon,lat;lon,lat;...}?sigma=15&k=8`.
/// OSRM-shaped: returns one matched waypoint per input ping (snapped location,
/// node, snap distance, segment continuity) plus an overall confidence.
fn handle_match(
    stream: &mut TcpStream,
    svc: &RoutingService,
    coords_part: &str,
    query: &str,
) -> std::io::Result<()> {
    let coords_part = coords_part.split('?').next().unwrap_or(coords_part);
    let trace: Vec<(f32, f32)> = coords_part
        .split(';')
        .filter_map(|wp| {
            let mut it = wp.split(',');
            let lon = it.next()?.parse::<f32>().ok()?;
            let lat = it.next()?.parse::<f32>().ok()?;
            Some((lat, lon))
        })
        .collect();
    if trace.len() < 2 {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"match requires at least 2 trace points"}"#,
        );
    }
    if trace.len() > 5000 {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"match caps at 5000 trace points"}"#,
        );
    }
    let sigma: f32 = query_param(query, "sigma")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(15.0)
        .clamp(1.0, 200.0);
    let k: usize = query_param(query, "k")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .clamp(2, 32);
    let t = Instant::now();
    let res = svc.match_trace(&trace, k, sigma);
    let elapsed_us = t.elapsed().as_micros();
    match res {
        Some(m) => {
            let pts: Vec<String> = m
                .points
                .iter()
                .map(|p| {
                    format!(
                        r#"{{"location":[{lon},{lat}],"node":{node},"distance":{d:.1},"connected":{conn}}}"#,
                        lon = p.lon,
                        lat = p.lat,
                        node = p.node,
                        d = p.snap_distance_m,
                        conn = p.connected,
                    )
                })
                .collect();
            let body = format!(
                r#"{{"code":"Ok","confidence":{:.3},"tracepoints":[{}],"elapsed_us":{elapsed_us}}}"#,
                m.confidence,
                pts.join(","),
            );
            write_response(stream, 200, "OK", "application/json", &body)
        }
        None => write_response(
            stream,
            200,
            "OK",
            "application/json",
            r#"{"code":"NoMatch","message":"no routing graph mounted"}"#,
        ),
    }
}

/// Isochrones: `GET /isochrone/v1/{profile}/{lon},{lat}?contours=300,600,900`
/// (seconds; `&metric=distance` switches to metres, `&cell=0.0015` sets the
/// polygon resolution in degrees). GeoJSON FeatureCollection out, one
/// MultiPolygon Feature per contour, Valhalla-style.
fn handle_isochrone(
    stream: &mut TcpStream,
    svc: &RoutingService,
    coords_part: &str,
    query: &str,
) -> std::io::Result<()> {
    let coords_part = coords_part.split('?').next().unwrap_or(coords_part);
    let mut it = coords_part.split(',');
    let (Some(lon_s), Some(lat_s)) = (it.next(), it.next()) else {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"expected lon,lat"}"#,
        );
    };
    let (Ok(lon), Ok(lat)) = (lon_s.parse::<f32>(), lat_s.parse::<f32>()) else {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"unparseable coordinate"}"#,
        );
    };
    let mut contours: Vec<f32> = query_param(query, "contours")
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![300.0, 600.0, 900.0]);
    contours.retain(|c| *c > 0.0 && *c <= 14_400.0);
    contours.sort_by(|a, b| a.partial_cmp(b).unwrap());
    contours.dedup();
    if contours.is_empty() || contours.len() > 8 {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidOptions","message":"contours: 1-8 positive values (seconds, <=14400)"}"#,
        );
    }
    let cell: f32 = query_param(query, "cell")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0015)
        .clamp(0.0003, 0.02);
    let metric_dist = query_param(query, "metric") == Some("distance");

    let t = Instant::now();
    let bands = svc.isochrone(lat, lon, &contours, cell, metric_dist);
    let elapsed_us = t.elapsed().as_micros();
    match bands {
        Some(bands) => {
            let features: Vec<String> = bands
                .iter()
                .map(|b| {
                    let polys: Vec<String> = b
                        .rings
                        .iter()
                        .map(|ring| {
                            let pts: Vec<String> = ring
                                .iter()
                                .map(|&(la, lo)| format!("[{lo},{la}]"))
                                .collect();
                            format!("[[{}]]", pts.join(","))
                        })
                        .collect();
                    format!(
                        r#"{{"type":"Feature","properties":{{"contour":{}}},"geometry":{{"type":"MultiPolygon","coordinates":[{}]}}}}"#,
                        b.limit,
                        polys.join(",")
                    )
                })
                .collect();
            let body = format!(
                r#"{{"type":"FeatureCollection","features":[{}],"elapsed_us":{elapsed_us}}}"#,
                features.join(",")
            );
            write_response(stream, 200, "OK", "application/json", &body)
        }
        None => write_response(
            stream,
            200,
            "OK",
            "application/json",
            r#"{"code":"NoIsochrone","message":"no routing graph mounted"}"#,
        ),
    }
}

/// Extract a `key=value` pair from a raw query string.
fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|kv| {
        let mut it = kv.splitn(2, '=');
        (it.next()? == key).then(|| it.next().unwrap_or(""))
    })
}

/// Trip service: `GET /trip/v1/{profile}/{lon,lat;lon,lat;...}?roundtrip=true`.
/// Orders the waypoints into the shortest visiting sequence (exact for ≤13,
/// 2-opt/Or-opt heuristic above) and returns per-leg routes.
fn handle_trip(
    stream: &mut TcpStream,
    svc: &RoutingService,
    coords_part: &str,
    query: &str,
) -> std::io::Result<()> {
    let coords_part = coords_part.split('?').next().unwrap_or(coords_part);
    let waypoints: Vec<(f32, f32)> = coords_part
        .split(';')
        .filter_map(|wp| {
            let mut it = wp.split(',');
            let lon = it.next()?.parse::<f32>().ok()?;
            let lat = it.next()?.parse::<f32>().ok()?;
            Some((lat, lon))
        })
        .collect();
    if waypoints.len() < 2 {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"trip requires at least 2 waypoints"}"#,
        );
    }
    if waypoints.len() > 500 {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"trip caps at 500 waypoints"}"#,
        );
    }
    let roundtrip = query_param(query, "roundtrip") != Some("false");
    let t = Instant::now();
    let trip = svc.trip(&waypoints, roundtrip);
    let elapsed_us = t.elapsed().as_micros();
    match trip {
        Some(tr) => {
            let legs: Vec<String> = tr
                .legs
                .iter()
                .map(|leg| {
                    let geom: Vec<String> = leg
                        .geometry
                        .iter()
                        .map(|&(la, lo)| format!("[{lo},{la}]"))
                        .collect();
                    format!(
                        r#"{{"duration":{:.1},"distance":{:.1},"geometry":{{"type":"LineString","coordinates":[{}]}}}}"#,
                        leg.duration_s,
                        leg.distance_m,
                        geom.join(",")
                    )
                })
                .collect();
            let order: Vec<String> = tr.order.iter().map(|i| i.to_string()).collect();
            let body = format!(
                r#"{{"code":"Ok","order":[{}],"roundtrip":{roundtrip},"duration":{:.1},"distance":{:.1},"legs":[{}],"elapsed_us":{elapsed_us}}}"#,
                order.join(","),
                tr.duration_s,
                tr.distance_m,
                legs.join(","),
            );
            write_response(stream, 200, "OK", "application/json", &body)
        }
        None => {
            let body = format!(
                r#"{{"code":"NoTrip","message":"no feasible ordering (disconnected waypoints?)","elapsed_us":{elapsed_us}}}"#,
            );
            write_response(stream, 200, "OK", "application/json", &body)
        }
    }
}

fn haversine_m(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let r = 6_371_000.0_f64;
    let l1 = (lat1 as f64).to_radians();
    let l2 = (lat2 as f64).to_radians();
    let dlat = (lat2 as f64 - lat1 as f64).to_radians();
    let dlon = (lon2 as f64 - lon1 as f64).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + l1.cos() * l2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    (r * c) as f32
}

fn handle_table(
    stream: &mut TcpStream,
    svc: &RoutingService,
    coords_part: &str,
    query: &str,
) -> std::io::Result<()> {
    if coords_part.is_empty() {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"missing coordinates"}"#,
        );
    }
    let coords: Vec<(f32, f32)> = coords_part
        .split(';')
        .filter_map(|wp| {
            let mut it = wp.split(',');
            let lon = it.next()?.parse::<f32>().ok()?;
            let lat = it.next()?.parse::<f32>().ok()?;
            Some((lat, lon))
        })
        .collect();
    if coords.is_empty() {
        return write_response(
            stream,
            400,
            "Bad Request",
            "application/json",
            r#"{"code":"InvalidUrl","message":"no coordinates"}"#,
        );
    }

    // sources/destinations parameters select subsets by index. "all" means all.
    // annotations=duration|distance|duration,distance picks which matrices
    // to return. VROOM and OSRM v5+ default to both.
    let mut src_idx: Vec<usize> = (0..coords.len()).collect();
    let mut dst_idx: Vec<usize> = (0..coords.len()).collect();
    let mut want_duration = true;
    let mut want_distance = false;
    let mut explicit_annotations = false;
    // `stream=true` opt-in for HTTP/1.1 chunked encoding. Default off because
    // some clients (VROOM 1.15) don't accept chunked /table responses.
    let mut stream_response = false;
    for kv in query.split('&') {
        let mut it = kv.splitn(2, '=');
        let k = it.next().unwrap_or("");
        let v = it.next().unwrap_or("");
        match k {
            "sources" => {
                if v != "all" && !v.is_empty() {
                    src_idx = parse_index_list(v, coords.len());
                }
            }
            "destinations" => {
                if v != "all" && !v.is_empty() {
                    dst_idx = parse_index_list(v, coords.len());
                }
            }
            "annotations" => {
                explicit_annotations = true;
                want_duration = false;
                want_distance = false;
                for ann in v.split(',') {
                    match ann {
                        "duration" => want_duration = true,
                        "distance" => want_distance = true,
                        _ => {}
                    }
                }
            }
            "stream" => {
                stream_response = matches!(v, "true" | "1" | "yes");
            }
            _ => {}
        }
    }
    if !explicit_annotations {
        // Default: return both, matches OSRM v5 behaviour expected by VROOM.
        want_duration = true;
        want_distance = true;
    }

    let srcs: Vec<(f32, f32)> = src_idx.iter().map(|&i| coords[i]).collect();
    let dsts: Vec<(f32, f32)> = dst_idx.iter().map(|&i| coords[i]).collect();

    let budget_mb = dijeng::budget::resolve_matrix_budget_mb(
        dijeng::budget::DEFAULT_MATRIX_BUDGET_MB,
    );
    let t = Instant::now();
    let (durations, distances_opt, snap_src, snap_dst) = if want_distance {
        let (du, di, ss, sd) =
            svc.matrix_with_distance_budgeted_full(&srcs, &dsts, budget_mb);
        (du, Some(di), ss, sd)
    } else if budget_mb > 0 {
        let (du, _di, ss, sd) =
            svc.matrix_with_distance_budgeted_full(&srcs, &dsts, budget_mb);
        (du, None, ss, sd)
    } else {
        let (du, ss, sd) = svc.matrix(&srcs, &dsts);
        (du, None, ss, sd)
    };
    let elapsed_us = t.elapsed().as_micros();

    if stream_response {
        // Opt-in: HTTP/1.1 chunked. Lowest peak RAM but rejected by VROOM 1.15.
        write_table_streaming(
            stream,
            if want_duration { Some(&durations) } else { None },
            distances_opt.as_deref(),
            srcs.len(),
            dsts.len(),
            &snap_src,
            &snap_dst,
            elapsed_us,
        )
    } else {
        // Default: two-pass. Pass 1 measures the body length without keeping
        // any text in memory; pass 2 emits Content-Length header followed by
        // body bytes written straight to the socket. Result: VROOM-compatible
        // (Content-Length is honoured) AND zero JSON-string allocation, so
        // peak RAM is bounded by the matrices themselves.
        write_table_content_length(
            stream,
            if want_duration { Some(&durations) } else { None },
            distances_opt.as_deref(),
            srcs.len(),
            dsts.len(),
            &snap_src,
            &snap_dst,
            elapsed_us,
        )
    }
}

/// Two-pass response: count body length, then write Content-Length header +
/// body. Memory peak is bounded by the matrices themselves — no big JSON
/// string is materialised. Compatible with HTTP clients that require
/// Content-Length (e.g. VROOM 1.15).
fn write_table_content_length(
    stream: &mut TcpStream,
    durations: Option<&[f32]>,
    distances: Option<&[f32]>,
    n_src: usize,
    n_dst: usize,
    snap_src: &[(f32, f32)],
    snap_dst: &[(f32, f32)],
    elapsed_us: u128,
) -> std::io::Result<()> {
    // Pass 1 — count.
    let mut counter = CountingWriter(0);
    write_table_body(
        &mut counter,
        durations,
        distances,
        n_src,
        n_dst,
        snap_src,
        snap_dst,
        elapsed_us,
    )?;
    let body_len = counter.0;

    // Pass 2 — emit headers + body straight to the socket.
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {body_len}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
    );
    stream.write_all(header.as_bytes())?;

    let mut bw = BufWriter::with_capacity(64 * 1024, stream);
    write_table_body(
        &mut bw,
        durations,
        distances,
        n_src,
        n_dst,
        snap_src,
        snap_dst,
        elapsed_us,
    )?;
    bw.flush()?;
    Ok(())
}

/// Format the JSON body of a /table response into any `io::Write`. Used for
/// both byte-count pass and actual socket-write pass — guarantees the two
/// produce identical lengths.
fn write_table_body<W: Write>(
    w: &mut W,
    durations: Option<&[f32]>,
    distances: Option<&[f32]>,
    n_src: usize,
    n_dst: usize,
    snap_src: &[(f32, f32)],
    snap_dst: &[(f32, f32)],
    elapsed_us: u128,
) -> std::io::Result<()> {
    w.write_all(br#"{"code":"Ok""#)?;
    if let Some(du) = durations {
        w.write_all(br#","durations":["#)?;
        write_matrix_body(w, du, n_src, n_dst)?;
        w.write_all(b"]")?;
    }
    if let Some(di) = distances {
        w.write_all(br#","distances":["#)?;
        write_matrix_body(w, di, n_src, n_dst)?;
        w.write_all(b"]")?;
    }
    w.write_all(br#","sources":["#)?;
    for (i, &(la, lo)) in snap_src.iter().enumerate() {
        if i > 0 {
            w.write_all(b",")?;
        }
        write!(w, r#"{{"location":[{},{}],"name":""}}"#, lo, la)?;
    }
    w.write_all(br#"],"destinations":["#)?;
    for (i, &(la, lo)) in snap_dst.iter().enumerate() {
        if i > 0 {
            w.write_all(b",")?;
        }
        write!(w, r#"{{"location":[{},{}],"name":""}}"#, lo, la)?;
    }
    write!(w, r#"],"elapsed_us":{}}}"#, elapsed_us)?;
    Ok(())
}

fn write_matrix_body<W: Write>(
    w: &mut W,
    matrix: &[f32],
    n_src: usize,
    n_dst: usize,
) -> std::io::Result<()> {
    for i in 0..n_src {
        if i > 0 {
            w.write_all(b",")?;
        }
        w.write_all(b"[")?;
        for j in 0..n_dst {
            if j > 0 {
                w.write_all(b",")?;
            }
            let v = matrix[i * n_dst + j];
            if v.is_finite() {
                write!(w, "{:.1}", v)?;
            } else {
                w.write_all(b"null")?;
            }
        }
        w.write_all(b"]")?;
    }
    Ok(())
}

/// `io::Write` adapter that just counts bytes, used for the Content-Length
/// measuring pass.
struct CountingWriter(usize);
impl Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len();
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Stream the /table response using HTTP/1.1 chunked transfer-encoding. JSON
/// is emitted row-by-row so we never hold the full text in memory.
fn write_table_streaming(
    stream: &mut TcpStream,
    durations: Option<&[f32]>,
    distances: Option<&[f32]>,
    n_src: usize,
    n_dst: usize,
    snap_src: &[(f32, f32)],
    snap_dst: &[(f32, f32)],
    elapsed_us: u128,
) -> std::io::Result<()> {
    write_chunked_header(stream, "application/json")?;
    let mut buf = String::with_capacity(1 << 14); // 16 KB row scratch

    buf.push_str(r#"{"code":"Ok""#);
    let mut needs_comma = true;

    if let Some(du) = durations {
        if needs_comma {
            buf.push(',');
        }
        buf.push_str(r#""durations":["#);
        write_chunk(stream, &buf)?;
        buf.clear();
        write_matrix_rows(stream, &mut buf, du, n_src, n_dst)?;
        buf.push(']');
        needs_comma = true;
    }

    if let Some(di) = distances {
        if needs_comma {
            buf.push(',');
        }
        buf.push_str(r#""distances":["#);
        write_chunk(stream, &buf)?;
        buf.clear();
        write_matrix_rows(stream, &mut buf, di, n_src, n_dst)?;
        buf.push(']');
    }

    buf.push_str(r#","sources":["#);
    for (i, &(la, lo)) in snap_src.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&format!(r#"{{"location":[{},{}],"name":""}}"#, lo, la));
        if buf.len() > 16 * 1024 {
            write_chunk(stream, &buf)?;
            buf.clear();
        }
    }
    buf.push_str(r#"],"destinations":["#);
    for (i, &(la, lo)) in snap_dst.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&format!(r#"{{"location":[{},{}],"name":""}}"#, lo, la));
        if buf.len() > 16 * 1024 {
            write_chunk(stream, &buf)?;
            buf.clear();
        }
    }
    buf.push_str(&format!(r#"],"elapsed_us":{}}}"#, elapsed_us));
    write_chunk(stream, &buf)?;

    // Final 0-length chunk closes the stream.
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}

/// Write all rows of a single matrix (`durations` or `distances`) flushing
/// every ~64 KB so the OS-level send buffer never grows large.
fn write_matrix_rows(
    stream: &mut TcpStream,
    buf: &mut String,
    matrix: &[f32],
    n_src: usize,
    n_dst: usize,
) -> std::io::Result<()> {
    use std::fmt::Write;
    for i in 0..n_src {
        if i > 0 {
            buf.push(',');
        }
        buf.push('[');
        for j in 0..n_dst {
            if j > 0 {
                buf.push(',');
            }
            let v = matrix[i * n_dst + j];
            if v.is_finite() {
                let _ = write!(buf, "{:.1}", v);
            } else {
                buf.push_str("null");
            }
        }
        buf.push(']');
        if buf.len() > 64 * 1024 {
            write_chunk(stream, buf)?;
            buf.clear();
        }
    }
    Ok(())
}

fn write_chunked_header(stream: &mut TcpStream, content_type: &str) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nTransfer-Encoding: chunked\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        ct = content_type,
    );
    stream.write_all(header.as_bytes())?;
    Ok(())
}

fn write_chunk(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    if body.is_empty() {
        return Ok(());
    }
    let header = format!("{:x}\r\n", body.len());
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.write_all(b"\r\n")?;
    Ok(())
}

fn parse_index_list(v: &str, max: usize) -> Vec<usize> {
    v.split(';')
        .filter_map(|s| s.parse::<usize>().ok())
        .filter(|&i| i < max)
        .collect()
}

fn render_table(
    durations: Option<&[f32]>,
    distances: Option<&[f32]>,
    n_src: usize,
    n_dst: usize,
    snap_src: &[(f32, f32)],
    snap_dst: &[(f32, f32)],
    elapsed_us: u128,
) -> String {
    fn render_matrix(s: &mut String, name: &str, m: &[f32], n_src: usize, n_dst: usize) {
        s.push('"');
        s.push_str(name);
        s.push_str("\":[");
        for i in 0..n_src {
            if i > 0 {
                s.push(',');
            }
            s.push('[');
            for j in 0..n_dst {
                if j > 0 {
                    s.push(',');
                }
                let v = m[i * n_dst + j];
                if v.is_finite() {
                    s.push_str(&format!("{:.1}", v));
                } else {
                    s.push_str("null");
                }
            }
            s.push(']');
        }
        s.push(']');
    }

    let cap = 64 + 24 * n_src * n_dst + 48 * (n_src + n_dst);
    let mut s = String::with_capacity(cap);
    s.push_str(r#"{"code":"Ok","#);
    let mut first = true;
    if let Some(du) = durations {
        render_matrix(&mut s, "durations", du, n_src, n_dst);
        first = false;
    }
    if let Some(di) = distances {
        if !first {
            s.push(',');
        }
        render_matrix(&mut s, "distances", di, n_src, n_dst);
    }
    s.push_str(",\"sources\":[");
    for (i, &(la, lo)) in snap_src.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(r#"{{"location":[{},{}],"name":""}}"#, lo, la));
    }
    s.push_str("],\"destinations\":[");
    for (i, &(la, lo)) in snap_dst.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(r#"{{"location":[{},{}],"name":""}}"#, lo, la));
    }
    s.push_str(&format!("],\"elapsed_us\":{}}}", elapsed_us));
    s
}

/// OSRM-compatible multi-route body (best first, like `alternatives=true`).
fn render_routes(rs: &[RouteResponse], elapsed_us: u128) -> String {
    let routes: Vec<String> = rs
        .iter()
        .map(|r| {
            let geom = polyline::encode(&r.geometry, 5);
            format!(
                r#"{{"distance":{dist:.1},"duration":{dur:.1},"weight":{dur:.1},"weight_name":"duration","geometry":{geom:?},"legs":[{{"distance":{dist:.1},"duration":{dur:.1},"summary":"","steps":[],"weight":{dur:.1}}}]}}"#,
                dist = r.distance_m,
                dur = r.duration_s,
                geom = geom,
            )
        })
        .collect();
    let (slat, slon) = rs[0].source_snapped;
    let (dlat, dlon) = rs[0].destination_snapped;
    format!(
        r#"{{"code":"Ok","routes":[{}],"waypoints":[{{"location":[{slon},{slat}],"name":""}},{{"location":[{dlon},{dlat}],"name":""}}],"elapsed_us":{elapsed_us}}}"#,
        routes.join(","),
    )
}

fn render_ok(r: &RouteResponse, elapsed_us: u128) -> String {
    let geom = polyline::encode(&r.geometry, 5);
    let (slat, slon) = r.source_snapped;
    let (dlat, dlon) = r.destination_snapped;
    // OSRM-compatible shape (single route, single leg, no steps).
    format!(
        r#"{{"code":"Ok","routes":[{{"distance":{dist:.1},"duration":{dur:.1},"weight":{dur:.1},"weight_name":"duration","geometry":{geom:?},"legs":[{{"distance":{dist:.1},"duration":{dur:.1},"summary":"","steps":[],"weight":{dur:.1}}}]}}],"waypoints":[{{"location":[{slon},{slat}],"name":""}},{{"location":[{dlon},{dlat}],"name":""}}],"elapsed_us":{eus}}}"#,
        dist = r.distance_m,
        dur = r.duration_s,
        geom = geom,
        slon = slon,
        slat = slat,
        dlon = dlon,
        dlat = dlat,
        eus = elapsed_us,
    )
}

fn write_response(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ct}\r\nContent-Length: {len}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
        code = code,
        reason = reason,
        ct = content_type,
        len = body.len(),
        body = body,
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()?;
    Ok(())
}

