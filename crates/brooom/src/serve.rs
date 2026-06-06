//! Minimal hand-rolled HTTP/1.1 API exposing the solver — no new dependencies
//! (the outbound webhook reuses `ureq`, already in the tree via the `osrm`
//! feature). Started with `brooom --serve <port>`.
//!
//! Endpoints:
//!   * `GET  /health`      → `ok` (liveness).
//!   * `POST /solve`       → a VROOM-style problem JSON (the same body `-i`
//!                           accepts, incl. an `options` object for objective /
//!                           dimensions). Returns the solution JSON synchronously.
//!                           If the body has a top-level `"webhook": "<url>"`,
//!                           the server instead returns **202** with
//!                           `{"job_id": <n>}` immediately, solves in the
//!                           background, and POSTs the solution JSON to `<url>`
//!                           when done (the async / webhook path).
//!   * `GET  /jobs/<id>`   → `{"status":"running|done|error", ...}` for polling
//!                           an async job.
//!
//! Same JSON contract as the CLI and the JSON-config surface — this is just
//! that surface over HTTP.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use brooom::io::to_output;
use brooom::matrix::{HaversineMatrix, MatrixSource, OsrmClient};
use brooom::solver::{solve_full, SolverConfig};

/// Server configuration, built from the CLI flags.
#[derive(Clone)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub time_limit_s: Option<f64>,
    /// `true` ⇒ build the matrix from an OSRM `/table` endpoint, else haversine.
    pub use_osrm: bool,
    pub osrm_host: String,
    pub osrm_profile: String,
    pub speed_mps: f64,
    pub detour: f64,
    pub verbose: bool,
}

impl ServeConfig {
    fn source(&self) -> Box<dyn MatrixSource> {
        if self.use_osrm {
            Box::new(OsrmClient::new(self.osrm_host.clone(), self.osrm_profile.clone()))
        } else {
            Box::new(HaversineMatrix { speed_mps: self.speed_mps, detour: self.detour })
        }
    }
}

#[derive(Clone)]
enum JobState {
    Running,
    Done(String),
    Error(String),
}

type Jobs = Arc<Mutex<HashMap<u64, JobState>>>;
static JOB_SEQ: AtomicU64 = AtomicU64::new(1);

/// Start the HTTP server. Blocks forever (one thread per connection).
pub fn run(cfg: ServeConfig) -> std::io::Result<()> {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr)?;
    let jobs: Jobs = Arc::new(Mutex::new(HashMap::new()));
    eprintln!(
        "brooom: serving on http://{addr}  (POST /solve · GET /jobs/<id> · GET /health)"
    );
    let cfg = Arc::new(cfg);
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let cfg = Arc::clone(&cfg);
        let jobs = Arc::clone(&jobs);
        thread::spawn(move || {
            if let Err(e) = handle(stream, &cfg, &jobs) {
                if cfg.verbose {
                    eprintln!("brooom: connection error: {e}");
                }
            }
        });
    }
    Ok(())
}

fn handle(mut stream: TcpStream, cfg: &ServeConfig, jobs: &Jobs) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(120)))?;
    let (method, path, body) = match read_request(&mut stream) {
        Ok(r) => r,
        Err(_) => return respond(&mut stream, 400, "Bad Request", "text/plain", b"bad request"),
    };

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut stream, 200, "OK", "text/plain", b"ok"),
        ("POST", "/solve") => handle_solve(&mut stream, cfg, jobs, &body),
        ("GET", p) if p.starts_with("/jobs/") => handle_job(&mut stream, jobs, &p[6..]),
        _ => respond(
            &mut stream,
            404,
            "Not Found",
            "application/json",
            br#"{"error":"unknown route; use POST /solve, GET /jobs/<id>, GET /health"}"#,
        ),
    }
}

fn handle_solve(
    stream: &mut TcpStream,
    cfg: &ServeConfig,
    jobs: &Jobs,
    body: &[u8],
) -> std::io::Result<()> {
    // Peek for an async webhook URL without disturbing the solver's own parse.
    let webhook: Option<String> = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("webhook").and_then(|w| w.as_str()).map(str::to_owned));

    match webhook {
        // Synchronous: solve now, return the solution.
        None => match solve_json(body, cfg) {
            Ok(out) => respond(stream, 200, "OK", "application/json", out.as_bytes()),
            Err(e) => respond(stream, 422, "Unprocessable Entity", "application/json", err_json(&e).as_bytes()),
        },
        // Asynchronous: register a job, return 202 + id, solve + POST in a thread.
        Some(url) => {
            let id = JOB_SEQ.fetch_add(1, Ordering::Relaxed);
            jobs.lock().unwrap().insert(id, JobState::Running);
            let body = body.to_vec();
            let cfg = cfg.clone();
            let jobs = Arc::clone(jobs);
            thread::spawn(move || {
                let state = match solve_json(&body, &cfg) {
                    Ok(out) => {
                        post_webhook(&url, &out, id, cfg.verbose);
                        JobState::Done(out)
                    }
                    Err(e) => {
                        post_webhook(&url, &err_json(&e), id, cfg.verbose);
                        JobState::Error(e)
                    }
                };
                jobs.lock().unwrap().insert(id, state);
            });
            let body = format!("{{\"job_id\":{id},\"status\":\"running\"}}");
            respond(stream, 202, "Accepted", "application/json", body.as_bytes())
        }
    }
}

fn handle_job(stream: &mut TcpStream, jobs: &Jobs, id_str: &str) -> std::io::Result<()> {
    let Ok(id) = id_str.parse::<u64>() else {
        return respond(stream, 400, "Bad Request", "application/json", br#"{"error":"bad job id"}"#);
    };
    let state = jobs.lock().unwrap().get(&id).cloned();
    match state {
        None => respond(stream, 404, "Not Found", "application/json", br#"{"error":"no such job"}"#),
        Some(JobState::Running) => {
            respond(stream, 200, "OK", "application/json", br#"{"status":"running"}"#)
        }
        Some(JobState::Done(out)) => {
            let body = format!("{{\"status\":\"done\",\"result\":{out}}}");
            respond(stream, 200, "OK", "application/json", body.as_bytes())
        }
        Some(JobState::Error(e)) => {
            let body = format!("{{\"status\":\"error\",\"error\":{}}}", json_string(&e));
            respond(stream, 200, "OK", "application/json", body.as_bytes())
        }
    }
}

/// Parse a VROOM-style problem (with optional `options`), solve, serialize.
fn solve_json(body: &[u8], cfg: &ServeConfig) -> Result<String, String> {
    let (mut problem, opts) =
        brooom::io::parse_input_reader_with_options(std::io::Cursor::new(body))
            .map_err(|e| format!("parse: {e}"))?;

    let objective_mode = opts.objective_mode().map_err(|e| format!("objective: {e}"))?;
    let dimensions = opts.build_dimensions().map_err(|e| format!("dimensions: {e}"))?;
    let _dim_guard = (!dimensions.is_empty()).then(|| brooom::DimensionGuard::install(dimensions));

    let config = SolverConfig {
        time_limit_ms: cfg.time_limit_s.map(|s| (s * 1000.0) as u64),
        objective_mode,
        ..Default::default()
    };

    let source = cfg.source();
    let solved = solve_full(&mut problem, Some(source.as_ref()), config)
        .map_err(|e| format!("solve: {e}"))?;
    let out = to_output(&problem, &solved.solution, Some(&solved.matrix));
    serde_json::to_string(&out).map_err(|e| format!("serialize: {e}"))
}

/// POST the result JSON back to the caller's webhook URL (best-effort).
fn post_webhook(url: &str, body: &str, job_id: u64, verbose: bool) {
    let res = ureq::post(url)
        .set("Content-Type", "application/json")
        .set("X-Brooom-Job", &job_id.to_string())
        .send_string(body);
    if verbose {
        match res {
            Ok(r) => eprintln!("brooom: webhook job {job_id} → {} {}", r.status(), url),
            Err(e) => eprintln!("brooom: webhook job {job_id} failed: {e}"),
        }
    }
}

// ---- tiny HTTP/1.1 helpers ------------------------------------------------

/// Read method, path, and body. Reads headers until the blank line, honours
/// `Content-Length`, and returns the body bytes.
fn read_request(stream: &mut TcpStream) -> std::io::Result<(String, String, Vec<u8>)> {
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut chunk = [0u8; 8192];
    // Read until we have the full header block.
    let header_end = loop {
        if let Some(i) = find(&buf, b"\r\n\r\n") {
            break i;
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof in headers"));
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > 1 << 20 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "headers too large"));
        }
    };

    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("").to_string();
    let path = raw_path.split('?').next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    for line in lines {
        if let Some(v) = line.strip_prefix("Content-Length:").or_else(|| line.strip_prefix("content-length:")) {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }

    // Body: bytes already read past the header, plus any remaining up to length.
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok((method, path, body))
}

fn respond(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn err_json(msg: &str) -> String {
    format!("{{\"error\":{}}}", json_string(msg))
}

/// Minimal JSON string escaping for error messages.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
