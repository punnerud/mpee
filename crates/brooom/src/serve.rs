//! Minimal hand-rolled HTTP/1.1 API exposing the solver — no new dependencies
//! (the outbound webhook reuses `ureq`, already in the tree via the `osrm`
//! feature). Started with `brooom --serve <port>`.
//!
//! Jobs run through a bounded worker pool (`--serve-workers N`): submissions
//! queue and are processed by N threads in parallel, so the box is never
//! oversubscribed. Each solve already uses every core via rayon, so the default
//! is one worker (a queue); raise it for many small jobs. Keep it at 1 when
//! solving on the GPU (a single device).
//!
//! Endpoints:
//!   * `GET  /health`        → `{"ok":true,"queued":n,"running":m,...}`.
//!   * `POST /solve`         → a VROOM-style problem JSON (the same body `-i`
//!                             accepts, incl. an `options` object). Synchronous
//!                             by default: waits for the result. Supply a
//!                             top-level `"webhook":"<url>"` (or `?async=1`) to
//!                             return **202** `{job_id,...}` immediately and POST
//!                             the solution to the webhook when done.
//!   * `GET  /jobs`          → list every job (id, key, status, timings).
//!   * `GET  /jobs/<id>`     → one job's status / result.
//!   * `GET  /jobs/by-key/<k>` → look a job up by its idempotency key.
//!
//! **Idempotency**: send an `Idempotency-Key: <k>` header (or a top-level
//! `"idempotency_key"` JSON field). A repeat with the same key never starts a
//! second solve — it returns the existing job, so the key doubles as a handle to
//! read status / progress.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use brooom::io::to_output;
use brooom::matrix::{HaversineMatrix, MatrixSource, OsrmClient};
use brooom::solver::{solve_full, SolverConfig};

/// Server configuration, built from the CLI flags.
#[derive(Clone)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
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

enum State {
    Queued,
    Running,
    Done(String),
    Error(String),
}

struct Job {
    key: Option<String>,
    body: Vec<u8>,
    webhook: Option<String>,
    state: State,
    submitted: Instant,
    started: Option<Instant>,
    finished: Option<Instant>,
}

/// Shared state: the registry (jobs + key index + FIFO queue) behind one Mutex,
/// plus two condvars — `work` (queue gained an item; workers wake) and `done`
/// (a job finished; synchronous callers wake).
struct Registry {
    jobs: HashMap<u64, Job>,
    by_key: HashMap<String, u64>,
    queue: VecDeque<u64>,
}

struct Shared {
    reg: Mutex<Registry>,
    work: Condvar,
    done: Condvar,
    cfg: ServeConfig,
    started_at: Instant,
}

static JOB_SEQ: AtomicU64 = AtomicU64::new(1);

/// Start the HTTP server. Spawns the worker pool, then accepts forever.
pub fn run(cfg: ServeConfig) -> std::io::Result<()> {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr)?;
    let workers = cfg.workers.max(1);
    let shared = Arc::new(Shared {
        reg: Mutex::new(Registry { jobs: HashMap::new(), by_key: HashMap::new(), queue: VecDeque::new() }),
        work: Condvar::new(),
        done: Condvar::new(),
        cfg: cfg.clone(),
        started_at: Instant::now(),
    });
    for w in 0..workers {
        let shared = Arc::clone(&shared);
        thread::Builder::new()
            .name(format!("brooom-worker-{w}"))
            .spawn(move || worker_loop(shared))?;
    }
    eprintln!(
        "brooom: serving on http://{addr}  ({workers} worker{}) — POST /solve · GET /jobs[/<id>] · GET /health",
        if workers == 1 { "" } else { "s" }
    );
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let shared = Arc::clone(&shared);
        thread::spawn(move || {
            if let Err(e) = handle(stream, &shared) {
                if shared.cfg.verbose {
                    eprintln!("brooom: connection error: {e}");
                }
            }
        });
    }
    Ok(())
}

/// One pool worker: pull a queued job, solve it, store the result, fire webhook.
fn worker_loop(shared: Arc<Shared>) {
    loop {
        let (id, body, webhook) = {
            let mut reg = shared.reg.lock().unwrap();
            let id = loop {
                if let Some(id) = reg.queue.pop_front() {
                    break id;
                }
                reg = shared.work.wait(reg).unwrap();
            };
            let now = Instant::now();
            let (body, webhook) = match reg.jobs.get_mut(&id) {
                Some(j) => {
                    j.state = State::Running;
                    j.started = Some(now);
                    (j.body.clone(), j.webhook.clone())
                }
                None => continue, // job vanished (shouldn't happen)
            };
            (id, body, webhook)
        };

        let result = solve_json(&body, &shared.cfg);

        {
            let mut reg = shared.reg.lock().unwrap();
            if let Some(j) = reg.jobs.get_mut(&id) {
                j.finished = Some(Instant::now());
                j.state = match &result {
                    Ok(out) => State::Done(out.clone()),
                    Err(e) => State::Error(e.clone()),
                };
                // The body is no longer needed once solved — free it.
                j.body = Vec::new();
            }
        }
        shared.done.notify_all();

        if let Some(url) = webhook {
            let payload = match &result {
                Ok(out) => out.clone(),
                Err(e) => err_json(e),
            };
            post_webhook(&url, &payload, id, shared.cfg.verbose);
        }
    }
}

fn handle(mut stream: TcpStream, shared: &Arc<Shared>) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    let req = match read_request(&mut stream) {
        Ok(r) => r,
        Err(_) => return respond(&mut stream, 400, "Bad Request", "text/plain", b"bad request"),
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => handle_health(&mut stream, shared),
        ("POST", "/solve") => handle_solve(&mut stream, shared, &req),
        ("GET", "/jobs") => handle_list(&mut stream, shared),
        ("GET", p) if p.starts_with("/jobs/by-key/") => {
            handle_by_key(&mut stream, shared, &p["/jobs/by-key/".len()..])
        }
        ("GET", p) if p.starts_with("/jobs/") => handle_job(&mut stream, shared, &p["/jobs/".len()..]),
        _ => respond(
            &mut stream,
            404,
            "Not Found",
            "application/json",
            br#"{"error":"unknown route; use POST /solve, GET /jobs[/<id>|/by-key/<k>], GET /health"}"#,
        ),
    }
}

fn handle_health(stream: &mut TcpStream, shared: &Arc<Shared>) -> std::io::Result<()> {
    let reg = shared.reg.lock().unwrap();
    let queued = reg.jobs.values().filter(|j| matches!(j.state, State::Queued)).count();
    let running = reg.jobs.values().filter(|j| matches!(j.state, State::Running)).count();
    let total = reg.jobs.len();
    let body = format!(
        "{{\"ok\":true,\"workers\":{},\"queued\":{queued},\"running\":{running},\"jobs\":{total},\"uptime_s\":{}}}",
        shared.cfg.workers.max(1),
        shared.started_at.elapsed().as_secs()
    );
    drop(reg);
    respond(stream, 200, "OK", "application/json", body.as_bytes())
}

fn handle_solve(stream: &mut TcpStream, shared: &Arc<Shared>, req: &Request) -> std::io::Result<()> {
    // Idempotency key: header wins, else a top-level JSON field.
    let body_val: Option<serde_json::Value> = serde_json::from_slice(&req.body).ok();
    let key = req.idempotency_key.clone().or_else(|| {
        body_val
            .as_ref()
            .and_then(|v| v.get("idempotency_key"))
            .and_then(|k| k.as_str())
            .map(str::to_owned)
    });
    let webhook = body_val
        .as_ref()
        .and_then(|v| v.get("webhook"))
        .and_then(|w| w.as_str())
        .map(str::to_owned);
    // Async when a webhook is given or `?async=1` is set.
    let want_async = webhook.is_some() || req.query.contains("async=1") || req.query.contains("async=true");

    // Idempotent hit: a job with this key already exists → return it, never re-run.
    if let Some(k) = key.as_ref() {
        let reg = shared.reg.lock().unwrap();
        if let Some(&id) = reg.by_key.get(k) {
            let (code, reason, body) = status_response(&reg, id);
            drop(reg);
            return respond(stream, code, reason, "application/json", body.as_bytes());
        }
    }

    // Register a new queued job and wake a worker.
    let id = JOB_SEQ.fetch_add(1, Ordering::Relaxed);
    {
        let mut reg = shared.reg.lock().unwrap();
        reg.jobs.insert(
            id,
            Job {
                key: key.clone(),
                body: req.body.clone(),
                webhook,
                state: State::Queued,
                submitted: Instant::now(),
                started: None,
                finished: None,
            },
        );
        if let Some(k) = key.clone() {
            reg.by_key.insert(k, id);
        }
        reg.queue.push_back(id);
    }
    shared.work.notify_one();

    if want_async {
        let key_field = key
            .as_ref()
            .map(|k| format!(",\"idempotency_key\":{}", json_string(k)))
            .unwrap_or_default();
        let body = format!("{{\"job_id\":{id},\"status\":\"queued\"{key_field}}}");
        return respond(stream, 202, "Accepted", "application/json", body.as_bytes());
    }

    // Synchronous: wait for the worker to finish this job (bounded), then reply.
    let cap = Duration::from_secs_f64(shared.cfg.time_limit_s.unwrap_or(60.0) + 120.0);
    let deadline = Instant::now() + cap;
    let mut reg = shared.reg.lock().unwrap();
    loop {
        match reg.jobs.get(&id).map(|j| &j.state) {
            Some(State::Done(out)) => {
                let out = out.clone();
                drop(reg);
                return respond(stream, 200, "OK", "application/json", out.as_bytes());
            }
            Some(State::Error(e)) => {
                let e = err_json(e);
                drop(reg);
                return respond(stream, 422, "Unprocessable Entity", "application/json", e.as_bytes());
            }
            _ => {}
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            // Still running — hand back the id so the client can poll.
            let (code, reason, body) = status_response(&reg, id);
            drop(reg);
            return respond(stream, code, reason, "application/json", body.as_bytes());
        }
        let (g, _) = shared.done.wait_timeout(reg, remaining).unwrap();
        reg = g;
    }
}

fn handle_job(stream: &mut TcpStream, shared: &Arc<Shared>, id_str: &str) -> std::io::Result<()> {
    let Ok(id) = id_str.parse::<u64>() else {
        return respond(stream, 400, "Bad Request", "application/json", br#"{"error":"bad job id"}"#);
    };
    let reg = shared.reg.lock().unwrap();
    if !reg.jobs.contains_key(&id) {
        drop(reg);
        return respond(stream, 404, "Not Found", "application/json", br#"{"error":"no such job"}"#);
    }
    let (code, reason, body) = status_response(&reg, id);
    drop(reg);
    respond(stream, code, reason, "application/json", body.as_bytes())
}

fn handle_by_key(stream: &mut TcpStream, shared: &Arc<Shared>, key: &str) -> std::io::Result<()> {
    let reg = shared.reg.lock().unwrap();
    match reg.by_key.get(key).copied() {
        Some(id) => {
            let (code, reason, body) = status_response(&reg, id);
            drop(reg);
            respond(stream, code, reason, "application/json", body.as_bytes())
        }
        None => {
            drop(reg);
            respond(stream, 404, "Not Found", "application/json", br#"{"error":"no job for that key"}"#)
        }
    }
}

fn handle_list(stream: &mut TcpStream, shared: &Arc<Shared>) -> std::io::Result<()> {
    let reg = shared.reg.lock().unwrap();
    let mut ids: Vec<u64> = reg.jobs.keys().copied().collect();
    ids.sort_unstable();
    let items: Vec<String> = ids.iter().map(|&id| job_summary_json(&reg, id)).collect();
    drop(reg);
    let body = format!("{{\"jobs\":[{}]}}", items.join(","));
    respond(stream, 200, "OK", "application/json", body.as_bytes())
}

/// Build the (code, reason, json) status reply for a job — includes the full
/// result when done. Used by /jobs/<id>, /jobs/by-key, idempotent hits and sync
/// timeouts.
fn status_response(reg: &Registry, id: u64) -> (u16, &'static str, String) {
    let Some(j) = reg.jobs.get(&id) else {
        return (404, "Not Found", r#"{"error":"no such job"}"#.to_string());
    };
    match &j.state {
        State::Done(out) => {
            let solve_ms = match (j.started, j.finished) {
                (Some(s), Some(f)) => f.duration_since(s).as_millis(),
                _ => 0,
            };
            (200, "OK", format!("{{\"job_id\":{id},\"status\":\"done\",\"solve_ms\":{solve_ms},\"result\":{out}}}"))
        }
        State::Error(e) => (200, "OK", format!("{{\"job_id\":{id},\"status\":\"error\",\"error\":{}}}", json_string(e))),
        State::Queued => {
            let pos = reg.queue.iter().position(|&q| q == id).map(|p| p + 1).unwrap_or(0);
            (200, "OK", format!("{{\"job_id\":{id},\"status\":\"queued\",\"queue_position\":{pos}}}"))
        }
        State::Running => {
            let ms = j.started.map(|s| s.elapsed().as_millis()).unwrap_or(0);
            (200, "OK", format!("{{\"job_id\":{id},\"status\":\"running\",\"elapsed_ms\":{ms}}}"))
        }
    }
}

/// Compact one-line summary for the /jobs list (no full result payload).
fn job_summary_json(reg: &Registry, id: u64) -> String {
    let Some(j) = reg.jobs.get(&id) else { return String::new() };
    let status = match &j.state {
        State::Queued => "queued",
        State::Running => "running",
        State::Done(_) => "done",
        State::Error(_) => "error",
    };
    let key = j.key.as_ref().map(|k| json_string(k)).unwrap_or_else(|| "null".into());
    let age = j.submitted.elapsed().as_millis();
    format!("{{\"job_id\":{id},\"status\":\"{status}\",\"idempotency_key\":{key},\"age_ms\":{age}}}")
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

struct Request {
    method: String,
    path: String,
    query: String,
    idempotency_key: Option<String>,
    body: Vec<u8>,
}

/// Read method, path, query, the `Idempotency-Key` header, and the body
/// (honouring `Content-Length`).
fn read_request(stream: &mut TcpStream) -> std::io::Result<Request> {
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut chunk = [0u8; 8192];
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
    let (path, query) = match raw_path.find('?') {
        Some(i) => (raw_path[..i].to_string(), raw_path[i + 1..].to_string()),
        None => (raw_path, String::new()),
    };

    let mut content_length = 0usize;
    let mut idempotency_key = None;
    for line in lines {
        if let Some(v) = ci_strip(line, "content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = ci_strip(line, "idempotency-key:") {
            idempotency_key = Some(v.trim().to_string());
        }
    }

    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok(Request { method, path, query, idempotency_key, body })
}

/// Case-insensitive header-prefix strip: returns the value if `line` starts with
/// `prefix` (compared lowercased).
fn ci_strip<'a>(line: &'a str, prefix_lower: &str) -> Option<&'a str> {
    if line.len() >= prefix_lower.len()
        && line[..prefix_lower.len()].eq_ignore_ascii_case(prefix_lower)
    {
        Some(&line[prefix_lower.len()..])
    } else {
        None
    }
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

/// Minimal JSON string escaping for error messages and keys.
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
