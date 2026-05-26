//! mpee-cli — unified driver for the mpee workspace.
//!
//! Subcommands:
//!   gen       — random Vroom-compatible problem inside a region's bbox
//!   download  — fetch an OSM PBF from Geofabrik
//!   build     — preprocess an OSM PBF → CSR + PP + CH caches (delegates
//!               to the standalone bench_pp / bench_ch binaries)
//!   solve     — load CH cache, snap customer coords, build the N×N
//!               routing matrix via dijeng's bucket-MMM, hand it
//!               directly into brooom — no IPC, no disk
//!   pipeline  — gen + solve in one shot
//!
//! The integration is in-process: the matrix that dijeng produces is
//! a `Vec<f32>` that lives in this process; it is converted once to the
//! `Vec<i32>` brooom wants and handed in by `&Matrix`. No serialisation
//! after that.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "mpee-cli", version, about = "mpee: routing + VRP in one process.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate a random Vroom-compatible VRP problem (inside a named region's
    /// bbox, or a disk around an arbitrary --center).
    Gen {
        /// Named region shortcut (london, oslo, manhattan, paris).
        #[arg(long, default_value = "london")]
        region: String,

        /// Generate around an ARBITRARY point instead of a named region, e.g.
        /// `--center 61.115,10.466` (Lillehammer). Overrides --region.
        #[arg(long, allow_hyphen_values = true)]
        center: Option<String>,

        /// Disk radius in km when --center is given (default 5).
        #[arg(long, default_value_t = 0.0)]
        radius_km: f64,

        /// Number of jobs (customers).
        #[arg(long, default_value_t = 500)]
        n_jobs: usize,

        /// Number of vehicles.
        #[arg(long, default_value_t = 20)]
        n_vehicles: usize,

        /// Capacity per vehicle (single-dimensional).
        #[arg(long, default_value_t = 100)]
        capacity: i64,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Output JSON path.
        #[arg(short = 'o', long)]
        output: PathBuf,
    },

    /// Download an OSM PBF extract from Geofabrik into ./data/.
    Download {
        /// Geofabrik path slug.
        region: String,
        #[arg(long, default_value = "data")]
        out_dir: PathBuf,
    },

    /// Preprocess a PBF into CSR + PP + CH caches (in-process).
    Build {
        pbf: PathBuf,
        /// Routing profile (optional positional, like bench_pp/bench_ch):
        /// `mpee build x.osm.pbf` or `mpee build x.osm.pbf bicycle`.
        #[arg(default_value = "car")]
        profile: String,
        /// Suppress the engine's parse/CH progress output.
        #[arg(long, short = 'q')]
        quiet: bool,
        /// Rebuild even if a cache already exists (default: reuse it).
        #[arg(long)]
        force: bool,
        /// Keep the intermediate .csr file (default: delete it to save disk).
        #[arg(long)]
        keep_csr: bool,
    },

    /// Solve a Vroom-compatible problem in-process against a CH cache.
    /// Problem JSON coords are [lon, lat] arrays (VROOM-style); see examples/problem.json.
    Solve {
        /// Vroom-style problem JSON (jobs + vehicles + [lon,lat] coordinates).
        problem: PathBuf,

        /// Cache prefix — uses <prefix>.ch + <prefix>.pp, e.g.
        /// `--cache data/greater-london-latest.osm.pbf`. Or pass --ch/--pp.
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Explicit CH cache file (overrides --cache).
        #[arg(long)]
        ch: Option<PathBuf>,
        /// Explicit PP cache file (overrides --cache).
        #[arg(long)]
        pp: Option<PathBuf>,

        /// Wall-time budget in seconds for the solver. None = no limit.
        #[arg(long)]
        time_limit_s: Option<f64>,

        /// Parallel multi-start count. Default 8; smaller = faster.
        #[arg(long, default_value_t = 8)]
        multi_start: usize,

        /// Output JSON path (omit for stdout).
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },

    /// Point-to-point driving route between two LAT,LON points — a quick
    /// sanity-check that the cache routes (no JSON needed).
    Route {
        /// Origin as "LAT,LON" (e.g. 61.115,10.466). Negative values OK.
        #[arg(allow_hyphen_values = true)]
        from: String,
        /// Destination as "LAT,LON".
        #[arg(allow_hyphen_values = true)]
        to: String,
        /// Cache prefix — uses <prefix>.ch + <prefix>.pp. Or pass --ch/--pp.
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long)]
        ch: Option<PathBuf>,
        #[arg(long)]
        pp: Option<PathBuf>,
    },

    /// Reverse-geocode: the nearest street name to a LAT,LON point. Offline,
    /// using the `.names` sidecar built alongside the cache (no extra index).
    Reverse {
        /// Point as "LAT,LON" (e.g. 59.913,10.752). Negative values OK.
        #[arg(allow_hyphen_values = true)]
        point: String,
        /// Cache prefix — uses <prefix>.pp + <prefix>.names. Or pass --ch/--pp.
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long)]
        ch: Option<PathBuf>,
        #[arg(long)]
        pp: Option<PathBuf>,
    },

    /// Forward-geocode: look up a street by name → its LAT,LON. Offline;
    /// case-insensitive and matches substrings (e.g. "karl johan").
    Geocode {
        /// Street name to look up.
        query: String,
        /// Disambiguate on a multi-city cache: return the match nearest this
        /// "LAT,LON" reference point (e.g. the city centre).
        #[arg(long, allow_hyphen_values = true)]
        near: Option<String>,
        /// Cache prefix — uses <prefix>.pp + <prefix>.names. Or pass --ch/--pp.
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long)]
        ch: Option<PathBuf>,
        #[arg(long)]
        pp: Option<PathBuf>,
    },

    /// Intersection search: where two named streets cross. Offline; prints one
    /// LAT,LON per shared road node (may be several).
    Crossing {
        /// First street name.
        a: String,
        /// Second street name.
        b: String,
        /// Disambiguate on a multi-city cache: sort crossings nearest-first to
        /// this "LAT,LON" reference point.
        #[arg(long, allow_hyphen_values = true)]
        near: Option<String>,
        /// With --near, keep only crossings within this many km of the point.
        #[arg(long)]
        radius_km: Option<f64>,
        /// Cache prefix — uses <prefix>.pp + <prefix>.names. Or pass --ch/--pp.
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long)]
        ch: Option<PathBuf>,
        #[arg(long)]
        pp: Option<PathBuf>,
    },

    /// End-to-end: gen + solve.
    Pipeline {
        #[arg(long, default_value = "london")]
        region: String,
        #[arg(long, default_value_t = 500)]
        n_jobs: usize,
        #[arg(long, default_value_t = 20)]
        n_vehicles: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 100)]
        capacity: i64,
        /// Cache prefix — uses <prefix>.ch + <prefix>.pp. Or pass --ch/--pp.
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long)]
        ch: Option<PathBuf>,
        #[arg(long)]
        pp: Option<PathBuf>,
        #[arg(long)]
        time_limit_s: Option<f64>,
        #[arg(long, default_value_t = 8)]
        multi_start: usize,
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Gen { region, center, radius_km, n_jobs, n_vehicles, capacity, seed, output } => {
            let center = match center.as_deref() {
                Some(s) => Some(parse_lat_lon(s)?),
                None => None,
            };
            cmd_gen(&region, center, radius_km, n_jobs, n_vehicles, capacity, seed, &output)
        }
        Cmd::Download { region, out_dir } => cmd_download(&region, &out_dir),
        Cmd::Build { pbf, profile, quiet, force, keep_csr } => cmd_build(&pbf, &profile, !quiet, force, keep_csr),
        Cmd::Solve { problem, cache, ch, pp, time_limit_s, multi_start, output } => {
            let (ch, pp) = resolve_cache(cache.as_deref(), ch.as_deref(), pp.as_deref())?;
            cmd_solve(&problem, &ch, &pp, time_limit_s, multi_start, output.as_deref())
        }
        Cmd::Route { from, to, cache, ch, pp } => {
            let (ch, pp) = resolve_cache(cache.as_deref(), ch.as_deref(), pp.as_deref())?;
            cmd_route(parse_lat_lon(&from)?, parse_lat_lon(&to)?, &ch, &pp)
        }
        // Geocoding never routes, so it only needs <prefix>.pp + <prefix>.names
        // (skip the large .ch). `--ch` is accepted but ignored here.
        Cmd::Reverse { point, cache, ch: _, pp } => {
            let pp = resolve_pp(cache.as_deref(), pp.as_deref())?;
            cmd_reverse(parse_lat_lon(&point)?, &pp)
        }
        Cmd::Geocode { query, near, cache, ch: _, pp } => {
            let pp = resolve_pp(cache.as_deref(), pp.as_deref())?;
            let near = match near.as_deref() { Some(s) => Some(parse_lat_lon(s)?), None => None };
            cmd_geocode(&query, near, &pp)
        }
        Cmd::Crossing { a, b, near, radius_km, cache, ch: _, pp } => {
            let pp = resolve_pp(cache.as_deref(), pp.as_deref())?;
            let near = match near.as_deref() { Some(s) => Some(parse_lat_lon(s)?), None => None };
            cmd_crossing(&a, &b, near, radius_km, &pp)
        }
        Cmd::Pipeline {
            region, n_jobs, n_vehicles, seed, capacity, cache, ch, pp,
            time_limit_s, multi_start, output,
        } => {
            let (ch, pp) = resolve_cache(cache.as_deref(), ch.as_deref(), pp.as_deref())?;
            let tmp = std::env::temp_dir().join(format!("mpee-pipeline-{}.json", seed));
            cmd_gen(&region, None, 0.0, n_jobs, n_vehicles, capacity, seed, &tmp)?;
            cmd_solve(&tmp, &ch, &pp, time_limit_s, multi_start, output.as_deref())
        }
    }
}

// -------------------------------------------------------------------------
// Region bounding boxes (lat_min, lat_max, lon_min, lon_max). These are
// rough rectangles; the snap layer will project points to the nearest road
// vertex, so points dropped over rivers or parks just snap to the nearest
// adjacent road.
// -------------------------------------------------------------------------

fn region_bbox(region: &str) -> Result<(f64, f64, f64, f64)> {
    // Tighter, road-dense rectangles. The looser Greater-London / Île-de-France
    // / Oslo-bygrense bboxes include water, parks, and isolated road
    // fragments where snap → car-routable node sometimes fails. For a clean
    // VRP demo we stick to the dense urban core. Callers wanting suburban
    // coverage should pass coords directly via the JSON instead of `gen`.
    Ok(match region {
        // Central London: Westminster + City + Camden + Islington + Lambeth.
        "london"    => (51.46, 51.56, -0.22,  0.02),
        // Inner Oslo.
        "oslo"      => (59.88, 59.95, 10.68, 10.82),
        // Manhattan from 23rd to 110th.
        "manhattan" => (40.74, 40.80, -74.00, -73.94),
        // Paris intra-muros.
        "paris"     => (48.82, 48.89,  2.27,  2.41),
        other => bail!("unknown region '{other}' (try: london, oslo, manhattan, paris)"),
    })
}

fn region_depot(region: &str) -> (f64, f64) {
    // (lat, lon) of a sensible city-centre depot.
    match region {
        "london"    => (51.5074, -0.1278),  // Charing Cross
        "oslo"      => (59.9139, 10.7522),  // Oslo Sentralstasjon
        "manhattan" => (40.7580, -73.9855), // Times Square
        "paris"     => (48.8566,  2.3522),  // Île de la Cité
        _ => (0.0, 0.0),
    }
}

// -------------------------------------------------------------------------
// gen: random problem inside a region's bbox.
// -------------------------------------------------------------------------

/// Parse a "lat,lon" string into (lat, lon), with a sanity check that catches
/// obviously-swapped input.
fn parse_lat_lon(s: &str) -> Result<(f64, f64)> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        bail!("expected LAT,LON, got {s:?}");
    }
    let lat: f64 = parts[0].parse().context("bad lat")?;
    let lon: f64 = parts[1].parse().context("bad lon")?;
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        bail!("{s:?} is not a valid LAT,LON (lat∈[-90,90], lon∈[-180,180]) — did you swap them?");
    }
    Ok((lat, lon))
}

/// Great-circle distance in metres between two (lat, lon) points.
fn haversine_m(a: (f64, f64), b: (f64, f64)) -> f64 {
    const R: f64 = 6_371_000.0;
    let (la1, la2) = (a.0.to_radians(), b.0.to_radians());
    let dlat = (b.0 - a.0).to_radians();
    let dlon = (b.1 - a.1).to_radians();
    let h = (dlat / 2.0).sin().powi(2) + la1.cos() * la2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * R * h.sqrt().clamp(0.0, 1.0).asin()
}

/// Resolve a CH+PP cache pair from `--cache <prefix>` (→ prefix.ch + prefix.pp)
/// or explicit `--ch`/`--pp`. Mirrors the pip CLI's single `--cache`.
fn resolve_cache(cache: Option<&Path>, ch: Option<&Path>, pp: Option<&Path>) -> Result<(PathBuf, PathBuf)> {
    if let (Some(c), Some(p)) = (ch, pp) {
        return Ok((c.to_path_buf(), p.to_path_buf()));
    }
    if let Some(base) = cache {
        let b = base.to_string_lossy();
        return Ok((PathBuf::from(format!("{b}.ch")), PathBuf::from(format!("{b}.pp"))));
    }
    bail!("give --cache <prefix> (uses <prefix>.ch + <prefix>.pp), or both --ch and --pp");
}

/// Resolve only the PP cache (for geocoding, which needs no `.ch`): from
/// `--cache <prefix>` → `<prefix>.pp`, or an explicit `--pp`.
fn resolve_pp(cache: Option<&Path>, pp: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = pp {
        return Ok(p.to_path_buf());
    }
    if let Some(base) = cache {
        return Ok(PathBuf::from(format!("{}.pp", base.to_string_lossy())));
    }
    bail!("give --cache <prefix> (uses <prefix>.pp + <prefix>.names), or --pp");
}

#[allow(clippy::too_many_arguments)]
fn cmd_gen(
    region: &str,
    center: Option<(f64, f64)>,
    radius_km: f64,
    n_jobs: usize,
    n_vehicles: usize,
    capacity: i64,
    seed: u64,
    output: &Path,
) -> Result<()> {
    // Depot + per-job sampler: either a disk around an arbitrary --center, or
    // a named region's bbox.
    let (depot_lat, depot_lon, disk_r_m, bbox) = if let Some((clat, clon)) = center {
        let r = if radius_km > 0.0 { radius_km } else { 5.0 };
        (clat, clon, Some(r * 1000.0), None)
    } else {
        let (depot_lat, depot_lon) = region_depot(region);
        (depot_lat, depot_lon, None, Some(region_bbox(region)?))
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut jobs = Vec::with_capacity(n_jobs);
    for i in 0..n_jobs {
        let (lat, lon) = if let Some(rm) = disk_r_m {
            // Uniform within a disk of radius `rm` metres around the depot:
            // r = sqrt(u)·R gives uniform area density; theta uniform.
            use std::f64::consts::PI;
            let u: f64 = rng.gen_range(0.0..1.0);
            let r = u.sqrt() * rm;
            let theta: f64 = rng.gen_range(0.0..(2.0 * PI));
            let lat = depot_lat + (r * theta.sin()) / 111_000.0;
            let lon = depot_lon + (r * theta.cos()) / (111_000.0 * depot_lat.to_radians().cos().max(1e-6));
            (lat, lon)
        } else {
            let (lat_min, lat_max, lon_min, lon_max) = bbox.unwrap();
            (rng.gen_range(lat_min..lat_max), rng.gen_range(lon_min..lon_max))
        };
        let delivery = rng.gen_range(1..=10_i64);
        jobs.push(serde_json::json!({
            "id": (i + 1) as u64,
            "location": [lon, lat],
            "service": 60,
            "delivery": [delivery],
        }));
    }

    let mut vehicles = Vec::with_capacity(n_vehicles);
    for v in 0..n_vehicles {
        vehicles.push(serde_json::json!({
            "id": (v + 1) as u64,
            "start": [depot_lon, depot_lat],
            "end":   [depot_lon, depot_lat],
            "capacity": [capacity],
            "profile": "car",
        }));
    }

    let problem = serde_json::json!({
        "description": match center {
            Some((clat, clon)) => format!(
                "mpee gen center={clat},{clon} radius_km={} n_jobs={n_jobs} n_vehicles={n_vehicles} seed={seed}",
                if radius_km > 0.0 { radius_km } else { 5.0 }
            ),
            None => format!("mpee gen {region} n_jobs={n_jobs} n_vehicles={n_vehicles} seed={seed}"),
        },
        "jobs": jobs,
        "vehicles": vehicles,
    });

    let bytes = serde_json::to_vec_pretty(&problem).context("serialise problem")?;
    std::fs::write(output, &bytes).with_context(|| format!("write {}", output.display()))?;
    eprintln!(
        "wrote {} ({} jobs, {} vehicles, {} bytes)",
        output.display(),
        n_jobs,
        n_vehicles,
        bytes.len()
    );
    Ok(())
}

// -------------------------------------------------------------------------
// download
// -------------------------------------------------------------------------

fn cmd_download(region: &str, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let url = format!("https://download.geofabrik.de/{}-latest.osm.pbf", region);
    let file_name = region.rsplit('/').next().unwrap_or("region");
    let out_path = out_dir.join(format!("{file_name}-latest.osm.pbf"));
    if out_path.exists() {
        eprintln!("already present: {}", out_path.display());
        return Ok(());
    }
    eprintln!("GET {url}");
    let resp = ureq::get(&url).call().context("Geofabrik request failed")?;
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(&out_path)
        .with_context(|| format!("create {}", out_path.display()))?;
    let bytes = std::io::copy(&mut reader, &mut file).context("download copy")?;
    eprintln!("wrote {} ({} MB)", out_path.display(), bytes / 1_048_576);
    Ok(())
}

// -------------------------------------------------------------------------
// build: delegates to bench_pp / bench_ch — they're already feature-complete
// and parallelised. The unified CLI keeps them visible as one command.
// -------------------------------------------------------------------------

fn cmd_build(pbf: &Path, profile: &str, progress: bool, force: bool, keep_csr: bool) -> Result<()> {
    // Build the cache IN-PROCESS via the shared dijeng helper — no
    // `cargo run` subprocess, no recompilation, and works as a distributed
    // binary (cargo/source not required). Same pipeline as Python's
    // Router.build, so the outputs (and their names) are identical.
    let prof = dijeng::osm_profile::Profile::from_name(profile)
        .ok_or_else(|| anyhow::anyhow!("unknown profile {profile:?} — use car/bicycle/foot"))?;
    eprintln!(
        "building cache from {} (profile={profile}) — seconds for a city, minutes for a country…",
        pbf.display()
    );
    // cmd_build's own status lines go to stderr, so they survive `--quiet`
    // (which only silences the engine's stdout progress chatter).
    let res = dijeng::build::build_cache(pbf, prof, progress, force, keep_csr).map_err(|e| anyhow::anyhow!(e))?;
    if res.cached {
        eprintln!("reused existing cache (pass --force to rebuild)");
    } else {
        eprintln!(
            "done: CH built in {:.1}s ({} nodes, {} edges)",
            res.build_secs, res.nodes, res.edges
        );
    }
    eprintln!("  pp: {}", res.pp_path.display());
    eprintln!("  ch: {}", res.ch_path.display());
    if !res.cached && res.names_path.is_file() {
        eprintln!("  names: {} (street names for `mpee reverse`/`geocode` — delete to save space)", res.names_path.display());
    }
    Ok(())
}

// -------------------------------------------------------------------------
// route: point-to-point sanity check against a cache (no JSON needed).
// Input is LAT,LON (matching the pip `mpee route`).
// -------------------------------------------------------------------------

fn cmd_route(from: (f64, f64), to: (f64, f64), ch_path: &Path, pp_path: &Path) -> Result<()> {
    let pp = dijeng::cache_pp::load_mmap(pp_path)
        .with_context(|| format!("load PP cache {}", pp_path.display()))?;
    let ch = dijeng::cache_ch::load_mmap(ch_path)
        .with_context(|| format!("load CH cache {}", ch_path.display()))?;
    let svc = dijeng::routing::RoutingService::new(ch, pp.coords);

    let resp = svc
        .route(from.0 as f32, from.1 as f32, to.0 as f32, to.1 as f32)
        .ok_or_else(|| anyhow::anyhow!("no route found — are the points reachable in this cache?"))?;

    let snap_from = haversine_m(from, (resp.source_snapped.0 as f64, resp.source_snapped.1 as f64));
    let snap_to = haversine_m(to, (resp.destination_snapped.0 as f64, resp.destination_snapped.1 as f64));

    println!("distance: {:.2} km", resp.distance_m / 1000.0);
    println!("duration: {:.1} min", resp.duration_s / 60.0);
    println!(
        "snap:     from {:.0} m, to {:.0} m  ({} geometry points)",
        snap_from, snap_to, resp.geometry.len()
    );
    if snap_from > 500.0 || snap_to > 500.0 {
        eprintln!(
            "WARN: a point snapped >500 m to the road network — check you passed LAT,LON \
             (not lon,lat) and that the cache covers this area"
        );
    }
    Ok(())
}

// -------------------------------------------------------------------------
// geocode: reverse (point → street) and forward (street → point), offline.
// Both reuse the snap grid the router already builds; the street names come
// from the `.names` sidecar written by `mpee build`. No separate index.
// -------------------------------------------------------------------------

/// Open a **geocoding-only** service from the `.pp` cache (no `.ch` — geocoding
/// never routes, so we skip the largest file), attaching the `.names` sidecar
/// derived from the PP path when present.
fn load_geocoder(pp_path: &Path) -> Result<dijeng::routing::RoutingService> {
    let pp = dijeng::cache_pp::load_mmap(pp_path)
        .with_context(|| format!("load PP cache {}", pp_path.display()))?;
    let n = pp.coords.as_slice().len();
    let mut svc = dijeng::routing::RoutingService::new_geocoding(pp.coords);

    let pp_str = pp_path.to_string_lossy();
    let names_path = pp_str
        .strip_suffix(".pp")
        .map(|b| format!("{b}.names"))
        .unwrap_or_else(|| format!("{pp_str}.names"));
    if Path::new(&names_path).is_file() {
        match dijeng::names::NameTable::load_mmap(&names_path, n) {
            Ok(nt) => svc.set_names(nt),
            Err(e) => eprintln!("WARN: ignoring names sidecar {names_path}: {e}"),
        }
    }
    Ok(svc)
}

fn cmd_reverse(point: (f64, f64), pp_path: &Path) -> Result<()> {
    let svc = load_geocoder(pp_path)?;
    if !svc.has_names() {
        bail!(
            "no .names sidecar next to {} — rebuild the cache (`mpee build`) with this \
             version to enable geocoding",
            pp_path.display()
        );
    }
    let (lat, lon) = (point.0 as f32, point.1 as f32);
    let snapped = svc.coords[svc.nearest_node(lat, lon) as usize];
    let snap_d = haversine_m(point, (snapped.0 as f64, snapped.1 as f64));
    match svc.reverse(lat, lon) {
        Some(name) => println!("{name}"),
        None => println!("(no street name on the nearest road)"),
    }
    println!("nearest road: ({:.5}, {:.5})  [{:.0} m away]", snapped.0, snapped.1, snap_d);
    if snap_d > 500.0 {
        eprintln!("WARN: nearest road is >500 m away — check LAT,LON order and cache coverage");
    }
    Ok(())
}

fn cmd_geocode(query: &str, near: Option<(f64, f64)>, pp_path: &Path) -> Result<()> {
    let svc = load_geocoder(pp_path)?;
    if !svc.has_names() {
        bail!(
            "no .names sidecar next to {} — rebuild the cache (`mpee build`) with this \
             version to enable geocoding",
            pp_path.display()
        );
    }
    let hit = match near {
        Some((la, lo)) => svc.geocode_near(query, la as f32, lo as f32),
        None => svc.geocode(query),
    };
    match hit {
        Some((lat, lon, name)) => {
            println!("{name}");
            println!("{lat:.6},{lon:.6}");
        }
        None => bail!("no street matching {query:?} found in this area"),
    }
    Ok(())
}

fn cmd_crossing(
    a: &str,
    b: &str,
    near: Option<(f64, f64)>,
    radius_km: Option<f64>,
    pp_path: &Path,
) -> Result<()> {
    let svc = load_geocoder(pp_path)?;
    if !svc.has_names() {
        bail!(
            "no .names sidecar next to {} — rebuild the cache (`mpee build`) with this \
             version to enable geocoding",
            pp_path.display()
        );
    }
    let hits = match near {
        Some((la, lo)) => svc.intersection_near(a, b, la as f32, lo as f32, radius_km),
        None => svc.intersection(a, b),
    };
    if hits.is_empty() {
        bail!("no intersection of {a:?} and {b:?} found (unknown street, or they share no node)");
    }
    println!("{a} × {b}: {} match(es)", hits.len());
    for (lat, lon) in hits {
        println!("{lat:.6},{lon:.6}");
    }
    Ok(())
}

// -------------------------------------------------------------------------
// solve: the actual shared-memory integration.
// -------------------------------------------------------------------------

fn cmd_solve(
    problem_path: &Path,
    ch_path: &Path,
    pp_path: &Path,
    time_limit_s: Option<f64>,
    multi_start: usize,
    output: Option<&Path>,
) -> Result<()> {
    let t_total = std::time::Instant::now();

    // 1. Read the Vroom-style problem JSON.
    eprintln!("[1/5] reading problem {}", problem_path.display());
    let json = std::fs::read_to_string(problem_path)
        .with_context(|| format!("read {}", problem_path.display()))?;
    let mut problem: brooom::Problem = brooom::io::parse_input(&json).context("parse problem")?;
    eprintln!(
        "      {} jobs, {} shipments, {} vehicles",
        problem.jobs.len(),
        problem.shipments.len(),
        problem.vehicles.len()
    );

    // 2. mmap-load the dijeng caches (~20 µs each regardless of size).
    eprintln!("[2/5] mmap CH {}", ch_path.display());
    let t = std::time::Instant::now();
    let pp = dijeng::cache_pp::load_mmap(pp_path)
        .with_context(|| format!("load PP cache {}", pp_path.display()))?;
    let ch = dijeng::cache_ch::load_mmap(ch_path)
        .with_context(|| format!("load CH cache {}", ch_path.display()))?;
    eprintln!(
        "      loaded in {:.2} ms (graph_n={}, edges={})",
        t.elapsed().as_secs_f64() * 1000.0,
        ch.graph_fwd.n,
        ch.graph_fwd.head.last().copied().unwrap_or(0),
    );

    // 3. Collect every (lat, lon) referenced by the problem: vehicles first
    //    (so vehicle starts/ends land at indices 0..2V), then jobs (V..V+J).
    //    brooom-style JSON uses [lon, lat]; dijeng wants (lat, lon).
    let (coords_latlon, vehicle_starts, vehicle_ends, job_indices) = collect_coords(&problem)?;
    let n_points = coords_latlon.len();
    eprintln!("[3/5] {} coords to snap", n_points);

    // 4. Build RoutingService and the routing matrix in one shot.
    //    `matrix_with_distance` runs dijeng's bucket-MMM: forward sweep
    //    per src, backward sweep per dst, all in parallel, dual-channel
    //    (dur + dist) at no extra Dijeng cost.
    eprintln!("[4/5] dijeng::matrix_with_distance ({n_points}×{n_points})");
    let svc = dijeng::routing::RoutingService::new(ch, pp.coords);
    let t = std::time::Instant::now();
    let (durs_f32, dists_f32, snap_src, _snap_dst) =
        svc.matrix_with_distance(&coords_latlon, &coords_latlon);
    let mmm_secs = t.elapsed().as_secs_f64();

    // Snap feedback: how far did each input point move to reach the road
    // network? A large max usually means swapped lon/lat or a point off-map.
    let (mut max_snap, mut far) = (0.0_f64, 0usize);
    for (orig, snapped) in coords_latlon.iter().zip(snap_src.iter()) {
        let d = haversine_m((orig.0 as f64, orig.1 as f64), (snapped.0 as f64, snapped.1 as f64));
        max_snap = max_snap.max(d);
        if d > 500.0 { far += 1; }
    }
    eprintln!(
        "      max snap distance {:.0} m{}",
        max_snap,
        if far > 0 {
            format!("  (WARN: {far} point(s) > 500 m — check [lon,lat] order / map coverage)")
        } else {
            String::new()
        }
    );
    let cells = (n_points as u64).pow(2);
    let mcells_s = cells as f64 / mmm_secs / 1e6;
    // Throughput is only meaningful for big matrices; for tiny ones it rounds
    // to 0.0, so just report the cell count there.
    if mcells_s >= 1.0 {
        eprintln!("      matrix built in {:.2} s ({:.0} M cells/s)", mmm_secs, mcells_s);
    } else {
        eprintln!("      matrix built in {:.2} s ({cells} cells)", mmm_secs);
    }

    // 5. Convert f32 → i32 (brooom uses i32 for cache density). One pass,
    //    no extra allocations beyond the destination Vec. f32::INFINITY pairs
    //    (graph-disconnected after snap) get a 7-day sentinel — large enough
    //    that the solver routes around them when a finite alternative exists,
    //    small enough that summing a handful of them doesn't blow up route
    //    cost on the final summary.
    let n_inf = durs_f32.iter().filter(|d| !d.is_finite()).count();
    if n_inf > 0 {
        eprintln!(
            "      WARN: {} / {} matrix cells unreachable after snap (placed at 7-day sentinel)",
            n_inf, durs_f32.len(),
        );
    }
    let durations: Vec<i32> = durs_f32.iter().map(narrow_pos_i32).collect();
    let distances: Vec<i32> = dists_f32.iter().map(narrow_pos_i32).collect();
    drop(durs_f32);
    drop(dists_f32);
    let matrix = brooom::Matrix {
        n: n_points,
        durations,
        distances: Some(distances),
    };

    // 6. Drop jobs whose nearest road snap landed on a fragment that can't
    //    reach the depot (or vice-versa). Without this, the solver still
    //    inserts those jobs because their feasibility check is purely TW +
    //    capacity, and sentinel edges look like "very slow but legal".
    //    A handful of sentinel edges per route would dominate the summary.
    let depot_idx_opt = vehicle_starts.iter().copied().find(|x| x.is_some()).flatten();
    let dropped: Vec<usize> = if let Some(d) = depot_idx_opt {
        let mut drop = Vec::new();
        for (j, &idx) in job_indices.iter().enumerate() {
            let dur_out = matrix.durations[d * n_points + idx];
            let dur_in = matrix.durations[idx * n_points + d];
            if dur_out >= SENTINEL_I32 || dur_in >= SENTINEL_I32 {
                drop.push(j);
            }
        }
        drop
    } else {
        Vec::new()
    };
    if !dropped.is_empty() {
        eprintln!(
            "      WARN: dropping {} jobs unreachable from depot (snapped to road fragments)",
            dropped.len()
        );
        // Remove from back to keep earlier indices valid.
        let drop_set: std::collections::HashSet<usize> = dropped.iter().copied().collect();
        problem.jobs = problem
            .jobs
            .into_iter()
            .enumerate()
            .filter_map(|(j, job)| if drop_set.contains(&j) { None } else { Some(job) })
            .collect();
        // Rebuild job_indices to match the surviving order.
    }
    let kept_job_indices: Vec<usize> = job_indices
        .iter()
        .enumerate()
        .filter_map(|(j, &idx)| {
            if dropped.iter().any(|&d| d == j) { None } else { Some(idx) }
        })
        .collect();

    // 7. Switch every Location reference in the problem from coord-mode to
    //    index-mode. brooom's `solve_with_matrix` keys off matrix indices,
    //    not lat/lon. The indices match the order in which we collected
    //    coords in step 3.
    rebind_problem_to_indices(&mut problem, &vehicle_starts, &vehicle_ends, &kept_job_indices);

    // 8. Solve.
    eprintln!(
        "[5/5] brooom::solve_with_matrix (multi_start={}, time_limit={}s)",
        multi_start,
        time_limit_s.map(|s| format!("{:.0}", s)).unwrap_or_else(|| "∞".into())
    );
    let cfg = brooom::solver::SolverConfig {
        multi_start: multi_start.max(1),
        time_limit_ms: time_limit_s.map(|s| (s * 1000.0) as u64),
        // Quiet: brooom's verbose line prints a raw internal cost
        // (runtime_s + 1e9·unassigned) that reads like an error. The summary
        // below reports the meaningful numbers instead.
        verbose: false,
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let solution = brooom::solver::solve_with_matrix(&problem, &matrix, &cfg);
    let solve_secs = t.elapsed().as_secs_f64();

    // 9. Print a short summary + write Vroom output JSON.
    let total = t_total.elapsed().as_secs_f64();
    print_summary(&problem, &matrix, &solution, mmm_secs, solve_secs, total);

    let vroom_out = brooom::io::to_output(&problem, &solution, Some(&matrix));
    let output_json = serde_json::to_string_pretty(&vroom_out).context("serialise output")?;
    match output {
        Some(path) => {
            std::fs::write(path, &output_json).with_context(|| format!("write {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
        None => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(output_json.as_bytes())?;
            stdout.write_all(b"\n")?;
        }
    }
    Ok(())
}

/// Sentinel for "unreachable in the road graph after snap" — 7 days in
/// seconds (or 7 days × 1 m/s in metres). Large enough that the solver
/// avoids the cell when any finite alternative exists, but small enough
/// not to dominate aggregate cost when one accidentally lands in a route.
const SENTINEL_I32: i32 = 7 * 24 * 60 * 60; // 604_800

fn narrow_pos_i32(v: &f32) -> i32 {
    let v = *v;
    if !v.is_finite() || v < 0.0 {
        SENTINEL_I32
    } else if v > SENTINEL_I32 as f32 {
        SENTINEL_I32
    } else {
        v.round() as i32
    }
}

/// Returns:
///   - coords_latlon: every distinct (lat, lon) referenced, in the order
///     vehicle-start, vehicle-end, job, job, ...
///   - vehicle_starts[v] / vehicle_ends[v] / job_indices[j] = index into
///     coords_latlon for that entity.
fn collect_coords(
    problem: &brooom::Problem,
) -> Result<(Vec<(f32, f32)>, Vec<Option<usize>>, Vec<Option<usize>>, Vec<usize>)> {
    let mut coords: Vec<(f32, f32)> = Vec::new();
    let push = |coords: &mut Vec<(f32, f32)>, lonlat: [f64; 2]| -> usize {
        let idx = coords.len();
        coords.push((lonlat[1] as f32, lonlat[0] as f32)); // (lat, lon)
        idx
    };
    let mut vehicle_starts = Vec::with_capacity(problem.vehicles.len());
    let mut vehicle_ends = Vec::with_capacity(problem.vehicles.len());
    for v in &problem.vehicles {
        let s = v.start.as_ref().and_then(|l| l.coord).map(|c| push(&mut coords, c));
        let e = v.end.as_ref().and_then(|l| l.coord).map(|c| push(&mut coords, c));
        vehicle_starts.push(s);
        vehicle_ends.push(e);
    }
    let mut job_indices = Vec::with_capacity(problem.jobs.len());
    for j in &problem.jobs {
        let c = j
            .location
            .coord
            .ok_or_else(|| anyhow::anyhow!("job {} is missing coord", j.id))?;
        job_indices.push(push(&mut coords, c));
    }
    if !problem.shipments.is_empty() {
        bail!("shipments not yet wired into mpee-cli — use brooom directly for now");
    }
    Ok((coords, vehicle_starts, vehicle_ends, job_indices))
}

fn rebind_problem_to_indices(
    problem: &mut brooom::Problem,
    vehicle_starts: &[Option<usize>],
    vehicle_ends: &[Option<usize>],
    job_indices: &[usize],
) {
    for (v, vh) in problem.vehicles.iter_mut().enumerate() {
        if let (Some(start), Some(idx)) = (vh.start.as_mut(), vehicle_starts[v]) {
            start.coord = None;
            start.index = Some(idx);
        }
        if let (Some(end), Some(idx)) = (vh.end.as_mut(), vehicle_ends[v]) {
            end.coord = None;
            end.index = Some(idx);
        }
    }
    for (j, job) in problem.jobs.iter_mut().enumerate() {
        job.location.coord = None;
        job.location.index = Some(job_indices[j]);
    }
}

fn print_summary(
    problem: &brooom::Problem,
    matrix: &brooom::Matrix,
    solution: &brooom::solution::Solution,
    mmm_secs: f64,
    solve_secs: f64,
    total_secs: f64,
) {
    let n_routes = solution.routes.len();
    let used: usize = solution.routes.iter().filter(|r| !r.steps.is_empty()).count();
    let assigned_jobs: usize = solution
        .routes
        .iter()
        .map(|r| {
            r.steps
                .iter()
                .filter(|t| matches!(t, brooom::solution::TaskRef::Job(_)))
                .count()
        })
        .sum();
    let unassigned = problem.jobs.len() - assigned_jobs;

    // Sum each route's total duration (driving + service) via the matrix.
    let mut total_dur: i64 = 0;
    let mut total_dist: i64 = 0;
    for route in &solution.routes {
        if route.steps.is_empty() {
            continue;
        }
        let v = &problem.vehicles[route.vehicle_idx];
        let start_idx = v.start.as_ref().and_then(|l| l.index);
        let end_idx = v.end.as_ref().and_then(|l| l.index);
        let mut prev = start_idx;
        for task in &route.steps {
            let here = match task {
                brooom::solution::TaskRef::Job(j) => problem.jobs[*j].location.index,
                _ => continue,
            };
            if let (Some(p), Some(h)) = (prev, here) {
                total_dur += matrix.duration(p, h);
                total_dist += matrix.distance(p, h);
            }
            prev = here;
        }
        if let (Some(p), Some(e)) = (prev, end_idx) {
            total_dur += matrix.duration(p, e);
            total_dist += matrix.distance(p, e);
        }
    }

    eprintln!();
    eprintln!("─── mpee solve summary ─────────────────────────────────────");
    eprintln!("jobs              : {} (assigned {} / unassigned {})", problem.jobs.len(), assigned_jobs, unassigned);
    eprintln!("vehicles          : {} (used {})", n_routes, used);
    eprintln!("matrix MMM time   : {:.2} s", mmm_secs);
    eprintln!("solver time       : {:.2} s", solve_secs);
    eprintln!("wall time         : {:.2} s", total_secs);
    eprintln!("total drive time  : {} s  (≈ {:.1} h)", total_dur, total_dur as f64 / 3600.0);
    eprintln!("total drive dist  : {} m  (≈ {:.1} km)", total_dist, total_dist as f64 / 1000.0);

    // Explain unassigned jobs so a correct run doesn't look broken.
    if unassigned > 0 {
        let demand: i64 = problem.jobs.iter().map(|j| j.delivery.iter().copied().sum::<i64>()).sum();
        let cap: i64 = problem.vehicles.iter().map(|v| v.capacity.iter().copied().sum::<i64>()).sum();
        if demand > cap {
            eprintln!(
                "⚠ {unassigned} unassigned: total demand {demand} > fleet capacity {cap} — \
                 raise --capacity or add vehicles."
            );
        } else {
            eprintln!(
                "⚠ {unassigned} unassigned despite enough capacity — those stops are likely \
                 unreachable (snapped to a disconnected road) or outside their time windows."
            );
        }
    }
    eprintln!("───────────────────────────────────────────────────────────");
}
