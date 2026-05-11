//! mpe — unified driver for the mpe-engine workspace.
//!
//! This binary is the single user-facing entry point. It is intentionally
//! thin: each subcommand wires together public APIs from `dijkstra`
//! (Contraction-Hierarchies routing — an OSRM-alternative) and `brooom`
//! (VRP solver — a Vroom-alternative). Both engines live as workspace
//! members and are designed to share memory in-process: a CH cache loaded
//! once is passed straight into the solver as `Arc<...>` / `&Vec<...>`,
//! with no IPC, no file round-trip, no serialization on the hot path.
//!
//! Workflow this CLI is being built to drive:
//!
//!   1. `mpe download <region>`  — fetch an OSM PBF from Geofabrik.
//!   2. `mpe build <pbf>`        — preprocess → CH cache (dijkstra side).
//!   3. `mpe solve <problem>`    — load CH, build K-NN on the fly, solve
//!                                  VRP (brooom side). No N×N matrix is
//!                                  ever materialised; brooom asks dijkstra
//!                                  for distances on demand via the K-NN
//!                                  hot path and CH single-pair queries
//!                                  for cold-path route evaluation.
//!   4. `mpe pipeline <region> <problem>` — end-to-end: 1 → 2 → 3.
//!
//! The integration design is documented in:
//!   - crates/brooom/integration.txt   (brooom side of the contract)
//!   - crates/dijkstra/integration.txt  (dijkstra side of the contract)
//!   - INTEGRATION.md at the workspace root (the bird's-eye view).
//!
//! At the time of this commit the subcommands are scaffolds. The dijkstra
//! and brooom crates can already be used directly via their own binaries
//! (`cargo run -p brooom`, `cargo run -p sssp_bench`); this CLI is the
//! seam where the next agent will plug them together.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "mpe",
    version,
    about = "mpe-engine: OSRM-alternative routing + Vroom-alternative VRP solver, in one process.",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Download an OSM PBF extract from Geofabrik into ./data/.
    Download {
        /// Region slug, e.g. "europe/norway", "europe/great-britain/england/greater-london".
        region: String,

        /// Output directory (default: ./data).
        #[arg(long, default_value = "data")]
        out_dir: PathBuf,
    },

    /// Build dijkstra's CSR + PP + CH caches from an OSM PBF.
    ///
    /// On first run this takes ~3-4 minutes for a city-sized graph
    /// (Greater London, n=1.16M). Subsequent loads are mmap and take ~0.02 ms.
    Build {
        /// Path to the .osm.pbf file.
        pbf: PathBuf,

        /// Routing profile: car / motorcycle / bicycle / foot.
        #[arg(long, default_value = "car")]
        profile: String,
    },

    /// Solve a Vroom-compatible VRP problem.
    ///
    /// When `--ch` is supplied, the CH cache is loaded in this process and
    /// distances are computed on the fly via dijkstra's K-NN and single-pair
    /// query — no full N×N matrix is materialised. Without `--ch`, brooom
    /// falls back to Haversine or its OSRM HTTP client (see `brooom --help`).
    Solve {
        /// Vroom-compatible JSON problem.
        problem: PathBuf,

        /// Path to a prebuilt dijkstra .ch cache. When set, dijkstra acts as
        /// the in-process routing engine (zero-copy K-NN feed into brooom).
        #[arg(long)]
        ch: Option<PathBuf>,

        /// Where to write the solution JSON (default: stdout).
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },

    /// End-to-end: download a region, build caches if missing, solve a problem.
    Pipeline {
        region: String,
        problem: PathBuf,

        #[arg(long, default_value = "data")]
        data_dir: PathBuf,

        #[arg(long, default_value = "car")]
        profile: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Download { region, out_dir } => download(&region, &out_dir),
        Cmd::Build { pbf, profile } => build(&pbf, &profile),
        Cmd::Solve { problem, ch, output } => solve(&problem, ch.as_deref(), output.as_deref()),
        Cmd::Pipeline { region, problem, data_dir, profile } => {
            pipeline(&region, &problem, &data_dir, &profile)
        }
    }
}

fn download(region: &str, out_dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating {}", out_dir.display()))?;

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
        .with_context(|| format!("creating {}", out_path.display()))?;
    let bytes = std::io::copy(&mut reader, &mut file).context("download stream copy")?;
    eprintln!("wrote {} ({} MB)", out_path.display(), bytes / 1_048_576);
    Ok(())
}

fn build(_pbf: &std::path::Path, _profile: &str) -> Result<()> {
    // Future: call into sssp_bench::cache, sssp_bench::preprocess,
    // sssp_bench::ch to materialise .csr / .pp / .ch caches.
    //
    // For now, the standalone `bench_pp` and `bench_ch` binaries already
    // do this — invoke them directly:
    //   cargo run --release -p sssp_bench --bin bench_pp -- <pbf> <profile>
    //   cargo run --release -p sssp_bench --bin bench_ch -- <pbf> <profile>
    bail!(
        "not yet wired up. Use the dijkstra binaries directly:\n  \
         cargo run --release -p sssp_bench --bin bench_pp -- <pbf> <profile>\n  \
         cargo run --release -p sssp_bench --bin bench_ch -- <pbf> <profile>"
    );
}

fn solve(_problem: &std::path::Path, _ch: Option<&std::path::Path>, _out: Option<&std::path::Path>) -> Result<()> {
    // Future, when path deps are uncommented in Cargo.toml:
    //
    //   let ch = sssp_bench::cache_ch::load_mmap(ch_path)?;
    //   let pp = sssp_bench::cache_pp::load_mmap(pp_path)?;
    //   let problem = brooom::io::read_problem(problem_path)?;
    //   let customers = problem.customer_node_ids(&pp);
    //   let knn = sssp_bench::knn::knn_matrix_flat(
    //       &pp.graph, &customers, 160, Some(&pp.edge_dist));
    //   // Zero-copy hand-off — same Vec, same address space:
    //   let granular = brooom::granular::Granular::from_knn_flat(
    //       &knn, customers.len(), 160);
    //   let solved = brooom::solver::solve_full(&problem, &granular, ...);
    //
    // For now, run brooom directly:
    //   cargo run --release -p brooom -- -i <problem.json>
    bail!(
        "not yet wired up. Use the brooom binary directly:\n  \
         cargo run --release -p brooom -- -i <problem.json>"
    );
}

fn pipeline(_region: &str, _problem: &std::path::Path, _data: &std::path::Path, _profile: &str) -> Result<()> {
    bail!("pipeline scaffolding only — see `mpe download`, `mpe build`, `mpe solve` individually");
}
