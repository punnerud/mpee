//! brooom CLI — Vroom-compatible VRP solver.
//!
//! Reads a Vroom-style JSON problem from a file or stdin, solves it, and
//! writes a Vroom-style JSON solution to a file or stdout.
//!
//! Examples:
//!   brooom -i problem.json -o solution.json
//!   cat problem.json | brooom -r osrm --osrm-host http://router.project-osrm.org > out.json
//!   brooom -i p.json -r haversine --speed-kmh 50

use std::io::Write;

use clap::{Parser, ValueEnum};

use brooom::{
    io::to_output,
    matrix::{HaversineMatrix, MatrixSource, OsrmClient},
    solver::{solve_full, SolverConfig},
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RoutingEngine {
    Haversine,
    Osrm,
}

#[derive(Debug, Parser)]
#[command(name = "brooom", about = "Vehicle routing optimizer (Vroom-compatible)")]
struct Cli {
    /// Input JSON file. Use `-` or omit for stdin.
    #[arg(short = 'i', long)]
    input: Option<String>,

    /// Output JSON file. Use `-` or omit for stdout.
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// Routing engine for matrix building (skipped if matrix is in input).
    #[arg(short = 'r', long, value_enum, default_value_t = RoutingEngine::Haversine)]
    routing: RoutingEngine,

    /// OSRM host (required when --routing osrm).
    #[arg(long, default_value = "http://router.project-osrm.org")]
    osrm_host: String,

    /// OSRM profile.
    #[arg(long, default_value = "driving")]
    osrm_profile: String,

    /// Haversine assumed speed in km/h.
    #[arg(long, default_value_t = 50.0)]
    speed_kmh: f64,

    /// Haversine detour multiplier (1.0 = straight-line).
    #[arg(long, default_value_t = 1.3)]
    detour: f64,

    /// Maximum local-search passes.
    #[arg(short = 'p', long, default_value_t = 50)]
    max_passes: usize,

    /// Granular neighborhood K. If unset, auto-tunes by problem size
    /// (N ≥ 500 → 80, N ≥ 100 → 40, else 20). Use 0 to disable. Pass an
    /// explicit value to override.
    #[arg(short = 'k', long)]
    granular_k: Option<usize>,

    /// Multi-start parallel attempts (best-of-K). 1 disables (single solve).
    /// K>1 trades wall time for ~5-10% better cost. Empirical sweet spot
    /// for N>=1000 is m=16 (mean -0.5% vs default; m>=20 has diminishing
    /// returns). For small N (<500), m=8 default is fine.
    #[arg(short = 'm', long, default_value_t = 8)]
    multi_start: usize,

    /// ILS iterations per multi-start variant. 0 disables.
    #[arg(long, default_value_t = 30)]
    ils_iters: usize,

    /// ILS kick size as a fraction of total tasks (0.0..1.0).
    #[arg(long, default_value_t = 0.4)]
    ils_kick_size: f64,

    /// Wall-time budget in seconds. ILS stops when elapsed time exceeds this.
    #[arg(short = 'l', long)]
    time_limit_s: Option<f64>,

    /// Print progress to stderr.
    #[arg(short = 'v', long, default_value_t = false)]
    verbose: bool,

    /// Pretty-print JSON output.
    #[arg(long, default_value_t = false)]
    pretty: bool,

    /// Disk-cache directory for solve results. When set, fingerprint-keyed
    /// hits skip the entire solve (matrix is normalized so proportional
    /// inputs collide). Falls back to env BROOOM_CACHE_DIR.
    #[arg(long)]
    cache_dir: Option<String>,

    /// Skip solve; instead, embed the input problem and print the K nearest
    /// cached entries (by L2 distance over standardized embeddings).
    /// Requires --cache-dir to point to a populated cache.
    #[arg(long)]
    find_similar: Option<usize>,

    /// Override solver hyperparameters with the median config of the K
    /// nearest cached neighbors. Requires --cache-dir. Time-limit is not
    /// transferred (wall-time is task-specific). CLI flags for the
    /// transferred parameters are ignored when this is set.
    #[arg(long)]
    use_similar_config: Option<usize>,

    /// Train a linear regressor (embedding → config, weighted by 1/cost)
    /// over the cache and use its prediction. Falls back to median if the
    /// corpus is too small or degenerate. Requires --cache-dir.
    #[arg(long, default_value_t = false)]
    use_regressed_config: bool,

    /// Run a GPU megakernel polish pass on the multi-start winner. Falls
    /// back silently to CPU-only if GPU init fails or no improvement is
    /// found. Most effective for N≥500 where GPU LS iters are far cheaper
    /// than CPU iters.
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Use Hybrid Genetic Search (HGS) with GPU-accelerated LS-education.
    /// Population-based metaheuristic that combines route-exchange
    /// crossover with batch LS on GPU. Effective on small-to-medium N
    /// (≤500) where pure LS gets stuck in local optima. Implies --gpu.
    #[arg(long, default_value_t = false)]
    hgs: bool,

    /// HGS population size (default 64). Larger = more diversity but
    /// slower per generation. 32-128 is a reasonable range.
    #[arg(long, default_value_t = 64)]
    hgs_pop: u32,

    /// Path to a Vroom-style solution JSON to use as warm-start. The solver
    /// skips its initial-insertion phase for the deterministic seed and
    /// drops straight into local search on this solution. Other multi-start
    /// seeds still run their normal starts (best-of-K guarantees we never
    /// regress vs no-warm-start).
    #[arg(long)]
    warm_start: Option<String>,

    /// Cluster-first decomposition: split jobs into K clusters via K-medoids
    /// on the distance matrix, solve each in parallel via the existing
    /// pipeline, then concatenate. Useful for large N where a flat solve is
    /// O(N²)-bound.
    ///
    /// Default 0 = auto: K=N/100 for N≥500, K=1 (disabled) otherwise.
    /// Cross-seed validering (rest_list 33c, 2026-05-10): N=1000 K=10 ga
    /// −2.43% mot flat med 9.4× speedup, og lukket Vroom-gap fra +2.56%
    /// til +0.08%. N=500 K=5 ga −0.72% med 3.7× speedup. N<500 ikke nyttig.
    /// Pass 1 explicitly to disable even on N>=500.
    #[arg(long, default_value_t = 0)]
    decompose: usize,

    /// Final polish pass: replace every route ≤ MAX_EXACT_LEN stops with
    /// its globally-optimal ordering (brute-force DFS with TW + capacity
    /// pruning). Idempotent and safe — LS-converged routes already match
    /// the exact optimum, so this typically reports zero improvements
    /// but takes ~10 ms per ≤ 14-stop route. Useful as a guarantee on
    /// LS-converged solutions and as recovery on insertion-only / weak
    /// LS solutions.
    #[arg(long, default_value_t = false)]
    exact_polish: bool,

    /// Post-solve population polish: spawn N parallel ILS trajectories
    /// that all start from the LS-converged solution and explore
    /// destroy-and-repair perturbations of it. Each trajectory does
    /// `--population-iters` kick→LS rounds. Best-of-N replaces the
    /// solution. 0 disables. 64 is the sweet spot on M3 (8-12 cores).
    /// Trades wall time for cost; targets the last 0.5–2% gap to BKS.
    #[arg(long, default_value_t = 0)]
    population: usize,

    /// ILS iterations per population trajectory. Higher = more
    /// exploration per trajectory; lower = wider population. 5 is a
    /// reasonable default; bump to 10-20 for small N, drop to 2-3 for
    /// large N where each LS pass is expensive.
    #[arg(long, default_value_t = 5)]
    population_iters: usize,

    /// Override kick fraction inside population polish. Smaller (0.1-0.2)
    /// is appropriate here than the main ILS default (0.4) because
    /// population trajectories explore around an already-good base —
    /// you want fine perturbations, not full destroy-and-rebuild.
    #[arg(long)]
    population_kick: Option<f64>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("brooom: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let t_main_start = std::time::Instant::now();

    // Streaming parse keeps peak memory low — for million-cell matrices the
    // full file as a String would dominate the working set.
    let mut problem = match cli.input.as_deref() {
        None | Some("-") => brooom::io::parse_input_reader(std::io::stdin().lock())?,
        Some(path) => {
            let f = std::fs::File::open(path)?;
            brooom::io::parse_input_reader(std::io::BufReader::new(f))?
        }
    };

    let source: Box<dyn MatrixSource> = match cli.routing {
        RoutingEngine::Haversine => Box::new(HaversineMatrix {
            speed_mps: cli.speed_kmh * 1000.0 / 3600.0,
            detour: cli.detour,
        }),
        RoutingEngine::Osrm => Box::new(OsrmClient::new(cli.osrm_host.clone(), cli.osrm_profile.clone())),
    };

    // Effective hyperparameters — start from CLI values, optionally override
    // with the median of cached neighbors (Level 3: hyperparameter transfer).
    let mut eff_max_passes = cli.max_passes;
    // None on the CLI means "auto-tune by N"; cache paths below set this to
    // false when they override granular_k explicitly.
    let mut auto_k_active = cli.granular_k.is_none();
    let mut eff_granular_k = cli.granular_k.unwrap_or(20);
    let mut eff_multi_start = cli.multi_start;
    let mut eff_ils_iters = cli.ils_iters;
    let mut eff_ils_kick = cli.ils_kick_size;

    if cli.use_regressed_config {
        let cache_dir = brooom::cache::resolve_dir(cli.cache_dir.as_deref())
            .ok_or("--use-regressed-config requires --cache-dir or BROOOM_CACHE_DIR")?;
        let matrix = brooom::solver::build_matrix(&mut problem, Some(source.as_ref()))?;
        let query = brooom::embedding::extract(&problem, &matrix);
        let entries = brooom::cache::list_meta(&cache_dir);
        if let Some(reg) = brooom::regression::ConfigRegressor::train(&entries) {
            let pred = reg.predict(&query);
            if cli.verbose {
                eprintln!(
                    "brooom: regressed config from {} corpus entries — \
                     k={} ms={} ils={} kick={:.2} max_passes={}",
                    entries.len(), pred.granular_k, pred.multi_start,
                    pred.ils_iters, pred.ils_kick_size, pred.max_passes,
                );
            }
            eff_max_passes = pred.max_passes;
            eff_granular_k = pred.granular_k;
            auto_k_active = false;
            eff_multi_start = pred.multi_start;
            eff_ils_iters = pred.ils_iters;
            eff_ils_kick = pred.ils_kick_size;
        } else {
            // Fall through to median-based path below if regression fails.
            if cli.verbose {
                eprintln!("brooom: regressor under-determined ({} entries < {}+5), falling back to median",
                    entries.len(), brooom::embedding::ProblemEmbedding::dim() + 1);
            }
            // Synthesize a use_similar_config request below by setting K = entries.len().
            // (Hack: reuse the median path as fallback by leaving eff_* and going
            // through use_similar_config below.)
        }
    }

    if let Some(k) = cli.use_similar_config {
        let cache_dir = brooom::cache::resolve_dir(cli.cache_dir.as_deref())
            .ok_or("--use-similar-config requires --cache-dir or BROOOM_CACHE_DIR")?;
        let matrix = brooom::solver::build_matrix(&mut problem, Some(source.as_ref()))?;
        let query = brooom::embedding::extract(&problem, &matrix);
        let neighbors = brooom::cache::nearest(&cache_dir, &query, k);
        let entries: Vec<brooom::cache::CacheMeta> =
            neighbors.iter().map(|(_, m)| m.clone()).collect();
        if let Some(med) = brooom::cache::median_config(&entries) {
            if cli.verbose {
                eprintln!(
                    "brooom: transferred config from {} neighbors — \
                     k={} ms={} ils={} kick={:.2} max_passes={}",
                    entries.len(), med.granular_k, med.multi_start,
                    med.ils_iters, med.ils_kick_size, med.max_passes,
                );
            }
            eff_max_passes = med.max_passes;
            eff_granular_k = med.granular_k;
            auto_k_active = false;
            eff_multi_start = med.multi_start;
            eff_ils_iters = med.ils_iters;
            eff_ils_kick = med.ils_kick_size;
        } else if cli.verbose {
            eprintln!("brooom: --use-similar-config: cache empty, falling back to CLI defaults");
        }
    }

    if auto_k_active {
        // Cross-seed validation (rest_list 26f/26h, 2026-05-10):
        //   N>=500 -> K=80 (brings N=500 near Vroom parity, +0.18%)
        //   N>=100 -> K=40 (wins cross-seed vs Vroom on N=100/250)
        //   else   -> K=20 (default, N=50 inconclusive)
        let n = problem.jobs.len() + problem.shipments.len();
        eff_granular_k = if n >= 500 { 80 }
                         else if n >= 100 { 40 }
                         else { 20 };
        if cli.verbose {
            eprintln!("brooom: auto-K → {} for N={}", eff_granular_k, n);
        }
    }

    let mut config = SolverConfig {
        max_local_search_passes: eff_max_passes,
        granular_k: if eff_granular_k == 0 { None } else { Some(eff_granular_k) },
        multi_start: eff_multi_start.max(1),
        ils_iters: eff_ils_iters,
        ils_kick_size: eff_ils_kick,
        time_limit_ms: cli.time_limit_s.map(|s| (s * 1000.0).round() as u64),
        verbose: cli.verbose,
        warm_start: None,
        use_gpu: cli.gpu,
        ..Default::default()
    };

    // --find-similar: short-circuit solve, just print top-K nearest cached
    // entries. Useful for inspecting which past solves the cache would treat
    // as "similar" — and later for transferring their hyperparameters.
    if let Some(k) = cli.find_similar {
        let cache_dir = brooom::cache::resolve_dir(cli.cache_dir.as_deref())
            .ok_or("--find-similar requires --cache-dir or BROOOM_CACHE_DIR")?;
        // Need a matrix to extract distance features — build one (haversine
        // works for embedding even if the user normally uses OSRM).
        let matrix = brooom::solver::build_matrix(&mut problem, Some(source.as_ref()))?;
        let query = brooom::embedding::extract(&problem, &matrix);
        let neighbors = brooom::cache::nearest(&cache_dir, &query, k);
        if neighbors.is_empty() {
            eprintln!("brooom: cache is empty (no .meta.json entries in {})", cache_dir.display());
        } else {
            println!("Top {} nearest cached entries:", neighbors.len());
            println!("{:<5}  {:<18}  {:>10}  {:>8}  config", "rank", "fingerprint", "distance", "cost");
            for (rank, (d, m)) in neighbors.iter().enumerate() {
                let cfg = format!(
                    "k={} ms={} ils={} kick={:.2}",
                    m.config.granular_k, m.config.multi_start,
                    m.config.ils_iters, m.config.ils_kick_size
                );
                println!("{:<5}  {:<18}  {:>10.3}  {:>8.0}  {}", rank + 1, m.fingerprint, d, m.cost, cfg);
            }
        }
        return Ok(());
    }

    // Cache lookup before any solve work. Fingerprint includes both the
    // (normalized) problem and the solve-affecting CLI flags.
    let cache_dir = brooom::cache::resolve_dir(cli.cache_dir.as_deref());
    let fingerprint_flags: Vec<(&str, String)> = vec![
        ("max_passes", eff_max_passes.to_string()),
        ("granular_k", eff_granular_k.to_string()),
        ("multi_start", eff_multi_start.to_string()),
        ("ils_iters", eff_ils_iters.to_string()),
        ("ils_kick_size", format!("{:.6}", eff_ils_kick)),
        ("time_limit_s", cli.time_limit_s.map(|s| format!("{s:.3}")).unwrap_or_default()),
    ];
    let fp = cache_dir.as_ref().map(|_| brooom::cache::fingerprint(&problem, &fingerprint_flags));
    if let (Some(dir), Some(fp)) = (cache_dir.as_ref(), fp.as_ref()) {
        if let Some(cached) = brooom::cache::load(dir, fp) {
            if cli.verbose {
                eprintln!("brooom: cache hit ({fp})");
            }
            match cli.output.as_deref() {
                None | Some("-") => {
                    let stdout = std::io::stdout();
                    let mut h = stdout.lock();
                    h.write_all(cached.as_bytes())?;
                    if !cached.ends_with('\n') { h.write_all(b"\n")?; }
                }
                Some(path) => {
                    let body = if cached.ends_with('\n') { cached } else { format!("{cached}\n") };
                    std::fs::write(path, body)?;
                }
            }
            return Ok(());
        }
    }

    // Resolve auto-decompose: 0 means "auto by N", 1 means "explicit off".
    let n_jobs = problem.jobs.len() + problem.shipments.len();
    let eff_decompose = if cli.decompose == 0 {
        if n_jobs >= 500 {
            let k = n_jobs / 100;
            if cli.verbose {
                eprintln!("brooom: auto-decompose K={} for N={}", k, n_jobs);
            }
            k
        } else {
            1
        }
    } else {
        cli.decompose
    };

    // If a warm-start file or decomposition is in play we need the matrix
    // in hand to operate on; build it now and use solve_with_matrix
    // directly. Otherwise let solve_full do its usual matrix-build.
    let mut solved = if cli.warm_start.is_some() || eff_decompose > 1 {
        problem.validate()?;
        let matrix = brooom::solver::build_matrix(&mut problem, Some(source.as_ref()))?;
        problem.matrices.clear();
        if let Some(ws_path) = cli.warm_start.as_ref() {
            let ws = brooom::warm_start::load_warm_start(&problem, &matrix, ws_path)?;
            if cli.verbose {
                eprintln!(
                    "brooom: warm-start loaded — {} routes, cost={:.2}, unassigned={}",
                    ws.routes.len(),
                    ws.summary.cost,
                    ws.unassigned.len()
                );
            }
            config.warm_start = Some(ws);
        }
        let solution = if eff_decompose > 1 {
            if cli.verbose {
                eprintln!("brooom: cluster-decompose into K={}", eff_decompose);
            }
            brooom::cluster_decompose::solve_decomposed(
                &problem, &matrix, &config, eff_decompose,
            )
        } else {
            brooom::solver::solve_with_matrix(&problem, &matrix, &config)
        };
        brooom::solver::Solved { matrix, solution }
    } else {
        solve_full(&mut problem, Some(source.as_ref()), config.clone())?
    };

    if cli.population > 0 {
        let pop_cfg = brooom::population::PopulationConfig {
            n_trajectories: cli.population,
            ils_iters_per_trajectory: cli.population_iters,
            kick_frac: cli.population_kick.unwrap_or(0.15),
            max_local_search_passes: eff_max_passes,
            granular_k: if eff_granular_k == 0 { None } else { Some(eff_granular_k) },
            deadline: cli.time_limit_s.map(|s|
                std::time::Instant::now() + std::time::Duration::from_secs_f64(s)),
            verbose: cli.verbose,
        };
        let (polished, _stats) = brooom::population::polish_with_population(
            &problem, &solved.matrix, &solved.solution, &pop_cfg,
        );
        solved.solution = polished;
    }

    // HGS: hybrid genetic search with GPU LS-education. Replaces the
    // top-level GPU+CPU polish with population-based search when --hgs
    // is requested. Uses the current solve's result as seed for the
    // initial population.
    if cli.hgs {
        let granular = if eff_granular_k > 0 {
            brooom::granular::Granular::build(&solved.matrix, eff_granular_k)
        } else {
            brooom::granular::Granular::build(&solved.matrix, 20)
        };
        let elapsed_ms = t_main_start.elapsed().as_millis() as u64;
        let hgs_budget = cli.time_limit_s
            .map(|s| ((s * 1000.0) as u64).saturating_sub(elapsed_ms + 2000))
            .unwrap_or(30_000);
        let hgs_cfg = brooom::hgs::HgsConfig {
            pop_size: cli.hgs_pop,
            max_generations: 1000,
            time_limit_ms: Some(hgs_budget),
            crossover_route_frac: 0.4,
            verbose: cli.verbose,
        };
        match brooom::hgs::solve_hgs(&problem, &solved.matrix, &granular, &solved.solution, &hgs_cfg) {
            Ok(best) => {
                if best.summary.cost + 1e-9 < solved.solution.summary.cost {
                    if cli.verbose {
                        eprintln!(
                            "brooom: HGS-GPU: {:.2} → {:.2} (Δ={:.2})",
                            solved.solution.summary.cost, best.summary.cost,
                            solved.solution.summary.cost - best.summary.cost
                        );
                    }
                    solved.solution = best;
                }
            }
            Err(e) => {
                if cli.verbose { eprintln!("brooom: HGS-GPU failed: {e}"); }
            }
        }
    } else
    // Top-level alternating GPU+CPU polish. After the main solve completes,
    // we loop:
    //   GPU batch polish (64 trajectories with diversification kicks)
    //     ↓
    //   CPU local_search_full polish (catches cross-route moves GPU missed)
    //     ↓
    //   repeat while time budget remains
    //
    // The alternation is the key insight: GPU diversifies (kick + LS), CPU
    // does deep deterministic LS. They find complementary improvements.
    if cli.gpu && solved.matrix.n >= 50 {
        let granular = eff_granular_k.checked_sub(0)
            .and_then(|k| if k == 0 { None } else { Some(brooom::granular::Granular::build(&solved.matrix, k)) });
        let max_iter = if solved.matrix.n >= 5000 { 2000 } else { 1500 };
        let deadline = cli.time_limit_s.map(|s| {
            t_main_start + std::time::Duration::from_secs_f64(s)
        });
        let margin = std::time::Duration::from_secs(2);
        let mut rounds = 0u32;
        let mut consecutive_no_improvement = 0u32;
        loop {
            if rounds > 0 {
                match deadline {
                    Some(d) if std::time::Instant::now() + margin < d => {}
                    Some(_) => break,
                    None => break,
                }
            }
            // GPU batch polish.
            let t_gpu = std::time::Instant::now();
            let pre_cost = solved.solution.summary.cost;
            if let Some(gpu_sol) = brooom::gpu_polish::gpu_polish(
                &problem, &solved.matrix, &solved.solution,
                granular.as_ref(), max_iter, cli.verbose,
            ) {
                if gpu_sol.summary.cost + 1e-9 < solved.solution.summary.cost {
                    if cli.verbose {
                        eprintln!(
                            "brooom: round {}: GPU polish {:.2} → {:.2} (Δ={:.2}, t={:.2}s)",
                            rounds, solved.solution.summary.cost, gpu_sol.summary.cost,
                            solved.solution.summary.cost - gpu_sol.summary.cost,
                            t_gpu.elapsed().as_secs_f64()
                        );
                    }
                    solved.solution = gpu_sol;
                }
            }
            // CPU LS polish.
            let t_cpu = std::time::Instant::now();
            let pre_cpu = solved.solution.summary.cost;
            brooom::cluster_decompose::polish_cpu_full(
                &problem, &solved.matrix, &mut solved.solution, &config,
            );
            if cli.verbose && solved.solution.summary.cost + 1e-9 < pre_cpu {
                eprintln!(
                    "brooom: round {}: CPU polish {:.2} → {:.2} (Δ={:.2}, t={:.2}s)",
                    rounds, pre_cpu, solved.solution.summary.cost,
                    pre_cpu - solved.solution.summary.cost,
                    t_cpu.elapsed().as_secs_f64()
                );
            }
            rounds += 1;
            // Stop early if neither GPU nor CPU found improvement.
            if solved.solution.summary.cost + 1e-9 >= pre_cost {
                consecutive_no_improvement += 1;
                if consecutive_no_improvement >= 3 { break; }  // 3 unproductive rounds
            } else {
                consecutive_no_improvement = 0;
            }
        }
        if cli.verbose && rounds > 0 {
            eprintln!("brooom: top-level alternating polish ran {rounds} round(s)");
        }
    }

    if cli.exact_polish {
        let stats = brooom::route_exact::polish_solution_with_exact(
            &problem, &solved.matrix, &mut solved.solution,
        );
        if cli.verbose {
            eprintln!(
                "exact-polish: {}/{} routes inspected, {} improved, savings={:.2} ({:.1} ms)",
                stats.tried,
                solved.solution.routes.len(),
                stats.improved,
                stats.total_cost_savings,
                stats.solver_us / 1000.0,
            );
        }
    }

    let out = to_output(&problem, &solved.solution, Some(&solved.matrix));
    let serialized = if cli.pretty {
        serde_json::to_string_pretty(&out)?
    } else {
        serde_json::to_string(&out)?
    };

    // Persist to cache before writing output (so a parallel reader sees it).
    if let (Some(dir), Some(fp)) = (cache_dir.as_ref(), fp.as_ref()) {
        brooom::cache::store(dir, fp, &serialized);
        // Sidecar meta for similarity search. Embedding is computed from
        // the matrix the solver actually used, so haversine fallback during
        // --find-similar gives consistent features for embedding-only runs.
        let embedding = brooom::embedding::extract(&problem, &solved.matrix);
        let meta = brooom::cache::CacheMeta {
            fingerprint: fp.clone(),
            embedding,
            cost: solved.solution.summary.cost,
            config: brooom::cache::SerializedConfig {
                max_passes: eff_max_passes,
                granular_k: eff_granular_k,
                multi_start: eff_multi_start,
                ils_iters: eff_ils_iters,
                ils_kick_size: eff_ils_kick,
                time_limit_s: cli.time_limit_s,
            },
        };
        brooom::cache::store_meta(dir, fp, &meta);
    }

    match cli.output.as_deref() {
        None | Some("-") => {
            let stdout = std::io::stdout();
            let mut h = stdout.lock();
            h.write_all(serialized.as_bytes())?;
            h.write_all(b"\n")?;
        }
        Some(path) => {
            std::fs::write(path, serialized + "\n")?;
        }
    };

    Ok(())
}
