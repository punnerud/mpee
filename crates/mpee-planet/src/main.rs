use mpee_planet::bitmap::BitMap;
use mpee_planet::build::{self, Paths};
use mpee_planet::addresses;
use mpee_planet::catalog::{self, Catalog};
use mpee_planet::cachecap;
use mpee_planet::codecstat;
use mpee_planet::pipeline::{self, BuildOpts};
use mpee_planet::overlay::{self, Overlay};
use mpee_planet::overrides::{self, OverrideDb, Overrides};
use mpee_planet::contract::{self, ContractCfg};
use mpee_planet::dataset::Dataset;
use mpee_planet::geocode::Geocoder;
use mpee_planet::graph;
use mpee_planet::serve;
use mpee_planet::toll::{self, TollIndex};
use mpee_planet::tolldb::{self, TollDb};
use mpee_planet::router::{self, Router};
use std::path::{Path, PathBuf};

fn print_rows(r: mpedb::ExecResult) {
    if let mpedb::ExecResult::Rows { columns, rows } = r {
        let short: Vec<String> =
            columns.iter().map(|c| c.rsplit('.').next().unwrap_or(c).to_string()).collect();
        println!("{}", short.join("\t"));
        for row in &rows {
            let cells: Vec<String> = row.iter().map(fmt_val).collect();
            println!("{}", cells.join("\t"));
        }
    }
}

/// Re-hash the live artifacts and compare against what the catalogue recorded.
fn verify_digests(cat: &Catalog, id: &str, live: &Path) -> std::io::Result<usize> {
    let mut bad = 0usize;
    if let mpedb::ExecResult::Rows { rows, .. } = cat.query(&format!(
        "SELECT name, digest FROM build_artifact WHERE build_id = '{id}' AND digest IS NOT NULL"
    ))? {
        for r in rows {
            let (mpedb::Value::Text(name), mpedb::Value::Text(want)) = (&r[0], &r[1]) else {
                continue;
            };
            match mpee_planet::catalog::digest_of(&live.join(name)) {
                Ok(got) if &got == want => {}
                Ok(_) => {
                    println!("  {name}: content differs from the recorded digest");
                    bad += 1;
                }
                Err(e) => {
                    println!("  {name}: {e}");
                    bad += 1;
                }
            }
        }
    }
    Ok(bad)
}

fn fmt_val(v: &mpedb::Value) -> String {
    match v {
        mpedb::Value::Null => "-".into(),
        mpedb::Value::Int(i) => i.to_string(),
        mpedb::Value::Float(f) => format!("{f}"),
        mpedb::Value::Text(s) => s.clone(),
        mpedb::Value::Numeric(s) => s.clone(),
        mpedb::Value::Bool(b) => b.to_string(),
        mpedb::Value::Time(us) => {
            let m = us / 60_000_000;
            format!("{:02}:{:02}", m / 60, m % 60)
        }
        mpedb::Value::Timestamp(us) => {
            // Seconds since the epoch: unambiguous, sortable, and no timezone
            // dependency to get wrong.
            format!("t{}", us / 1_000_000)
        }
        other => format!("{other:?}"),
    }
}

fn parse_ll(s: &str) -> (i32, i32) {
    let mut it = s.split(',');
    let la: f64 = it.next().unwrap().trim().parse().expect("lat");
    let lo: f64 = it.next().unwrap().trim().parse().expect("lon");
    ((la * 1e7).round() as i32, (lo * 1e7).round() as i32)
}

fn usage() -> ! {
    eprintln!(
        "mpee-planet — build a planet-scale offline routing dataset

  mpee-planet build    <planet.osm.pbf> <root> [simplify_m] [nvdb.json] [--no-overlay]
                                                    the whole pipeline, staged and published atomically
  mpee-planet builds   <root>                       what has been built, and what is live
  mpee-planet restat   <root>                       recompute the live build's derived facts
  mpee-planet prune    <root> [keep]                delete old build directories (never the live one)

  Local edits, applied without a rebuild:
  mpee-planet override add  <root> closed|speed|penalty <lat,lon>
                            [--radius M] [--street S] [--speed KMH] [--factor F]
                            [--note T] [--hours H]
  mpee-planet override list <root>
  mpee-planet override rm   <root> <id>
  mpee-planet verify   <root> [--deep]              check the live dataset is internally consistent
  mpee-planet adopt    <root> <source.pbf> [simplify_m]
                                                    bring a pre-catalogue dataset under management

  Individual passes (they write in place; `build` is the supported route):
  mpee-planet scan     <planet.osm.pbf> <workdir>    pass 1 + 2 (bitmaps, ways, coords)
  mpee-planet contract <workdir> [simplify_m]       pass 3 (segments, lengths, geometry)
  mpee-planet graph    <workdir>                    pass 4 (Hilbert order, CSR, snap index)
  mpee-planet route    <dir> <lat,lon> <lat,lon> [car|hgv] [HH:MM]
                                                    one route, with timings and toll cost
  mpee-planet addr     <workdir>                    pass 5 (addresses, geocoding index)
  mpee-planet toll     <workdir>                    pass 6 (toll gates)
  mpee-planet overlay  <dir> [target]               build the region overlay (CRP)
  mpee-planet xyz      <dir>                        write vxyz.bin (Cartesian vertex positions)
  mpee-planet traffic  <dir>                        fold use counters into traffic.bin
  mpee-planet osc      <dir> <f.osc.gz>...           refresh from OSM replication diffs
                       [nodes|nodes=N]               also place modified nodes (one snap each)
                       [apply]                       drop the affected rows so they recompute
                       [values]                      recompute and climb only where numbers moved
  mpee-planet ovcheck  <dir> [pairs]                overlay answers vs the plain router
  mpee-planet tolldb   <dir> [nvdb.json]            build the tariff database (mpedb)
  mpee-planet tollq    <dir> '<SQL>'                ask the tariff database anything
  mpee-planet find     <dir> <street> <no> [city]   address -> coordinate
  mpee-planet rev      <dir> <lat,lon>              coordinate -> address
  mpee-planet serve    <dir> [addr]                 offline map + routing UI
  mpee-planet stats    <dir>                        dataset size breakdown
  mpee-planet codecstat <dir> [samples]             what compression would buy, per array
"
    );
    std::process::exit(2)
}

fn main() -> std::io::Result<()> {
    // A worker inherits the QoS of whoever spawned it, and on Apple Silicon
    // that is what decides performance or efficiency cores — there is no
    // affinity to set. `MPEE_QOS=utility` is the one worth knowing: it keeps a
    // long batch job off the interactive path while still clocking the
    // efficiency cluster up when it spills there, instead of pinning it to the
    // low background frequency the way `taskpolicy -b` does.
    if let Ok(q) = std::env::var("MPEE_QOS") {
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let n = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(n);
        let q2 = q.clone();
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .start_handler(move |_| {
                mpee_planet::cachecap::set_thread_qos(&q2);
            })
            .build_global();
        mpee_planet::cachecap::set_thread_qos(&q);
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }
    match args[1].as_str() {
        "override" => {
            if args.len() < 4 {
                usage();
            }
            let root = PathBuf::from(&args[3]);
            let db = OverrideDb::open(&root)?;
            let flag = |name: &str| -> Option<String> {
                args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
            };
            match args[2].as_str() {
                "add" => {
                    if args.len() < 6 {
                        usage();
                    }
                    let kind = args[4].clone();
                    let ll = parse_ll(&args[5]);
                    let id = db.add(
                        &kind,
                        ll.0 as f64 * 1e-7,
                        ll.1 as f64 * 1e-7,
                        flag("--radius").and_then(|v| v.parse().ok()).unwrap_or(30),
                        flag("--street").as_deref(),
                        flag("--speed").and_then(|v| v.parse().ok()),
                        flag("--factor").and_then(|v| v.parse().ok()),
                        flag("--note").as_deref().unwrap_or(""),
                        flag("--author").as_deref().unwrap_or("local"),
                        flag("--hours")
                            .and_then(|v| v.parse::<f64>().ok())
                            .map(|h| overrides::now_us() + (h * 3.6e9) as i64),
                    )?;
                    // Report what it actually bound to, so a rule that matches
                    // nothing is visible now rather than at the next reroute.
                    let live = catalog::resolve(&root);
                    match Dataset::open(&live) {
                        Ok(ds) => {
                            let rules = db.list()?;
                            let r = Overrides::resolve(&ds, &rules, db.generation());
                            let n = r.matched.iter().find(|(i, _)| *i == id).map(|(_, n)| *n).unwrap_or(0);
                            println!("override {id} added — matches {n} segment(s)");
                            if n == 0 {
                                let radius = flag("--radius")
                                    .and_then(|v| v.parse::<f64>().ok())
                                    .unwrap_or(30.0);
                                let near = overrides::names_near(&ds, ll.0, ll.1, radius);
                                if near.is_empty() {
                                    println!("  nothing matched: no road within {radius:.0} m — widen --radius");
                                } else {
                                    println!("  nothing matched. Roads within {radius:.0} m:");
                                    for (name, k) in near.iter().take(6) {
                                        println!("    {name}  ({k} segment(s))");
                                    }
                                    println!("  motorways are identified by ref (E6), but the dataset stores name — try dropping --street");
                                }
                            }
                        }
                        Err(_) => println!("override {id} added (no dataset here to resolve it against)"),
                    }
                }
                "list" => {
                    let rules = db.list()?;
                    let live = catalog::resolve(&root);
                    let resolved = Dataset::open(&live)
                        .ok()
                        .map(|ds| Overrides::resolve(&ds, &rules, db.generation()));
                    println!("id\tkind\tsegments\tat\t\t\tradius\tstreet\tnote");
                    for r in &rules {
                        let n = resolved
                            .as_ref()
                            .and_then(|o| o.matched.iter().find(|(i, _)| *i == r.id))
                            .map(|(_, n)| n.to_string())
                            .unwrap_or_else(|| "?".into());
                        let detail = match r.kind.as_str() {
                            "speed" => format!("{} km/h", r.speed_kmh.unwrap_or(0)),
                            "penalty" => format!("x{}", r.factor.unwrap_or(1.0)),
                            _ => String::new(),
                        };
                        println!(
                            "{}\t{} {}\t{}\t{:.5},{:.5}\t{} m\t{}\t{}",
                            r.id, r.kind, detail, n,
                            r.lat_e7 as f64 * 1e-7, r.lon_e7 as f64 * 1e-7,
                            r.radius_m,
                            r.street.as_deref().unwrap_or("-"),
                            r.note
                        );
                    }
                    println!("generation {}", db.generation());
                }
                "rm" => {
                    if args.len() < 5 {
                        usage();
                    }
                    let id: i64 = args[4].parse().unwrap_or(-1);
                    println!(
                        "{}",
                        if db.remove(id)? { format!("override {id} removed") } else { "no such override".into() }
                    );
                }
                other => {
                    eprintln!("unknown override subcommand {other:?}");
                    usage();
                }
            }
        }
        "build" => {
            if args.len() < 4 {
                usage();
            }
            let mut o = BuildOpts::default();
            if let Some(v) = args.get(4).and_then(|s| s.parse::<f64>().ok()) {
                o.simplify_m = v;
            }
            o.nvdb = args.get(5).map(PathBuf::from).filter(|p| p.exists());
            o.keep_scratch = args.iter().any(|a| a == "--keep-scratch");
            o.overlay = !args.iter().any(|a| a == "--no-overlay");
            o.deep_digest = args.iter().any(|a| a == "--deep");
            pipeline::build_all(Path::new(&args[2]), Path::new(&args[3]), &o)?;
        }
        "prune" => {
            if args.len() < 3 {
                usage();
            }
            let keep = args.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
            let (n, freed) = pipeline::prune(Path::new(&args[2]), keep)?;
            println!("removed {n} build(s), freed {:.2} GB", freed as f64 / 1e9);
        }
        "restat" => {
            if args.len() < 3 {
                usage();
            }
            let n = pipeline::restat(Path::new(&args[2]))?;
            println!("recorded {n} artifacts and the counts the dataset implies");
        }
        "adopt" => {
            if args.len() < 4 {
                usage();
            }
            let simplify = args.get(4).and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
            pipeline::adopt(Path::new(&args[2]), Path::new(&args[3]), simplify)?;
        }
        "verify" => {
            if args.len() < 3 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let live = catalog::resolve(&root);
            println!("verifying {}", live.display());
            let problems = mpee_planet::dataset::verify(&live);
            if problems.is_empty() {
                println!("  structurally consistent — every array length agrees with the rest");
            } else {
                for p in &problems {
                    println!("  {}: {}", p.file, p.detail);
                }
            }
            if args.iter().any(|a| a == "--deep") {
                match Catalog::open(&root) {
                    Ok(cat) => match cat.head() {
                        Some(id) => {
                            let bad = verify_digests(&cat, &id, &live)?;
                            if bad == 0 {
                                println!("  digests match the catalogue");
                            } else {
                                println!("  {bad} artifact(s) differ from the recorded digest");
                            }
                        }
                        None => println!("  no HEAD — nothing to compare digests against"),
                    },
                    Err(e) => println!("  no catalogue: {e}"),
                }
            }
            if !problems.is_empty() {
                std::process::exit(1);
            }
        }
        "builds" => {
            if args.len() < 3 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let cat = Catalog::open(&root)?;
            let head = cat.head().unwrap_or_default();
            println!("live: {}", if head.is_empty() { "(none)".into() } else { head.clone() });
            print_rows(cat.query(
                "SELECT build_id, status, started, finished, simplify_m, source_bytes, source_path \
                 FROM build ORDER BY started DESC",
            )?);
            println!();
            print_rows(cat.query(
                "SELECT b.build_id, p.pass, p.status, p.wall_s FROM build_pass p \
                 JOIN build b ON b.build_id = p.build_id ORDER BY b.started DESC, p.started",
            )?);
            println!();
            print_rows(cat.query(
                "SELECT build_id, key, value FROM build_stat ORDER BY build_id DESC, key",
            )?);
        }
        "scan" => {
            if args.len() < 4 {
                usage();
            }
            let pbf = PathBuf::from(&args[2]);
            let paths = Paths::new(&PathBuf::from(&args[3]));
            let t0 = std::time::Instant::now();
            let dir = build::open_dir(&pbf, &paths)?;
            let mut st = build::pass1(&pbf, &paths, &dir)?;
            build::finish_pass1(&mut st);
            let p2 = build::pass2(&pbf, &paths, &dir, &mut st)?;
            st.junc.build_rank();
            let n_vertices = st.junc.total();
            eprintln!(
                "[scan] complete in {:.1} s — {} coords, {} graph vertices",
                t0.elapsed().as_secs_f64(),
                p2.n_coords,
                n_vertices
            );
            st.junc.save(&paths.f("junc.bm"))?;
            st.need.save(&paths.f("need.bm"))?;
            eprintln!("[scan] bitmaps saved");
        }
        "contract" => {
            if args.len() < 3 {
                usage();
            }
            let paths = Paths::new(&PathBuf::from(&args[2]));
            let simplify_m: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            let t = std::time::Instant::now();
            let mut need = BitMap::load(&paths.f("need.bm"))?;
            let mut junc = BitMap::load(&paths.f("junc.bm"))?;
            need.build_rank();
            junc.build_rank();
            eprintln!(
                "[contract] bitmaps loaded in {:.1} s — {} coords, {} vertices",
                t.elapsed().as_secs_f64(),
                need.total(),
                junc.total()
            );
            contract::pass3(&paths, &need, &junc, &ContractCfg { simplify_m })?;
        }
        "graph" => {
            if args.len() < 3 {
                usage();
            }
            let paths = Paths::new(&PathBuf::from(&args[2]));
            let mut need = BitMap::load(&paths.f("need.bm"))?;
            let mut junc = BitMap::load(&paths.f("junc.bm"))?;
            need.build_rank();
            junc.build_rank();
            graph::pass4(&paths, &need, &junc)?;
        }
        "route" => {
            if args.len() < 5 {
                usage();
            }
            let live = catalog::resolve(Path::new(&args[2]));
            let ds = Dataset::open(&live)?;
            eprintln!(
                "[route] {} vertices, {} edges, {} segments",
                ds.n_vertices(),
                ds.n_edges(),
                ds.n_segments()
            );
            let a = parse_ll(&args[3]);
            let b = parse_ll(&args[4]);
            let t = std::time::Instant::now();
            let sa = ds.snap(a.0, a.1, 2000.0).expect("no road near origin");
            let sb = ds.snap(b.0, b.1, 2000.0).expect("no road near destination");
            eprintln!(
                "[route] snapped in {:.1} ms — origin {:.1} m off road, destination {:.1} m off",
                t.elapsed().as_secs_f64() * 1000.0,
                sa.off_m,
                sb.off_m
            );
            // Overrides live at the root, not in the build directory: they
            // are meant to outlive the dataset they were written against.
            let root = PathBuf::from(&args[2]);
            let ov = OverrideDb::open(&root).ok().map(|db| {
                let rules = db.list().unwrap_or_default();
                Overrides::resolve(&ds, &rules, db.generation())
            });
            if let Some(o) = ov.as_ref().filter(|o| !o.is_empty()) {
                eprintln!("[route] {} segment(s) overridden", o.len());
            }
            if std::env::var("MPEE_ROUTER").as_deref() == Ok("overlay") {
                let ovl = Overlay::open(&live)?;
                let mut o = overlay::OverlayRouter::new();
                // MPEE_CACHE_MB caps the resident page cache, so a query can be
                // run as it would be on a machine that cannot hold the tables.
                if let Ok(mb) = std::env::var("MPEE_CACHE_MB") {
                    let mb: usize = mb.parse().unwrap_or(512);
                    let mut cap = cachecap::CacheCap::new(mb * 1_000_000);
                    cap.govern(ds.maps());
                    cap.govern(ovl.maps());
                    o.cache = Some(cap);
                }
                // MPEE_NO_L2 forces the single-level search, which is how the
                // second level's contribution gets separated from the first's.
                o.use_l2 = std::env::var("MPEE_NO_L2").is_err();
                let t = std::time::Instant::now();
                let got = o.search(&ds, &ovl, &sa, &sb);
                // Anonymous vs file-backed matters here: the search state is
                // memory the process must have, while mapped table pages are a
                // cache the OS evicts under pressure.
                println!(
                    "overlay-only: {:?} ds, {} settled, {} pruned, {:.0} ms, search state {:.1} MB",
                    got.as_ref().map(|(c, _)| *c),
                    o.settled,
                    o.pruned,
                    t.elapsed().as_secs_f64() * 1000.0,
                    o.bytes() as f64 / 1e6
                );
                if o.batches > 0 || o.late_rows > 0 {
                    println!(
                        "  rows: {} filled in {} parallel batches (avg {:.0}), {} filled late by the search",
                        o.batch_rows,
                        o.batches,
                        o.batch_rows as f64 / o.batches.max(1) as f64,
                        o.late_rows
                    );
                }
                if let Some(c) = o.cache.as_ref() {
                    println!(
                        "  cache cap {} MB: {} flush(es), peak RSS {:.0} MB",
                        c.budget() / 1_000_000,
                        c.flushes,
                        c.peak as f64 / 1e6
                    );
                }
                return Ok(());
            }
            let mut r = Router::new(ds.n_vertices());
            let t = std::time::Instant::now();
            let route = r
                .route_with(&ds, &sa, &sb, ov.as_ref())
                .expect("no route found (an override may have closed the only way through)");
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let ms_plain = ms;
            println!(
                "distance {:.3} km   duration {:.1} min   {} legs   {} vertices settled   {:.0} ms",
                route.dist_m() / 1000.0,
                route.dur_s() / 60.0,
                route.legs.len(),
                r.settled,
                ms
            );
            let geom = router::route_geometry(&ds, &route);
            println!("geometry: {} points", geom.len());
            // When an overlay is present, answer the same query through it and
            // report both, so the two are always comparable on real routes
            // rather than only on synthetic pairs.
            // MPEE_ROUTER=plain|overlay isolates one router in its own
            // process, which is the only way to attribute page faults to it.
            let which = std::env::var("MPEE_ROUTER").unwrap_or_default();
            if which != "plain" {
            if let Ok(ov) = Overlay::open(&live) {
                let mut o = overlay::OverlayRouter::new();
                let t = std::time::Instant::now();
                if let Some((cost, path)) = o.search(&ds, &ov, &sa, &sb) {
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    println!(
                        "overlay: {:.1} min ({} ds) via {} boundary vertices, {} settled, {} regions entered, {ms:.0} ms",
                        cost as f64 / 600.0,
                        cost,
                        path.len(),
                        o.settled,
                        o.regions_entered
                    );
                    let gap = cost as i64 - route.dur_ds as i64;
                    println!(
                        "  vs plain: {} ds ({:+} ds, {:.1}x fewer settled, {:.1}x faster)",
                        route.dur_ds,
                        gap,
                        r.settled as f64 / o.settled.max(1) as f64,
                        ms_plain / ms.max(0.001)
                    );
                }
            }
            }
            if let Ok(t) = TollIndex::open(&live) {
                let hits = router::route_tolls(&route, &t);
                if !hits.is_empty() {
                    let class = args
                        .get(5)
                        .map(|s| s.as_str())
                        .filter(|s| *s == "car" || *s == "hgv")
                        .unwrap_or("car");
                    let at = args.get(6).and_then(|s| {
                        let (h, m) = s.split_once(':')?;
                        Some(tolldb::time_us(h.parse().ok()?, m.parse().ok()?))
                    });
                    let ids: Vec<u64> = hits.iter().map(|h| t.gates[h.gate].osm_id).collect();
                    println!(
                        "toll gates passed: {}   (class {class}{})",
                        hits.len(),
                        args.get(6).map(|s| format!(", at {s}")).unwrap_or_default()
                    );
                    match TollDb::open(&live) {
                        Ok(db) => {
                            let q = db.quote(&ids, class, at);
                            for l in &q.lines {
                                println!(
                                    "  {:<30} {:>9} {:<6} {:<8} {}",
                                    if l.gate.is_empty() { "(unnamed gate)" } else { &l.gate },
                                    format!("{:.2}", l.price),
                                    l.currency,
                                    l.rate_kind,
                                    if l.charged { l.scheme.clone() } else { format!("{} — covered by hour rule", l.scheme) }
                                );
                            }
                            for (cur, sum) in &q.totals {
                                println!("  TOLL: {sum:.2} {cur}");
                            }
                            if q.unpriced > 0 {
                                println!("  ({} gates crossed have no known tariff)", q.unpriced);
                            }
                        }
                        Err(e) => println!("  (no tariff database: {e})"),
                    }
                }
            }
            for (name, cm, cls) in router::route_steps(&ds, &route).iter().take(14) {
                println!(
                    "  {:>9.2} km  {:<14} {}",
                    *cm as f64 / 100_000.0,
                    router::class_name(*cls),
                    name
                );
            }
        }
        "addr" => {
            if args.len() < 3 {
                usage();
            }
            let paths = Paths::new(&PathBuf::from(&args[2]));
            let mut need = BitMap::load(&paths.f("need.bm"))?;
            need.build_rank();
            addresses::pass5(&paths, &need)?;
        }
        "toll" => {
            if args.len() < 3 {
                usage();
            }
            let paths = Paths::new(&PathBuf::from(&args[2]));
            let mut junc = BitMap::load(&paths.f("junc.bm"))?;
            junc.build_rank();
            toll::pass6(&paths, &junc)?;
        }
        "overlay" => {
            if args.len() < 3 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let live = catalog::resolve(&root);
            let target = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(overlay::TARGET);
            let ds = Dataset::open(&live)?;
            let paths = Paths::new(&live);
            // `lazy` leaves every level's values to be computed on demand,
            // which is what lets an update forget a region instead of
            // rebuilding one.
            let lazy = args.iter().any(|a| a == "lazy");
            let st = overlay::build_with(&paths, &ds, target, lazy)?;
            println!(
                "overlay built in {:.1} s\n  {} regions, {} boundary vertices\n  {} table entries ({:.2} GB)",
                st.secs,
                st.cells,
                st.boundary,
                st.matrix_entries,
                st.matrix_entries as f64 * 4.0 / 1e9
            );
            // Before anything opens the dataset: level 0 just changed, so the
            // rungs above it describe a partition that no longer exists.
            drop_levels_above_0(&live);
            let ov = Overlay::open(&live)?;
            let mut few = 0usize;
            let mut cov = 0usize;
            for c in 0..ov.cells() as u32 {
                let b = ov.boundary(c).len();
                if b <= 4 {
                    few += 1;
                    cov += b;
                }
            }
            println!("  regions reachable by <=4 roads: {few}");
            let _ = cov;
            drop(ov);
            let t2 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(overlay::TARGET2);
            // How small the top rung's boundary has to get before the ladder
            // stops. Overridable so a small extract can be made to build a real
            // ladder — Norway's 14 174 gates are under the default on the first
            // rung, so nothing above level 0 is ever exercised there.
            // No limit by default: the shed rule decides the height. Set this
            // to force a short ladder, which is the only way a small extract
            // builds an upper rung at all.
            let top: usize = std::env::var("MPEE_LADDER_TOP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            build_ladder(&live, &paths, &ds, t2, top, false, lazy)?;
        }
        // Level 2 alone, on a finished level 1. Rebuilding the first level of
        // the planet is four minutes of work that the second does not change.
        "overlay2" => {
            if args.len() < 3 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let live = catalog::resolve(&root);
            let t2 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(overlay::TARGET2);
            let ds = Dataset::open(&live)?;
            let paths = Paths::new(&live);
            // Same default as `overlay`: no absolute limit, the shed rule
            // decides. An explicit argument still forces a short ladder.
            let top = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0usize);
            // Rungs are expensive — the planet's first is eighteen minutes —
            // so `continue` keeps what is already built and picks up above it.
            let resume = args.iter().any(|a| a == "continue");
            // `lazy` writes each rung's shape and leaves its numbers to be
            // filled by the queries that actually need them.
            let lazy = args.iter().any(|a| a == "lazy");
            build_ladder(&live, &paths, &ds, t2, top, resume, lazy)?;
        }
        // Warm the lazy cache on purpose, rather than waiting for traffic.
        // `warm <dir> [rows] [seconds]` — both budgets, whichever comes first.
        // Cartesian vertex positions, so the A* potential never calls a
        // trigonometric function. Separate from the graph build on purpose: it
        // is derived entirely from `vcoord.bin`, so an existing dataset can
        // gain it — or drop it — without being rebuilt.
        "xyz" => {
            if args.len() < 3 {
                usage();
            }
            use rayon::prelude::*;
            use std::io::Write;
            let live = catalog::resolve(Path::new(&args[2]));
            let ds = Dataset::open(&live)?;
            let n = ds.n_vertices();
            let out = live.join("vxyz.bin");
            let t = std::time::Instant::now();
            let mut w = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&out)?);
            // In chunks, so a planet's 3.2 GB never has to be resident at once.
            const CHUNK: usize = 4 << 20;
            let mut buf: Vec<[f32; 3]> = Vec::with_capacity(CHUNK);
            let mut done = 0usize;
            while done < n {
                let end = (done + CHUNK).min(n);
                buf.clear();
                buf.resize(end - done, [0.0; 3]);
                buf.par_iter_mut().enumerate().for_each(|(i, p)| {
                    let (a, o) = ds.vcoord[done + i];
                    let q = overlay::xyz_of(a, o);
                    *p = [q[0] as f32, q[1] as f32, q[2] as f32];
                });
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 12)
                };
                w.write_all(bytes)?;
                done = end;
            }
            w.flush()?;
            drop(w);
            println!(
                "vxyz.bin: {n} vertices, {:.2} GB in {:.1} s",
                n as f64 * 12.0 / 1e9,
                t.elapsed().as_secs_f64()
            );
            // Read it back through the same path a query uses and check a
            // sample against the trigonometry it replaces. A silently wrong
            // position would not crash: it would quietly weaken or, worse,
            // overstate a bound that the search trusts to be a floor.
            let ds2 = Dataset::open(&live)?;
            let mut worst = 0.0f64;
            let step = (n / 100_000).max(1);
            for v in (0..n).step_by(step) {
                let (a, o) = ds2.vcoord[v];
                let want = overlay::xyz_of(a, o);
                let got = ds2.vxyz[v];
                for j in 0..3 {
                    worst = worst.max((got[j] as f64 - want[j]).abs());
                }
            }
            println!("  checked {} samples, worst axis error {worst:.3} m", n / step);
            if worst > 1.0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("vxyz.bin is {worst:.3} m off — the bound's slack is 4 m"),
                ));
            }
        }
        // Fold the use counters every level gathers while filling rows into
        // one vertex-keyed table, so the next partition can be cut where the
        // traffic is not. Run it after querying, before rebuilding.
        "traffic" => {
            if args.len() < 3 {
                usage();
            }
            let live = catalog::resolve(Path::new(&args[2]));
            let ov = Overlay::open(&live)?;
            let (n, peak) = overlay::write_traffic(&live, &ov)?;
            let before = overlay::Traffic::open(&live);
            println!(
                "traffic.bin: {n} vertices carry traffic, busiest settled {peak} times \
                 ({:.1} MB)",
                n as f64 * 8.0 / 1e6
            );
            if before.len() != n {
                println!("  (reopened as {} rows)", before.len());
            }
        }
        "warm" => {
            if args.len() < 3 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let live = catalog::resolve(&root);
            let rows: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(u64::MAX);
            let secs: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(u64::MAX);
            let ds = Dataset::open(&live)?;
            let ov = Overlay::open(&live)?;
            for k in 0..ov.levels() {
                if let Some((have, tot)) = ov.filled(k) {
                    println!("  level {k}: {have} of {tot} rows ({:.2} %)", 100.0 * have as f64 / tot as f64);
                }
            }
            let (n, secs_taken) = overlay::warm(
                &ds,
                None,
                &ov,
                rows,
                std::time::Duration::from_secs(secs.saturating_mul(1)),
            );
            println!("warmed {n} rows in {secs_taken:.1} s ({:.0} rows/s)", n as f64 / secs_taken.max(1e-9));
            for k in 0..ov.levels() {
                if let Some((have, tot)) = ov.filled(k) {
                    println!("  level {k}: {have} of {tot} rows ({:.2} %)", 100.0 * have as f64 / tot as f64);
                }
            }
        }
        // What changed since we built, and what that costs to fix.
        // `diff <dataset> <new.osm.pbf> [apply]`
        // The cheap refresh path: OSM's replication diffs instead of a planet
        // file. Four days of edits are four 95 MB files against 94.7 GB, and
        // they name the ids that changed rather than making us derive them.
        "osc" => {
            if args.len() < 4 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let live = catalog::resolve(&root);
            let files: Vec<PathBuf> = args[3..]
                .iter()
                .filter(|a| a.ends_with(".osc") || a.ends_with(".osc.gz"))
                .map(PathBuf::from)
                .collect();
            if files.is_empty() {
                usage();
            }
            if let Err(e) = build::check_way_stamp(&live) {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
            }
            let t = std::time::Instant::now();
            let touched = mpee_planet::osc::read(&live, &files)?;
            println!(
                "read {} change file(s) in {:.1} s",
                files.len(),
                t.elapsed().as_secs_f64()
            );
            println!(
                "  ways   created {} modified {} deleted {}  ({} distinct)",
                touched.way_counts[0],
                touched.way_counts[1],
                touched.way_counts[2],
                touched.ways.len()
            );
            println!(
                "  nodes  created {} modified {} deleted {}  ({} placed, {} without a position)",
                touched.node_counts[0],
                touched.node_counts[1],
                touched.node_counts[2],
                touched.nodes.len(),
                touched.unplaced
            );

            if touched.cost_known {
                println!(
                    "  of {} edited ways: {} not in the index, {} unchanged cost, \
                     {} no longer routable, {} moved a cost",
                    touched.ways.len(),
                    touched.unknown,
                    touched.cost_same,
                    touched.cost_unroutable,
                    touched.cost_changed.len()
                );
            } else {
                println!("  no way index — whether a cost moved cannot be told");
            }
            let ds = Dataset::open(&live)?;
            let ovl = Overlay::open(&live).ok();
            // One budget over everything the refresh maps: the graph, the
            // overlay, and the way index. Without it the kernel keeps the
            // 1.34 GB of way ids the join reads, which is fine here and fatal
            // on the machine this is for.
            let mut cap = std::env::var("MPEE_CACHE_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map(|mb| {
                    let mut c = cachecap::CacheCap::new(mb * 1_000_000);
                    c.govern(ds.maps());
                    if let Some(o) = ovl.as_ref() {
                        c.govern(o.maps());
                    }
                    c
                });
            let t2 = std::time::Instant::now();
            let by_way = mpee_planet::osc::segments_of(&live, &touched.ways, cap.as_mut())?;
            println!(
                "  {} segments from {} ways in {:.1} s",
                by_way.len(),
                touched.ways.len(),
                t2.elapsed().as_secs_f64()
            );
            let mut segs = by_way.clone();

            // The node path is opt-in. A way edit names its segments outright;
            // a node that moved has to be found geographically, because
            // `way.topo` hashes a way's node list rather than keeping it. That
            // is one snap per node, and nine days of planet edits hold 4.8
            // million modified nodes — minutes of work for the last few tenths
            // of a percent. `nodes` asks for it, `nodes=N` samples N of them.
            let want = args.iter().find(|a| a.starts_with("nodes"));
            if let Some(w) = want {
                let limit: usize = w
                    .split_once('=')
                    .and_then(|(_, v)| v.parse().ok())
                    .unwrap_or(usize::MAX);
                let t3 = std::time::Instant::now();
                let (by_node, looked) =
                    mpee_planet::osc::segments_near(&ds, &touched.nodes, 30.0, limit);
                println!(
                    "  {} segments from {} of {} modified nodes in {:.1} s",
                    by_node.len(),
                    looked,
                    touched.nodes.len(),
                    t3.elapsed().as_secs_f64()
                );
                segs.extend_from_slice(&by_node);
                segs.sort_unstable();
                segs.dedup();
            } else if !touched.nodes.is_empty() {
                println!(
                    "  {} modified nodes not placed — pass `nodes` to snap them, \
                     and see the note in osc.rs for what that misses",
                    touched.nodes.len()
                );
            }
            println!("  {} distinct segments affected", segs.len());

            // `apply` forgets the affected rows so they are computed again from
            // the graph. Note what that does and does not do: the graph here is
            // still the old one, because reading a change file does not rebuild
            // segments. So until that half exists, this measures what an update
            // costs and proves the forget-and-recompute path is faithful — the
            // rows come back identical — rather than bringing in new data.
            let apply = args.iter().any(|a| a == "apply");
            // `values` carries the update up by measured change instead of by
            // containment: recompute the affected regions, compare, and climb
            // only where a number actually moved. Sound for metric changes,
            // which is what this path sees; a topology change needs the graph
            // rebuilt and then nothing here transfers. See `diff::propagate`.
            if args.iter().any(|a| a == "values") {
                let Some(ov) = ovl.as_ref() else {
                    println!("  no overlay to update");
                    return Ok(());
                };
                let ovr = OverrideDb::open(&root).ok().map(|db| {
                    let rules = db.list().unwrap_or_default();
                    Overrides::resolve(&ds, &rules, db.generation())
                });
                // Only the ways whose *cost* moved. A change file names what an
                // editor touched; `way.hash` says whether that touch could have
                // moved a table entry. Without the index the question cannot be
                // answered, and then every edit has to be assumed to count.
                let work = if touched.cost_known {
                    let cs = mpee_planet::osc::segments_of(
                        &live,
                        &touched.cost_changed,
                        cap.as_mut(),
                    )?;
                    println!("  -> {} of {} segments to examine", cs.len(), segs.len());
                    cs
                } else {
                    println!("  no way index — every edit assumed to move a cost");
                    segs.clone()
                };
                let t4 = std::time::Instant::now();
                let steps = mpee_planet::diff::propagate(
                    &ds,
                    ovr.as_ref().filter(|o| !o.is_empty()),
                    ov,
                    &work,
                );
                let mut rows = 0usize;
                let mut moved = 0usize;
                for st in &steps {
                    rows += st.rows;
                    moved += st.moved;
                    println!(
                        "  level {}: {} regions examined, {} moved, {} of {} rows changed{}",
                        st.level,
                        st.regions,
                        st.changed,
                        st.moved,
                        st.rows,
                        if st.eager { "  — eager, assumed changed" } else { "" }
                    );
                }
                let tot: usize =
                    (0..ov.levels()).map(|k| ov.level(k).boundary_total() * 2).sum();
                println!(
                    "{rows} rows recomputed, {moved} moved, of {tot} in the ladder \
                     ({:.2} % recomputed) in {:.1} s",
                    100.0 * rows as f64 / tot.max(1) as f64,
                    t4.elapsed().as_secs_f64()
                );
                if steps.len() < ov.levels() {
                    println!(
                        "  the climb stopped after level {} — nothing above it could move",
                        steps.len() - 1
                    );
                }
                return Ok(());
            }
            match ovl {
                Some(ov) => {
                    let per = mpee_planet::diff::regions_of(&ds, &ov, &segs, cap.as_mut());
                    let mut dropped = 0usize;
                    for (k, set) in per.iter().enumerate() {
                        let tot = ov.level(k).regions().max(1);
                        let lazy = ov.level(k).is_lazy();
                        // Counted whether or not it is applied: the row count
                        // is the size of an update, and a dry run that cannot
                        // say it is no use for choosing a partition.
                        let mut rows = 0usize;
                        for &c in set {
                            rows += if apply && lazy {
                                ov.invalidate(k, c)
                            } else {
                                ov.level(k).boundary(c).len() * 2
                            };
                        }
                        dropped += rows;
                        println!(
                            "  level {k}: {} of {} regions to recompute ({:.3} %){}",
                            set.len(),
                            tot,
                            100.0 * set.len() as f64 / tot as f64,
                            if !lazy {
                                "  — eager, needs a rebuild rather than a forget".into()
                            } else if apply {
                                format!("  — {rows} rows dropped")
                            } else {
                                format!("  — {rows} rows would drop")
                            }
                        );
                    }
                    if apply {
                        println!(
                            "dropped {dropped} cached rows; they will be recomputed on demand \
                             or by `warm`"
                        );
                    } else {
                        let tot: usize = (0..ov.levels())
                            .map(|k| ov.level(k).boundary_total() * 2)
                            .sum();
                        println!(
                            "{dropped} of {tot} rows would drop ({:.1} % of the ladder) \
                             — dry run, pass `apply` to do it",
                            100.0 * dropped as f64 / tot.max(1) as f64
                        );
                    }
                }
                None => println!("  no overlay to invalidate"),
            }
        }
        "diff" => {
            if args.len() < 4 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let live = catalog::resolve(&root);
            let pbf = PathBuf::from(&args[3]);
            let apply = args.iter().any(|a| a == "apply");
            // The diff's temporaries go beside the dataset but must not collide
            // with the build's: a blob directory belongs to one PBF, and this
            // is deliberately a different one.
            // Before anything is hashed: the index has to be in a revision whose
            // way.hash means the same thing this build computes.
            if let Err(e) = build::check_way_stamp(&live) {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
            }
            let paths = Paths::new(&root.join("diff.tmp"));
            let dir = build::open_dir(&pbf, &paths)?;
            let t = std::time::Instant::now();
            let n = mpee_planet::diff::hash_ways(&pbf, &paths, &dir)?;
            println!("hashed {n} routable ways in {:.1} s", t.elapsed().as_secs_f64());
            let ch = mpee_planet::diff::compare(&live, &paths)?;
            println!(
                "  unchanged {}   changed {}   added {}   deleted {}",
                ch.unchanged, ch.changed, ch.added, ch.deleted
            );
            println!("  {} segments affected", ch.segments.len());
            for (n, lbl) in [(0usize, "unchanged"), (1, "changed"), (2, "added"), (3, "deleted")] {
                if !ch.sample[n].is_empty() {
                    println!("  sample {lbl}: {:?}", ch.sample[n]);
                }
            }
            let ds = Dataset::open(&live)?;
            let ov = Overlay::open(&live)?;
            // Same budget as the `osc` path, and for the same reason: the walk
            // reads `seg_u`, `seg_v` and `cell.of`, which is over 3 GB between
            // them on the planet.
            let mut cap = std::env::var("MPEE_CACHE_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map(|mb| {
                    let mut c = cachecap::CacheCap::new(mb * 1_000_000);
                    c.govern(ds.maps());
                    c.govern(ov.maps());
                    c
                });
            let per = mpee_planet::diff::regions_of(&ds, &ov, &ch.segments, cap.as_mut());
            let mut dropped = 0usize;
            for (k, set) in per.iter().enumerate() {
                let lazy = ov.level(k).is_lazy();
                println!(
                    "  level {k}: {} of {} regions affected ({:.4} %){}",
                    set.len(),
                    ov.level(k).regions(),
                    100.0 * set.len() as f64 / ov.level(k).regions().max(1) as f64,
                    if lazy { "" } else { "  — eager, needs a rebuild rather than a forget" }
                );
                if apply {
                    for &c in set {
                        dropped += ov.invalidate(k, c);
                    }
                }
            }
            if apply {
                println!("dropped {dropped} cached rows; they will be recomputed on demand");
            } else {
                println!("(dry run — pass `apply` to drop the affected rows)");
            }
            std::fs::remove_file(paths.f("diff.wayhash")).ok();
        }
        // How concentrated is road use? `betweenness <root> [routes]`
        //
        // Sample random routes and count how often each segment carries one.
        // If a small fraction of segments carries most traffic, then where a
        // partition cuts matters a great deal — cutting across a corridor
        // makes every route through it a boundary vertex. If use is flat, the
        // topology is all there is to go on and a gate cap is the whole story.
        "betweenness" => {
            if args.len() < 3 {
                usage();
            }
            let live = catalog::resolve(&PathBuf::from(&args[2]));
            let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(400);
            let ds = Dataset::open(&live)?;
            let nv = ds.n_vertices();
            let mut count: Vec<u32> = vec![0; ds.seg_len.len()];
            let mut r = Router::new(nv);
            let mut seed = 0x243F_6A88_85A3_08D3u64;
            let mut rng = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let (mut done, mut tries, mut legs) = (0usize, 0usize, 0u64);
            let t = std::time::Instant::now();
            while done < n && tries < n * 50 {
                tries += 1;
                let a = (rng() % nv as u64) as u32;
                let b = (rng() % nv as u64) as u32;
                let (Some(sa), Some(sb)) = (vertex_snap(&ds, a), vertex_snap(&ds, b)) else {
                    continue;
                };
                if let Some(rt) = r.route(&ds, &sa, &sb) {
                    if rt.legs.len() < 20 {
                        continue; // too short to say anything about corridors
                    }
                    for l in &rt.legs {
                        count[l.seg as usize] += 1;
                    }
                    legs += rt.legs.len() as u64;
                    done += 1;
                }
            }
            println!(
                "{done} routes, {legs} segment traversals in {:.1} s",
                t.elapsed().as_secs_f64()
            );
            let mut used: Vec<u32> = count.into_iter().filter(|&c| c > 0).collect();
            used.sort_unstable_by(|a, b| b.cmp(a));
            let total: u64 = used.iter().map(|&c| c as u64).sum();
            println!("  {} distinct segments carried a route", used.len());
            let mut acc = 0u64;
            for (i, &c) in used.iter().enumerate() {
                acc += c as u64;
                let f = (i + 1) as f64 / used.len() as f64;
                if [0.001, 0.01, 0.05, 0.1, 0.25, 0.5].iter().any(|t| {
                    let k = (used.len() as f64 * t) as usize;
                    k == i + 1
                }) {
                    println!(
                        "    the busiest {:5.1} % of them carry {:5.1} % of all traversals",
                        f * 100.0,
                        acc as f64 / total as f64 * 100.0
                    );
                }
            }
        }
        // Split the worst regions of a level.
        // `split <root> <level> <max-gates> [limit]`
        "split" => {
            if args.len() < 5 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let live = catalog::resolve(&root);
            let k: usize = args[3].parse().unwrap_or(0);
            let max_gates: usize = args[4].parse().unwrap_or(256);
            let limit: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let ds = Dataset::open(&live)?;
            let ov = Overlay::open(&live)?;
            let paths = Paths::new(&live);
            // `bisect` is the naive comparison: halve the region by breadth-first
            // sweep and let the cut fall where it may.
            let plan = if args.iter().any(|a| a == "bisect") {
                overlay::plan_splits(&ds, &ov, k, max_gates, limit)
            } else {
                overlay::plan_splits_by_cost(&ds, &ov, k, max_gates, limit)
            };
            println!("{} region(s) over {max_gates} gates", plan.cuts.len());
            if plan.cuts.is_empty() {
                return Ok(());
            }
            let st = overlay::split_level(&paths, &ds, &ov, k, &plan)?;
            println!(
                "  {} -> {} regions in {:.1} s",
                st.regions_before, st.regions_after, st.secs
            );
            println!(
                "  gates {} -> {} ({:+.1} %)",
                st.gates_before,
                st.gates_after,
                100.0 * (st.gates_after as f64 / st.gates_before as f64 - 1.0)
            );
            println!(
                "  rows: {} kept, {} to recompute ({:.2} % of the level)",
                st.rows_kept,
                st.rows_dropped,
                100.0 * st.rows_dropped as f64 / (st.rows_kept + st.rows_dropped).max(1) as f64
            );
        }
        // One adaptive cycle: clone, split, warm, publish.
        // `adapt <root> <level> <max-gates> [limit] [warm-seconds]`
        "adapt" => {
            if args.len() < 5 {
                usage();
            }
            let root = PathBuf::from(&args[2]);
            let live = catalog::resolve(&root);
            let k: usize = args[3].parse().unwrap_or(0);
            let max_gates: usize = args[4].parse().unwrap_or(256);
            let limit: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let warm_s: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(600);

            // Plan against the live dataset before touching anything, so a
            // cycle with nothing to do costs a read and no clone.
            {
                let ds = Dataset::open(&live)?;
                let ov = Overlay::open(&live)?;
                let plan = overlay::plan_splits_by_cost(&ds, &ov, k, max_gates, limit);
                if plan.cuts.is_empty() {
                    println!("nothing over {max_gates} gates at level {k} — nothing to do");
                    return Ok(());
                }
                println!("{} region(s) to split at level {k}", plan.cuts.len());
            }

            // The clone. On APFS this is copy-on-write: the planet's 33 GB of
            // row cache took 0.146 s and consumed nothing until it diverged.
            // Everything below happens in the copy, so the live dataset keeps
            // answering from a consistent structure the whole time.
            let id = catalog::new_build_id();
            let dir = catalog::build_dir(&root, &id);
            if dir.exists() {
                println!("build {id} already exists — wait a second and retry");
                return Ok(());
            }
            let t = std::time::Instant::now();
            let ok = std::process::Command::new("cp")
                .args(["-c", "-R"])
                .arg(&live)
                .arg(&dir)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !ok {
                return Err(std::io::Error::other("could not clone the live build"));
            }
            println!("cloned to {id} in {:.2} s", t.elapsed().as_secs_f64());

            let ds = Dataset::open(&dir)?;
            let ov = Overlay::open(&dir)?;
            let paths = Paths::new(&dir);
            let plan = overlay::plan_splits_by_cost(&ds, &ov, k, max_gates, limit);
            let st = overlay::split_level(&paths, &ds, &ov, k, &plan)?;
            println!(
                "  {} -> {} regions, gates {} -> {} ({:+.1} %), {} rows kept, {} to recompute",
                st.regions_before,
                st.regions_after,
                st.gates_before,
                st.gates_after,
                100.0 * (st.gates_after as f64 / st.gates_before as f64 - 1.0),
                st.rows_kept,
                st.rows_dropped
            );
            drop(ov);
            drop(ds);

            // Warm what the split invalidated, before anyone can be served
            // from it. A published dataset with cold rows in the corridors
            // people actually use is the failure this whole ordering avoids.
            let ds = Dataset::open(&dir)?;
            let ov = Overlay::open(&dir)?;
            let (rows, secs) = overlay::warm(
                &ds,
                None,
                &ov,
                u64::MAX,
                std::time::Duration::from_secs(warm_s),
            );
            println!("  warmed {rows} rows in {secs:.1} s");
            let left: usize = (0..ov.levels())
                .filter_map(|l| ov.filled(l))
                .map(|(have, tot)| tot - have)
                .sum();
            drop(ov);
            drop(ds);
            if left > 0 {
                println!(
                    "  {left} rows still cold — publishing anyway; they fill on demand"
                );
            }

            // Verify, then swap. `rename` is atomic: a reader sees the old id
            // or the new one, never a half-written HEAD.
            let problems = mpee_planet::dataset::verify(&dir);
            if !problems.is_empty() {
                for pr in problems.iter().take(5) {
                    eprintln!("  {}: {}", pr.file, pr.detail);
                }
                return Err(std::io::Error::other("the split build does not verify"));
            }
            let cat = Catalog::open(&root)?;
            cat.begin_build(&id, &live, "adaptive", 0.0)?;
            for pass in catalog::PASSES {
                cat.pass_start(&id, pass)?;
                cat.pass_end(&id, pass, true, 0.0)?;
            }
            cat.publish(&id)?;
            println!("published {id}");
        }
        "ovcheck" => {
            if args.len() < 3 {
                usage();
            }
            let live = catalog::resolve(Path::new(&args[2]));
            let ds = Dataset::open(&live)?;
            let ov = Overlay::open(&live)?;
            let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
            let mut over = overlay::OverlayRouter::new();
            let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let nv = ds.n_vertices();
            let (mut checked, mut agree, mut worst) = (0usize, 0usize, 0i64);
            let (mut set_ref, mut set_ov) = (0u64, 0u64);
            let (mut t_ref, mut t_ov) = (0.0f64, 0.0f64);
            // Vertex to vertex, so no snapping or partial end segment can
            // account for a difference: the reference is a plain Dijkstra.
            while checked < n {
                let s0 = (next() as usize) % nv;
                let t0 = (next() as usize) % nv;
                if s0 == t0 || ov.cell(s0 as u32) == ov.cell(t0 as u32) {
                    continue;
                }
                let t = std::time::Instant::now();
                let (d, settled) = mpee_planet::overlay::reference_cost(&ds, s0 as u32, t0 as u32);
                t_ref += t.elapsed().as_secs_f64();
                set_ref += settled;
                let Some(exact) = d else { continue };
                let t = std::time::Instant::now();
                let got = over.search_vertices(&ds, &ov, s0 as u32, t0 as u32);
                t_ov += t.elapsed().as_secs_f64();
                set_ov += over.settled;
                checked += 1;
                match got {
                    Some(c) if c == exact => agree += 1,
                    Some(c) => {
                        let gap = c as i64 - exact as i64;
                        if gap.abs() > worst.abs() {
                            worst = gap;
                        }
                        if checked <= 3 {
                            println!("  {s0} -> {t0}: overlay {c} vs exact {exact} ({gap:+})");
                        }
                    }
                    None => println!("  {s0} -> {t0}: overlay found nothing, exact says {exact}"),
                }
            }
            println!(
                "{agree}/{checked} vertex pairs exact (worst gap {worst} ds)\n  \
                 settled: reference {set_ref} vs overlay {set_ov} ({:.1}x fewer)\n  \
                 time:    reference {:.1} ms vs overlay {:.2} ms per query",
                set_ref as f64 / set_ov.max(1) as f64,
                t_ref * 1000.0 / checked.max(1) as f64,
                t_ov * 1000.0 / checked.max(1) as f64
            );
        }
        "tolldb" => {
            if args.len() < 3 {
                usage();
            }
            let dir = PathBuf::from(&args[2]);
            let idx = TollIndex::open(&dir)?;
            let nvdb = args.get(3).map(PathBuf::from).filter(|p| p.exists());
            let t = std::time::Instant::now();
            let st = tolldb::ingest(&dir, &idx.gates, &idx.vertex, nvdb.as_deref())
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            println!(
                "toll.mpedb built in {:.1} s\n  {} gates\n  {} schemes\n  {} rates ({} from NVDB stations, {} from OSM charge tags)\n  {} rush windows",
                t.elapsed().as_secs_f64(),
                st.gates, st.schemes, st.rates,
                st.matched_nvdb, st.from_osm_tag, st.rush_windows
            );
            if !st.unmatched_nvdb.is_empty() {
                println!("  {} NVDB stations unmatched, e.g.:", st.unmatched_nvdb.len());
                for (n, why) in st.unmatched_nvdb.iter().take(4) {
                    println!("    {n} — {why}");
                }
            }
        }
        "tollq" => {
            if args.len() < 4 {
                usage();
            }
            let db = TollDb::open(&catalog::resolve(Path::new(&args[2])))
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let t = std::time::Instant::now();
            let r = db.query(&args[3]).map_err(|e| std::io::Error::other(e.to_string()))?;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            match r {
                mpedb::ExecResult::Rows { columns, rows } => {
                    println!("{}", columns.join("\t"));
                    for row in rows.iter().take(40) {
                        let cells: Vec<String> = row.iter().map(fmt_val).collect();
                        println!("{}", cells.join("\t"));
                    }
                    eprintln!("[tollq] {} rows in {ms:.1} ms", rows.len());
                }
                other => println!("{other:?}"),
            }
        }
        "find" => {
            if args.len() < 5 {
                usage();
            }
            let g = Geocoder::open(&catalog::resolve(Path::new(&args[2])))?;
            eprintln!("[find] {} address points, {} streets", g.len(), g.street_count());
            let t = std::time::Instant::now();
            let city = args.get(5).map(|s| s.trim()).filter(|s| !s.is_empty());
            let hits = g.forward(&args[3], &args[4], city, 8);
            let us = t.elapsed().as_secs_f64() * 1e6;
            if hits.is_empty() {
                println!("no match");
                for s in g.suggest(&args[3], 8) {
                    println!("  did you mean: {s}");
                }
            }
            for h in &hits {
                println!(
                    "{:.7},{:.7}  {} {}{}{}{}",
                    h.lat,
                    h.lon,
                    h.street,
                    h.housenumber,
                    h.city.as_deref().map(|c| format!(", {c}")).unwrap_or_default(),
                    h.postcode.as_deref().map(|c| format!(" ({c})")).unwrap_or_default(),
                    if h.approximate { "  [nearest number]" } else { "" }
                );
            }
            eprintln!("[find] {us:.0} us");
        }
        "rev" => {
            if args.len() < 4 {
                usage();
            }
            let g = Geocoder::open(&catalog::resolve(Path::new(&args[2])))?;
            let ll = parse_ll(&args[3]);
            let t = std::time::Instant::now();
            let h = g.reverse(ll.0 as f64 * 1e-7, ll.1 as f64 * 1e-7, 500.0);
            let us = t.elapsed().as_secs_f64() * 1e6;
            match h {
                Some(h) => println!(
                    "{} {}{}{}  ({:.1} m away)",
                    h.street,
                    h.housenumber,
                    h.city.as_deref().map(|c| format!(", {c}")).unwrap_or_default(),
                    h.postcode.as_deref().map(|c| format!(" {c}")).unwrap_or_default(),
                    h.distance_m
                ),
                None => println!("no address within 500 m"),
            }
            eprintln!("[rev] {us:.0} us");
        }
        "serve" => {
            if args.len() < 3 {
                usage();
            }
            let addr = args.get(3).cloned().unwrap_or_else(|| "127.0.0.1:8099".into());
            // The server takes the catalogue *root*: it resolves HEAD itself and
            // reads overrides from alongside it.
            let app = std::sync::Arc::new(serve::App::open(Path::new(&args[2]))?);
            serve::run(app, &addr)?;
        }
        "codecstat" => {
            if args.len() < 3 {
                usage();
            }
            let live = catalog::resolve(Path::new(&args[2]));
            let n = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(400);
            codecstat::bench_codec(&live)?;
            println!("\nsampling {n} blocks of {} records per array\n", codecstat::BLOCK);
            let ms = codecstat::analyse(&live, n)?;
            let raw: u64 = ms.iter().map(|m| m.raw_bytes).sum();
            let best: u64 = ms
                .iter()
                .map(|m| m.deflate_bytes.min(m.predicted_deflate_bytes))
                .sum();
            println!(
                "\n  measured arrays: {:.2} GB raw -> {:.2} GB at best scheme ({:.2}x)",
                raw as f64 / 1e9,
                best as f64 / 1e9,
                raw as f64 / best as f64
            );
        }
        "stats" => {
            if args.len() < 3 {
                usage();
            }
            mpee_planet::dataset::report_sizes(&catalog::resolve(Path::new(&args[2])))?;
        }
        _ => usage(),
    }
    Ok(())
}

/// Build the ladder above level 0, rung by rung, until the top is small
/// enough to search directly.
///
/// Each rung is `STEP` times coarser than the last, and the loop stops on
/// whichever comes first: a top level small enough that a plain Dijkstra over
/// it is trivial, a rung that sheds almost nothing (so the next would cost
/// more than it saves), or the ceiling on levels. Existing rungs are removed
/// first, so a rebuild is a rebuild and never a ladder with two different
/// shapes stacked on top of each other.
/// Snap a vertex to itself, so a random vertex can be routed from.
fn vertex_snap(ds: &Dataset, v: u32) -> Option<router::Snap> {
    let (a, b) = (ds.head[v as usize] as usize, ds.head[v as usize + 1] as usize);
    if a == b {
        return None;
    }
    let seg = (ds.eseg[a] & router::SEG_MASK) as usize;
    Some(router::Snap {
        seg: seg as u32,
        u: ds.seg_u[seg],
        v: ds.seg_v[seg],
        t: if ds.seg_u[seg] == v { 0.0 } else { 1.0 },
        lat_e7: ds.vcoord[v as usize].0,
        lon_e7: ds.vcoord[v as usize].1,
        off_m: 0.0,
    })
}

/// Take away every rung above level 0.
///
/// Called the moment level 0 is rebuilt, not when the next rung is about to be
/// written. A rung is built from the boundary of the one below it, so a fresh
/// level 0 makes every level above it stale in the same instant — and until
/// they are gone, opening the dataset means opening a ladder whose bottom is
/// new and whose top is not. That was true before the layout stamp existed;
/// the stamp only made it stop quietly succeeding.
/// Take away one rung. Used when a rung has been built and then judged not
/// worth keeping — the judgement needs the rung's boundary count, which only
/// exists once it is written.
fn drop_level(live: &std::path::Path, k: usize) {
    for f in overlay::level_files(k) {
        std::fs::remove_file(live.join(&f)).ok();
        std::fs::remove_file(live.join(format!("{f}.wide"))).ok();
    }
    for f in overlay::lazy_files(k) {
        std::fs::remove_file(live.join(&f)).ok();
    }
    std::fs::remove_file(live.join(overlay::fmt_file(k))).ok();
    std::fs::remove_file(live.join(overlay::use_file(k))).ok();
    std::fs::remove_file(live.join(overlay::geo_file(k))).ok();
}

fn drop_levels_above_0(live: &std::path::Path) {
    for k in 1..overlay::MAX_LEVELS {
        for f in overlay::level_files(k) {
            std::fs::remove_file(live.join(&f)).ok();
            std::fs::remove_file(live.join(format!("{f}.wide"))).ok();
        }
        for f in overlay::lazy_files(k) {
            std::fs::remove_file(live.join(&f)).ok();
        }
        std::fs::remove_file(live.join(overlay::fmt_file(k))).ok();
        std::fs::remove_file(live.join(overlay::use_file(k))).ok();
    }
}

fn build_ladder(
    live: &std::path::Path,
    paths: &Paths,
    ds: &Dataset,
    first_target: usize,
    top: usize,
    resume: bool,
    lazy: bool,
) -> std::io::Result<()> {
    if resume {
        // Only the scratch of a rung that was interrupted mid-table. The six
        // real files are written last and together, so a rung either exists
        // whole or not at all — which is what makes resuming safe.
        for k in 1..overlay::MAX_LEVELS {
            std::fs::remove_file(live.join(format!("{}.wide", overlay::level_files(k)[4]))).ok();
        }
    } else {
        drop_levels_above_0(live);
    }
    // Each rung is `STEP` coarser than the last, so resuming has to start at
    // the size the interrupted rung would have had, not at the first one.
    let mut target = first_target;
    if resume {
        let have = Overlay::open(live)?.levels();
        for _ in 1..have {
            target = target.saturating_mul(overlay::STEP);
        }
        if have > 1 {
            println!("resuming with {} level(s) already built, next target {target}", have);
        }
    }
    loop {
        let ov = Overlay::open(live)?;
        let k = ov.levels();
        let below = ov.level(k - 1).boundary_total();
        if k >= overlay::MAX_LEVELS {
            println!("  ladder full at {k} levels");
            break;
        }
        // An absolute boundary size is not a reason to stop, and measuring it
        // showed why. The planet's second rung came out at 167 697 gates, under
        // the 200 000 this used to compare against, so the ladder stopped there
        // — while two more rungs were available and both paid: 287 -> 274 -> 264
        // ms on Lisboa-Warszawa. The rule was not finding a natural end, it was
        // cutting in front of one.
        //
        // What a rung is worth depends on how much boundary it *removes*, not on
        // how much is left, and the shed rule below says that directly. `top` is
        // kept only as an override for forcing a short ladder on purpose, which
        // is how a small extract can be made to exercise the upper rungs at all.
        if top > 0 && below <= top {
            println!("  stopping at {k} level(s): told to stop below {top} boundary vertices");
            break;
        }
        drop(ov);
        let ov = Overlay::open(live)?;
        let st = if lazy {
            // Hours, not rows: what an operator can say is how long they are
            // prepared to warm for.
            let budget = std::env::var("MPEE_MAX_FILL_HOURS")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(8.0);
            overlay::build_level_lazy(paths, ds, &ov, k, target, budget * 3600.0)?
        } else {
            overlay::build_level(paths, ds, None, &ov, k, target)?
        };
        // Nothing was written when a rung is refused, so nothing is reported
        // as built — the refusal has already said why on stderr.
        if st.refused {
            break;
        }
        println!(
            "level {} built in {:.1} s: {} regions, {} boundary vertices, {:.2} GB",
            k + 1,
            st.secs,
            st.cells,
            st.boundary,
            st.entries as f64 * 4.0 / 1e9
        );
        // Shed less than this and the rung is not worth its disk or its warm.
        //
        // 10 % is where this started and it is measurably too generous. Each
        // rung on the planet shed 80.6 %, 86.7 %, 45.6 %, 21.3 %, 0.0 % — and
        // the query times went 1059 -> 521 -> 289 -> 274 -> 264 ms. The two
        // rungs that halved the time shed over 80 %; the two that shed under
        // half bought 8 % between them, for 2.32 GB and four times the cold
        // fill. A threshold of 0.5 would have stopped at three rungs.
        //
        // Left at 0.9 deliberately: which side of that trade is right depends on
        // whether the dataset is warmed once and queried often, and that is not
        // a question the build can answer. `MPEE_LADDER_SHED` moves it.
        let shed_floor: f64 = std::env::var("MPEE_LADDER_SHED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.9);
        if st.boundary as f64 > below as f64 * shed_floor {
            println!(
                "  stopping: that rung shed only {:.1} % of the boundary — \
                 another would cost more than it saves",
                100.0 - 100.0 * st.boundary as f64 / below as f64
            );
            // And take it away again. The test needs the rung's boundary count,
            // which only exists once it is written, so a rejected rung has
            // already reached the disk by the time it is rejected. Leaving it
            // there is not harmless: the planet kept a sixth rung byte-identical
            // to its fifth, reserving 1.04 GB of table that a warm would have
            // spent an hour filling to no purpose.
            drop_level(live, k);
            break;
        }
        target = target.saturating_mul(overlay::STEP);
    }
    Ok(())
}
