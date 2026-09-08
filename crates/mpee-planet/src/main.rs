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
                    "overlay-only: {:?} ds, {} settled, {:.0} ms, search state {:.1} MB",
                    got.as_ref().map(|(c, _)| *c),
                    o.settled,
                    t.elapsed().as_secs_f64() * 1000.0,
                    o.bytes() as f64 / 1e6
                );
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
            let st = overlay::build(&paths, &ds, target)?;
            println!(
                "overlay built in {:.1} s\n  {} regions, {} boundary vertices\n  {} table entries ({:.2} GB)",
                st.secs,
                st.cells,
                st.boundary,
                st.matrix_entries,
                st.matrix_entries as f64 * 4.0 / 1e9
            );
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
            build_ladder(&live, &paths, &ds, t2, 200_000, false)?;
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
            let top = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200_000usize);
            // Rungs are expensive — the planet's first is eighteen minutes —
            // so `continue` keeps what is already built and picks up above it.
            let resume = args.iter().any(|a| a == "continue");
            build_ladder(&live, &paths, &ds, t2, top, resume)?;
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
fn build_ladder(
    live: &std::path::Path,
    paths: &Paths,
    ds: &Dataset,
    first_target: usize,
    top: usize,
    resume: bool,
) -> std::io::Result<()> {
    if resume {
        // Only the scratch of a rung that was interrupted mid-table. The six
        // real files are written last and together, so a rung either exists
        // whole or not at all — which is what makes resuming safe.
        for k in 1..overlay::MAX_LEVELS {
            std::fs::remove_file(live.join(format!("{}.wide", overlay::level_files(k)[4]))).ok();
        }
    } else {
        for k in 1..overlay::MAX_LEVELS {
            for f in overlay::level_files(k) {
                std::fs::remove_file(live.join(&f)).ok();
                std::fs::remove_file(live.join(format!("{f}.wide"))).ok();
            }
        }
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
        if below <= top {
            println!(
                "  stopping at {k} level(s): the top has {below} boundary vertices, \
                 small enough to search directly"
            );
            break;
        }
        drop(ov);
        let ov = Overlay::open(live)?;
        let st = overlay::build_level(paths, ds, &ov, k, target)?;
        println!(
            "level {} built in {:.1} s: {} regions, {} boundary vertices, {:.2} GB",
            k + 1,
            st.secs,
            st.cells,
            st.boundary,
            st.entries as f64 * 4.0 / 1e9
        );
        if st.boundary as f64 > below as f64 * 0.9 {
            println!(
                "  stopping: that rung shed only {:.1} % of the boundary — \
                 another would cost more than it saves",
                100.0 - 100.0 * st.boundary as f64 / below as f64
            );
            break;
        }
        target = target.saturating_mul(overlay::STEP);
    }
    Ok(())
}
