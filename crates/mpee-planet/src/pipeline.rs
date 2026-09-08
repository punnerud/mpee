//! One-call build: PBF in, published dataset out, or nothing at all.
//!
//! The passes were always separate commands, which is convenient for
//! development and wrong for operations: nothing tied them together, so a
//! build that died between two of them left a directory that looked finished.
//! This runs the whole sequence against a *staging* directory, records each
//! pass in the catalogue as it commits, checks the result is internally
//! consistent, and only then makes it live.
//!
//! Every pass keeps its own scope so the memory it needs — the three id
//! bitmaps are ~1.8 GB each — is released before the next one starts, exactly
//! as running them as separate processes did.

use crate::bitmap::BitMap;
use crate::build::{self, Paths};
use crate::catalog::{self, Catalog};
use crate::contract::{self, ContractCfg};
use crate::dataset::Dataset;
use crate::{addresses, dataset, graph, overlay, toll, tolldb};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct BuildOpts {
    pub simplify_m: f64,
    pub nvdb: Option<PathBuf>,
    pub keep_scratch: bool,
    pub deep_digest: bool,
    /// Build the region overlay. It costs a few minutes and ~5 GB, and buys a
    /// router that runs in 58 MB instead of 4.4 GB.
    pub overlay: bool,
}

impl Default for BuildOpts {
    fn default() -> Self {
        BuildOpts {
            simplify_m: 1.0,
            nvdb: None,
            keep_scratch: false,
            deep_digest: false,
            overlay: true,
        }
    }
}

/// Run one pass, bracketing it with the catalogue so a crash is visible.
fn pass<T>(
    cat: &Catalog,
    id: &str,
    name: &str,
    f: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    eprintln!("[build/{id}] ── {name}");
    cat.pass_start(id, name)?;
    let t = Instant::now();
    match f() {
        Ok(v) => {
            cat.pass_end(id, name, true, t.elapsed().as_secs_f64())?;
            Ok(v)
        }
        Err(e) => {
            cat.pass_end(id, name, false, t.elapsed().as_secs_f64())?;
            let _ = cat.fail_build(id, &format!("{name}: {e}"));
            Err(e)
        }
    }
}

pub fn build_all(pbf: &Path, root: &Path, o: &BuildOpts) -> io::Result<String> {
    let cat = Catalog::open(root)?;
    let id = catalog::new_build_id();
    let dir = cat.begin_build(&id, pbf, "car", o.simplify_m)?;
    let paths = Paths::new(&dir);
    let t0 = Instant::now();
    eprintln!(
        "[build/{id}] staging in {} — the live dataset is untouched until this finishes",
        dir.display()
    );

    pass(&cat, &id, "scan", || {
        let dir = build::open_dir(pbf, &paths)?;
        let mut st = build::pass1(pbf, &paths, &dir)?;
        let (road, need, junc) = build::finish_pass1(&mut st);
        cat.stat(&id, "road_nodes", road)?;
        cat.stat(&id, "coords", need)?;
        let p2 = build::pass2(pbf, &paths, &dir, &mut st)?;
        st.junc.build_rank();
        cat.stat(&id, "vertices", st.junc.total())?;
        let _ = (junc, p2);
        st.junc.save(&paths.f("junc.bm"))?;
        st.need.save(&paths.f("need.bm"))?;
        Ok(())
    })?;

    pass(&cat, &id, "contract", || {
        let (mut need, mut junc) =
            (BitMap::load(&paths.f("need.bm"))?, BitMap::load(&paths.f("junc.bm"))?);
        need.build_rank();
        junc.build_rank();
        let out = contract::pass3(&paths, &need, &junc, &ContractCfg { simplify_m: o.simplify_m })?;
        cat.stat(&id, "segments", out.segments)?;
        cat.stat(&id, "edges", out.directed_edges)?;
        cat.stat(&id, "street_names", out.names)?;
        cat.stat(&id, "geometry_points", out.geom_points)?;
        cat.stat(&id, "shape_points_seen", out.raw_points)?;
        Ok(())
    })?;

    pass(&cat, &id, "graph", || {
        let (mut need, mut junc) =
            (BitMap::load(&paths.f("need.bm"))?, BitMap::load(&paths.f("junc.bm"))?);
        need.build_rank();
        junc.build_rank();
        let g = graph::pass4(&paths, &need, &junc)?;
        cat.stat(&id, "snap_cells", g.snap_entries)?;
        Ok(())
    })?;

    pass(&cat, &id, "addr", || {
        let mut need = BitMap::load(&paths.f("need.bm"))?;
        need.build_rank();
        let a = addresses::pass5(&paths, &need)?;
        cat.stat(&id, "addresses", a.points)?;
        cat.stat(&id, "address_streets", a.streets)?;
        cat.stat(&id, "places", a.places)?;
        Ok(())
    })?;

    pass(&cat, &id, "toll", || {
        let mut junc = BitMap::load(&paths.f("junc.bm"))?;
        junc.build_rank();
        let n = toll::pass6(&paths, &junc)?;
        cat.stat(&id, "toll_gates", n)?;
        Ok(())
    })?;

    pass(&cat, &id, "tolldb", || {
        let idx = toll::TollIndex::open(&dir)?;
        let st = tolldb::ingest(&dir, &idx.gates, &idx.vertex, o.nvdb.as_deref())
            .map_err(|e| io::Error::other(e.to_string()))?;
        cat.stat(&id, "toll_rates", st.rates)?;
        cat.stat(&id, "toll_schemes", st.schemes)?;
        cat.stat(&id, "toll_rush_windows", st.rush_windows)?;
        Ok(())
    })?;

    pass(&cat, &id, "overlay", || {
        if !o.overlay {
            eprintln!("[build/{id}] overlay skipped (--no-overlay)");
            return Ok(());
        }
        // Needs the finished dataset, so it runs last.
        let ds = Dataset::open(&dir)?;
        let st = overlay::build(&paths, &ds, overlay::TARGET)?;
        cat.stat(&id, "overlay_regions", st.cells as u64)?;
        cat.stat(&id, "overlay_boundary", st.boundary as u64)?;
        cat.stat(&id, "overlay_entries", st.matrix_entries)?;
        Ok(())
    })?;

    // Structural check before anything becomes visible. A build that got this
    // far but is internally inconsistent is a bug we want to hear about now,
    // not from a router answering nonsense.
    let problems = dataset::verify(&dir);
    if !problems.is_empty() {
        for p in &problems {
            eprintln!("[build/{id}] INCONSISTENT {}: {}", p.file, p.detail);
        }
        cat.fail_build(&id, &format!("{} structural problems", problems.len()))?;
        return Err(io::Error::other(format!(
            "build {id} is internally inconsistent and was not published"
        )));
    }

    if !o.keep_scratch {
        prune_scratch(&dir)?;
    }
    let n = cat.record_artifacts(&id, &dir, o.deep_digest)?;
    cat.publish(&id)?;
    eprintln!(
        "[build/{id}] published in {:.1} min — {n} artifacts, verified consistent",
        t0.elapsed().as_secs_f64() / 60.0
    );
    Ok(id)
}

/// Delete the intermediates. They are ~44 GB on a planet and are dead the
/// moment the dataset is published.
pub fn prune_scratch(dir: &Path) -> io::Result<u64> {
    let mut freed = 0u64;
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        if dataset::is_runtime_artifact(&name) || name == "HEAD" {
            continue;
        }
        let Ok(md) = e.metadata() else { continue };
        if md.is_file() {
            freed += md.len();
            std::fs::remove_file(e.path())?;
        }
    }
    Ok(freed)
}

/// Delete build directories that are not live, newest kept first.
///
/// A dead build is kept on purpose — it is the evidence of what went wrong —
/// but not forever: a planet build is 38 GB published and another 44 GB of
/// intermediates, so "keep everything" is not an operating policy.
pub fn prune(root: &Path, keep: usize) -> io::Result<(usize, u64)> {
    let cat = Catalog::open(root)?;
    let live = cat.head().unwrap_or_default();
    let builds = root.join("builds");
    if !builds.is_dir() {
        return Ok((0, 0));
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&builds)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    // Build ids sort chronologically by construction.
    dirs.sort();
    dirs.reverse();
    let mut kept = 0usize;
    let (mut removed, mut freed) = (0usize, 0u64);
    for d in dirs {
        let name = d.file_name().unwrap_or_default().to_string_lossy().to_string();
        if name == live {
            continue; // never the live one, whatever the count says
        }
        if kept < keep {
            kept += 1;
            continue;
        }
        freed += dir_bytes(&d);
        std::fs::remove_dir_all(&d)?;
        removed += 1;
    }
    Ok((removed, freed))
}

fn dir_bytes(d: &Path) -> u64 {
    std::fs::read_dir(d)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Recompute the statistics a dataset implies about itself, for the live
/// build. Useful after adopting with an older binary, or whenever the
/// catalogue's derived facts have fallen behind the code that derives them.
pub fn restat(root: &Path) -> io::Result<u64> {
    let cat = Catalog::open(root)?;
    let Some(id) = cat.head() else {
        return Err(io::Error::other("no live build to restate"));
    };
    let live = catalog::resolve(root);
    let Some(c) = dataset::counts(&live) else {
        return Err(io::Error::other("the live dataset has no CSR to measure"));
    };
    for (k, v) in [
        ("vertices", c.vertices),
        ("edges", c.edges),
        ("segments", c.segments),
        ("addresses", c.addresses),
        ("geometry_points", c.geometry_points),
        ("street_names", c.street_names),
    ] {
        cat.stat(&id, k, v)?;
    }
    cat.record_artifacts(&id, &live, false)
}

/// Bring a dataset that predates the catalogue under management, without
/// rebuilding it.
///
/// The planet took 55 minutes to produce; adopting it is a directory move and
/// a few rows, and it is what makes the catalogue adoptable rather than a
/// reason to start over.
pub fn adopt(root: &Path, source: &Path, simplify_m: f64) -> io::Result<String> {
    let problems = dataset::verify(root);
    if !problems.is_empty() {
        for p in &problems {
            eprintln!("[adopt] INCONSISTENT {}: {}", p.file, p.detail);
        }
        return Err(io::Error::other("refusing to adopt an inconsistent dataset"));
    }
    let cat = Catalog::open(root)?;
    let id = catalog::new_build_id();
    let dir = cat.begin_build(&id, source, "car", simplify_m)?;
    // Move, do not copy: the files are tens of gigabytes and a rename within
    // one filesystem is free.
    let mut moved = 0u64;
    for e in std::fs::read_dir(root)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        if !e.metadata().map(|m| m.is_file()).unwrap_or(false) {
            continue;
        }
        if name == "HEAD" || name == "catalog.mpedb" || name.starts_with("catalog.mpedb") {
            continue;
        }
        std::fs::rename(e.path(), dir.join(&name))?;
        moved += 1;
    }
    for p in catalog::PASSES {
        cat.pass_start(&id, p)?;
        cat.pass_end(&id, p, true, 0.0)?;
    }
    // Statistics derived from the arrays themselves, not copied from a log.
    if let Some(c) = dataset::counts(&dir) {
        cat.stat(&id, "vertices", c.vertices)?;
        cat.stat(&id, "edges", c.edges)?;
        cat.stat(&id, "segments", c.segments)?;
        cat.stat(&id, "addresses", c.addresses)?;
        cat.stat(&id, "geometry_points", c.geometry_points)?;
        cat.stat(&id, "street_names", c.street_names)?;
    }
    let n = cat.record_artifacts(&id, &dir, false)?;
    cat.publish(&id)?;
    eprintln!("[adopt] {moved} files moved into build {id}; {n} artifacts recorded");
    Ok(id)
}
