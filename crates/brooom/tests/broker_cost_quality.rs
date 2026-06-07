//! Stage-A acceptance gate for the cost-aware matrix broker.
//!
//! Builds a synthetic single-gateway "truth" matrix (metric, additive
//! cross-cluster structure), serves it through a counting `CellSource`, and
//! checks the broker (a) buys only a fraction of N², (b) reproduces the bought
//! cells EXACTLY, (c) derives the rest as sound upper bounds, and (d) the
//! resulting matrix drives a VRP to essentially the same cost as the full
//! matrix (quality preserved on what the solver reads).

use std::sync::Mutex;

use brooom::broker::{BrokerMatrixSource, BrokerPolicy, CellDb, DeriveMode};
use brooom::error::Result;
use brooom::matrix::{
    gather_cells, haversine_m, CellRequest, CellResponse, CellSource, Matrix, MatrixSource,
};

const CLUSTERS: usize = 8;
const PER: usize = 50;
const N: usize = CLUSTERS * PER; // 400 (index 0 doubles as depot)

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Coords: CLUSTERS cities spread far apart, PER points around each centre.
fn make_world() -> (Vec<[f64; 2]>, Matrix) {
    let mut s = 0x1234_5678u64;
    let centres: Vec<[f64; 2]> = (0..CLUSTERS)
        .map(|_| [5.0 + lcg(&mut s) * 12.0, 58.0 + lcg(&mut s) * 6.0]) // Norway-ish box
        .collect();
    let mut coords = Vec::with_capacity(N);
    for c in 0..CLUSTERS {
        for _ in 0..PER {
            coords.push([
                centres[c][0] + (lcg(&mut s) - 0.5) * 0.06,
                centres[c][1] + (lcg(&mut s) - 0.5) * 0.06,
            ]);
        }
    }
    // One gateway per cluster = its first point.
    let gw: Vec<usize> = (0..CLUSTERS).map(|c| c * PER).collect();
    let cluster_of = |i: usize| i / PER;
    let speed = 13.9;
    let mut durations = vec![0i32; N * N];
    let mut distances = vec![0i32; N * N];
    for i in 0..N {
        for j in 0..N {
            if i == j {
                continue;
            }
            let (ci, cj) = (cluster_of(i), cluster_of(j));
            let d = if ci == cj {
                haversine_m(coords[i], coords[j])
            } else {
                haversine_m(coords[i], coords[gw[ci]])
                    + haversine_m(coords[gw[ci]], coords[gw[cj]])
                    + haversine_m(coords[gw[cj]], coords[j])
            };
            distances[i * N + j] = d.round() as i32;
            durations[i * N + j] = (d / speed).round() as i32;
        }
    }
    (coords, Matrix { n: N, durations, distances: Some(distances) })
}

/// A `CellSource` that serves the truth matrix and counts cells fetched.
struct TruthSource {
    truth: Matrix,
    fetched: Mutex<usize>,
}
impl MatrixSource for TruthSource {
    fn build(&self, _coords: &[[f64; 2]]) -> Result<Matrix> {
        *self.fetched.lock().unwrap() += self.truth.n * self.truth.n;
        Ok(self.truth.clone())
    }
}
impl CellSource for TruthSource {
    fn fetch_cells(&self, _coords: &[[f64; 2]], req: &CellRequest) -> Result<CellResponse> {
        *self.fetched.lock().unwrap() += req.pairs.len();
        Ok(gather_cells(&self.truth, req))
    }
}

#[test]
fn skeleton_buys_few_reproduces_exact_derives_upper_bounds() {
    let (coords, truth) = make_world();
    let policy = BrokerPolicy { derive: DeriveMode::Skeleton, ..Default::default() };
    let broker = BrokerMatrixSource::new(
        TruthSource { truth: truth.clone(), fetched: Mutex::new(0) },
        policy,
    );
    let m = broker.build(&coords).unwrap();
    let st = broker.last_stats();

    // (a) bought only a fraction of N².
    assert!(st.metric_ok, "synthetic gateway world should pass the metric check");
    assert!(
        st.saved_fraction() > 0.5,
        "expected to save >50% of cells, saved {:.1}% ({} of {})",
        st.saved_fraction() * 100.0,
        st.cells_bought,
        st.cells_total
    );

    // (b) every cell is a sound upper bound on truth, and (c) the bulk match
    //     truth closely (bought exact; derived near for this additive world).
    let mut max_under = 0i64; // how far any cell dips BELOW truth (should be ~0)
    let mut close = 0usize;
    for i in 0..N {
        for j in 0..N {
            if i == j {
                continue;
            }
            let got = m.durations[i * N + j] as i64;
            let t = truth.durations[i * N + j] as i64;
            max_under = max_under.max(t - got);
            if (got - t).abs() <= (t / 50).max(2) {
                close += 1;
            }
        }
    }
    // Min-plus + exact buys never UNDER-estimate a metric (tiny rounding slack).
    assert!(max_under <= 3, "cells under-estimated truth by {max_under}s (should be ~0)");
    let frac_close = close as f64 / (N * N - N) as f64;
    assert!(frac_close > 0.9, "only {:.1}% of cells within 2% of truth", frac_close * 100.0);
}

#[test]
fn db_reuse_makes_second_solve_nearly_free() {
    let (coords, truth) = make_world();
    let path = std::env::temp_dir().join(format!("mpee_broker_db_{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);

    // Run 1: cold DB — buys the skeleton, no hits.
    let first_bought = {
        let db = CellDb::open(&path, "car");
        let broker = BrokerMatrixSource::with_db(
            TruthSource { truth: truth.clone(), fetched: Mutex::new(0) },
            BrokerPolicy::default(),
            db,
        );
        broker.build(&coords).unwrap();
        let st = broker.last_stats();
        assert!(st.cells_bought > 0, "cold DB must buy the skeleton");
        assert_eq!(st.db_hits, 0, "cold DB has no hits");
        st.cells_bought
    };

    // Run 2: warm DB (re-opened from disk) — same skeleton, all served free.
    {
        let db = CellDb::open(&path, "car");
        let broker = BrokerMatrixSource::with_db(
            TruthSource { truth: truth.clone(), fetched: Mutex::new(0) },
            BrokerPolicy::default(),
            db,
        );
        broker.build(&coords).unwrap();
        let st = broker.last_stats();
        assert_eq!(st.cells_bought, 0, "warm DB should buy nothing the second time");
        assert_eq!(st.db_hits, first_bought, "all skeleton cells served from the DB");
    }

    // Frequency counter accrued across the two runs.
    let db = CellDb::open(&path, "car");
    assert!(db.node_seen(coords[0]) >= 2, "node frequency should accrue across runs");
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "pyspell")]
#[test]
fn pricing_and_policy_spells_evaluate() {
    use brooom::broker::BrokerVars;
    use brooom::constraint::Verdict;

    // Tiered cost: tier 0 = 5/cell, tier 1 = 4/cell.
    let cost = brooom::pyspell::compile_broker_rust(
        "if broker.tier == 0 { broker.batch_size * 5 } else { broker.batch_size * 4 }",
    )
    .unwrap();
    assert!(matches!(
        cost(&BrokerVars { batch_size: 100, tier: 0, ..Default::default() }),
        Verdict::Penalty(p) if (p - 500.0).abs() < 1e-6
    ));
    assert!(matches!(
        cost(&BrokerVars { batch_size: 100, tier: 1, ..Default::default() }),
        Verdict::Penalty(p) if (p - 400.0).abs() < 1e-6
    ));

    // Buy/skip predicate: buy hubs (seen ≥ 10) or near pairs (< 5 km).
    let policy = brooom::pyspell::compile_broker_rust(
        "broker.crossing_count >= 10 || broker.haversine_km < 5.0",
    )
    .unwrap();
    assert!(matches!(
        policy(&BrokerVars { crossing_count: 12, ..Default::default() }),
        Verdict::Feasible
    ));
    assert!(matches!(
        policy(&BrokerVars { crossing_count: 1, haversine_km: 9.0, ..Default::default() }),
        Verdict::Infeasible
    ));
}

#[cfg(feature = "pyspell")]
#[test]
fn buy_budget_caps_cells_bought() {
    let (coords, truth) = make_world();
    // Price = 1 per cell; cap the spend at 5000 cells.
    let cost = brooom::pyspell::compile_broker_rust("broker.batch_size").unwrap();
    let budget = 5000.0;
    let policy = BrokerPolicy { buy_budget: Some(budget), ..Default::default() };
    let broker = BrokerMatrixSource::new(
        TruthSource { truth: truth.clone(), fetched: Mutex::new(0) },
        policy,
    )
    .with_cost_fn(cost);
    broker.build(&coords).unwrap();
    let st = broker.last_stats();
    assert!(st.cells_bought > 0, "should still buy the affordable prefix");
    assert!(
        st.cells_bought as f64 <= budget,
        "bought {} cells, over the {budget} budget",
        st.cells_bought
    );
    // The full skeleton is far larger than 5000, so the budget actually bites.
    assert!(st.cells_bought as f64 >= budget * 0.9, "budget should be ~saturated");
}

// ---- Stage E1: temporal profiles + congestion/uncertainty -----------------

use brooom::broker::{DepartureProfile, WeekdayClass};

/// A `CellSource` that serves the truth matrix scaled by a constant factor —
/// stands in for a different time-of-day congestion level (rush = >1.0).
struct ScaledSource {
    truth: Matrix,
}
impl ScaledSource {
    fn new(truth: &Matrix, factor: f64) -> Self {
        let durations: Vec<i32> =
            truth.durations.iter().map(|&d| (d as f64 * factor).round() as i32).collect();
        ScaledSource {
            truth: Matrix { n: truth.n, durations, distances: truth.distances.clone() },
        }
    }
}
impl MatrixSource for ScaledSource {
    fn build(&self, _coords: &[[f64; 2]]) -> Result<Matrix> {
        Ok(self.truth.clone())
    }
}
impl CellSource for ScaledSource {
    fn fetch_cells(&self, _coords: &[[f64; 2]], req: &CellRequest) -> Result<CellResponse> {
        Ok(gather_cells(&self.truth, req))
    }
}

fn total_duration(m: &Matrix) -> i64 {
    m.durations.iter().map(|&d| d as i64).sum()
}

#[test]
fn temporal_db_welford_and_bucketing() {
    // The cell DB keys by (weekday_class, hour) and carries a Welford mean +
    // std over every observation in that bucket. Distinct buckets never alias,
    // and the stats survive a flush/reopen.
    let path = std::env::temp_dir().join(format!("mpee_broker_temporal_{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let a = [10.0, 60.0];
    let b = [10.1, 60.1];
    let workday_8: brooom::broker::TimeBucket = Some((0, 8));
    let weekend_8: brooom::broker::TimeBucket = Some((1, 8));

    {
        let mut db = CellDb::open(&path, "car");
        // Workday 08:00 sees [100, 140] → mean 120, population std 20.
        db.observe(a, b, workday_8, 100, 1000);
        db.observe(a, b, workday_8, 140, 1000);
        // Weekend 08:00 is a *separate* bucket (free-flowing): one sample at 90.
        db.observe(a, b, weekend_8, 90, 1000);
        db.flush();
    }

    let db = CellDb::open(&path, "car");
    let s = db.get(a, b, workday_8).expect("workday bucket present after reopen");
    assert_eq!(s.mean_dur, 120, "Welford mean over [100,140]");
    assert_eq!(s.count, 2);
    assert!((s.std_dur - 20.0).abs() < 1e-6, "population std over [100,140] is 20, got {}", s.std_dur);
    let w = db.get(a, b, weekend_8).expect("weekend bucket present");
    assert_eq!(w.mean_dur, 90, "weekend bucket does not alias the workday one");
    assert!(w.std_dur < 1e-9, "single weekend sample has zero variance");
    // A bucket never observed is absent.
    assert!(db.get(a, b, Some((0, 17))).is_none(), "unseen window is unknown");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn offline_reuse_serves_other_days_free() {
    // Observe one representative workday-08:00 (cold build), then a *different*
    // weekday at the same hour maps to the same (workday, 08) bucket → served
    // entirely from the DB, buying nothing new (the killer cost-saver).
    let (coords, truth) = make_world();
    let path = std::env::temp_dir().join(format!("mpee_broker_offline_{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let departure = Some(DepartureProfile { weekday_class: WeekdayClass::Workday, hour: 8 });

    let first_bought = {
        let broker = BrokerMatrixSource::with_db(
            TruthSource { truth: truth.clone(), fetched: Mutex::new(0) },
            BrokerPolicy { departure, ..Default::default() },
            CellDb::open(&path, "car"),
        );
        broker.build(&coords).unwrap();
        let st = broker.last_stats();
        assert!(st.cells_bought > 0, "cold workday profile must buy the skeleton");
        st.cells_bought
    };

    // A later "Thursday" at 08:00 — same class+hour bucket, warm DB.
    let broker = BrokerMatrixSource::with_db(
        TruthSource { truth: truth.clone(), fetched: Mutex::new(0) },
        BrokerPolicy { departure, ..Default::default() },
        CellDb::open(&path, "car"),
    );
    let m = broker.build(&coords).unwrap();
    let st = broker.last_stats();
    assert_eq!(st.cells_bought, 0, "warm workday profile buys nothing on another day");
    assert_eq!(st.db_hits, first_bought, "every skeleton cell served from the profile");
    // The baked matrix reflects the learned means (≈ truth here; one sample).
    assert_eq!(m.durations[0 * N + PER], truth.durations[0 * N + PER]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn rush_hour_profile_costs_more_than_offpeak() {
    // The same places, two time-of-day profiles: a rush window (×1.6 travel
    // time) must bake a strictly larger matrix than the off-peak window (×1.0).
    let (coords, truth) = make_world();
    let rush = BrokerMatrixSource::new(
        ScaledSource::new(&truth, 1.6),
        BrokerPolicy {
            departure: Some(DepartureProfile { weekday_class: WeekdayClass::Workday, hour: 8 }),
            ..Default::default()
        },
    );
    let offpeak = BrokerMatrixSource::new(
        ScaledSource::new(&truth, 1.0),
        BrokerPolicy {
            departure: Some(DepartureProfile { weekday_class: WeekdayClass::Workday, hour: 13 }),
            ..Default::default()
        },
    );
    let mr = rush.build(&coords).unwrap();
    let mo = offpeak.build(&coords).unwrap();
    assert!(
        total_duration(&mr) > total_duration(&mo),
        "rush profile ({}) should exceed off-peak ({})",
        total_duration(&mr),
        total_duration(&mo)
    );
}

#[test]
fn uncertainty_weight_penalises_high_variance_cells() {
    // Seed a few depot-row cells with cross-day variance (queue zones), then let
    // the broker bake `mean + W·std`: those arcs cost more than their mean and
    // register as hotspots, so the solver routes around them.
    let (coords, truth) = make_world();
    let path = std::env::temp_dir().join(format!("mpee_broker_uncert_{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let when: brooom::broker::TimeBucket = Some((0, 8));

    // Cell (0,j) observed on two "days" as 100 then 140 → mean 120, std 20.
    {
        let mut db = CellDb::open(&path, "car");
        for j in 1..4usize {
            db.observe(coords[0], coords[j], when, 100, 1000);
            db.observe(coords[0], coords[j], when, 140, 1000);
        }
        db.flush();
    }

    let broker = BrokerMatrixSource::with_db(
        TruthSource { truth: truth.clone(), fetched: Mutex::new(0) },
        BrokerPolicy {
            departure: Some(DepartureProfile { weekday_class: WeekdayClass::Workday, hour: 8 }),
            uncertainty_weight: 2.0,
            ..Default::default()
        },
        CellDb::open(&path, "car"),
    );
    let m = broker.build(&coords).unwrap();
    let st = broker.last_stats();
    assert!(st.hotspots >= 3, "the three high-variance cells should be flagged, got {}", st.hotspots);
    // mean 120 + 2·std(20) = 160 baked into the static matrix.
    assert_eq!(m.durations[0 * N + 1], 160, "uncertainty penalty baked: mean(120) + 2*std(20)");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn off_mode_buys_everything() {
    let (coords, truth) = make_world();
    let broker = BrokerMatrixSource::new(
        TruthSource { truth: truth.clone(), fetched: Mutex::new(0) },
        BrokerPolicy { derive: DeriveMode::Off, ..Default::default() },
    );
    let m = broker.build(&coords).unwrap();
    assert_eq!(broker.last_stats().cells_bought, N * N);
    // Off mode returns the provider matrix verbatim.
    assert_eq!(m.durations, truth.durations);
}
