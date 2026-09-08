//! The tariff layer's behaviour, held to cases a flat file got wrong.
//!
//! Each test here corresponds to something the previous `toll.tariff.tsv`
//! either could not express or answered incorrectly:
//!
//!   * one price per gate, so a large vehicle was quoted the car rate;
//!   * no time dimension, so rush hour was invisible;
//!   * `.unwrap_or(0.0)` on the price column, so a typo became a free toll;
//!   * an hour rule approximated per *route* rather than per *scheme*.

use mpee_planet::toll::Gate;
use mpee_planet::tolldb::{self, TollDb, RATE_RUSH, VEHICLE_CAR, VEHICLE_HGV};
use std::path::PathBuf;

/// A throwaway directory that survives a panicking test.
struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tmp(name: &str) -> Tmp {
    let p = std::env::temp_dir().join(format!("mpee-toll-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    Tmp(p)
}

fn gate(osm_id: u64, name: &str, lat: f64, lon: f64, charge: &str) -> Gate {
    Gate {
        vertex: osm_id as u32,
        osm_id,
        lat_e7: (lat * 1e7) as i32,
        lon_e7: (lon * 1e7) as i32,
        name: name.into(),
        operator: String::new(),
        charge: charge.into(),
        direction: String::new(),
    }
}

/// One NVDB station object, shaped exactly like the live API returns them.
// A tariff row is genuinely this wide — name, scheme, position, class prices
// and a rush window. Bundling them into a struct would only move the same
// arguments behind a literal, and make each call site longer to read.
#[allow(clippy::too_many_arguments)]
fn station(
    name: &str,
    scheme: &str,
    lat: f64,
    lon: f64,
    car: f64,
    car_rush: Option<f64>,
    hgv: f64,
    hgv_rush: Option<f64>,
    hour_rule: i64,
    rush_from: Option<&str>,
    rush_to: Option<&str>,
) -> serde_json::Value {
    let mut props = vec![
        serde_json::json!({"navn": "Navn bomstasjon", "verdi": name}),
        serde_json::json!({"navn": "Navn bompengeanlegg", "verdi": scheme}),
        serde_json::json!({"navn": "Takst liten bil", "verdi": car}),
        serde_json::json!({"navn": "Takst stor bil", "verdi": hgv}),
        serde_json::json!({"navn": "Timesregel, varighet", "verdi": hour_rule}),
    ];
    if let Some(v) = car_rush {
        props.push(serde_json::json!({"navn": "Rushtidstakst liten bil", "verdi": v}));
    }
    if let Some(v) = hgv_rush {
        props.push(serde_json::json!({"navn": "Rushtidstakst stor bil", "verdi": v}));
    }
    if let (Some(a), Some(b)) = (rush_from, rush_to) {
        props.push(serde_json::json!({"navn": "Rushtid morgen, fra", "verdi": a}));
        props.push(serde_json::json!({"navn": "Rushtid morgen, til", "verdi": b}));
    }
    serde_json::json!({
        "egenskaper": props,
        "geometri": {"wkt": format!("POINT Z ({lat} {lon} 0)")}
    })
}

/// Two gates in one toll ring plus one standalone gate on the motorway.
fn fixture(dir: &std::path::Path) -> Vec<Gate> {
    let gates = vec![
        gate(1, "Ring West", 59.90, 10.70, ""),
        gate(2, "Ring East", 59.91, 10.75, ""),
        gate(3, "Motorway North", 60.50, 10.90, ""),
        gate(4, "Untagged", 61.00, 11.00, ""),
    ];
    let nvdb = serde_json::json!([
        // Same scheme, 60-minute hour rule; East is the dearer passage.
        station("Ring West", "City Ring", 59.90, 10.70, 40.0, Some(70.0), 90.0, Some(160.0), 60,
                Some("06:30"), Some("08:59")),
        station("Ring East", "City Ring", 59.91, 10.75, 45.0, Some(80.0), 100.0, Some(180.0), 60,
                Some("06:30"), Some("08:59")),
        // Outside any ring: no rush rate, no hour rule.
        station("Motorway North", "Motorway Project", 60.50, 10.90, 30.0, None, 75.0, None, 0,
                None, None),
    ]);
    let p = dir.join("nvdb.json");
    std::fs::write(&p, serde_json::to_vec(&nvdb).unwrap()).unwrap();
    let vertices: Vec<u32> = gates.iter().map(|g| g.vertex).collect();
    tolldb::ingest(dir, &gates, &vertices, Some(&p)).unwrap();
    gates
}

#[test]
fn vehicle_class_selects_its_own_rate() {
    let t = tmp("class");
    fixture(&t.0);
    let db = TollDb::open(&t.0).unwrap();
    let car = db.quote(&[3], VEHICLE_CAR, None);
    let hgv = db.quote(&[3], VEHICLE_HGV, None);
    assert_eq!(car.totals, vec![("NOK".to_string(), 30.0)]);
    assert_eq!(hgv.totals, vec![("NOK".to_string(), 75.0)], "a lorry is not a car");
}

#[test]
fn rush_rate_applies_only_inside_the_window() {
    let t = tmp("rush");
    fixture(&t.0);
    let db = TollDb::open(&t.0).unwrap();
    let inside = db.quote(&[1], VEHICLE_CAR, Some(tolldb::time_us(7, 15)));
    let edge = db.quote(&[1], VEHICLE_CAR, Some(tolldb::time_us(8, 59)));
    let outside = db.quote(&[1], VEHICLE_CAR, Some(tolldb::time_us(9, 30)));
    let none = db.quote(&[1], VEHICLE_CAR, None);
    assert_eq!(inside.totals[0].1, 70.0, "07:15 is inside 06:30–08:59");
    assert_eq!(edge.totals[0].1, 70.0, "the closing minute is still rush");
    assert_eq!(outside.totals[0].1, 40.0, "09:30 is not");
    assert_eq!(none.totals[0].1, 40.0, "no time given ⇒ standard rate");
    assert_eq!(inside.lines[0].rate_kind, RATE_RUSH);
}

#[test]
fn a_gate_without_a_rush_rate_falls_back_to_standard() {
    let t = tmp("fallback");
    fixture(&t.0);
    let db = TollDb::open(&t.0).unwrap();
    // Motorway North has no rush rate at all; asking at 07:15 must not lose it.
    let q = db.quote(&[3], VEHICLE_CAR, Some(tolldb::time_us(7, 15)));
    assert_eq!(q.totals, vec![("NOK".to_string(), 30.0)]);
    assert_eq!(q.unpriced, 0);
}

#[test]
fn the_hour_rule_charges_the_dearest_passage_once_per_scheme() {
    let t = tmp("hourrule");
    fixture(&t.0);
    let db = TollDb::open(&t.0).unwrap();
    // Both ring gates plus the standalone one: 45 (dearest in the ring) + 30.
    let q = db.quote(&[1, 2, 3], VEHICLE_CAR, None);
    assert_eq!(q.totals, vec![("NOK".to_string(), 75.0)]);
    // Order must not change the total — the rule is "once per scheme", not
    // "the first one wins".
    let rev = db.quote(&[2, 1, 3], VEHICLE_CAR, None);
    assert_eq!(rev.totals, q.totals);
    // …and exactly one of the two ring passages is reported as charged.
    let ring_charged = q.lines.iter().filter(|l| l.scheme == "City Ring" && l.charged).count();
    assert_eq!(ring_charged, 1);
}

#[test]
fn the_hour_rule_does_not_leak_between_schemes() {
    let t = tmp("schemes");
    fixture(&t.0);
    let db = TollDb::open(&t.0).unwrap();
    // Motorway North shares no scheme with the ring, so it is always charged.
    let q = db.quote(&[2, 3], VEHICLE_CAR, None);
    assert_eq!(q.totals, vec![("NOK".to_string(), 75.0)]);
    assert!(q.lines.iter().all(|l| l.charged));
}

#[test]
fn gates_without_a_tariff_are_reported_not_priced_at_zero() {
    let t = tmp("unpriced");
    fixture(&t.0);
    let db = TollDb::open(&t.0).unwrap();
    // Gate 4 has neither an NVDB match nor an OSM charge tag. The old parser
    // ended in `.unwrap_or(0.0)` and would have called it free.
    let q = db.quote(&[4], VEHICLE_CAR, None);
    assert_eq!(q.unpriced, 1, "an unknown tariff must be visible, not zero");
    assert!(q.lines.is_empty());
    assert!(q.totals.is_empty(), "no currency total is invented");
}

#[test]
fn an_osm_charge_tag_is_used_where_nvdb_has_nothing() {
    let t = tmp("osmtag");
    let gates = vec![gate(9, "Bridge", 45.0, 9.0, "2.50 EUR")];
    let vertices = vec![9u32];
    tolldb::ingest(&t.0, &gates, &vertices, None).unwrap();
    let db = TollDb::open(&t.0).unwrap();
    let q = db.quote(&[9], VEHICLE_CAR, None);
    assert_eq!(q.totals, vec![("EUR".to_string(), 2.5)]);
    assert_eq!(q.lines[0].price, 2.5);
}

#[test]
fn prices_are_exact_decimals_not_floats() {
    let t = tmp("exact");
    let gates = vec![gate(7, "Cents", 45.0, 9.0, "0.10 EUR")];
    tolldb::ingest(&t.0, &gates, &[7u32], None).unwrap();
    let db = TollDb::open(&t.0).unwrap();
    // Ten 0.10 passages must be exactly 1.00. Summed as f32/f64 accumulators
    // over a decimal literal this is where cent drift shows up.
    let q = db.quote(&[7; 10], VEHICLE_CAR, None);
    let stored = db
        .query("SELECT price FROM toll_rate WHERE osm_id = 7")
        .unwrap();
    if let mpedb::ExecResult::Rows { rows, .. } = stored {
        assert_eq!(
            rows[0][0],
            mpedb::Value::Numeric("0.10".into()),
            "the price is stored as an exact decimal, not a float"
        );
    } else {
        panic!("expected rows");
    }
    assert!((q.totals[0].1 - 1.0).abs() < 1e-9);
}
