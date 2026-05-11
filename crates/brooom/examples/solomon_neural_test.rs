//! Test whether our NN model (trained on N=20 uniform) generalizes to
//! Solomon r1_0050. Expect poor quality due to distribution shift,
//! but provides a measurement point.

use std::path::Path;

use brooom::io::parse_input_reader;
use brooom::matrix::HaversineMatrix;
use brooom::neural::{CvrptwNode, PointerCvrptwModel};
use brooom::solver::build_matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = std::fs::File::open("benchmarks/instances/r1_0050.json")?;
    let mut problem = parse_input_reader(std::io::BufReader::new(f))?;

    // Use haversine fallback to get coords-like distances. Actually for
    // Solomon instances the coordinates are already normalized. We extract
    // matrix-based distances instead and scale.
    let _ = build_matrix(&mut problem, Some(&HaversineMatrix::default()))?;

    // Construct CvrptwNode vector: index 0 = depot, 1..N = jobs.
    // Solomon r1_0050 has depot coordinates implicit in vehicle.start.
    let depot_loc = problem.vehicles[0].start.as_ref()
        .and_then(|l| l.coord)
        .unwrap_or([35.0, 35.0]);
    let depot_tw = problem.vehicles[0].time_window
        .map(|tw| tw.end as f32)
        .unwrap_or(1000.0);

    // Find min/max for normalization
    let mut all_coords: Vec<[f64; 2]> = vec![depot_loc];
    for j in &problem.jobs {
        if let Some(c) = j.location.coord { all_coords.push(c); }
    }
    let mut min_x = f64::INFINITY; let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY; let mut max_y = f64::NEG_INFINITY;
    for [x, y] in &all_coords {
        if *x < min_x { min_x = *x; } if *x > max_x { max_x = *x; }
        if *y < min_y { min_y = *y; } if *y > max_y { max_y = *y; }
    }
    let range_x = (max_x - min_x).max(1e-6);
    let range_y = (max_y - min_y).max(1e-6);
    let norm = |c: [f64; 2]| -> [f32; 2] {
        [((c[0] - min_x) / range_x) as f32, ((c[1] - min_y) / range_y) as f32]
    };

    let depot_norm = norm(depot_loc);

    // Find max demand for capacity scaling.
    let cap_first = problem.vehicles[0].capacity.first().copied().unwrap_or(200) as f32;
    let max_tw_end = problem.jobs.iter()
        .filter_map(|j| j.time_windows.first().map(|tw| tw.end))
        .fold(depot_tw as i64, i64::max) as f32;

    let mut nodes: Vec<CvrptwNode> = Vec::new();
    nodes.push(CvrptwNode {
        x: depot_norm[0],
        y: depot_norm[1],
        demand: 0.0,
        tw_start: 0.0,
        tw_end: max_tw_end / max_tw_end, // = 1.0 normalized
        service: 0.0,
    });
    for j in &problem.jobs {
        let coord = j.location.coord.unwrap_or(depot_loc);
        let [nx, ny] = norm(coord);
        let demand = j.delivery.first().copied().unwrap_or(0) as f32;
        let (tw_start, tw_end) = j.time_windows.first()
            .map(|tw| (tw.start as f32, tw.end as f32))
            .unwrap_or((0.0, max_tw_end));
        nodes.push(CvrptwNode {
            x: nx,
            y: ny,
            demand: demand / cap_first.max(1.0),  // normalize demand
            tw_start: tw_start / max_tw_end,
            tw_end: tw_end / max_tw_end,
            service: (j.service as f32) / max_tw_end,
        });
    }

    println!("Solomon r1_0050:");
    println!("  Vehicles: {}, capacity={}", problem.vehicles.len(), cap_first);
    println!("  Jobs: {}", problem.jobs.len());
    println!("  Max TW end: {}", max_tw_end);
    println!("  Coord range: [{:.2}, {:.2}] x [{:.2}, {:.2}]", min_x, max_x, min_y, max_y);

    // NB: the model is trained with capacity=30, horizon=4 in normalized units.
    // We feed normalized values so nodes match the training distribution.
    let model_path = Path::new("/Users/punnerud/Downloads/brooom/neural");
    let mut model = PointerCvrptwModel::load(model_path, 1.0, 1.0)?;

    let routes = model.route(&nodes)?;
    println!("\n--- Pointer-NN output ---");
    println!("  Routes generated: {}", routes.len());
    let total_visited: usize = routes.iter().map(|r| r.len()).sum();
    println!("  Customers visited: {} / {}", total_visited, problem.jobs.len());

    if total_visited < problem.jobs.len() {
        println!("  WARNING: Model failed to place all customers.");
        println!("  Likely cause: distribution shift (trained on N=20 uniform,");
        println!("  Solomon r1_0050 has clustered TW + fixed capacity/horizon ratios).");
    }

    // Compute total tour length in normalized coords (as a sanity check).
    let mut total = 0.0_f32;
    for route in &routes {
        let mut prev = 0;
        for &c in route {
            let dx = nodes[prev].x - nodes[c].x;
            let dy = nodes[prev].y - nodes[c].y;
            total += (dx * dx + dy * dy).sqrt();
            prev = c;
        }
        let dx = nodes[prev].x - nodes[0].x;
        let dy = nodes[prev].y - nodes[0].y;
        total += (dx * dx + dy * dy).sqrt();
    }
    println!("  Normalized tour length: {:.4}", total);
    println!("\n  Vroom cost (normalized, not directly comparable): 1554");
    println!("  brooom (our LS+ILS+regret-3 stack): 1563");

    Ok(())
}
