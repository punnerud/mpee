use brooom::neural::{CvrptwNode, PointerCvrptwModel};
use std::path::Path;

fn main() {
    let dir = Path::new("/Users/punnerud/Downloads/brooom/neural");
    let mut model = PointerCvrptwModel::load(dir, 30.0, 4.0)
        .expect("load CVRPTW model");

    // Generate the same kind of random instance the model was trained on:
    // depot at (0.5, 0.5), 20 customers uniform in [0,1]^2.
    let mut nodes: Vec<CvrptwNode> = Vec::new();
    nodes.push(CvrptwNode {
        x: 0.5, y: 0.5,
        demand: 0.0, tw_start: 0.0, tw_end: 4.0, service: 0.0,
    });
    let n_cust = 20;
    for i in 0..n_cust {
        let x = ((i * 13 + 7) % 97) as f32 / 100.0;
        let y = ((i * 17 + 23) % 89) as f32 / 100.0;
        let dx = x - 0.5;
        let dy = y - 0.5;
        let d_depot = (dx * dx + dy * dy).sqrt();
        let center = 2.0 * d_depot + 0.5;
        let width = 0.6;
        let demand = ((i % 9) + 1) as f32;
        nodes.push(CvrptwNode {
            x, y,
            demand,
            tw_start: (center - width / 2.0).max(0.0),
            tw_end: (center + width / 2.0).min(3.5),
            service: 0.1,
        });
    }

    let routes = model.route(&nodes).expect("route");
    println!("Trent CVRPTW pointer-NN på N=20:");
    println!("  Routes: {}", routes.len());
    let total_visited: usize = routes.iter().map(|r| r.len()).sum();
    println!("  Customers visited: {} / {}", total_visited, n_cust);

    // Compute total tour length (depot→r1[0]→...→r1[end]→depot→r2[0]...).
    let mut total = 0.0_f32;
    for route in &routes {
        let mut prev_idx = 0;
        for &cust in route {
            let dx = nodes[prev_idx].x - nodes[cust].x;
            let dy = nodes[prev_idx].y - nodes[cust].y;
            total += (dx * dx + dy * dy).sqrt();
            prev_idx = cust;
        }
        // Return to depot.
        let dx = nodes[prev_idx].x - nodes[0].x;
        let dy = nodes[prev_idx].y - nodes[0].y;
        total += (dx * dx + dy * dy).sqrt();
    }
    println!("  Total travel cost: {:.4}", total);

    // Print routes for inspection.
    for (i, r) in routes.iter().enumerate() {
        println!("  Route {}: {:?}", i, r);
    }
}
