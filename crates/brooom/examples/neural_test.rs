use brooom::neural::{PointerModel, tour_length, nearest_neighbor_tour};
use std::path::Path;

fn main() {
    let dir = Path::new("/Users/punnerud/Downloads/brooom/neural");
    let mut model = PointerModel::load(dir).expect("load model");

    // Random N=20 instans (matchet til treningen)
    let coords: Vec<[f32; 2]> = (0..20)
        .map(|i| {
            let x = ((i * 13 + 7) % 100) as f32 / 100.0;
            let y = ((i * 17 + 23) % 100) as f32 / 100.0;
            [x, y]
        })
        .collect();

    let nn_tour = nearest_neighbor_tour(&coords);
    let nn_cost = tour_length(&coords, &nn_tour);

    let nn_model_tour = model.route(&coords, true).expect("route");
    let nn_model_cost = tour_length(&coords, &nn_model_tour);

    println!("Nearest-neighbor tour cost : {:.4}", nn_cost);
    println!("Pointer-NN tour cost       : {:.4}", nn_model_cost);
    println!("Δ                          : {:+.1}%", (nn_model_cost - nn_cost) / nn_cost * 100.0);
}
