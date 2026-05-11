//! Neural Combinatorial Optimization warm-start via ONNX.
//!
//! Loads a pre-trained Pointer Network (encoder + decoder) and uses it to
//! generate a candidate route autoregressively — the "LLM-in-loop" pattern.
//! Encoder runs once over the full coord set; decoder runs N times, each
//! producing a logit distribution over remaining nodes, and we either
//! greedy-pick or sample.
//!
//! Training (PyTorch) happens out-of-band — see `neural/train_pointer_tsp.py`.
//! The Rust side is purely inference; it never sees training data.
//!
//! Current limitation: the proof-of-concept model is trained on N=20 random
//! 2D-TSP only. Real CVRPTW integration requires:
//!   1. Retraining with TW + capacity features per node
//!   2. Variable-N support (the encoder already supports it via dynamic axes;
//!      training data must include varied N)
//!   3. Multi-vehicle output (current model produces one tour over all nodes)

use std::path::Path;

use ndarray::{Array, Array2, Array3};
use ort::{session::Session, value::Tensor};

use crate::error::{Error, Result};

/// Loaded model handles. Encoder is run once per query; decoder is run
/// once per step in the autoregressive loop.
pub struct PointerModel {
    encoder: Session,
    decoder: Session,
    embed_dim: usize,
}

impl PointerModel {
    /// Load encoder + decoder from `dir/pointer_tsp_encoder.onnx` and
    /// `dir/pointer_tsp_decoder.onnx`.
    pub fn load(dir: &Path) -> Result<Self> {
        let enc_path = dir.join("pointer_tsp_encoder.onnx");
        let dec_path = dir.join("pointer_tsp_decoder.onnx");
        let encoder = Session::builder()
            .map_err(|e| Error::Other(format!("ort builder: {e}")))?
            .commit_from_file(&enc_path)
            .map_err(|e| Error::Other(format!("load encoder {}: {e}", enc_path.display())))?;
        let decoder = Session::builder()
            .map_err(|e| Error::Other(format!("ort builder: {e}")))?
            .commit_from_file(&dec_path)
            .map_err(|e| Error::Other(format!("load decoder {}: {e}", dec_path.display())))?;
        // Embed dim is hardcoded in training (64). Caller can override if a
        // differently-trained model is dropped in.
        Ok(Self { encoder, decoder, embed_dim: 64 })
    }

    /// Run the full pipeline on a set of normalized 2D coords. Returns a
    /// permutation (visit order). Coords should be in [0, 1]^2.
    pub fn route(&mut self, coords: &[[f32; 2]], greedy: bool) -> Result<Vec<usize>> {
        let n = coords.len();
        if n == 0 { return Ok(Vec::new()); }

        // Encoder: (1, N, 2) → (1, N, E)
        let mut coords_arr = Array3::<f32>::zeros((1, n, 2));
        for (i, c) in coords.iter().enumerate() {
            coords_arr[[0, i, 0]] = c[0];
            coords_arr[[0, i, 1]] = c[1];
        }
        let coords_tensor = Tensor::from_array(coords_arr)
            .map_err(|e| Error::Other(format!("coords tensor: {e}")))?;
        let enc_out = self.encoder.run(ort::inputs![coords_tensor])
            .map_err(|e| Error::Other(format!("encoder run: {e}")))?;
        let h_nodes = enc_out["h_nodes"].try_extract_array::<f32>()
            .map_err(|e| Error::Other(format!("h_nodes extract: {e}")))?;
        let h_nodes = h_nodes.as_standard_layout();
        let h_nodes_arr: Array3<f32> = h_nodes.to_owned().into_dimensionality()
            .map_err(|e| Error::Other(format!("h_nodes shape: {e}")))?;

        // Global graph embedding = mean over nodes.
        let mut h_graph = Array2::<f32>::zeros((1, self.embed_dim));
        for i in 0..n {
            for k in 0..self.embed_dim {
                h_graph[[0, k]] += h_nodes_arr[[0, i, k]];
            }
        }
        for k in 0..self.embed_dim { h_graph[[0, k]] /= n as f32; }

        // Autoregressive decode.
        let mut visited = vec![false; n];
        let mut tour: Vec<usize> = Vec::with_capacity(n);
        let mut last_emb = Array2::<f32>::zeros((1, self.embed_dim));

        for _step in 0..n {
            // Build mask: True where node is ALREADY VISITED (matches Python
            // export: mask_neg = mask.float() * 1e9, subtracted from logits).
            let mut mask_arr = Array2::<bool>::default((1, n));
            for (i, &v) in visited.iter().enumerate() { mask_arr[[0, i]] = v; }

            let h_nodes_t = Tensor::from_array(h_nodes_arr.clone())
                .map_err(|e| Error::Other(format!("h_nodes tensor: {e}")))?;
            let h_graph_t = Tensor::from_array(h_graph.clone())
                .map_err(|e| Error::Other(format!("h_graph tensor: {e}")))?;
            let last_emb_t = Tensor::from_array(last_emb.clone())
                .map_err(|e| Error::Other(format!("last_emb tensor: {e}")))?;
            let mask_t = Tensor::from_array(mask_arr)
                .map_err(|e| Error::Other(format!("mask tensor: {e}")))?;

            let dec_out = self.decoder.run(ort::inputs![h_nodes_t, h_graph_t, last_emb_t, mask_t])
                .map_err(|e| Error::Other(format!("decoder run: {e}")))?;
            let logits = dec_out["logits"].try_extract_array::<f32>()
                .map_err(|e| Error::Other(format!("logits extract: {e}")))?;
            let logits = logits.as_standard_layout();
            let logits_arr: Array2<f32> = logits.to_owned().into_dimensionality()
                .map_err(|e| Error::Other(format!("logits shape: {e}")))?;

            // Pick next node: greedy argmax (or sample — left as future work).
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for i in 0..n {
                if visited[i] { continue; }
                let v = logits_arr[[0, i]];
                if v > best_val { best_val = v; best_idx = i; }
            }
            let _ = greedy;

            tour.push(best_idx);
            visited[best_idx] = true;
            // Update last_emb = h_nodes[0, best_idx, :].
            for k in 0..self.embed_dim {
                last_emb[[0, k]] = h_nodes_arr[[0, best_idx, k]];
            }
        }

        Ok(tour)
    }
}

/// Compute total tour length (Euclidean closed loop). Uses the same coords
/// that were fed to `route`, so caller is responsible for de-normalizing if
/// they need real-world distances.
pub fn tour_length(coords: &[[f32; 2]], tour: &[usize]) -> f32 {
    if tour.is_empty() { return 0.0; }
    let mut total = 0.0;
    for i in 0..tour.len() {
        let a = coords[tour[i]];
        let b = coords[tour[(i + 1) % tour.len()]];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        total += (dx * dx + dy * dy).sqrt();
    }
    total
}

/// CVRPTW node features: (x, y, demand, tw_start, tw_end, service).
/// Index 0 must be the depot (demand=0, tw=horizon).
#[derive(Debug, Clone, Copy)]
pub struct CvrptwNode {
    pub x: f32,
    pub y: f32,
    pub demand: f32,
    pub tw_start: f32,
    pub tw_end: f32,
    pub service: f32,
}

/// Multi-vehicle CVRPTW pointer model. Loads from
/// `dir/pointer_cvrptw_encoder.onnx` + `_decoder.onnx`.
pub struct PointerCvrptwModel {
    encoder: Session,
    decoder: Session,
    embed_dim: usize,
    capacity: f32,
    horizon: f32,
}

impl PointerCvrptwModel {
    pub fn load(dir: &Path, capacity: f32, horizon: f32) -> Result<Self> {
        let enc_path = dir.join("pointer_cvrptw_encoder.onnx");
        let dec_path = dir.join("pointer_cvrptw_decoder.onnx");
        let encoder = Session::builder()
            .map_err(|e| Error::Other(format!("ort builder: {e}")))?
            .commit_from_file(&enc_path)
            .map_err(|e| Error::Other(format!("load enc {}: {e}", enc_path.display())))?;
        let decoder = Session::builder()
            .map_err(|e| Error::Other(format!("ort builder: {e}")))?
            .commit_from_file(&dec_path)
            .map_err(|e| Error::Other(format!("load dec {}: {e}", dec_path.display())))?;
        Ok(Self { encoder, decoder, embed_dim: 64, capacity, horizon })
    }

    /// Generate a multi-route CVRPTW solution. Returns Vec of routes —
    /// each route is a sequence of customer indices (depot returns are
    /// implicit as route boundaries).
    ///
    /// `nodes[0]` MUST be the depot. Returns customer indices (1..N).
    pub fn route(&mut self, nodes: &[CvrptwNode]) -> Result<Vec<Vec<usize>>> {
        let n = nodes.len();
        if n < 2 { return Ok(Vec::new()); }

        // Build (1, N, 6) feature tensor.
        let mut feats = Array3::<f32>::zeros((1, n, 6));
        for (i, node) in nodes.iter().enumerate() {
            feats[[0, i, 0]] = node.x;
            feats[[0, i, 1]] = node.y;
            feats[[0, i, 2]] = node.demand;
            feats[[0, i, 3]] = node.tw_start;
            feats[[0, i, 4]] = node.tw_end;
            feats[[0, i, 5]] = node.service;
        }
        let feats_t = Tensor::from_array(feats)
            .map_err(|e| Error::Other(format!("feats tensor: {e}")))?;
        let enc_out = self.encoder.run(ort::inputs![feats_t])
            .map_err(|e| Error::Other(format!("enc run: {e}")))?;
        let h_nodes = enc_out["h_nodes"].try_extract_array::<f32>()
            .map_err(|e| Error::Other(format!("h_nodes extract: {e}")))?;
        let h_nodes = h_nodes.as_standard_layout();
        let h_nodes_arr: Array3<f32> = h_nodes.to_owned().into_dimensionality()
            .map_err(|e| Error::Other(format!("h_nodes shape: {e}")))?;

        let mut h_graph = Array2::<f32>::zeros((1, self.embed_dim));
        for i in 0..n {
            for k in 0..self.embed_dim {
                h_graph[[0, k]] += h_nodes_arr[[0, i, k]];
            }
        }
        for k in 0..self.embed_dim { h_graph[[0, k]] /= n as f32; }

        let mut visited = vec![false; n];
        let mut last = 0usize; // start at depot
        let mut load = 0.0_f32;
        let mut time_now = 0.0_f32;
        let mut routes: Vec<Vec<usize>> = Vec::new();
        let mut current_route: Vec<usize> = Vec::new();

        let max_steps = 2 * n;
        for _ in 0..max_steps {
            // Build feasibility mask.
            let mut mask = vec![false; n];
            let last_node = &nodes[last];
            for i in 0..n {
                if i == 0 {
                    // Depot is selectable only when not currently at depot
                    // (avoid infinite loop). If we're at customer, depot
                    // is always allowed (close current route).
                    mask[i] = last != 0;
                } else if visited[i] {
                    mask[i] = false;
                } else {
                    let dx = last_node.x - nodes[i].x;
                    let dy = last_node.y - nodes[i].y;
                    let travel = (dx * dx + dy * dy).sqrt();
                    let arrival = time_now + travel;
                    let cap_ok = load + nodes[i].demand <= self.capacity;
                    let tw_ok = arrival <= nodes[i].tw_end;
                    mask[i] = cap_ok && tw_ok;
                }
            }
            if !mask.iter().any(|&b| b) {
                // Forced to depot.
                mask[0] = true;
            }

            // Forward decoder.
            let mut h_nodes_in = h_nodes_arr.clone();
            let h_graph_in = h_graph.clone();
            let mut last_emb = Array2::<f32>::zeros((1, self.embed_dim));
            for k in 0..self.embed_dim { last_emb[[0, k]] = h_nodes_arr[[0, last, k]]; }
            let mut state = Array2::<f32>::zeros((1, 2));
            state[[0, 0]] = load / self.capacity;
            state[[0, 1]] = time_now / self.horizon;
            let mut mask_arr = Array2::<bool>::default((1, n));
            for i in 0..n { mask_arr[[0, i]] = mask[i]; }

            let h_t = Tensor::from_array(std::mem::replace(&mut h_nodes_in, Array3::zeros((0,0,0))))
                .map_err(|e| Error::Other(format!("h tensor: {e}")))?;
            let g_t = Tensor::from_array(h_graph_in)
                .map_err(|e| Error::Other(format!("g tensor: {e}")))?;
            let l_t = Tensor::from_array(last_emb)
                .map_err(|e| Error::Other(format!("l tensor: {e}")))?;
            let s_t = Tensor::from_array(state)
                .map_err(|e| Error::Other(format!("s tensor: {e}")))?;
            let m_t = Tensor::from_array(mask_arr)
                .map_err(|e| Error::Other(format!("m tensor: {e}")))?;
            let dec_out = self.decoder.run(ort::inputs![h_t, g_t, l_t, s_t, m_t])
                .map_err(|e| Error::Other(format!("dec run: {e}")))?;
            let logits = dec_out["logits"].try_extract_array::<f32>()
                .map_err(|e| Error::Other(format!("logits: {e}")))?;
            let logits = logits.as_standard_layout();
            let logits_arr: Array2<f32> = logits.to_owned().into_dimensionality()
                .map_err(|e| Error::Other(format!("logits shape: {e}")))?;

            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for i in 0..n {
                if !mask[i] { continue; }
                let v = logits_arr[[0, i]];
                if v > best_val { best_val = v; best_idx = i; }
            }

            // Apply transition.
            let dx = nodes[last].x - nodes[best_idx].x;
            let dy = nodes[last].y - nodes[best_idx].y;
            let travel = (dx * dx + dy * dy).sqrt();
            let arrival = time_now + travel;
            let wait = (nodes[best_idx].tw_start - arrival).max(0.0);
            time_now = arrival + wait + nodes[best_idx].service;

            if best_idx == 0 {
                // Return to depot — close current route.
                if !current_route.is_empty() {
                    routes.push(std::mem::take(&mut current_route));
                }
                load = 0.0;
                time_now = 0.0;
                last = 0;
            } else {
                visited[best_idx] = true;
                load += nodes[best_idx].demand;
                current_route.push(best_idx);
                last = best_idx;
            }

            // Done?
            if visited[1..].iter().all(|&v| v) && last == 0 { break; }
        }
        if !current_route.is_empty() { routes.push(current_route); }

        Ok(routes)
    }
}

/// Simple nearest-neighbor baseline for comparison.
pub fn nearest_neighbor_tour(coords: &[[f32; 2]]) -> Vec<usize> {
    let n = coords.len();
    if n == 0 { return Vec::new(); }
    let mut visited = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    let mut current = 0usize;
    tour.push(current);
    visited[current] = true;
    for _ in 1..n {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for j in 0..n {
            if visited[j] { continue; }
            let dx = coords[current][0] - coords[j][0];
            let dy = coords[current][1] - coords[j][1];
            let d = (dx * dx + dy * dy).sqrt();
            if d < best_d { best_d = d; best = j; }
        }
        tour.push(best);
        visited[best] = true;
        current = best;
    }
    tour
}
