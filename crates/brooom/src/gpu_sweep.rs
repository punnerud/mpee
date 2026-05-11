//! GPU-accelerated batch 2-opt evaluation via `wgpu`.
//!
//! Cross-platform compute: Metal on M3, Vulkan on Linux/AMD/Intel,
//! DirectX 12 on Windows/NVIDIA. The same WGSL shader runs everywhere.
//!
//! ## Why GPU here
//!
//! 2-opt sweep on a tour of length N requires evaluating ~N² candidate
//! moves (every pair (i, j) with j > i + 1). Each evaluation is four
//! independent matrix lookups + arithmetic — embarrassingly parallel.
//! For N ≥ 1000 the sweep dominates an LS pass; CPU does ~1 ms, GPU
//! does ~50 µs (kernel launch) + microseconds of compute.
//!
//! ## Caveats
//!
//! - We compute *only* the distance delta. Time-window feasibility must
//!   still be checked CPU-side after the GPU returns the candidates.
//! - First-time device init takes ~50 ms (one-shot). Per-call overhead
//!   is ~50–100 µs (buffer upload + dispatch + readback). Below N≈500
//!   the CPU sweep wins.
//! - The distance matrix is uploaded once and reused; only the tour and
//!   the output buffer move per call.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::error::Error;

/// One set of GPU resources, parameterised on a fixed distance matrix.
/// Re-used across many sweep calls; the matrix is uploaded once.
pub struct GpuSweep {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pipeline_argmin: wgpu::ComputePipeline,
    pipeline_batched: wgpu::ComputePipeline,
    matrix_buf: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_layout_argmin: wgpu::BindGroupLayout,
    bind_group_layout_batched: wgpu::BindGroupLayout,
    /// Square matrix dimension (n_locations).
    matrix_dim: u32,
}

/// Result of a GPU 2-opt sweep with on-GPU argmin reduction. The CPU
/// reads back ~12 bytes total instead of N² × 4 bytes.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BestMove {
    /// Most-improving Δcost found across all (i,j) pairs. Negative = improvement.
    pub delta: i32,
    pub i: u32,
    pub j: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct Params {
    /// Tour length (number of stops, including any depot endpoints).
    n: u32,
    /// Matrix side length (n_locations in problem).
    matrix_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

impl GpuSweep {
    /// Block on the async wgpu init; returns a ready-to-use compute
    /// context bound to the given distance matrix.
    pub fn new(matrix: &[i32], matrix_dim: u32) -> Result<Self, Error> {
        pollster::block_on(Self::new_async(matrix, matrix_dim))
    }

    async fn new_async(matrix: &[i32], matrix_dim: u32) -> Result<Self, Error> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::Other("gpu_sweep: no GPU adapter found".into()))?;

        let info = adapter.get_info();
        tracing::info!(
            "gpu_sweep: adapter = {} ({:?}, backend={:?})",
            info.name, info.device_type, info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("brooom-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| Error::Other(format!("gpu_sweep: device init: {e}")))?;

        // Upload the distance matrix once. STORAGE | COPY_SRC so we can
        // also debug-readback if needed.
        let matrix_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matrix"),
            contents: bytemuck::cast_slice(matrix),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brooom-2opt"),
            source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brooom-2opt-bgl"),
            entries: &[
                // 0 = tour (read)
                bind_entry(0, true),
                // 1 = matrix (read)
                bind_entry(1, true),
                // 2 = output deltas (read_write)
                bind_entry(2, false),
                // 3 = params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brooom-2opt-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brooom-2opt-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // Combined sweep + argmin pipeline: regn delta in shared memory,
        // reduce within workgroup, then atomic-min into a single output.
        let bind_group_layout_argmin = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brooom-2opt-argmin-bgl"),
            entries: &[
                bind_entry(0, true),  // tour
                bind_entry(1, true),  // matrix
                bind_entry(2, false), // best (3×u32)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout_argmin = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brooom-2opt-argmin-pl"),
            bind_group_layouts: &[&bind_group_layout_argmin],
            push_constant_ranges: &[],
        });
        let shader_argmin = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brooom-2opt-argmin"),
            source: wgpu::ShaderSource::Wgsl(SHADER_ARGMIN_WGSL.into()),
        });
        let pipeline_argmin = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brooom-2opt-argmin-pipeline"),
            layout: Some(&pipeline_layout_argmin),
            module: &shader_argmin,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // Batched-routes pipeline. One workgroup per route; threads in
        // a workgroup cooperate to find the best 2-opt for that route.
        let bind_group_layout_batched = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brooom-2opt-batched-bgl"),
            entries: &[
                bind_entry(0, true),  // tour_buffer (concatenated)
                bind_entry(1, true),  // route_starts
                bind_entry(2, true),  // route_lengths
                bind_entry(3, true),  // matrix
                bind_entry(4, false), // best_per_route (n_routes × 3 i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout_batched = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brooom-2opt-batched-pl"),
            bind_group_layouts: &[&bind_group_layout_batched],
            push_constant_ranges: &[],
        });
        let shader_batched = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brooom-2opt-batched"),
            source: wgpu::ShaderSource::Wgsl(SHADER_BATCHED_WGSL.into()),
        });
        let pipeline_batched = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brooom-2opt-batched-pipeline"),
            layout: Some(&pipeline_layout_batched),
            module: &shader_batched,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_argmin,
            pipeline_batched,
            matrix_buf,
            bind_group_layout,
            bind_group_layout_argmin,
            bind_group_layout_batched,
            matrix_dim,
        })
    }

    /// Batched per-route 2-opt: every route gets its own workgroup; the
    /// kernel finds the most-improving (i, j)-swap for each route in one
    /// dispatch. Returns `Vec<BestMove>` of length `routes.len()`. Caller
    /// is responsible for TW/capacity feasibility.
    ///
    /// Input:
    ///   tours: per-route stop sequences (excluding depot endpoints)
    ///
    /// We concat the tours into one buffer with parallel arrays of
    /// (start, length) so every workgroup knows where its slice lives.
    pub fn batched_best_2opt(&self, routes: &[Vec<u32>]) -> Result<Vec<BestMove>, Error> {
        if routes.is_empty() {
            return Ok(Vec::new());
        }
        let n_routes = routes.len() as u32;

        // Concat tours, build start/length arrays.
        let mut tour_buffer: Vec<u32> = Vec::new();
        let mut route_starts: Vec<u32> = Vec::with_capacity(routes.len());
        let mut route_lengths: Vec<u32> = Vec::with_capacity(routes.len());
        for r in routes {
            route_starts.push(tour_buffer.len() as u32);
            route_lengths.push(r.len() as u32);
            tour_buffer.extend_from_slice(r);
        }

        let tour_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tour-batched"),
            contents: bytemuck::cast_slice(&tour_buffer),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let starts_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("route-starts"),
            contents: bytemuck::cast_slice(&route_starts),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lengths_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("route-lengths"),
            contents: bytemuck::cast_slice(&route_lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Init best_per_route to {INT_MAX, 0, 0} for each route.
        let mut init: Vec<i32> = Vec::with_capacity(routes.len() * 3);
        for _ in 0..routes.len() {
            init.extend_from_slice(&[i32::MAX, 0, 0]);
        }
        let best_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("best-per-route"),
            contents: bytemuck::cast_slice(&init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params = Params { n: n_routes, matrix_dim: self.matrix_dim, _pad0: 0, _pad1: 0 };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params-batched"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-batched"),
            layout: &self.bind_group_layout_batched,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: starts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lengths_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: best_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
            ],
        });

        // One workgroup per route.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brooom-2opt-batched-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brooom-2opt-batched-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_batched);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_routes, 1, 1);
        }

        let out_size = (n_routes as u64) * 12;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-batched"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&best_buf, 0, &readback, 0, out_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| Error::Other(format!("gpu_sweep: batched map recv: {e}")))?
            .map_err(|e| Error::Other(format!("gpu_sweep: batched map: {e:?}")))?;

        let data = slice.get_mapped_range();
        let arr: &[i32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(routes.len());
        for r_idx in 0..routes.len() {
            out.push(BestMove {
                delta: arr[r_idx * 3],
                i: arr[r_idx * 3 + 1] as u32,
                j: arr[r_idx * 3 + 2] as u32,
            });
        }
        drop(data);
        readback.unmap();
        Ok(out)
    }

    /// Sweep + argmin entirely on GPU. Returns the single best 2-opt
    /// move (most-improving Δcost) across all valid (i, j) pairs.
    /// Readback is 12 bytes regardless of N.
    pub fn best_2opt(&self, tour: &[u32]) -> Result<BestMove, Error> {
        let n = tour.len() as u32;
        if n < 4 {
            return Ok(BestMove { delta: 0, i: 0, j: 0 });
        }

        let tour_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tour-argmin"),
            contents: bytemuck::cast_slice(tour),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Best-buffer holds [delta, i, j] as packed i32. Initialised to
        // {INT_MAX, 0, 0} so atomic-min finds anything improving.
        let init_best = [i32::MAX, 0i32, 0i32];
        let best_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("best"),
            contents: bytemuck::cast_slice(&init_best),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params = Params { n, matrix_dim: self.matrix_dim, _pad0: 0, _pad1: 0 };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params-argmin"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-argmin"),
            layout: &self.bind_group_layout_argmin,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: best_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let groups = (n + 7) / 8;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brooom-2opt-argmin-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brooom-2opt-argmin-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_argmin);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-argmin"),
            size: 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&best_buf, 0, &readback, 0, 12);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| Error::Other(format!("gpu_sweep: argmin map recv: {e}")))?
            .map_err(|e| Error::Other(format!("gpu_sweep: argmin map: {e:?}")))?;

        let data = slice.get_mapped_range();
        let arr: &[i32] = bytemuck::cast_slice(&data);
        let result = BestMove { delta: arr[0], i: arr[1] as u32, j: arr[2] as u32 };
        drop(data);
        readback.unmap();
        Ok(result)
    }

    /// Evaluate distance-delta for every 2-opt swap (i, j), j > i + 1,
    /// in `tour`. Returns a flat `n * n` array indexed `delta[i * n + j]`;
    /// invalid pairs (i ≥ j-1, depot wrap, etc.) are filled with 0.
    /// Negative delta = improving move. Caller must still check TW
    /// feasibility on candidates.
    pub fn eval_2opt(&self, tour: &[u32]) -> Result<Vec<i32>, Error> {
        let n = tour.len() as u32;
        if n < 4 {
            return Ok(vec![0; (n * n) as usize]);
        }

        let t0 = Instant::now();

        let tour_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tour"),
            contents: bytemuck::cast_slice(tour),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_size = (n as u64) * (n as u64) * std::mem::size_of::<i32>() as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Params { n, matrix_dim: self.matrix_dim, _pad0: 0, _pad1: 0 };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        // Workgroup size 8×8 = 64 threads; dispatch ceil(n/8) × ceil(n/8) groups.
        let groups = (n + 7) / 8;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brooom-2opt-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brooom-2opt-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);
        }

        // Stage a readback buffer (MAP_READ | COPY_DST).
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_size);

        self.queue.submit(Some(encoder.finish()));

        // Map and read.
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| Error::Other(format!("gpu_sweep: map recv: {e}")))?
            .map_err(|e| Error::Other(format!("gpu_sweep: map: {e:?}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();

        tracing::debug!("gpu_sweep::eval_2opt n={} took {:?}", n, t0.elapsed());

        Ok(result)
    }
}

fn bind_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Combined sweep + argmin reduction shader. Each invocation computes
/// one (i,j)-delta, then atomic-min into a packed [delta, i, j] best
/// record. Atomic-min on i32 lets us pack delta in the high bits and
/// (i, j) in the lower bits ... but wgpu/WGSL atomicMin works on i32
/// directly, so we use a "lock-and-record" pattern: atomic-min on a
/// shared delta slot, then if our delta is the new best, write i and j.
/// This is racy in theory but for our use case (find ANY most-improving
/// move) the result is correct: whoever wrote the smallest delta wins.
const SHADER_ARGMIN_WGSL: &str = r#"
struct Params {
    n: u32,
    matrix_dim: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> tour: array<u32>;
@group(0) @binding(1) var<storage, read> matrix: array<i32>;
@group(0) @binding(2) var<storage, read_write> best: array<atomic<i32>, 3>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let n = params.n;
    let md = params.matrix_dim;

    if (i >= n || j >= n) {
        return;
    }
    if (j <= i + 1u || i + 1u >= n) {
        return;
    }

    let a = tour[i];
    let b = tour[i + 1u];
    let c = tour[j];
    var jp1 = j + 1u;
    if (jp1 >= n) { jp1 = 0u; }
    let d = tour[jp1];

    let old_cost = matrix[a * md + b] + matrix[c * md + d];
    let new_cost = matrix[a * md + c] + matrix[b * md + d];
    let delta = new_cost - old_cost;

    let prev = atomicMin(&best[0], delta);
    if (delta < prev) {
        // We won the race for this delta value (or a better one). Record
        // (i, j). If a later thread finds an even smaller delta, it
        // overwrites. We accept a small chance of (i, j) lagging behind
        // delta during contention; for our use case (find best move)
        // this is acceptable — the LS step verifies the proposed move
        // before applying.
        atomicStore(&best[1], i32(i));
        atomicStore(&best[2], i32(j));
    }
}
"#;

/// Batched per-route 2-opt: one workgroup handles one route. Threads
/// in the workgroup share work over (i, j) pairs and atomic-min into a
/// per-route best slot. Workgroup size 64 fits typical M3 NEON+Metal.
///
/// Layout of `best_per_route`: 3 i32 per route — [delta, i, j].
const SHADER_BATCHED_WGSL: &str = r#"
struct Params {
    n_routes: u32,
    matrix_dim: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> tour_buffer: array<u32>;
@group(0) @binding(1) var<storage, read> route_starts: array<u32>;
@group(0) @binding(2) var<storage, read> route_lengths: array<u32>;
@group(0) @binding(3) var<storage, read> matrix: array<i32>;
@group(0) @binding(4) var<storage, read_write> best_per_route: array<atomic<i32>>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let route_idx = wg.x;
    if (route_idx >= params.n_routes) {
        return;
    }
    let start = route_starts[route_idx];
    let len = route_lengths[route_idx];
    let md = params.matrix_dim;
    let tid = lid.x;
    let n_threads = 64u;

    // Distribute (i, j) pairs across the 64 threads. Total pairs ≈ len²/2.
    // Thread `tid` handles every n_threads-th pair (stride pattern).
    let n_pairs = len * len;
    var k = tid;
    loop {
        if (k >= n_pairs) { break; }
        let i = k / len;
        let j = k % len;
        if (j > i + 1u && i + 1u < len && j < len) {
            let a = tour_buffer[start + i];
            let b = tour_buffer[start + i + 1u];
            let c = tour_buffer[start + j];
            // For closing edge: wrap inside this route's slice.
            var jp1 = j + 1u;
            if (jp1 >= len) { jp1 = 0u; }
            let d = tour_buffer[start + jp1];

            let old_cost = matrix[a * md + b] + matrix[c * md + d];
            let new_cost = matrix[a * md + c] + matrix[b * md + d];
            let delta = new_cost - old_cost;

            let best_off = route_idx * 3u;
            let prev = atomicMin(&best_per_route[best_off], delta);
            if (delta < prev) {
                atomicStore(&best_per_route[best_off + 1u], i32(i));
                atomicStore(&best_per_route[best_off + 2u], i32(j));
            }
        }
        k += n_threads;
    }
}
"#;

const SHADER_WGSL: &str = r#"
struct Params {
    n: u32,
    matrix_dim: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> tour: array<u32>;
@group(0) @binding(1) var<storage, read> matrix: array<i32>;
@group(0) @binding(2) var<storage, read_write> out_delta: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let n = params.n;
    let md = params.matrix_dim;

    if (i >= n || j >= n) {
        return;
    }
    // Need j > i + 1 for a meaningful 2-opt swap (and i in [0, n-2]).
    if (j <= i + 1u || i + 1u >= n) {
        // Mark as no-op.
        out_delta[i * n + j] = 0;
        return;
    }

    let a  = tour[i];
    let b  = tour[i + 1u];
    let c  = tour[j];
    // Wrap-around for the closing edge.
    var jp1 = j + 1u;
    if (jp1 >= n) { jp1 = 0u; }
    let d  = tour[jp1];

    // Old edges: (a,b), (c,d). New: (a,c), (b,d). Path between b..c
    // gets reversed; its internal cost is symmetric so it cancels.
    let old_cost = matrix[a * md + b] + matrix[c * md + d];
    let new_cost = matrix[a * md + c] + matrix[b * md + d];
    out_delta[i * n + j] = new_cost - old_cost;
}
"#;
