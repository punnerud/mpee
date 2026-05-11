//! Persistent, multi-trajectory tour state on the GPU.
//!
//! `GpuSweep` is stateless — it ships a single tour to the device, runs
//! one kernel, reads back the answer. That's fine for one-off sweeps but
//! pays the ~2 ms round-trip cost on every iteration of the LS loop.
//!
//! `GpuPopulation` keeps `pop_size` solutions resident on the device so
//! kernels can mutate them in place across many iterations without ever
//! crossing the PCIe / unified-memory boundary. It is the foundation for
//! the planned full-GPU LS loop:
//!   1. Upload initial solutions once.
//!   2. Run kernels (precompute, eval, search-operators, apply-move) that
//!      read and write the persistent buffers in place.
//!   3. Read back the best solution at the end.
//!
//! Phase 1 here only covers (1) and the read-back side of (3) — i.e. a
//! verifiable round-trip. A small distance-summing kernel is included as
//! proof-of-life that the persistent buffers are visible to a shader.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::error::Error;

/// One trajectory's view of its routes: per-route stop sequences. Each
/// stop is a *location index* (the column index into the distance matrix).
/// Depot endpoints are included if the caller wants them tracked — this
/// type makes no assumption either way.
pub type TrajectoryTours = Vec<Vec<u32>>;

/// GPU-resident state for `pop_size` trajectories. Layout:
///   tour_buf       : pop_size × tour_capacity u32
///   route_starts   : pop_size × max_routes  u32 (offset into tour slot)
///   route_lengths  : pop_size × max_routes  u32 (stops per route)
///   n_routes       : pop_size              u32 (active routes per traj)
///   matrix         : n_locations²          i32 (uploaded once)
///
/// Slot strides (`tour_capacity`, `max_routes`) are fixed at construction
/// so kernels can index by `(traj_idx, slot_offset)` without indirections.
pub struct GpuPopulation {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pop_size: u32,
    max_routes: u32,
    tour_capacity: u32,
    n_locations: u32,

    tour_buf: wgpu::Buffer,
    route_starts_buf: wgpu::Buffer,
    route_lengths_buf: wgpu::Buffer,
    n_routes_buf: wgpu::Buffer,
    matrix_buf: wgpu::Buffer,

    /// Proof-of-life kernel: per-route Manhattan-style sum of consecutive
    /// matrix lookups. Used by `route_distances()`.
    pipeline_distances: wgpu::ComputePipeline,
    bgl_distances: wgpu::BindGroupLayout,

    // ---- Phase 2: per-location problem data + per-trajectory vehicle data
    //      and the precompute kernel. Optional — created lazily.
    problem_uploaded: std::cell::Cell<bool>,
    vehicle_uploaded: std::cell::Cell<bool>,
    /// Per-location problem data, length n_locations. Uploaded once.
    problem_service_buf: wgpu::Buffer,
    problem_demand_buf: wgpu::Buffer,
    problem_tw_start_buf: wgpu::Buffer,
    problem_tw_end_buf: wgpu::Buffer,
    /// Per-trajectory vehicle data, length pop_size. Uploaded per call.
    vehicle_capacity_buf: wgpu::Buffer,
    vehicle_tw_start_buf: wgpu::Buffer,
    vehicle_tw_end_buf: wgpu::Buffer,
    /// Per-position outputs of the precompute kernel.
    depart_buf: wgpu::Buffer,
    latest_arrival_buf: wgpu::Buffer,
    load_at_buf: wgpu::Buffer,
    /// Per-route flag: 1 = feasible under TW + capacity, 0 = infeasible.
    feasible_buf: wgpu::Buffer,
    /// Per-route metrics: 5 i32 per route, layout
    /// `[travel_time, service_time, waiting_time, distance, end_time]`.
    /// Unused slots stay zero.
    route_metrics_buf: wgpu::Buffer,
    pipeline_precompute: wgpu::ComputePipeline,
    bgl_precompute: wgpu::BindGroupLayout,

    // ---- Phase 4: 2-opt search + apply on persistent tour state ----
    /// Per-route best 2-opt move: 3 i32 per route, layout
    /// `[delta, i, j]` with `delta` initialised to i32::MAX before each
    /// dispatch. Distance-only — no TW check.
    best_2opt_buf: wgpu::Buffer,
    pipeline_find_2opt: wgpu::ComputePipeline,
    bgl_find_2opt: wgpu::BindGroupLayout,
    pipeline_apply_2opt: wgpu::ComputePipeline,
    bgl_apply_2opt: wgpu::BindGroupLayout,

    // ---- Megakernel: full 2-opt LS loop in one dispatch ----
    /// Stores [iter_count, applied_count, final_best_delta] for telemetry.
    megakernel_status_buf: wgpu::Buffer,
    pipeline_megakernel: wgpu::ComputePipeline,
    bgl_megakernel: wgpu::BindGroupLayout,

    // ---- Granular K-nearest-neighbour table for scaling to N≥1000 ----
    /// Flat: `granular[i * k + r]` = r-th nearest location index to i.
    /// Replaces O(N²) operator search with O(N×K). Optional — if not
    /// uploaded, megakernel falls back to full N²-search.
    granular_buf: wgpu::Buffer,
    granular_uploaded: std::cell::Cell<bool>,
    granular_k: std::cell::Cell<u32>,
    /// Per-task → (route_idx, position) lookup. Maintained at the start
    /// of each LS iter inside the megakernel. Lets granular-driven
    /// relocate/exchange resolve "where does this neighbor live" in O(1).
    /// Stored as packed (route << 16 | pos), with 0xFFFFFFFF = unassigned.
    task_pos_buf: wgpu::Buffer,

    // ---- Coord mode: per-location (x, y) as f32×2 ----
    /// When uploaded, megakernel computes distances on-the-fly from coords
    /// (Euclidean × 100) instead of reading from `matrix`. Needed for
    /// N≥16K where the N² i32 matrix exceeds buffer limits. matrix_buf
    /// still bound (as a stub) so the bind group layout stays uniform.
    coords_buf: wgpu::Buffer,
    coords_uploaded: std::cell::Cell<bool>,
}

/// One trajectory's best 2-opt move per route, as returned by
/// `find_best_2opt_all` + `read_best_2opt_per_route`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBest2opt {
    /// Most-improving distance delta. Negative = improvement.
    pub delta: i32,
    pub i: u32,
    pub j: u32,
}

/// Per-trajectory status returned by megakernel batch calls.
#[derive(Debug, Clone, Copy, Default)]
pub struct MegakernelStatus {
    pub iters: u32,
    pub applies: u32,
    pub final_delta: i32,
    /// Number of pulled tasks dropped because no feasible insertion was
    /// found. > 0 means this trajectory is incomplete and should be
    /// excluded from best-of-N selection.
    pub dropped: u32,
}

/// Per-route metrics read back from `read_route_metrics`. All times are
/// in the same unit as the matrix (typically seconds for VRPTW).
#[derive(Debug, Clone, Default)]
pub struct GpuRouteMetrics {
    pub travel_time: i32,
    pub service_time: i32,
    pub waiting_time: i32,
    pub distance: i32,
    pub end_time: i32,
    pub feasible: bool,
}

/// Per-position precompute outputs read back from the GPU.
#[derive(Debug, Clone, Default)]
pub struct GpuRoutePrecomp {
    /// Time the vehicle leaves each position (post-wait, post-service).
    /// `depart[0]` = vehicle TW start; `depart[len-1]` = arrival at end depot.
    pub depart: Vec<i32>,
    /// Latest time the vehicle may arrive at each position without
    /// breaking downstream TW or vehicle horizon.
    pub latest_arrival: Vec<i32>,
    /// Cumulative load at each position. `load_at[0]` = sum-of-deliveries
    /// (initial load); decreases as customer demands are dropped off.
    pub load_at: Vec<i32>,
    /// True iff the route is TW + capacity feasible end-to-end.
    pub feasible: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct PopParams {
    pop_size: u32,
    max_routes: u32,
    tour_capacity: u32,
    matrix_dim: u32,
}

impl GpuPopulation {
    pub fn new(
        matrix: &[i32],
        n_locations: u32,
        pop_size: u32,
        max_routes: u32,
        tour_capacity: u32,
    ) -> Result<Self, Error> {
        pollster::block_on(Self::new_async(
            matrix,
            n_locations,
            pop_size,
            max_routes,
            tour_capacity,
        ))
    }

    async fn new_async(
        matrix: &[i32],
        n_locations: u32,
        pop_size: u32,
        max_routes: u32,
        tour_capacity: u32,
    ) -> Result<Self, Error> {
        // Allow matrix to be empty (coord-mode caller will upload coords
        // instead). Otherwise it must be N².
        if !matrix.is_empty() && (matrix.len() as u32) != n_locations * n_locations {
            return Err(Error::Other(format!(
                "gpu_population: matrix len {} != n_locations² ({})",
                matrix.len(),
                n_locations * n_locations
            )));
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::Other("gpu_population: no GPU adapter found".into()))?;

        let info = adapter.get_info();
        tracing::info!(
            "gpu_population: adapter = {} ({:?}, backend={:?})",
            info.name,
            info.device_type,
            info.backend
        );

        // Phase 2's precompute kernel binds 14 storage buffers + 1 uniform.
        // Default `max_storage_buffers_per_shader_stage` is 8 (WebGPU low
        // bar). M3 / Vulkan / DX12 natively support far more — bump the
        // limit to whatever the adapter actually supports.
        // Megakernel uses workgroup_size=1024 (default cap is 256).
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage.max(16);
        limits.max_compute_invocations_per_workgroup =
            adapter_limits.max_compute_invocations_per_workgroup.max(1024);
        limits.max_compute_workgroup_size_x =
            adapter_limits.max_compute_workgroup_size_x.max(1024);
        // Allow large distance matrices for N≥10000 (N²×4 bytes).
        // Take whatever the adapter supports — typically 1-2 GB on M3.
        limits.max_buffer_size =
            adapter_limits.max_buffer_size.max(1 << 30); // ≥1 GiB
        limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size.max(1 << 30);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("brooom-gpu-pop"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .map_err(|e| Error::Other(format!("gpu_population: device init: {e}")))?;

        // Persistent buffers — sized once, mutated by future kernels.
        let tour_zero = vec![0u32; (pop_size * tour_capacity) as usize];
        let tour_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-tour"),
            contents: bytemuck::cast_slice(&tour_zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let routes_zero = vec![0u32; (pop_size * max_routes) as usize];
        let route_starts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-route-starts"),
            contents: bytemuck::cast_slice(&routes_zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let route_lengths_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-route-lengths"),
            contents: bytemuck::cast_slice(&routes_zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let n_zero = vec![0u32; pop_size as usize];
        let n_routes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-n-routes"),
            contents: bytemuck::cast_slice(&n_zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        // Matrix buffer must be non-empty (wgpu requires size > 0). In
        // coord-mode the caller passes &[] — we substitute a tiny stub.
        let stub_matrix: Vec<i32> = vec![0; 4];
        let matrix_slice: &[i32] = if matrix.is_empty() { &stub_matrix } else { matrix };
        let matrix_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-matrix"),
            contents: bytemuck::cast_slice(matrix_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Distances proof-of-life pipeline.
        let bgl_distances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pop-distances-bgl"),
            entries: &[
                bind_entry(0, true),  // tour
                bind_entry(1, true),  // route_starts
                bind_entry(2, true),  // route_lengths
                bind_entry(3, true),  // n_routes
                bind_entry(4, true),  // matrix
                bind_entry(5, false), // out_distances
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
        let pl_distances = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pop-distances-pl"),
            bind_group_layouts: &[&bgl_distances],
            push_constant_ranges: &[],
        });
        let shader_distances = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pop-distances"),
            source: wgpu::ShaderSource::Wgsl(SHADER_DISTANCES_WGSL.into()),
        });
        let pipeline_distances = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pop-distances-pipeline"),
            layout: Some(&pl_distances),
            module: &shader_distances,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // ---- Phase 2 buffers ----
        let loc_zero_i32 = vec![0i32; n_locations as usize];
        let mk_loc_buf = |label: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&loc_zero_i32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let problem_service_buf = mk_loc_buf("pop-service");
        let problem_demand_buf = mk_loc_buf("pop-demand");
        let problem_tw_start_buf = mk_loc_buf("pop-tw-start");
        let problem_tw_end_buf = mk_loc_buf("pop-tw-end");

        let pop_zero_i32 = vec![0i32; pop_size as usize];
        let mk_traj_buf = |label: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&pop_zero_i32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let vehicle_capacity_buf = mk_traj_buf("pop-veh-cap");
        let vehicle_tw_start_buf = mk_traj_buf("pop-veh-tw-start");
        let vehicle_tw_end_buf = mk_traj_buf("pop-veh-tw-end");

        let pos_zero_i32 = vec![0i32; (pop_size * tour_capacity) as usize];
        let mk_pos_buf = |label: &str| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&pos_zero_i32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let depart_buf = mk_pos_buf("pop-depart");
        let latest_arrival_buf = mk_pos_buf("pop-latest");
        let load_at_buf = mk_pos_buf("pop-loadat");

        let feas_zero = vec![0u32; (pop_size * max_routes) as usize];
        let feasible_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-feasible"),
            contents: bytemuck::cast_slice(&feas_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let metrics_zero = vec![0i32; (pop_size * max_routes * 5) as usize];
        let route_metrics_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-route-metrics"),
            contents: bytemuck::cast_slice(&metrics_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Phase 4: best-2opt buffer (3 i32 per route).
        let best_zero = vec![0i32; (pop_size * max_routes * 3) as usize];
        let best_2opt_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-best-2opt"),
            contents: bytemuck::cast_slice(&best_zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let bgl_precompute = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pop-precompute-bgl"),
            entries: &[
                bind_entry(0, true),  // tour
                bind_entry(1, true),  // route_starts
                bind_entry(2, true),  // route_lengths
                bind_entry(3, true),  // n_routes
                bind_entry(4, true),  // matrix
                bind_entry(5, true),  // service
                bind_entry(6, true),  // demand
                bind_entry(7, true),  // tw_start
                bind_entry(8, true),  // tw_end
                bind_entry(9, true),  // vehicle_capacity
                bind_entry(10, true), // vehicle_tw_start
                bind_entry(11, true), // vehicle_tw_end
                bind_entry(12, false), // depart
                bind_entry(13, false), // latest_arrival
                bind_entry(14, false), // load_at
                bind_entry(15, false), // feasible
                bind_entry(16, false), // route_metrics
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
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
        let pl_precompute = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pop-precompute-pl"),
            bind_group_layouts: &[&bgl_precompute],
            push_constant_ranges: &[],
        });
        let shader_precompute = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pop-precompute"),
            source: wgpu::ShaderSource::Wgsl(SHADER_PRECOMPUTE_WGSL.into()),
        });
        let pipeline_precompute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pop-precompute-pipeline"),
            layout: Some(&pl_precompute),
            module: &shader_precompute,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // Phase 4: find best 2-opt per route. Reads tour, matrix, route
        // metadata; writes best_2opt (atomic).
        let bgl_find_2opt = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pop-find2opt-bgl"),
            entries: &[
                bind_entry(0, true),  // tour
                bind_entry(1, true),  // route_starts
                bind_entry(2, true),  // route_lengths
                bind_entry(3, true),  // n_routes
                bind_entry(4, true),  // matrix
                bind_entry(5, false), // best_2opt (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
        let pl_find_2opt = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pop-find2opt-pl"),
            bind_group_layouts: &[&bgl_find_2opt],
            push_constant_ranges: &[],
        });
        let shader_find_2opt = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pop-find2opt"),
            source: wgpu::ShaderSource::Wgsl(SHADER_FIND_2OPT_WGSL.into()),
        });
        let pipeline_find_2opt = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pop-find2opt-pipeline"),
            layout: Some(&pl_find_2opt),
            module: &shader_find_2opt,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // Phase 4: apply one 2-opt move (reverse segment in place).
        let bgl_apply_2opt = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pop-apply2opt-bgl"),
            entries: &[
                bind_entry(0, false), // tour (read_write)
                bind_entry(1, true),  // route_starts
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let pl_apply_2opt = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pop-apply2opt-pl"),
            bind_group_layouts: &[&bgl_apply_2opt],
            push_constant_ranges: &[],
        });
        let shader_apply_2opt = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pop-apply2opt"),
            source: wgpu::ShaderSource::Wgsl(SHADER_APPLY_2OPT_WGSL.into()),
        });
        let pipeline_apply_2opt = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pop-apply2opt-pipeline"),
            layout: Some(&pl_apply_2opt),
            module: &shader_apply_2opt,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // ---- Granular K-NN buffer ----
        // Pre-allocated for max K=64. Actual K filled in at upload time.
        // Size = n_locations × max_k. Index format: granular[i * k + r].
        let max_granular_k = 64u32;
        let granular_zero = vec![0u32; (n_locations * max_granular_k) as usize];
        let granular_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-granular"),
            contents: bytemuck::cast_slice(&granular_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // task_pos: per-trajectory N-sized lookup. Stored in storage so
        // it scales beyond workgroup-memory limit when N grows.
        let task_pos_zero = vec![0u32; (pop_size * n_locations) as usize];
        let task_pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-task-pos"),
            contents: bytemuck::cast_slice(&task_pos_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // coords: per-location (x, y) as f32×2. Only used when coord_mode=1.
        let coords_zero = vec![0.0f32; (n_locations * 2) as usize];
        let coords_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-coords"),
            contents: bytemuck::cast_slice(&coords_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ---- Megakernel: hele LS-loopen i én dispatch ----
        // Per-trajectory status slots so multi-workgroup batch mode can
        // each write its own [iter, applies, final_delta, _] without
        // contention.
        let status_zero = vec![0i32; (pop_size * 4) as usize];
        let megakernel_status_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("megakernel-status"),
            contents: bytemuck::cast_slice(&status_zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let bgl_megakernel = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pop-megakernel-bgl"),
            entries: &[
                bind_entry(0, false), // tour (read_write — apply mutates)
                bind_entry(1, true),  // route_starts
                bind_entry(2, false), // route_lengths (read_write — kick mutates)
                bind_entry(3, true),  // n_routes
                bind_entry(4, true),  // matrix
                bind_entry(5, false), // status (out)
                bind_entry(6, true),  // service
                bind_entry(7, true),  // tw_start
                bind_entry(8, true),  // tw_end
                bind_entry(9, true),  // vehicle_tw_start
                bind_entry(10, true), // vehicle_tw_end
                bind_entry(11, true), // demand
                bind_entry(12, true), // vehicle_capacity
                bind_entry(13, true), // granular (read-only)
                bind_entry(14, false), // task_pos (read_write — maintained per iter)
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                bind_entry(16, true), // coords (f32×2 per location, read-only)
            ],
        });
        let pl_megakernel = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pop-megakernel-pl"),
            bind_group_layouts: &[&bgl_megakernel],
            push_constant_ranges: &[],
        });
        let shader_megakernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pop-megakernel"),
            source: wgpu::ShaderSource::Wgsl(SHADER_MEGAKERNEL_WGSL.into()),
        });
        let pipeline_megakernel = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pop-megakernel-pipeline"),
            layout: Some(&pl_megakernel),
            module: &shader_megakernel,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pop_size,
            max_routes,
            tour_capacity,
            n_locations,
            tour_buf,
            route_starts_buf,
            route_lengths_buf,
            n_routes_buf,
            matrix_buf,
            pipeline_distances,
            bgl_distances,
            problem_uploaded: std::cell::Cell::new(false),
            vehicle_uploaded: std::cell::Cell::new(false),
            problem_service_buf,
            problem_demand_buf,
            problem_tw_start_buf,
            problem_tw_end_buf,
            vehicle_capacity_buf,
            vehicle_tw_start_buf,
            vehicle_tw_end_buf,
            depart_buf,
            latest_arrival_buf,
            load_at_buf,
            feasible_buf,
            route_metrics_buf,
            pipeline_precompute,
            bgl_precompute,
            best_2opt_buf,
            pipeline_find_2opt,
            bgl_find_2opt,
            pipeline_apply_2opt,
            bgl_apply_2opt,
            megakernel_status_buf,
            pipeline_megakernel,
            bgl_megakernel,
            granular_buf,
            granular_uploaded: std::cell::Cell::new(false),
            granular_k: std::cell::Cell::new(0),
            task_pos_buf,
            coords_buf,
            coords_uploaded: std::cell::Cell::new(false),
        })
    }

    /// Upload per-location (x, y) coordinates as f32 pairs (flat array
    /// `coords_xy[i*2..i*2+2]`). When uploaded, megakernel switches to
    /// Euclidean-from-coords distance evaluation, bypassing the i32 matrix
    /// buffer entirely. Distance returned in shader: `i32(sqrt(dx²+dy²) * 100)`.
    pub fn upload_coords(&self, coords_xy: &[f32]) -> Result<(), Error> {
        if (coords_xy.len() as u32) != self.n_locations * 2 {
            return Err(Error::Other(format!(
                "upload_coords: len {} != n_locations × 2 ({})",
                coords_xy.len(),
                self.n_locations * 2
            )));
        }
        self.queue.write_buffer(&self.coords_buf, 0, bytemuck::cast_slice(coords_xy));
        self.coords_uploaded.set(true);
        Ok(())
    }

    pub fn coords_uploaded(&self) -> bool { self.coords_uploaded.get() }

    /// Upload a granular K-nearest-neighbour table. Each location `i`
    /// has up to `k` nearest neighbours stored at `near[i * k + r]`.
    /// Once uploaded, megakernel relocate/exchange will limit candidates
    /// to positions adjacent to a task's K nearest — O(N×K) instead of
    /// O(N²). Essential for scaling to N ≥ 1000.
    ///
    /// `near.len()` must be `n_locations × k`. `k` must be ≤ 64 (MAX_GRANULAR_K).
    pub fn upload_granular(&self, near: &[u32], k: u32) -> Result<(), Error> {
        const MAX_GRANULAR_K: u32 = 64;
        if k == 0 || k > MAX_GRANULAR_K {
            return Err(Error::Other(format!(
                "upload_granular: k={k} out of range [1, {MAX_GRANULAR_K}]"
            )));
        }
        let expected = (self.n_locations * k) as usize;
        if near.len() != expected {
            return Err(Error::Other(format!(
                "upload_granular: near.len()={} != n_locations × k = {expected}",
                near.len()
            )));
        }
        // Pad to MAX_GRANULAR_K stride so kernel can use a constant stride.
        let mut padded = vec![0u32; (self.n_locations * MAX_GRANULAR_K) as usize];
        for i in 0..self.n_locations as usize {
            let src_off = i * k as usize;
            let dst_off = i * MAX_GRANULAR_K as usize;
            padded[dst_off..dst_off + k as usize]
                .copy_from_slice(&near[src_off..src_off + k as usize]);
        }
        self.queue.write_buffer(&self.granular_buf, 0, bytemuck::cast_slice(&padded));
        self.queue.submit(std::iter::empty());
        self.granular_uploaded.set(true);
        self.granular_k.set(k);
        Ok(())
    }

    /// True once `upload_granular` has been called successfully.
    pub fn granular_uploaded(&self) -> bool { self.granular_uploaded.get() }
    pub fn granular_k(&self) -> u32 { self.granular_k.get() }

    pub fn pop_size(&self) -> u32 { self.pop_size }
    pub fn max_routes(&self) -> u32 { self.max_routes }
    pub fn tour_capacity(&self) -> u32 { self.tour_capacity }
    pub fn n_locations(&self) -> u32 { self.n_locations }

    /// Upload `pop_size` trajectories using **fixed-slot layout**: each
    /// route gets a fixed slot of size `slot_size = tour_capacity /
    /// max_routes` regardless of actual length. This avoids cascading
    /// offset updates when remove/insert kernels grow/shrink routes —
    /// each route's slot is independent.
    ///
    /// Constraint: each route must have ≤ `slot_size` stops. The caller
    /// chooses `tour_capacity` at construction time as
    /// `max_routes × max_route_len_with_headroom`.
    pub fn upload(&self, trajectories: &[TrajectoryTours]) -> Result<(), Error> {
        if trajectories.len() != self.pop_size as usize {
            return Err(Error::Other(format!(
                "gpu_population::upload: got {} trajectories, expected pop_size={}",
                trajectories.len(),
                self.pop_size
            )));
        }
        let slot_size = (self.tour_capacity / self.max_routes) as usize;
        if slot_size == 0 {
            return Err(Error::Other(format!(
                "gpu_population::upload: tour_capacity ({}) < max_routes ({}); slot_size = 0",
                self.tour_capacity, self.max_routes
            )));
        }

        let tour_slots = (self.pop_size * self.tour_capacity) as usize;
        let route_slots = (self.pop_size * self.max_routes) as usize;
        let mut tour = vec![0u32; tour_slots];
        let mut starts = vec![0u32; route_slots];
        let mut lengths = vec![0u32; route_slots];
        let mut n_routes = vec![0u32; self.pop_size as usize];

        for (t, traj) in trajectories.iter().enumerate() {
            if traj.len() > self.max_routes as usize {
                return Err(Error::Other(format!(
                    "gpu_population::upload: traj {} has {} routes, max_routes={}",
                    t,
                    traj.len(),
                    self.max_routes
                )));
            }
            let tour_base = t * self.tour_capacity as usize;
            let route_base = t * self.max_routes as usize;
            for (r, route) in traj.iter().enumerate() {
                if route.len() > slot_size {
                    return Err(Error::Other(format!(
                        "gpu_population::upload: traj {t} route {r} has {} stops, \
                         slot_size = {slot_size} (tour_capacity / max_routes = {} / {})",
                        route.len(), self.tour_capacity, self.max_routes
                    )));
                }
                let slot_offset = r * slot_size;
                starts[route_base + r] = slot_offset as u32;
                lengths[route_base + r] = route.len() as u32;
                tour[tour_base + slot_offset..tour_base + slot_offset + route.len()]
                    .copy_from_slice(route);
            }
            n_routes[t] = traj.len() as u32;
        }

        self.queue.write_buffer(&self.tour_buf, 0, bytemuck::cast_slice(&tour));
        self.queue.write_buffer(&self.route_starts_buf, 0, bytemuck::cast_slice(&starts));
        self.queue.write_buffer(&self.route_lengths_buf, 0, bytemuck::cast_slice(&lengths));
        self.queue.write_buffer(&self.n_routes_buf, 0, bytemuck::cast_slice(&n_routes));
        self.queue.submit(std::iter::empty());
        Ok(())
    }

    /// Slot size used by the fixed-slot layout (== `tour_capacity / max_routes`).
    pub fn slot_size(&self) -> u32 {
        self.tour_capacity / self.max_routes
    }

    /// Read back trajectory `traj_idx`, reconstructing it as
    /// `Vec<Vec<u32>>` (the same shape that was uploaded).
    pub fn read_back(&self, traj_idx: u32) -> Result<TrajectoryTours, Error> {
        if traj_idx >= self.pop_size {
            return Err(Error::Other(format!(
                "gpu_population::read_back: traj_idx {} >= pop_size {}",
                traj_idx, self.pop_size
            )));
        }

        // Pull all four arrays at once and slice locally; one round-trip.
        let tour = self.read_buffer_u32(&self.tour_buf, self.pop_size * self.tour_capacity)?;
        let starts = self.read_buffer_u32(&self.route_starts_buf, self.pop_size * self.max_routes)?;
        let lengths = self.read_buffer_u32(&self.route_lengths_buf, self.pop_size * self.max_routes)?;
        let n_routes = self.read_buffer_u32(&self.n_routes_buf, self.pop_size)?;

        let t = traj_idx as usize;
        let nr = n_routes[t] as usize;
        let route_base = t * self.max_routes as usize;
        let tour_base = t * self.tour_capacity as usize;
        let mut out = Vec::with_capacity(nr);
        for r in 0..nr {
            let s = starts[route_base + r] as usize;
            let l = lengths[route_base + r] as usize;
            out.push(tour[tour_base + s..tour_base + s + l].to_vec());
        }
        Ok(out)
    }

    /// Read back every trajectory.
    pub fn read_back_all(&self) -> Result<Vec<TrajectoryTours>, Error> {
        let tour = self.read_buffer_u32(&self.tour_buf, self.pop_size * self.tour_capacity)?;
        let starts = self.read_buffer_u32(&self.route_starts_buf, self.pop_size * self.max_routes)?;
        let lengths = self.read_buffer_u32(&self.route_lengths_buf, self.pop_size * self.max_routes)?;
        let n_routes = self.read_buffer_u32(&self.n_routes_buf, self.pop_size)?;

        let mut all = Vec::with_capacity(self.pop_size as usize);
        for t in 0..self.pop_size as usize {
            let nr = n_routes[t] as usize;
            let route_base = t * self.max_routes as usize;
            let tour_base = t * self.tour_capacity as usize;
            let mut traj = Vec::with_capacity(nr);
            for r in 0..nr {
                let s = starts[route_base + r] as usize;
                let l = lengths[route_base + r] as usize;
                traj.push(tour[tour_base + s..tour_base + s + l].to_vec());
            }
            all.push(traj);
        }
        Ok(all)
    }

    /// Compute the sum of consecutive matrix lookups along each route, on
    /// the GPU, using the persistent state. Returns a flat `pop_size ×
    /// max_routes` array (zero-padded for inactive route slots). Used as
    /// proof-of-life that kernels can correctly read the buffers.
    pub fn route_distances(&self) -> Result<Vec<i32>, Error> {
        let total = (self.pop_size * self.max_routes) as usize;
        let init = vec![0i32; total];
        let out_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-distances-out"),
            contents: bytemuck::cast_slice(&init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params = PopParams {
            pop_size: self.pop_size,
            max_routes: self.max_routes,
            tour_capacity: self.tour_capacity,
            matrix_dim: self.n_locations,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-distances-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pop-distances-bg"),
            layout: &self.bgl_distances,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.route_starts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.route_lengths_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.n_routes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pop-distances-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pop-distances-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_distances);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.pop_size, self.max_routes, 1);
        }

        let out_size = (total * std::mem::size_of::<i32>()) as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pop-distances-readback"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| Error::Other(format!("gpu_population: distances map recv: {e}")))?
            .map_err(|e| Error::Other(format!("gpu_population: distances map: {e:?}")))?;

        let data = slice.get_mapped_range();
        let arr: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        Ok(arr)
    }

    fn read_buffer_u32(&self, buf: &wgpu::Buffer, n_u32: u32) -> Result<Vec<u32>, Error> {
        let size = (n_u32 as u64) * 4;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pop-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pop-readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(buf, 0, &readback, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| Error::Other(format!("gpu_population: read map recv: {e}")))?
            .map_err(|e| Error::Other(format!("gpu_population: read map: {e:?}")))?;
        let data = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        Ok(v)
    }
}

// ---- Phase 2 API on GpuPopulation ----

impl GpuPopulation {
    /// Upload per-location problem data. Each slice must have length
    /// `n_locations`. Depots should have service=0, demand=0, and TW set
    /// to the vehicle's TW (or any value — depots are never read for TW
    /// in the kernel since position 0 and len-1 use vehicle TW directly).
    pub fn upload_problem_data(
        &self,
        service: &[i32],
        demand: &[i32],
        tw_start: &[i32],
        tw_end: &[i32],
    ) -> Result<(), Error> {
        let n = self.n_locations as usize;
        for (name, slc) in [
            ("service", service),
            ("demand", demand),
            ("tw_start", tw_start),
            ("tw_end", tw_end),
        ] {
            if slc.len() != n {
                return Err(Error::Other(format!(
                    "gpu_population::upload_problem_data: {name} len {} != n_locations {n}",
                    slc.len()
                )));
            }
        }
        self.queue.write_buffer(&self.problem_service_buf, 0, bytemuck::cast_slice(service));
        self.queue.write_buffer(&self.problem_demand_buf, 0, bytemuck::cast_slice(demand));
        self.queue.write_buffer(&self.problem_tw_start_buf, 0, bytemuck::cast_slice(tw_start));
        self.queue.write_buffer(&self.problem_tw_end_buf, 0, bytemuck::cast_slice(tw_end));
        self.queue.submit(std::iter::empty());
        self.problem_uploaded.set(true);
        Ok(())
    }

    /// Upload per-trajectory vehicle data. Phase 2 assumes a single
    /// (homogeneous) vehicle config per trajectory. `capacity` is one
    /// value per trajectory; later phases will switch to per-route
    /// indirection.
    pub fn upload_vehicle_data(
        &self,
        capacity: &[i32],
        tw_start: &[i32],
        tw_end: &[i32],
    ) -> Result<(), Error> {
        let p = self.pop_size as usize;
        for (name, slc) in [
            ("capacity", capacity),
            ("tw_start", tw_start),
            ("tw_end", tw_end),
        ] {
            if slc.len() != p {
                return Err(Error::Other(format!(
                    "gpu_population::upload_vehicle_data: {name} len {} != pop_size {p}",
                    slc.len()
                )));
            }
        }
        self.queue.write_buffer(&self.vehicle_capacity_buf, 0, bytemuck::cast_slice(capacity));
        self.queue.write_buffer(&self.vehicle_tw_start_buf, 0, bytemuck::cast_slice(tw_start));
        self.queue.write_buffer(&self.vehicle_tw_end_buf, 0, bytemuck::cast_slice(tw_end));
        self.queue.submit(std::iter::empty());
        self.vehicle_uploaded.set(true);
        Ok(())
    }

    /// Run forward + backward TW pass and load simulation on every active
    /// route across all trajectories. Each route slot in the tour buffer
    /// must be laid out as `[depot, stop_1, ..., stop_L, depot]`. Updates
    /// the resident `depart`, `latest_arrival`, `load_at`, `feasible`
    /// buffers in place. CPU does not see anything until `read_precompute`.
    pub fn precompute_all(&self) -> Result<(), Error> {
        if !self.problem_uploaded.get() {
            return Err(Error::Other(
                "gpu_population::precompute_all: call upload_problem_data first".into(),
            ));
        }
        if !self.vehicle_uploaded.get() {
            return Err(Error::Other(
                "gpu_population::precompute_all: call upload_vehicle_data first".into(),
            ));
        }

        let params = PopParams {
            pop_size: self.pop_size,
            max_routes: self.max_routes,
            tour_capacity: self.tour_capacity,
            matrix_dim: self.n_locations,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-precompute-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pop-precompute-bg"),
            layout: &self.bgl_precompute,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.route_starts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.route_lengths_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.n_routes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.problem_service_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.problem_demand_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.problem_tw_start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.problem_tw_end_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.vehicle_capacity_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.vehicle_tw_start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.vehicle_tw_end_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.depart_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: self.latest_arrival_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: self.load_at_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: self.feasible_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: self.route_metrics_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 17, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pop-precompute-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pop-precompute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_precompute);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.pop_size, self.max_routes, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// Read back the precompute outputs for one route.
    pub fn read_precompute(&self, traj_idx: u32, route_idx: u32) -> Result<GpuRoutePrecomp, Error> {
        if traj_idx >= self.pop_size || route_idx >= self.max_routes {
            return Err(Error::Other(format!(
                "gpu_population::read_precompute: out of range ({}, {})",
                traj_idx, route_idx
            )));
        }
        let starts = self.read_buffer_u32(&self.route_starts_buf, self.pop_size * self.max_routes)?;
        let lengths = self.read_buffer_u32(&self.route_lengths_buf, self.pop_size * self.max_routes)?;
        let n_routes = self.read_buffer_u32(&self.n_routes_buf, self.pop_size)?;
        let depart = self.read_buffer_i32(&self.depart_buf, self.pop_size * self.tour_capacity)?;
        let latest = self.read_buffer_i32(&self.latest_arrival_buf, self.pop_size * self.tour_capacity)?;
        let load_at = self.read_buffer_i32(&self.load_at_buf, self.pop_size * self.tour_capacity)?;
        let feasible = self.read_buffer_u32(&self.feasible_buf, self.pop_size * self.max_routes)?;

        let t = traj_idx as usize;
        let r = route_idx as usize;
        if (r as u32) >= n_routes[t] {
            return Ok(GpuRoutePrecomp::default());
        }
        let route_off = t * self.max_routes as usize + r;
        let s = starts[route_off] as usize;
        let l = lengths[route_off] as usize;
        let tour_base = t * self.tour_capacity as usize;
        Ok(GpuRoutePrecomp {
            depart: depart[tour_base + s..tour_base + s + l].to_vec(),
            latest_arrival: latest[tour_base + s..tour_base + s + l].to_vec(),
            load_at: load_at[tour_base + s..tour_base + s + l].to_vec(),
            feasible: feasible[route_off] != 0,
        })
    }

    fn read_buffer_i32(&self, buf: &wgpu::Buffer, n: u32) -> Result<Vec<i32>, Error> {
        let v = self.read_buffer_u32(buf, n)?;
        Ok(v.into_iter().map(|x| x as i32).collect())
    }

    /// Read back per-route metrics for one route. Inactive routes return
    /// the default-zero struct.
    pub fn read_route_metrics(&self, traj_idx: u32, route_idx: u32) -> Result<GpuRouteMetrics, Error> {
        if traj_idx >= self.pop_size || route_idx >= self.max_routes {
            return Err(Error::Other(format!(
                "gpu_population::read_route_metrics: out of range ({}, {})",
                traj_idx, route_idx
            )));
        }
        let metrics = self.read_buffer_i32(&self.route_metrics_buf, self.pop_size * self.max_routes * 5)?;
        let feasible = self.read_buffer_u32(&self.feasible_buf, self.pop_size * self.max_routes)?;
        let n_routes = self.read_buffer_u32(&self.n_routes_buf, self.pop_size)?;
        if route_idx >= n_routes[traj_idx as usize] {
            return Ok(GpuRouteMetrics::default());
        }
        let off = (traj_idx as usize * self.max_routes as usize + route_idx as usize) * 5;
        let f_off = traj_idx as usize * self.max_routes as usize + route_idx as usize;
        Ok(GpuRouteMetrics {
            travel_time: metrics[off],
            service_time: metrics[off + 1],
            waiting_time: metrics[off + 2],
            distance: metrics[off + 3],
            end_time: metrics[off + 4],
            feasible: feasible[f_off] != 0,
        })
    }

    /// Find the best 2-opt move (distance-only) for every active route
    /// across all trajectories. The result is written to a per-route
    /// best buffer that you can read with `read_best_2opt_per_route` or
    /// `read_best_2opt_all`. Distance-only — does NOT check TW
    /// feasibility; the caller must validate after `apply_2opt`.
    pub fn find_best_2opt_all(&self) -> Result<(), Error> {
        // Reset best buffer to {INT_MAX, 0, 0} per route.
        let total_routes = (self.pop_size * self.max_routes) as usize;
        let mut init = Vec::with_capacity(total_routes * 3);
        for _ in 0..total_routes {
            init.extend_from_slice(&[i32::MAX, 0, 0]);
        }
        self.queue.write_buffer(&self.best_2opt_buf, 0, bytemuck::cast_slice(&init));

        let params = PopParams {
            pop_size: self.pop_size,
            max_routes: self.max_routes,
            tour_capacity: self.tour_capacity,
            matrix_dim: self.n_locations,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-find2opt-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pop-find2opt-bg"),
            layout: &self.bgl_find_2opt,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.route_starts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.route_lengths_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.n_routes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.best_2opt_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pop-find2opt-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pop-find2opt-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_find_2opt);
            pass.set_bind_group(0, &bg, &[]);
            // One workgroup per (trajectory, route).
            pass.dispatch_workgroups(self.pop_size, self.max_routes, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// Read back the per-route best-2opt records into a flat
    /// `pop_size × max_routes` Vec, one entry per route slot.
    pub fn read_best_2opt_all(&self) -> Result<Vec<GpuBest2opt>, Error> {
        let raw = self.read_buffer_i32(&self.best_2opt_buf, self.pop_size * self.max_routes * 3)?;
        let n_routes = self.read_buffer_u32(&self.n_routes_buf, self.pop_size)?;
        let mut out = Vec::with_capacity((self.pop_size * self.max_routes) as usize);
        for t in 0..self.pop_size as usize {
            for r in 0..self.max_routes as usize {
                if (r as u32) >= n_routes[t] {
                    out.push(GpuBest2opt::default());
                    continue;
                }
                let off = (t * self.max_routes as usize + r) * 3;
                let delta = raw[off];
                let i = raw[off + 1] as u32;
                let j = raw[off + 2] as u32;
                out.push(GpuBest2opt { delta, i, j });
            }
        }
        Ok(out)
    }

    /// Apply a 2-opt move (reverse the segment `tour[start+i+1..=start+j]`)
    /// to one route on the GPU, in place. Caller is responsible for
    /// re-running `precompute_all` if they want updated metrics.
    pub fn apply_2opt(&self, traj_idx: u32, route_idx: u32, i: u32, j: u32) -> Result<(), Error> {
        if traj_idx >= self.pop_size || route_idx >= self.max_routes {
            return Err(Error::Other(format!(
                "apply_2opt: out of range ({}, {})",
                traj_idx, route_idx
            )));
        }
        if j <= i + 1 {
            return Err(Error::Other(format!(
                "apply_2opt: j must be > i+1 (i={}, j={})",
                i, j
            )));
        }

        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable, Debug)]
        struct ApplyParams {
            traj: u32,
            route: u32,
            i: u32,
            j: u32,
            max_routes: u32,
            tour_capacity: u32,
            _pad0: u32,
            _pad1: u32,
        }
        let params = ApplyParams {
            traj: traj_idx,
            route: route_idx,
            i, j,
            max_routes: self.max_routes,
            tour_capacity: self.tour_capacity,
            _pad0: 0, _pad1: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-apply2opt-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pop-apply2opt-bg"),
            layout: &self.bgl_apply_2opt,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.route_starts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pop-apply2opt-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pop-apply2opt-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_apply_2opt);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// Run the **entire** 2-opt LS loop in a single GPU dispatch via
    /// the megakernel. Eliminates per-iteration dispatch barriers (~5
    /// sync points × 1.5 ms each) at the cost of constraining the loop
    /// to a single 1024-thread workgroup.
    ///
    /// **TW-aware**: each candidate move is checked against per-stop and
    /// vehicle time-windows in-kernel (uses `service` + `tw_start` +
    /// `tw_end` + `vehicle_tw_*`). Capacity is unaffected by 2-opt so we
    /// don't re-validate. Caller must `upload_problem_data` and
    /// `upload_vehicle_data` first.
    ///
    /// Returns (iters_run, applies, final_best_delta) read from the
    /// status buffer.
    pub fn run_megakernel_2opt(
        &self,
        traj_idx: u32,
        max_iters: u32,
    ) -> Result<(u32, u32, i32), Error> {
        if traj_idx >= self.pop_size {
            return Err(Error::Other(format!(
                "run_megakernel_2opt: traj_idx {} >= pop_size {}",
                traj_idx, self.pop_size
            )));
        }
        if !self.problem_uploaded.get() {
            return Err(Error::Other(
                "run_megakernel_2opt: call upload_problem_data first".into(),
            ));
        }
        if !self.vehicle_uploaded.get() {
            return Err(Error::Other(
                "run_megakernel_2opt: call upload_vehicle_data first".into(),
            ));
        }

        // Reset status buffer (entire pop_size × 4 i32 array).
        let zero = vec![0i32; (self.pop_size * 4) as usize];
        self.queue.write_buffer(&self.megakernel_status_buf, 0, bytemuck::cast_slice(&zero));

        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable, Debug)]
        struct MegaParams {
            traj: u32,
            max_routes: u32,
            tour_capacity: u32,
            matrix_dim: u32,
            max_iters: u32,
            batch_mode: u32,
            kick_count: u32,
            kick_seed: u32,
            gk: u32,
            coord_mode: u32,
            penalty_mode: u32,
            _pad3: u32,
        }
        let params = MegaParams {
            traj: traj_idx,
            max_routes: self.max_routes,
            tour_capacity: self.tour_capacity,
            matrix_dim: self.n_locations,
            max_iters,
            batch_mode: 0,
            kick_count: 0,
            kick_seed: 0,
            gk: if self.granular_uploaded.get() { self.granular_k.get() } else { 0 },
            coord_mode: if self.coords_uploaded.get() { 1 } else { 0 },
            penalty_mode: if std::env::var("BROOOM_PENALTY_LS").map(|v| v == "1").unwrap_or(false) { 1 } else { 0 },
            _pad3: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-megakernel-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pop-megakernel-bg"),
            layout: &self.bgl_megakernel,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.route_starts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.route_lengths_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.n_routes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.megakernel_status_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.problem_service_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.problem_tw_start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.problem_tw_end_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.vehicle_tw_start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.vehicle_tw_end_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.problem_demand_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.vehicle_capacity_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: self.granular_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: self.task_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: self.coords_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pop-megakernel-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pop-megakernel-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_megakernel);
            pass.set_bind_group(0, &bg, &[]);
            // Single workgroup — everything happens in one cooperating cohort.
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let status = self.read_buffer_i32(&self.megakernel_status_buf, self.pop_size * 4)?;
        let off = (traj_idx as usize) * 4;
        Ok((status[off] as u32, status[off + 1] as u32, status[off + 2]))
    }

    /// Run megakernel in chunks of `chunk_iters` per dispatch, totaling at
    /// most `max_iters` iterations. Stops early when the kernel reports no
    /// further improving move (final_delta == 0). This avoids Metal's GPU
    /// watchdog (≈5-10s per command buffer on macOS) for large instances
    /// where a single dispatch would otherwise time out.
    ///
    /// Returns (total_iters, total_applies, last_delta).
    pub fn run_megakernel_2opt_chunked(
        &self,
        traj_idx: u32,
        max_iters: u32,
        chunk_iters: u32,
    ) -> Result<(u32, u32, i32), Error> {
        let chunk = chunk_iters.max(1);
        let mut total_iters: u32 = 0;
        let mut total_applies: u32 = 0;
        let mut last_delta: i32 = 0;
        while total_iters < max_iters {
            let remaining = max_iters - total_iters;
            let this_chunk = chunk.min(remaining);
            let (iters, applies, delta) = self.run_megakernel_2opt(traj_idx, this_chunk)?;
            total_iters += iters;
            total_applies += applies;
            last_delta = delta;
            // If the chunk didn't use all its iterations or found no improving move, we've converged.
            if iters < this_chunk || delta == 0 {
                break;
            }
        }
        Ok((total_iters, total_applies, last_delta))
    }

    /// Batch megakernel WITH ILS-kick (proper destroy-and-repair): each
    /// workgroup removes `kick_count` random tasks, then reinserts each
    /// at its cheapest TW-feasible position via parallel argmin across
    /// all (route, position) candidates. Then runs the 2-opt LS-loop on
    /// the perturbed-and-repaired starting point.
    ///
    /// If a pulled task has no feasible insertion anywhere, it is
    /// dropped and counted in the `dropped` field of each result tuple.
    /// Trajectories with dropped > 0 should be excluded from best-of-N.
    pub fn run_megakernel_2opt_batch_with_kick(
        &self,
        max_iters: u32,
        kick_count: u32,
        kick_seed: u32,
    ) -> Result<Vec<MegakernelStatus>, Error> {
        self.run_megakernel_internal(max_iters, true, kick_count, kick_seed)
    }

    /// Run the megakernel in **batch mode**: dispatches `pop_size`
    /// workgroups, each running the LS loop on its own trajectory in
    /// parallel. All trajectories share the same matrix and per-location
    /// problem data, so this is true Phase-8 population mode (different
    /// initial tours / perturbations exploring the same problem).
    ///
    /// Returns per-trajectory `(iters, applies, final_delta)` triples.
    pub fn run_megakernel_2opt_batch(
        &self,
        max_iters: u32,
    ) -> Result<Vec<MegakernelStatus>, Error> {
        self.run_megakernel_internal(max_iters, true, 0, 0)
    }

    fn run_megakernel_internal(
        &self,
        max_iters: u32,
        batch: bool,
        kick_count: u32,
        kick_seed: u32,
    ) -> Result<Vec<MegakernelStatus>, Error> {
        if !self.problem_uploaded.get() {
            return Err(Error::Other(
                "run_megakernel_2opt_batch: call upload_problem_data first".into(),
            ));
        }
        if !self.vehicle_uploaded.get() {
            return Err(Error::Other(
                "run_megakernel_2opt_batch: call upload_vehicle_data first".into(),
            ));
        }

        let zero = vec![0i32; (self.pop_size * 4) as usize];
        self.queue.write_buffer(&self.megakernel_status_buf, 0, bytemuck::cast_slice(&zero));

        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable, Debug)]
        struct MegaParams {
            traj: u32,
            max_routes: u32,
            tour_capacity: u32,
            matrix_dim: u32,
            max_iters: u32,
            batch_mode: u32,
            kick_count: u32,
            kick_seed: u32,
            gk: u32,
            coord_mode: u32,
            penalty_mode: u32,
            _pad3: u32,
        }
        let params = MegaParams {
            traj: 0,
            max_routes: self.max_routes,
            tour_capacity: self.tour_capacity,
            matrix_dim: self.n_locations,
            max_iters,
            batch_mode: if batch { 1 } else { 0 },
            kick_count,
            kick_seed,
            gk: if self.granular_uploaded.get() { self.granular_k.get() } else { 0 },
            coord_mode: if self.coords_uploaded.get() { 1 } else { 0 },
            penalty_mode: if std::env::var("BROOOM_PENALTY_LS").map(|v| v == "1").unwrap_or(false) { 1 } else { 0 },
            _pad3: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pop-megakernel-batch-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pop-megakernel-batch-bg"),
            layout: &self.bgl_megakernel,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tour_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.route_starts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.route_lengths_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.n_routes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.megakernel_status_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.problem_service_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.problem_tw_start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.problem_tw_end_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.vehicle_tw_start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.vehicle_tw_end_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.problem_demand_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.vehicle_capacity_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: self.granular_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: self.task_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: self.coords_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pop-megakernel-batch-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pop-megakernel-batch-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_megakernel);
            pass.set_bind_group(0, &bg, &[]);
            // pop_size workgroups — one trajectory per workgroup.
            pass.dispatch_workgroups(self.pop_size, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let status = self.read_buffer_i32(&self.megakernel_status_buf, self.pop_size * 4)?;
        let mut out = Vec::with_capacity(self.pop_size as usize);
        for t in 0..self.pop_size as usize {
            let off = t * 4;
            out.push(MegakernelStatus {
                iters: status[off] as u32,
                applies: status[off + 1] as u32,
                final_delta: status[off + 2],
                dropped: status[off + 3] as u32,
            });
        }
        Ok(out)
    }

    /// Read back metrics for every active route in every trajectory.
    /// Returns `Vec<Vec<GpuRouteMetrics>>` with the outer length =
    /// pop_size and inner length = `n_routes[t]`.
    pub fn read_all_route_metrics(&self) -> Result<Vec<Vec<GpuRouteMetrics>>, Error> {
        let metrics = self.read_buffer_i32(&self.route_metrics_buf, self.pop_size * self.max_routes * 5)?;
        let feasible = self.read_buffer_u32(&self.feasible_buf, self.pop_size * self.max_routes)?;
        let n_routes = self.read_buffer_u32(&self.n_routes_buf, self.pop_size)?;
        let mut out = Vec::with_capacity(self.pop_size as usize);
        for t in 0..self.pop_size as usize {
            let nr = n_routes[t] as usize;
            let mut traj = Vec::with_capacity(nr);
            for r in 0..nr {
                let f_off = t * self.max_routes as usize + r;
                let off = f_off * 5;
                traj.push(GpuRouteMetrics {
                    travel_time: metrics[off],
                    service_time: metrics[off + 1],
                    waiting_time: metrics[off + 2],
                    distance: metrics[off + 3],
                    end_time: metrics[off + 4],
                    feasible: feasible[f_off] != 0,
                });
            }
            out.push(traj);
        }
        Ok(out)
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

/// Per-route, single-thread distance-sum kernel. One workgroup of size 1
/// per (trajectory, route) cell. Inactive cells (route_idx ≥ n_routes[t])
/// emit 0. Used to verify that persistent buffers are correctly indexed.
const SHADER_DISTANCES_WGSL: &str = r#"
struct PopParams {
    pop_size: u32,
    max_routes: u32,
    tour_capacity: u32,
    matrix_dim: u32,
};

@group(0) @binding(0) var<storage, read> tour: array<u32>;
@group(0) @binding(1) var<storage, read> route_starts: array<u32>;
@group(0) @binding(2) var<storage, read> route_lengths: array<u32>;
@group(0) @binding(3) var<storage, read> n_routes: array<u32>;
@group(0) @binding(4) var<storage, read> matrix: array<i32>;
@group(0) @binding(5) var<storage, read_write> out_distances: array<i32>;
@group(0) @binding(6) var<uniform> params: PopParams;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let t = wg.x;
    let r = wg.y;
    if (t >= params.pop_size) { return; }
    if (r >= params.max_routes) { return; }

    let out_idx = t * params.max_routes + r;
    if (r >= n_routes[t]) {
        out_distances[out_idx] = 0;
        return;
    }

    let route_idx = t * params.max_routes + r;
    let start = route_starts[route_idx];
    let len = route_lengths[route_idx];
    if (len < 2u) {
        out_distances[out_idx] = 0;
        return;
    }

    let tour_base = t * params.tour_capacity;
    var sum: i32 = 0;
    var i: u32 = 0u;
    loop {
        if (i + 1u >= len) { break; }
        let a = tour[tour_base + start + i];
        let b = tour[tour_base + start + i + 1u];
        sum = sum + matrix[a * params.matrix_dim + b];
        i = i + 1u;
    }
    out_distances[out_idx] = sum;
}
"#;

/// Phase-2 precompute kernel.
///
/// One workgroup of size 1 per `(trajectory, route)` cell. The body is
/// purely sequential within a route: forward TW pass, then backward TW
/// pass, then load simulation. Routes are short enough (< 50 stops) that
/// 64-thread cooperation isn't worthwhile yet.
///
/// Tour layout: `[depot, stop_1, ..., stop_L, depot]`. The kernel
/// recognises position 0 and len-1 as depots — they have no service,
/// no demand, and use the vehicle's TW.
///
/// Outputs are written into the persistent buffers in place; the slot
/// for an inactive route (route_idx ≥ n_routes[t]) is left untouched.
const SHADER_PRECOMPUTE_WGSL: &str = r#"
struct PopParams {
    pop_size: u32,
    max_routes: u32,
    tour_capacity: u32,
    matrix_dim: u32,
};

@group(0) @binding(0) var<storage, read> tour: array<u32>;
@group(0) @binding(1) var<storage, read> route_starts: array<u32>;
@group(0) @binding(2) var<storage, read> route_lengths: array<u32>;
@group(0) @binding(3) var<storage, read> n_routes: array<u32>;
@group(0) @binding(4) var<storage, read> matrix: array<i32>;
@group(0) @binding(5) var<storage, read> service: array<i32>;
@group(0) @binding(6) var<storage, read> demand: array<i32>;
@group(0) @binding(7) var<storage, read> tw_start: array<i32>;
@group(0) @binding(8) var<storage, read> tw_end: array<i32>;
@group(0) @binding(9) var<storage, read> vehicle_capacity: array<i32>;
@group(0) @binding(10) var<storage, read> vehicle_tw_start: array<i32>;
@group(0) @binding(11) var<storage, read> vehicle_tw_end: array<i32>;
@group(0) @binding(12) var<storage, read_write> depart: array<i32>;
@group(0) @binding(13) var<storage, read_write> latest_arrival: array<i32>;
@group(0) @binding(14) var<storage, read_write> load_at: array<i32>;
@group(0) @binding(15) var<storage, read_write> feasible: array<u32>;
@group(0) @binding(16) var<storage, read_write> route_metrics: array<i32>;
@group(0) @binding(17) var<uniform> params: PopParams;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let t = wg.x;
    let r = wg.y;
    if (t >= params.pop_size) { return; }
    if (r >= params.max_routes) { return; }
    if (r >= n_routes[t]) { return; }

    let route_off = t * params.max_routes + r;
    let s = route_starts[route_off];
    let len = route_lengths[route_off];
    if (len < 2u) {
        feasible[route_off] = 0u;
        return;
    }

    let tour_base = t * params.tour_capacity;
    let md = params.matrix_dim;

    let veh_cap = vehicle_capacity[t];
    let veh_tw_s = vehicle_tw_start[t];
    let veh_tw_e = vehicle_tw_end[t];

    var feas: u32 = 1u;

    // Initial load = sum of all delivery amounts (all interior stops).
    var init_load: i32 = 0;
    for (var k: u32 = 1u; k + 1u < len; k = k + 1u) {
        let loc = tour[tour_base + s + k];
        init_load = init_load + demand[loc];
    }
    if (init_load > veh_cap) { feas = 0u; }

    // Forward TW pass + load_at + per-route metric accumulation.
    var current_t: i32 = veh_tw_s;
    var cur_load: i32 = init_load;
    var travel_total: i32 = 0;
    var service_total: i32 = 0;
    var waiting_total: i32 = 0;
    var distance_total: i32 = 0;

    // Position 0 = start depot.
    depart[tour_base + s + 0u] = current_t;
    load_at[tour_base + s + 0u] = cur_load;
    var prev_loc: u32 = tour[tour_base + s + 0u];

    for (var k: u32 = 1u; k < len; k = k + 1u) {
        let here = tour[tour_base + s + k];
        let edge = matrix[prev_loc * md + here];
        current_t = current_t + edge;
        travel_total = travel_total + edge;
        distance_total = distance_total + edge;

        if (k + 1u < len) {
            // Customer stop.
            let tw_s_here = tw_start[here];
            let tw_e_here = tw_end[here];
            if (current_t < tw_s_here) {
                waiting_total = waiting_total + (tw_s_here - current_t);
                current_t = tw_s_here;
            }
            if (current_t > tw_e_here) { feas = 0u; }
            cur_load = cur_load - demand[here];
            if (cur_load < 0) { feas = 0u; }
            load_at[tour_base + s + k] = cur_load;
            let svc = service[here];
            current_t = current_t + svc;
            service_total = service_total + svc;
        } else {
            // End depot — no service, no demand change.
            load_at[tour_base + s + k] = cur_load;
        }

        depart[tour_base + s + k] = current_t;
        prev_loc = here;
    }

    if (current_t > veh_tw_e) { feas = 0u; }

    // Write metrics (5 i32 per route).
    let m_off = route_off * 5u;
    route_metrics[m_off + 0u] = travel_total;
    route_metrics[m_off + 1u] = service_total;
    route_metrics[m_off + 2u] = waiting_total;
    route_metrics[m_off + 3u] = distance_total;
    route_metrics[m_off + 4u] = current_t;

    // Backward TW pass.
    latest_arrival[tour_base + s + len - 1u] = veh_tw_e;
    // Iterate k = len-2, len-3, ..., 0.
    for (var k_off: u32 = 0u; k_off + 1u < len; k_off = k_off + 1u) {
        let k = len - 2u - k_off;
        let here = tour[tour_base + s + k];
        let next_loc = tour[tour_base + s + k + 1u];
        let edge = matrix[here * md + next_loc];

        // service at this position (depots have 0).
        var s_here: i32 = 0;
        if (k > 0u && k + 1u < len) { s_here = service[here]; }
        let chain = latest_arrival[tour_base + s + k + 1u] - s_here - edge;

        // TW end at this position (depot uses vehicle TW end).
        var tw_e_here: i32 = veh_tw_e;
        if (k > 0u && k + 1u < len) { tw_e_here = tw_end[here]; }

        var lat: i32 = chain;
        if (tw_e_here < lat) { lat = tw_e_here; }
        latest_arrival[tour_base + s + k] = lat;
    }

    feasible[route_off] = feas;
}
"#;

/// Phase-4 find-best-2opt kernel.
///
/// One workgroup of size 64 per `(trajectory, route)` cell. Threads
/// stride over `(i, j)` pairs and atomic-min into the per-route best
/// slot. Tour layout: `[depot, stop_1, ..., stop_L, depot]` — we only
/// consider swaps with `j+1 < len` so the closing depot edge stays
/// pinned. Distance-only (no TW check); caller validates after apply.
const SHADER_FIND_2OPT_WGSL: &str = r#"
struct PopParams {
    pop_size: u32,
    max_routes: u32,
    tour_capacity: u32,
    matrix_dim: u32,
};

@group(0) @binding(0) var<storage, read> tour: array<u32>;
@group(0) @binding(1) var<storage, read> route_starts: array<u32>;
@group(0) @binding(2) var<storage, read> route_lengths: array<u32>;
@group(0) @binding(3) var<storage, read> n_routes: array<u32>;
@group(0) @binding(4) var<storage, read> matrix: array<i32>;
@group(0) @binding(5) var<storage, read_write> best_2opt: array<atomic<i32>>;
@group(0) @binding(6) var<uniform> params: PopParams;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = wg.x;
    let r = wg.y;
    if (t >= params.pop_size || r >= params.max_routes) { return; }
    if (r >= n_routes[t]) { return; }

    let route_off = t * params.max_routes + r;
    let s = route_starts[route_off];
    let len = route_lengths[route_off];
    if (len < 4u) { return; }  // need depot + 2 stops + depot at least

    let tour_base = t * params.tour_capacity;
    let md = params.matrix_dim;
    let tid = lid.x;
    let n_pairs = len * len;

    var k: u32 = tid;
    loop {
        if (k >= n_pairs) { break; }
        let i = k / len;
        let j = k % len;
        // valid 2-opt: j > i + 1, j + 1 < len (don't reverse closing depot edge),
        // and i not on the closing depot.
        if (j > i + 1u && j + 1u < len && i + 1u < len) {
            let a = tour[tour_base + s + i];
            let b = tour[tour_base + s + i + 1u];
            let c = tour[tour_base + s + j];
            let d = tour[tour_base + s + j + 1u];

            let old_cost = matrix[a * md + b] + matrix[c * md + d];
            let new_cost = matrix[a * md + c] + matrix[b * md + d];
            let delta = new_cost - old_cost;

            let bo = route_off * 3u;
            let prev = atomicMin(&best_2opt[bo], delta);
            if (delta < prev) {
                atomicStore(&best_2opt[bo + 1u], i32(i));
                atomicStore(&best_2opt[bo + 2u], i32(j));
            }
        }
        k = k + 64u;
    }
}
"#;

/// **Megakernel:** the entire 2-opt LS loop in one dispatch. Runs in a
/// single 1024-thread workgroup; threads cooperate over (route, i, j)
/// candidates via workgroup-shared atomic argmin, then thread 0 applies
/// the winning move sequentially. Loop continues until no improving move
/// remains or `max_iters` is exceeded.
///
/// Distance-only — does NOT check TW feasibility inside the loop. The
/// caller must validate via `precompute_all` after the kernel returns
/// and rollback if necessary.
///
/// Status output (4 i32 at binding 5):
///   [0] = iterations run
///   [1] = moves applied (== iter_count if every iter improved)
///   [2] = final best delta (>= 0 if converged, otherwise == max_iters)
///   [3] = reserved
const SHADER_MEGAKERNEL_WGSL: &str = r#"
struct MegaParams {
    traj: u32,
    max_routes: u32,
    tour_capacity: u32,
    matrix_dim: u32,
    max_iters: u32,
    /// 0 = single-trajectory (use `traj` field). 1 = batch mode (each
    /// workgroup handles a different trajectory via workgroup_id.x).
    batch_mode: u32,
    /// Number of random inter-route task swaps to perform before LS.
    kick_count: u32,
    kick_seed: u32,
    /// Granular K. 0 disables granular (full N²-search fallback).
    /// Stride in `granular[]` is always 64 (MAX_GRANULAR_K).
    gk: u32,
    /// 0 = read distances from `dist_at()`. 1 = compute Euclidean × 100
    /// from `coords[]` on the fly (needed for N≥16K where matrix would
    /// exceed buffer limits).
    coord_mode: u32,
    /// 0 = strict feasibility (reject moves that violate TW or capacity).
    /// 1 = penalty mode (accept violations with a cost penalty so LS can
    /// hop through infeasible intermediates to better feasible optima).
    /// Penalty mode is currently honored by Phase 2b granular relocate.
    penalty_mode: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read_write> tour: array<u32>;
@group(0) @binding(1) var<storage, read> route_starts: array<u32>;
@group(0) @binding(2) var<storage, read_write> route_lengths: array<u32>;
@group(0) @binding(3) var<storage, read> n_routes: array<u32>;
@group(0) @binding(4) var<storage, read> matrix: array<i32>;
@group(0) @binding(5) var<storage, read_write> status: array<i32>;
@group(0) @binding(6) var<storage, read> service: array<i32>;
@group(0) @binding(7) var<storage, read> tw_start: array<i32>;
@group(0) @binding(8) var<storage, read> tw_end: array<i32>;
@group(0) @binding(9) var<storage, read> vehicle_tw_start: array<i32>;
@group(0) @binding(10) var<storage, read> vehicle_tw_end: array<i32>;
@group(0) @binding(11) var<storage, read> demand: array<i32>;
@group(0) @binding(12) var<storage, read> vehicle_capacity: array<i32>;
@group(0) @binding(13) var<storage, read> granular: array<u32>;
@group(0) @binding(14) var<storage, read_write> task_pos: array<u32>;
@group(0) @binding(15) var<uniform> params: MegaParams;
@group(0) @binding(16) var<storage, read> coords: array<f32>;

// Distance accessor: returns the i32 distance between locations a and b.
// In matrix-mode (default) reads from `dist_at(a * matrix_dim + b)`. In
// coord-mode (params.coord_mode == 1) computes Euclidean × 100 from
// (x,y) pairs in `coords[]`. The branch is on a uniform value, so the
// scheduler treats both paths uniformly across the warp.
fn dist_at(idx: u32) -> i32 {
    if (params.coord_mode == 0u) {
        return matrix[idx];
    }
    let a = idx / params.matrix_dim;
    let b = idx % params.matrix_dim;
    let ax = coords[a * 2u];
    let ay = coords[a * 2u + 1u];
    let bx = coords[b * 2u];
    let by = coords[b * 2u + 1u];
    let dx = ax - bx;
    let dy = ay - by;
    return i32(sqrt(dx * dx + dy * dy) * 100.0);
}

// Workgroup-shared state: best candidate this iter + done flag.
// Op-type encodes which operator owns the best slot:
//   0 = 2-opt   (best_route, best_i, best_j; best_d unused)
//   1 = relocate (best_route=src_r, best_i=src_i, best_j=dst_r, best_d=dst_p)
var<workgroup> wg_best_delta: atomic<i32>;
var<workgroup> wg_best_op: atomic<u32>;
var<workgroup> wg_best_route: atomic<u32>;
var<workgroup> wg_best_i: atomic<u32>;
var<workgroup> wg_best_j: atomic<u32>;
var<workgroup> wg_best_d: atomic<u32>;
var<workgroup> wg_done: atomic<u32>;

// ILS-kick: pulled tasks' workgroup-shared limbo (max 16 per kick).
var<workgroup> wg_limbo: array<u32, 16>;
var<workgroup> wg_limbo_count: u32;
var<workgroup> wg_dropped_count: u32;

// Reinsert phase atomics.
var<workgroup> wg_insert_cost: atomic<i32>;
var<workgroup> wg_insert_route: atomic<u32>;
var<workgroup> wg_insert_pos: atomic<u32>;

// Per-route current capacity load (sum of demands of interior tasks),
// recomputed at the start of each LS iter so relocate-search can check
// capacity in O(1) per candidate. Capped at 64 routes.
var<workgroup> wg_route_load: array<i32, 512>;

// Scratch buffers for 2-opt* apply: save A's tail and B's tail before
// overwriting in place. Sized at 64 to fit any reasonable route slot.
var<workgroup> wg_a_tail: array<u32, 64>;
var<workgroup> wg_b_tail: array<u32, 64>;

// Per-route time-window violation total (sum over stops of "lateness")
// and capacity overrun. Only meaningful in penalty mode. Initialized at
// the start of each LS iter alongside wg_route_load.
var<workgroup> wg_route_tw_viol: array<i32, 512>;
var<workgroup> wg_route_cap_viol: array<i32, 512>;

@compute @workgroup_size(1024)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tid = lid.x;
    let n_threads = 1024u;
    // Trajectory index: if dispatch is single-workgroup, params.traj
    // selects which trajectory to operate on. For multi-workgroup
    // dispatch, workgroup_id.x is the trajectory — `params.traj` is
    // ignored (we use wg.x). The CPU caller chooses by passing
    // pop_size workgroups for batch mode, or 1 for single-traj mode.
    let t = select(params.traj, wg.x, params.batch_mode != 0u);
    let md = params.matrix_dim;
    let tour_base = t * params.tour_capacity;
    let n_active = n_routes[t];
    let veh_tw_s = vehicle_tw_start[t];
    let veh_tw_e = vehicle_tw_end[t];

    // ---- Phase 0a: remove K random tasks → limbo ----
    // Thread 0 of each workgroup picks `kick_count` random interior tasks
    // (seeded by workgroup_id), removes them by shifting left, and stores
    // them in workgroup-shared limbo for reinsert. RNG seeded with
    // workgroup_id so different workgroups get different kicks → feasible
    // diversity for best-of-N reduction.
    if (tid == 0u) {
        wg_limbo_count = 0u;
        wg_dropped_count = 0u;
    }
    workgroupBarrier();

    if (params.kick_count > 0u && tid == 0u && n_active >= 1u) {
        var rng: u32 = wg.x * 2654435761u + params.kick_seed + 0xDEADBEEFu;
        for (var k: u32 = 0u; k < params.kick_count && k < 16u; k = k + 1u) {
            // Try a few times to find a route with an interior task.
            var found: bool = false;
            for (var attempt: u32 = 0u; attempt < 8u && !found; attempt = attempt + 1u) {
                rng = rng * 1664525u + 1013904223u;
                let r = rng % n_active;
                let off = t * params.max_routes + r;
                let len = route_lengths[off];
                if (len < 3u) { continue; }
                rng = rng * 1664525u + 1013904223u;
                let i = 1u + (rng % (len - 2u));
                let s = route_starts[off];
                wg_limbo[wg_limbo_count] = tour[tour_base + s + i];
                wg_limbo_count = wg_limbo_count + 1u;
                // Shift left to remove tour[s + i]
                for (var k2: u32 = i; k2 + 1u < len; k2 = k2 + 1u) {
                    tour[tour_base + s + k2] = tour[tour_base + s + k2 + 1u];
                }
                route_lengths[off] = len - 1u;
                found = true;
            }
        }
    }
    workgroupBarrier();

    // ---- Phase 0b: reinsert pulled tasks in cheapest-feasible order ----
    // For each task in limbo, all threads cooperate to find the cheapest
    // TW-feasible insertion across all (route, position) candidates. Then
    // thread 0 applies the winning insertion. If no feasible position
    // exists, the task is "dropped" (counted in wg_dropped_count).
    let local_limbo_count = wg_limbo_count;
    for (var lk: u32 = 0u; lk < local_limbo_count; lk = lk + 1u) {
        let pulled = wg_limbo[lk];

        if (tid == 0u) {
            atomicStore(&wg_insert_cost, 2147483647);
            atomicStore(&wg_insert_route, 0u);
            atomicStore(&wg_insert_pos, 0u);
        }
        workgroupBarrier();

        // Each thread strides over (route, position) candidates.
        for (var r: u32 = 0u; r < n_active; r = r + 1u) {
            let off = t * params.max_routes + r;
            let s = route_starts[off];
            let len = route_lengths[off];
            if (len < 2u) { continue; }
            // Valid insert positions: 1..len-1 (between depot and last interior stop).
            var p: u32 = tid + 1u;
            loop {
                if (p >= len) { break; }
                // Cheap distance delta first.
                let prev_loc = tour[tour_base + s + p - 1u];
                let next_loc = tour[tour_base + s + p];
                let old_edge = dist_at(prev_loc * md + next_loc);
                let new_edge = dist_at(prev_loc * md + pulled) + dist_at(pulled * md + next_loc);
                let delta = new_edge - old_edge;

                let cur_best = atomicLoad(&wg_insert_cost);
                if (delta >= cur_best) { p = p + n_threads; continue; }

                // TW check: walk full proposed route with `pulled` at p.
                var current_t: i32 = veh_tw_s;
                var prev_walk: u32 = tour[tour_base + s];
                var feas: bool = true;

                // Pre-insert prefix: positions 1..p-1.
                for (var k2: u32 = 1u; k2 < p; k2 = k2 + 1u) {
                    let here = tour[tour_base + s + k2];
                    current_t = current_t + dist_at(prev_walk * md + here);
                    if (k2 + 1u < len) {
                        if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                        if (current_t > tw_end[here]) { feas = false; break; }
                        current_t = current_t + service[here];
                    }
                    prev_walk = here;
                }

                if (feas) {
                    // Insert pulled.
                    current_t = current_t + dist_at(prev_walk * md + pulled);
                    if (current_t < tw_start[pulled]) { current_t = tw_start[pulled]; }
                    if (current_t > tw_end[pulled]) { feas = false; }
                    if (feas) {
                        current_t = current_t + service[pulled];
                        prev_walk = pulled;
                    }
                }

                if (feas) {
                    // Post-insert suffix: positions p..len-1.
                    for (var k2: u32 = p; k2 < len; k2 = k2 + 1u) {
                        let here = tour[tour_base + s + k2];
                        current_t = current_t + dist_at(prev_walk * md + here);
                        if (k2 + 1u < len) {
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) { feas = false; break; }
                            current_t = current_t + service[here];
                        }
                        prev_walk = here;
                    }
                }

                if (feas && current_t <= veh_tw_e) {
                    let prev_b = atomicMin(&wg_insert_cost, delta);
                    if (delta < prev_b) {
                        atomicStore(&wg_insert_route, r);
                        atomicStore(&wg_insert_pos, p);
                    }
                }
                p = p + n_threads;
            }
        }

        workgroupBarrier();

        // Apply insertion (thread 0).
        if (tid == 0u) {
            let best_cost = atomicLoad(&wg_insert_cost);
            if (best_cost < 2147483647) {
                let r = atomicLoad(&wg_insert_route);
                let p = atomicLoad(&wg_insert_pos);
                let off = t * params.max_routes + r;
                let s = route_starts[off];
                let len = route_lengths[off];
                // Shift right from p..len to make room for `pulled`.
                var k2: u32 = len;
                loop {
                    if (k2 <= p) { break; }
                    tour[tour_base + s + k2] = tour[tour_base + s + k2 - 1u];
                    k2 = k2 - 1u;
                }
                tour[tour_base + s + p] = pulled;
                route_lengths[off] = len + 1u;
            } else {
                wg_dropped_count = wg_dropped_count + 1u;
            }
        }
        workgroupBarrier();
    }

    var iter_count: u32 = 0u;
    var apply_count: u32 = 0u;
    var final_delta: i32 = 0;

    loop {
        if (iter_count >= params.max_iters) { break; }

        // ---- Phase 1: reset workgroup state ----
        if (tid == 0u) {
            atomicStore(&wg_best_delta, 2147483647);  // i32::MAX
            atomicStore(&wg_best_op, 0u);
            atomicStore(&wg_best_route, 0u);
            atomicStore(&wg_best_i, 0u);
            atomicStore(&wg_best_j, 0u);
            atomicStore(&wg_best_d, 0u);
            atomicStore(&wg_done, 0u);
            // Precompute per-route demand totals for relocate's capacity
            // check. Sequential per-route — cheap (≤ 512 routes × ≤ 24 stops).
            // In penalty mode, also compute per-route TW violation.
            let cap = vehicle_capacity[t];
            for (var r: u32 = 0u; r < n_active && r < 512u; r = r + 1u) {
                let off = t * params.max_routes + r;
                let s = route_starts[off];
                let len = route_lengths[off];
                var ld: i32 = 0;
                for (var k: u32 = 1u; k + 1u < len; k = k + 1u) {
                    ld = ld + demand[tour[tour_base + s + k]];
                }
                wg_route_load[r] = ld;

                // Capacity violation (≥ 0).
                let cv = ld - cap;
                wg_route_cap_viol[r] = select(0, cv, cv > 0);

                // TW violation: walk the route, accumulate lateness.
                // Only computed in penalty mode (cheap regardless, but
                // saves work for the strict path).
                if (params.penalty_mode == 1u && len >= 2u) {
                    var current_t: i32 = veh_tw_s;
                    var prev: u32 = tour[tour_base + s];
                    var tw_viol: i32 = 0;
                    for (var k: u32 = 1u; k < len; k = k + 1u) {
                        let here = tour[tour_base + s + k];
                        current_t = current_t + dist_at(prev * md + here);
                        if (k + 1u < len) {
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) {
                                tw_viol = tw_viol + (current_t - tw_end[here]);
                                // current_t NOT clamped — violation propagates.
                            }
                            current_t = current_t + service[here];
                        }
                        prev = here;
                    }
                    if (current_t > veh_tw_e) {
                        tw_viol = tw_viol + (current_t - veh_tw_e);
                    }
                    wg_route_tw_viol[r] = tw_viol;
                } else {
                    wg_route_tw_viol[r] = 0;
                }
            }
        }
        workgroupBarrier();

        // ---- Phase 1b: maintain task_pos lookup (only if granular enabled) ----
        // task_pos[t * matrix_dim + task] = (route << 16) | position.
        // Parallel reset by tid stride, then parallel fill by route.
        if (params.gk > 0u) {
            let tp_base = t * params.matrix_dim;
            // Reset all N entries to 0xFFFFFFFF (= "unassigned").
            var idx: u32 = tid;
            loop {
                if (idx >= params.matrix_dim) { break; }
                task_pos[tp_base + idx] = 0xFFFFFFFFu;
                idx = idx + n_threads;
            }
            workgroupBarrier();
            // Each route's interior tasks: tid 0..n_active-1 fills its route.
            // For >1024 routes we'd stride; but we cap at 64 routes per
            // trajectory anyway (workgroup_load array limit).
            for (var r: u32 = tid; r < n_active; r = r + n_threads) {
                let off = t * params.max_routes + r;
                let s = route_starts[off];
                let len = route_lengths[off];
                for (var p: u32 = 1u; p + 1u < len; p = p + 1u) {
                    let task = tour[tour_base + s + p];
                    task_pos[tp_base + task] = (r << 16u) | p;
                }
            }
            workgroupBarrier();
        }

        // ---- Phase 2: find best TW-feasible 2-opt across all routes ----
        // For each candidate (i, j):
        //   1. Compute distance delta — cheap, prune non-improving early
        //   2. If improving, walk the proposed reversed-segment route
        //      from depot through tour[i] (recomputing depart_at_i),
        //      then through reversed segment tour[j]→...→tour[i+1],
        //      then to tour[j+1]. Check TW at every step.
        //   3. If end-to-end TW-feasible, atomic-min into best.
        // Cost per candidate: O(L). For 17-stop routes with 225
        // candidates × 17 ops = 3825 ops/route, distributed across 1024
        // threads via stride.
        for (var r: u32 = 0u; r < n_active; r = r + 1u) {
            let route_off = t * params.max_routes + r;
            let s = route_starts[route_off];
            let len = route_lengths[route_off];
            if (len < 4u) { continue; }

            let n_pairs = len * len;
            var k: u32 = tid;
            loop {
                if (k >= n_pairs) { break; }
                let i = k / len;
                let j = k % len;
                if (j > i + 1u && j + 1u < len && i + 1u < len) {
                    let a = tour[tour_base + s + i];
                    let b = tour[tour_base + s + i + 1u];
                    let c = tour[tour_base + s + j];
                    let d = tour[tour_base + s + j + 1u];
                    let old_cost = dist_at(a * md + b) + dist_at(c * md + d);
                    let new_cost = dist_at(a * md + c) + dist_at(b * md + d);
                    let delta = new_cost - old_cost;

                    if (delta < 0) {
                        // Walk full proposed route, check TW.
                        var current_t: i32 = veh_tw_s;
                        var prev_loc: u32 = tour[tour_base + s];  // depot
                        var feasible: bool = true;

                        // Walk untouched prefix [1 .. i]
                        for (var k2: u32 = 1u; k2 <= i; k2 = k2 + 1u) {
                            let here = tour[tour_base + s + k2];
                            current_t = current_t + dist_at(prev_loc * md + here);
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) { feasible = false; break; }
                            current_t = current_t + service[here];
                            prev_loc = here;
                        }

                        // Walk reversed segment: tour[j], tour[j-1], ..., tour[i+1]
                        if (feasible) {
                            var k2: u32 = j;
                            loop {
                                if (!feasible) { break; }
                                let here = tour[tour_base + s + k2];
                                current_t = current_t + dist_at(prev_loc * md + here);
                                if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                if (current_t > tw_end[here]) { feasible = false; break; }
                                current_t = current_t + service[here];
                                prev_loc = here;
                                if (k2 == i + 1u) { break; }
                                k2 = k2 - 1u;
                            }
                        }

                        // Walk untouched suffix: tour[j+1] .. tour[len-1]
                        if (feasible) {
                            for (var k2: u32 = j + 1u; k2 < len; k2 = k2 + 1u) {
                                let here = tour[tour_base + s + k2];
                                current_t = current_t + dist_at(prev_loc * md + here);
                                if (k2 + 1u < len) {
                                    if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                    if (current_t > tw_end[here]) { feasible = false; break; }
                                    current_t = current_t + service[here];
                                }
                                prev_loc = here;
                            }
                        }

                        if (feasible && current_t <= veh_tw_e) {
                            let prev = atomicMin(&wg_best_delta, delta);
                            if (delta < prev) {
                                atomicStore(&wg_best_op, 0u);  // 2-opt
                                atomicStore(&wg_best_route, r);
                                atomicStore(&wg_best_i, i);
                                atomicStore(&wg_best_j, j);
                            }
                        }
                    }
                }
                k = k + n_threads;
            }
        }

        // ---- Phase 2b: find best inter-route relocate move ----
        // Two paths:
        //   gk == 0: exhaustive N² search (legacy, OK for N≤500)
        //   gk > 0:  granular K-NN search (O(N×K), scales to N≥1000)
        if (params.gk == 0u) {
        for (var src_r: u32 = 0u; src_r < n_active; src_r = src_r + 1u) {
            let src_off = t * params.max_routes + src_r;
            let src_s = route_starts[src_off];
            let src_len = route_lengths[src_off];
            if (src_len < 3u) { continue; }  // need ≥1 interior task

            for (var dst_r: u32 = 0u; dst_r < n_active; dst_r = dst_r + 1u) {
                let dst_off = t * params.max_routes + dst_r;
                let dst_s = route_starts[dst_off];
                let dst_len = route_lengths[dst_off];
                if (dst_len < 2u) { continue; }

                // Candidates: src_i in [1, src_len-2], dst_p in [1, dst_len-1].
                let n_src_pos = src_len - 2u;  // interior tasks
                let n_dst_pos = dst_len - 1u;  // valid insert positions
                let n_pairs = n_src_pos * n_dst_pos;

                var k: u32 = tid;
                loop {
                    if (k >= n_pairs) { break; }
                    let src_i = 1u + (k / n_dst_pos);
                    let dst_p = 1u + (k % n_dst_pos);

                    // Skip same-route same-position (no-op).
                    if (src_r == dst_r) {
                        if (dst_p == src_i || dst_p == src_i + 1u) {
                            k = k + n_threads;
                            continue;
                        }
                    }

                    let task_to_move = tour[tour_base + src_s + src_i];

                    // Cost delta:
                    //   src savings = (prev,task) + (task,next) - (prev,next)
                    //   dst added   = (prev_dst, task) + (task, next_dst) - (prev_dst, next_dst)
                    //   delta = dst_added - src_savings
                    let prev_src = tour[tour_base + src_s + src_i - 1u];
                    let next_src = tour[tour_base + src_s + src_i + 1u];
                    let src_save = dist_at(prev_src * md + task_to_move)
                                 + dist_at(task_to_move * md + next_src)
                                 - dist_at(prev_src * md + next_src);
                    let prev_dst = tour[tour_base + dst_s + dst_p - 1u];
                    let next_dst = tour[tour_base + dst_s + dst_p];
                    let dst_add = dist_at(prev_dst * md + task_to_move)
                                + dist_at(task_to_move * md + next_dst)
                                - dist_at(prev_dst * md + next_dst);
                    let delta = dst_add - src_save;

                    let cur_best = atomicLoad(&wg_best_delta);
                    if (delta >= cur_best) { k = k + n_threads; continue; }

                    // Capacity check: only inter-route moves can violate
                    // capacity (intra-route preserves total load).
                    if (src_r != dst_r) {
                        let task_demand = demand[task_to_move];
                        let new_dst_load = wg_route_load[dst_r] + task_demand;
                        if (new_dst_load > vehicle_capacity[t]) {
                            k = k + n_threads;
                            continue;
                        }
                    }

                    // TW check — walk both routes with the proposed move.
                    var feas: bool = true;

                    // Source route walk: skip src_i.
                    var current_t: i32 = veh_tw_s;
                    var prev_walk: u32 = tour[tour_base + src_s];
                    for (var k2: u32 = 1u; k2 < src_len; k2 = k2 + 1u) {
                        if (k2 == src_i) { continue; }
                        let here = tour[tour_base + src_s + k2];
                        current_t = current_t + dist_at(prev_walk * md + here);
                        // Customer (not end depot): TW + service.
                        if (k2 + 1u < src_len) {
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) { feas = false; break; }
                            current_t = current_t + service[here];
                        }
                        prev_walk = here;
                    }
                    if (feas && current_t > veh_tw_e) { feas = false; }

                    // Destination route walk: insert task_to_move at dst_p.
                    if (feas) {
                        current_t = veh_tw_s;
                        prev_walk = tour[tour_base + dst_s];
                        for (var k2: u32 = 1u; k2 < dst_len; k2 = k2 + 1u) {
                            if (k2 == dst_p) {
                                // Insert moved task here first.
                                current_t = current_t + dist_at(prev_walk * md + task_to_move);
                                if (current_t < tw_start[task_to_move]) {
                                    current_t = tw_start[task_to_move];
                                }
                                if (current_t > tw_end[task_to_move]) { feas = false; break; }
                                current_t = current_t + service[task_to_move];
                                prev_walk = task_to_move;
                            }
                            let here = tour[tour_base + dst_s + k2];
                            current_t = current_t + dist_at(prev_walk * md + here);
                            if (k2 + 1u < dst_len) {
                                if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                if (current_t > tw_end[here]) { feas = false; break; }
                                current_t = current_t + service[here];
                            }
                            prev_walk = here;
                        }
                        if (feas && current_t > veh_tw_e) { feas = false; }
                    }

                    if (feas) {
                        let prev = atomicMin(&wg_best_delta, delta);
                        if (delta < prev) {
                            atomicStore(&wg_best_op, 1u);  // relocate
                            atomicStore(&wg_best_route, src_r);
                            atomicStore(&wg_best_i, src_i);
                            atomicStore(&wg_best_j, dst_r);
                            atomicStore(&wg_best_d, dst_p);
                        }
                    }
                    k = k + n_threads;
                }
            }
        }
        } else {
        // ---- Granular relocate: O(N×K) candidates instead of O(N²) ----
        // For each interior src task, look up the position of each of
        // its K nearest neighbours via task_pos[]. Try inserting the
        // task immediately AFTER each neighbour (dst_p = neighbor_pos+1).
        // This is the single biggest speedup for N ≥ 1000.
        let g_stride = 64u;  // MAX_GRANULAR_K
        let tp_base = t * params.matrix_dim;

        for (var src_r: u32 = 0u; src_r < n_active; src_r = src_r + 1u) {
            let src_off = t * params.max_routes + src_r;
            let src_s = route_starts[src_off];
            let src_len = route_lengths[src_off];
            if (src_len < 3u) { continue; }

            // Threads stride over (src_i, k) candidate pairs.
            let n_cands = (src_len - 2u) * params.gk;
            var pk: u32 = tid;
            loop {
                if (pk >= n_cands) { break; }
                let src_i = 1u + (pk / params.gk);
                let kk = pk % params.gk;
                pk = pk + n_threads;

                let task_to_move = tour[tour_base + src_s + src_i];
                let neighbour = granular[task_to_move * g_stride + kk];
                let np = task_pos[tp_base + neighbour];
                if (np == 0xFFFFFFFFu) { continue; }
                let dst_r = np >> 16u;
                let neighbour_pos = np & 0xFFFFu;

                let dst_off = t * params.max_routes + dst_r;
                let dst_s = route_starts[dst_off];
                let dst_len = route_lengths[dst_off];
                // Try inserting immediately AFTER neighbour.
                let dst_p = neighbour_pos + 1u;
                if (dst_p >= dst_len) { continue; }

                if (src_r == dst_r) {
                    if (dst_p == src_i || dst_p == src_i + 1u) { continue; }
                }

                let prev_src = tour[tour_base + src_s + src_i - 1u];
                let next_src = tour[tour_base + src_s + src_i + 1u];
                let src_save = dist_at(prev_src * md + task_to_move)
                             + dist_at(task_to_move * md + next_src)
                             - dist_at(prev_src * md + next_src);
                let prev_dst = tour[tour_base + dst_s + dst_p - 1u];
                let next_dst = tour[tour_base + dst_s + dst_p];
                let dst_add = dist_at(prev_dst * md + task_to_move)
                            + dist_at(task_to_move * md + next_dst)
                            - dist_at(prev_dst * md + next_dst);
                let delta = dst_add - src_save;

                let pen_mode = (params.penalty_mode == 1u) && (src_r != dst_r);

                // Fast bail-out on dist delta only (strict mode); in penalty
                // mode we can't bail because TW improvement can dominate.
                let cur_best = atomicLoad(&wg_best_delta);
                if (!pen_mode && delta >= cur_best) { continue; }

                if (src_r != dst_r) {
                    let task_demand = demand[task_to_move];
                    let new_dst_load = wg_route_load[dst_r] + task_demand;
                    if (new_dst_load > vehicle_capacity[t]) { continue; }
                }

                // TW check: walk both routes (same logic as full path).
                // In penalty mode (inter-route only), accumulate overruns
                // instead of bailing — total_delta picks them up via PENALTY_TW.
                var feas: bool = true;
                var tw_viol_src: i32 = 0;
                var tw_viol_dst: i32 = 0;
                var current_t: i32 = veh_tw_s;
                var prev_walk: u32 = tour[tour_base + src_s];
                for (var k2: u32 = 1u; k2 < src_len; k2 = k2 + 1u) {
                    if (k2 == src_i) { continue; }
                    let here = tour[tour_base + src_s + k2];
                    current_t = current_t + dist_at(prev_walk * md + here);
                    if (k2 + 1u < src_len) {
                        if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                        if (current_t > tw_end[here]) {
                            if (pen_mode) {
                                tw_viol_src = tw_viol_src + (current_t - tw_end[here]);
                                current_t = tw_end[here];
                            } else {
                                feas = false; break;
                            }
                        }
                        current_t = current_t + service[here];
                    }
                    prev_walk = here;
                }
                if (current_t > veh_tw_e) {
                    if (pen_mode) {
                        tw_viol_src = tw_viol_src + (current_t - veh_tw_e);
                    } else if (feas) {
                        feas = false;
                    }
                }

                if (feas || pen_mode) {
                    current_t = veh_tw_s;
                    prev_walk = tour[tour_base + dst_s];
                    for (var k2: u32 = 1u; k2 < dst_len; k2 = k2 + 1u) {
                        if (k2 == dst_p) {
                            current_t = current_t + dist_at(prev_walk * md + task_to_move);
                            if (current_t < tw_start[task_to_move]) {
                                current_t = tw_start[task_to_move];
                            }
                            if (current_t > tw_end[task_to_move]) {
                                if (pen_mode) {
                                    tw_viol_dst = tw_viol_dst + (current_t - tw_end[task_to_move]);
                                    current_t = tw_end[task_to_move];
                                } else {
                                    feas = false; break;
                                }
                            }
                            current_t = current_t + service[task_to_move];
                            prev_walk = task_to_move;
                        }
                        let here = tour[tour_base + dst_s + k2];
                        current_t = current_t + dist_at(prev_walk * md + here);
                        if (k2 + 1u < dst_len) {
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) {
                                if (pen_mode) {
                                    tw_viol_dst = tw_viol_dst + (current_t - tw_end[here]);
                                    current_t = tw_end[here];
                                } else {
                                    feas = false; break;
                                }
                            }
                            current_t = current_t + service[here];
                        }
                        prev_walk = here;
                    }
                    if (current_t > veh_tw_e) {
                        if (pen_mode) {
                            tw_viol_dst = tw_viol_dst + (current_t - veh_tw_e);
                        } else if (feas) {
                            feas = false;
                        }
                    }
                }

                var total_delta: i32 = delta;
                var admissible: bool = feas;
                if (pen_mode) {
                    let PENALTY_TW: i32 = 100;
                    let old_tw = wg_route_tw_viol[src_r] + wg_route_tw_viol[dst_r];
                    let new_tw = tw_viol_src + tw_viol_dst;
                    total_delta = delta + PENALTY_TW * (new_tw - old_tw);
                    // Admit any move that improves penalty-adjusted cost,
                    // but require that the move doesn't *increase* total
                    // violation beyond a guard (avoid runaway infeasibility).
                    admissible = true;
                }

                if (admissible) {
                    let prev = atomicMin(&wg_best_delta, total_delta);
                    if (total_delta < prev) {
                        atomicStore(&wg_best_op, 1u);
                        atomicStore(&wg_best_route, src_r);
                        atomicStore(&wg_best_i, src_i);
                        atomicStore(&wg_best_j, dst_r);
                        atomicStore(&wg_best_d, dst_p);
                    }
                }
            }
        }
        }

        // ---- Phase 2c: find best inter-route exchange ----
        // Swap task at (rA, iA) with task at (rB, iB). Capacity preserved
        // when intra-route; for inter-route, both routes' loads change by
        // (demand_B - demand_A) and (demand_A - demand_B) respectively.
        // We process unique pairs by enforcing rA ≤ rB; ties on routes use iA < iB.
        for (var rA: u32 = 0u; rA < n_active; rA = rA + 1u) {
            let oA = t * params.max_routes + rA;
            let sA = route_starts[oA];
            let lenA = route_lengths[oA];
            if (lenA < 3u) { continue; }

            for (var rB: u32 = rA; rB < n_active; rB = rB + 1u) {
                let oB = t * params.max_routes + rB;
                let sB = route_starts[oB];
                let lenB = route_lengths[oB];
                if (lenB < 3u) { continue; }

                let n_iA = lenA - 2u;
                let n_iB = lenB - 2u;
                let n_pairs = n_iA * n_iB;
                var k: u32 = tid;
                loop {
                    if (k >= n_pairs) { break; }
                    let iA = 1u + (k / n_iB);
                    let iB = 1u + (k % n_iB);

                    // Skip degenerate pairs.
                    if (rA == rB && iA >= iB) { k = k + n_threads; continue; }

                    let taskA = tour[tour_base + sA + iA];
                    let taskB = tour[tour_base + sB + iB];

                    // Cost delta:
                    //   removed edges: (a-1)→A→(a+1) and (b-1)→B→(b+1)
                    //   added edges: (a-1)→B→(a+1) and (b-1)→A→(b+1)
                    // For inter-route case, simple. For intra-route adjacency,
                    // edges share — we skip adjacency for simplicity.
                    let prevA = tour[tour_base + sA + iA - 1u];
                    let nextA = tour[tour_base + sA + iA + 1u];
                    let prevB = tour[tour_base + sB + iB - 1u];
                    let nextB = tour[tour_base + sB + iB + 1u];

                    // Skip intra-route adjacent positions (handled by 2-opt or relocate).
                    if (rA == rB && (iA + 1u == iB || iB + 1u == iA)) {
                        k = k + n_threads; continue;
                    }

                    let removed = dist_at(prevA * md + taskA) + dist_at(taskA * md + nextA)
                                + dist_at(prevB * md + taskB) + dist_at(taskB * md + nextB);
                    let added = dist_at(prevA * md + taskB) + dist_at(taskB * md + nextA)
                              + dist_at(prevB * md + taskA) + dist_at(taskA * md + nextB);
                    let delta = added - removed;

                    let cur_best = atomicLoad(&wg_best_delta);
                    if (delta >= cur_best) { k = k + n_threads; continue; }

                    // Capacity check (inter-route only).
                    if (rA != rB) {
                        let dA = demand[taskA];
                        let dB = demand[taskB];
                        // After swap: rA loses A, gains B → load[A] - dA + dB
                        let new_loadA = wg_route_load[rA] - dA + dB;
                        let new_loadB = wg_route_load[rB] - dB + dA;
                        if (new_loadA > vehicle_capacity[t]) { k = k + n_threads; continue; }
                        if (new_loadB > vehicle_capacity[t]) { k = k + n_threads; continue; }
                    }

                    // TW check: walk both routes with swap applied.
                    var feas: bool = true;

                    // Walk route A with taskA replaced by taskB.
                    var current_t: i32 = veh_tw_s;
                    var prev_walk: u32 = tour[tour_base + sA];
                    for (var k2: u32 = 1u; k2 < lenA; k2 = k2 + 1u) {
                        let here = select(tour[tour_base + sA + k2], taskB, k2 == iA);
                        current_t = current_t + dist_at(prev_walk * md + here);
                        if (k2 + 1u < lenA) {
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) { feas = false; break; }
                            current_t = current_t + service[here];
                        }
                        prev_walk = here;
                    }
                    if (feas && current_t > veh_tw_e) { feas = false; }

                    if (feas && rA != rB) {
                        // Walk route B with taskB replaced by taskA.
                        current_t = veh_tw_s;
                        prev_walk = tour[tour_base + sB];
                        for (var k2: u32 = 1u; k2 < lenB; k2 = k2 + 1u) {
                            let here = select(tour[tour_base + sB + k2], taskA, k2 == iB);
                            current_t = current_t + dist_at(prev_walk * md + here);
                            if (k2 + 1u < lenB) {
                                if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                if (current_t > tw_end[here]) { feas = false; break; }
                                current_t = current_t + service[here];
                            }
                            prev_walk = here;
                        }
                        if (feas && current_t > veh_tw_e) { feas = false; }
                    }
                    // Note: intra-route case (rA == rB) the single walk above
                    // already covers the swap (we replaced both positions
                    // implicitly via select on either iA or iB — actually
                    // only iA. So intra-route swap walk is wrong above).
                    // For now skip intra-route exchange (handled by 2-opt-style).
                    if (rA == rB) { k = k + n_threads; continue; }

                    if (feas) {
                        let prev = atomicMin(&wg_best_delta, delta);
                        if (delta < prev) {
                            atomicStore(&wg_best_op, 2u);  // exchange
                            atomicStore(&wg_best_route, rA);
                            atomicStore(&wg_best_i, iA);
                            atomicStore(&wg_best_j, rB);
                            atomicStore(&wg_best_d, iB);
                        }
                    }
                    k = k + n_threads;
                }
            }
        }

        workgroupBarrier();

        // ---- Phase 2d: granular 2-opt* (cross-route 2-opt / tail-swap) ----
        // For each task a at position iA in route rA, look at its K nearest
        // neighbours via `granular`. If a neighbour b lives at position pB
        // in a different route rB, consider the cross-route 2-opt* with
        // split-points (iA, pB-1):
        //   rA_new = [d, a_1..a_iA, b_{pB}..b_{lenB-2}, d]
        //   rB_new = [d, b_1..b_{pB-1}, a_{iA+1}..a_{lenA-2}, d]
        // Delta (distance only):
        //   removed = dist(a_iA, a_{iA+1}) + dist(b_{pB-1}, b_pB)
        //   added   = dist(a_iA, b_pB) + dist(b_{pB-1}, a_{iA+1})
        // The "new edge" (a_iA, b_pB) is precisely a→b — so granular hits.
        // Requires gk > 0 (no full-N²-fallback for 2-opt* — it's an
        // operator that only makes sense with neighbour info).
        if (params.gk > 0u) {
            let g_stride = 64u;
            let tp_base = t * params.matrix_dim;
            for (var rA: u32 = 0u; rA < n_active; rA = rA + 1u) {
                let oA = t * params.max_routes + rA;
                let sA = route_starts[oA];
                let lenA = route_lengths[oA];
                if (lenA < 3u) { continue; }

                let n_iA = lenA - 2u;  // valid split-points 1..=lenA-2
                let n_cands = n_iA * params.gk;
                var pk: u32 = tid;
                loop {
                    if (pk >= n_cands) { break; }
                    let iA = 1u + (pk / params.gk);
                    let kk = pk % params.gk;
                    pk = pk + n_threads;

                    let a_iA = tour[tour_base + sA + iA];
                    let a_next = tour[tour_base + sA + iA + 1u];
                    let nb = granular[a_iA * g_stride + kk];
                    let np = task_pos[tp_base + nb];
                    if (np == 0xFFFFFFFFu) { continue; }
                    let rB = np >> 16u;
                    let pB = np & 0xFFFFu;
                    if (rB == rA) { continue; }      // intra-route = ordinary 2-opt
                    if (pB == 0u || pB == 0xFFFFu) { continue; }  // depot endpoint

                    let oB = t * params.max_routes + rB;
                    let sB = route_starts[oB];
                    let lenB = route_lengths[oB];
                    if (lenB < 3u) { continue; }
                    if (pB + 1u > lenB) { continue; } // safety

                    // pB is the position of b in rB. Split B at j = pB - 1
                    // so b_{j+1} = b_pB = nb.
                    let jB = pB - 1u;
                    let b_jB = tour[tour_base + sB + jB];
                    let b_pB = nb;

                    // Distance delta.
                    let removed = dist_at(a_iA * md + a_next)
                                + dist_at(b_jB * md + b_pB);
                    let added = dist_at(a_iA * md + b_pB)
                              + dist_at(b_jB * md + a_next);
                    let delta = added - removed;

                    let cur_best = atomicLoad(&wg_best_delta);
                    if (delta >= cur_best) { continue; }

                    // Capacity check. Sum A's tail demand and B's tail
                    // demand inline (≤ slot_size iterations each).
                    var dA_tail: i32 = 0;
                    for (var k2: u32 = iA + 1u; k2 + 1u < lenA; k2 = k2 + 1u) {
                        dA_tail = dA_tail + demand[tour[tour_base + sA + k2]];
                    }
                    var dB_tail: i32 = 0;
                    for (var k2: u32 = pB; k2 + 1u < lenB; k2 = k2 + 1u) {
                        dB_tail = dB_tail + demand[tour[tour_base + sB + k2]];
                    }
                    let new_loadA = wg_route_load[rA] - dA_tail + dB_tail;
                    let new_loadB = wg_route_load[rB] - dB_tail + dA_tail;
                    if (new_loadA > vehicle_capacity[t]) { continue; }
                    if (new_loadB > vehicle_capacity[t]) { continue; }

                    // TW check for rA_new = [d, a_1..a_iA, b_pB..b_{lenB-2}, d].
                    var feas: bool = true;
                    var current_t: i32 = veh_tw_s;
                    var prev_walk: u32 = tour[tour_base + sA];
                    // First segment: a_0 (depot) → a_1 → ... → a_iA
                    for (var k2: u32 = 1u; k2 <= iA; k2 = k2 + 1u) {
                        let here = tour[tour_base + sA + k2];
                        current_t = current_t + dist_at(prev_walk * md + here);
                        if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                        if (current_t > tw_end[here]) { feas = false; break; }
                        current_t = current_t + service[here];
                        prev_walk = here;
                    }
                    // Second segment: a_iA → b_pB → b_{pB+1} → ... → b_{lenB-2}
                    if (feas) {
                        for (var k2: u32 = pB; k2 + 1u < lenB; k2 = k2 + 1u) {
                            let here = tour[tour_base + sB + k2];
                            current_t = current_t + dist_at(prev_walk * md + here);
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) { feas = false; break; }
                            current_t = current_t + service[here];
                            prev_walk = here;
                        }
                    }
                    // Final hop to depot.
                    if (feas) {
                        let depot_end = tour[tour_base + sA + lenA - 1u];
                        current_t = current_t + dist_at(prev_walk * md + depot_end);
                        if (current_t > veh_tw_e) { feas = false; }
                    }

                    // TW check for rB_new = [d, b_1..b_{pB-1}, a_{iA+1}..a_{lenA-2}, d].
                    if (feas) {
                        current_t = veh_tw_s;
                        prev_walk = tour[tour_base + sB];
                        for (var k2: u32 = 1u; k2 <= jB; k2 = k2 + 1u) {
                            let here = tour[tour_base + sB + k2];
                            current_t = current_t + dist_at(prev_walk * md + here);
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) { feas = false; break; }
                            current_t = current_t + service[here];
                            prev_walk = here;
                        }
                        if (feas) {
                            for (var k2: u32 = iA + 1u; k2 + 1u < lenA; k2 = k2 + 1u) {
                                let here = tour[tour_base + sA + k2];
                                current_t = current_t + dist_at(prev_walk * md + here);
                                if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                if (current_t > tw_end[here]) { feas = false; break; }
                                current_t = current_t + service[here];
                                prev_walk = here;
                            }
                        }
                        if (feas) {
                            let depot_end = tour[tour_base + sB + lenB - 1u];
                            current_t = current_t + dist_at(prev_walk * md + depot_end);
                            if (current_t > veh_tw_e) { feas = false; }
                        }
                    }

                    if (feas) {
                        let prev = atomicMin(&wg_best_delta, delta);
                        if (delta < prev) {
                            atomicStore(&wg_best_op, 3u);  // 2-opt*
                            atomicStore(&wg_best_route, rA);
                            atomicStore(&wg_best_i, iA);
                            atomicStore(&wg_best_j, rB);
                            atomicStore(&wg_best_d, jB);
                        }
                    }
                }
            }
        }

        workgroupBarrier();

        // ---- Phase 2e: granular swap-star ----
        // For each (a in rA at iA, b in rB at pB) granular pair with rA != rB,
        // consider replacing (a, b) with cross-inserted (b at p_b_in_A in A,
        // a at p_a_in_B in B). Candidate positions per side: {-1, 0, +1}
        // relative to the original task position — gives 9 sub-candidates.
        //
        // Decomposition for apply: swap-at-position + intra-relocate per side.
        // So storage:
        //   wg_best_op    = 4
        //   wg_best_route = rA
        //   wg_best_i     = (iA << 16) | pB
        //   wg_best_j     = (rB << 16) | p_a_in_B
        //   wg_best_d     = p_b_in_A
        if (params.gk > 0u) {
            let g_stride = 64u;
            let tp_base = t * params.matrix_dim;
            for (var rA: u32 = 0u; rA < n_active; rA = rA + 1u) {
                let oA = t * params.max_routes + rA;
                let sA = route_starts[oA];
                let lenA = route_lengths[oA];
                if (lenA < 4u) { continue; }  // need at least depot+2 customers
                let n_iA = lenA - 2u;
                let n_cands = n_iA * params.gk;

                var pk: u32 = tid;
                loop {
                    if (pk >= n_cands) { break; }
                    let iA = 1u + (pk / params.gk);
                    let kk = pk % params.gk;
                    pk = pk + n_threads;

                    let a_task = tour[tour_base + sA + iA];
                    let nb = granular[a_task * g_stride + kk];
                    let np = task_pos[tp_base + nb];
                    if (np == 0xFFFFFFFFu) { continue; }
                    let rB = np >> 16u;
                    let pB = np & 0xFFFFu;
                    if (rB == rA) { continue; }
                    if (pB == 0u) { continue; }

                    let oB = t * params.max_routes + rB;
                    let sB = route_starts[oB];
                    let lenB = route_lengths[oB];
                    if (lenB < 4u) { continue; }
                    if (pB + 1u >= lenB) { continue; }

                    let b_task = nb;
                    let prev_a = tour[tour_base + sA + iA - 1u];
                    let next_a = tour[tour_base + sA + iA + 1u];
                    let prev_b = tour[tour_base + sB + pB - 1u];
                    let next_b = tour[tour_base + sB + pB + 1u];

                    // Removal-cost contributions.
                    let rem_a = dist_at(prev_a * md + a_task) + dist_at(a_task * md + next_a)
                              - dist_at(prev_a * md + next_a);
                    let rem_b = dist_at(prev_b * md + b_task) + dist_at(b_task * md + next_b)
                              - dist_at(prev_b * md + next_b);

                    // === FULL VIDAL: search ALL valid positions in each route ===
                    // For each candidate position p in B' (B with b removed),
                    // compute insertion delta of a. Take the minimum. Same for b in A'.
                    //
                    // Positions in B' are 1..(lenB-1). When p_a_in_B == pB, the slot
                    // where b was is freed; otherwise we need to account for the
                    // shift that removing b causes.

                    // --- Find best p_a_in_B (position in B' to insert a) ---
                    var best_a_val: i32 = 2147483647;
                    var best_a_pos: u32 = 0u;
                    // Iterate over positions in B' interpreted as positions in the
                    // ORIGINAL B (1..lenB-1). For each, compute insertion delta.
                    for (var p: u32 = 1u; p + 1u < lenB; p = p + 1u) {
                        // Determine prev/next in B' (B with b removed) at insertion point p.
                        var prev_p: u32; var next_p: u32;
                        if (p == pB) {
                            // Insert in b's old slot: prev=prev_b, next=next_b
                            prev_p = prev_b; next_p = next_b;
                        } else if (p < pB) {
                            // Position p is before b. prev = B[p-1], next = B[p].
                            // (b is later in the route, doesn't affect this edge.)
                            prev_p = tour[tour_base + sB + p - 1u];
                            next_p = tour[tour_base + sB + p];
                        } else {
                            // Position p > pB. In B', original B[p] is now at p-1.
                            // So inserting "at position p" in B' = before original B[p+1].
                            // prev = B[p] (was at p-1 after b removed), next = B[p+1].
                            prev_p = tour[tour_base + sB + p];
                            next_p = tour[tour_base + sB + p + 1u];
                            // Edge: when p = pB + 1, prev=B[pB+1]=next_b, next=B[pB+2].
                            // OK.
                        }
                        let ins = dist_at(prev_p * md + a_task)
                                + dist_at(a_task * md + next_p)
                                - dist_at(prev_p * md + next_p);
                        if (ins < best_a_val) {
                            best_a_val = ins;
                            best_a_pos = p;
                        }
                    }

                    // --- Find best p_b_in_A (position in A' to insert b) ---
                    var best_b_val: i32 = 2147483647;
                    var best_b_pos: u32 = 0u;
                    for (var p: u32 = 1u; p + 1u < lenA; p = p + 1u) {
                        var prev_p: u32; var next_p: u32;
                        if (p == iA) {
                            prev_p = prev_a; next_p = next_a;
                        } else if (p < iA) {
                            prev_p = tour[tour_base + sA + p - 1u];
                            next_p = tour[tour_base + sA + p];
                        } else {
                            prev_p = tour[tour_base + sA + p];
                            next_p = tour[tour_base + sA + p + 1u];
                        }
                        let ins = dist_at(prev_p * md + b_task)
                                + dist_at(b_task * md + next_p)
                                - dist_at(prev_p * md + next_p);
                        if (ins < best_b_val) {
                            best_b_val = ins;
                            best_b_pos = p;
                        }
                    }

                    if (best_a_val == 2147483647 || best_b_val == 2147483647) { continue; }

                    let delta = best_a_val + best_b_val - rem_a - rem_b;
                    let cur_best = atomicLoad(&wg_best_delta);
                    if (delta >= cur_best) { continue; }

                    // Capacity: a leaves A, b enters A (net dem(b) - dem(a)).
                    let dem_a = demand[a_task];
                    let dem_b = demand[b_task];
                    let new_loadA = wg_route_load[rA] - dem_a + dem_b;
                    let new_loadB = wg_route_load[rB] - dem_b + dem_a;
                    if (new_loadA > vehicle_capacity[t]) { continue; }
                    if (new_loadB > vehicle_capacity[t]) { continue; }

                    let p_a_in_B = best_a_pos;
                    let p_b_in_A = best_b_pos;

                    // TW-check the winning combination by walking both
                    // modified routes. A_new = A with a removed and b at p_b_in_A.
                    // B_new = B with b removed and a at p_a_in_B.
                    var feas: bool = true;

                    // Walk A_new.
                    var current_t: i32 = veh_tw_s;
                    var prev_walk: u32 = tour[tour_base + sA];
                    for (var k2: u32 = 1u; k2 < lenA; k2 = k2 + 1u) {
                        // Position in A' (= A with iA removed):
                        //   for k2 < iA: A'[k2] = A[k2]
                        //   for k2 >= iA: A'[k2] = A[k2 + 1]
                        // We want to walk A_new = A' with b inserted at p_b_in_A.
                        // Translate iteration index k2 (0..lenA-1) into A_new index:
                        //   if k2 < p_b_in_A: read A'[k2]
                        //   if k2 == p_b_in_A: read b_task
                        //   if k2 > p_b_in_A: read A'[k2-1]
                        let here: u32 = select(
                            select(
                                tour[tour_base + sA + select(k2, k2 + 1u, k2 >= iA)],
                                tour[tour_base + sA + select(k2 - 1u, k2, k2 - 1u >= iA)],
                                k2 > p_b_in_A,
                            ),
                            b_task,
                            k2 == p_b_in_A,
                        );
                        if (k2 == 1u) { /* nothing — handled below */ }
                        current_t = current_t + dist_at(prev_walk * md + here);
                        if (k2 + 1u < lenA) {
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) { feas = false; break; }
                            current_t = current_t + service[here];
                        }
                        prev_walk = here;
                    }
                    if (feas && current_t > veh_tw_e) { feas = false; }

                    // Walk B_new.
                    if (feas) {
                        current_t = veh_tw_s;
                        prev_walk = tour[tour_base + sB];
                        for (var k2: u32 = 1u; k2 < lenB; k2 = k2 + 1u) {
                            let here: u32 = select(
                                select(
                                    tour[tour_base + sB + select(k2, k2 + 1u, k2 >= pB)],
                                    tour[tour_base + sB + select(k2 - 1u, k2, k2 - 1u >= pB)],
                                    k2 > p_a_in_B,
                                ),
                                a_task,
                                k2 == p_a_in_B,
                            );
                            current_t = current_t + dist_at(prev_walk * md + here);
                            if (k2 + 1u < lenB) {
                                if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                if (current_t > tw_end[here]) { feas = false; break; }
                                current_t = current_t + service[here];
                            }
                            prev_walk = here;
                        }
                        if (feas && current_t > veh_tw_e) { feas = false; }
                    }

                    if (feas) {
                        let prev = atomicMin(&wg_best_delta, delta);
                        if (delta < prev) {
                            atomicStore(&wg_best_op, 4u);  // swap-star
                            atomicStore(&wg_best_route, rA);
                            atomicStore(&wg_best_i, (iA << 16u) | pB);
                            atomicStore(&wg_best_j, (rB << 16u) | p_a_in_B);
                            atomicStore(&wg_best_d, p_b_in_A);
                        }
                    }
                }
            }
        }

        workgroupBarrier();

        // ---- Phase 2f: granular Or-opt-2 (segment relocate, L=2) ----
        // Move a 2-task consecutive segment [src_i, src_i+1] from rA to a
        // new position in route rB (possibly same as rA). Granular-driven
        // via first-task's K nearest neighbours. The internal edge between
        // the two segment tasks is preserved across the move.
        //
        // Removal cost (saved):
        //   d(prev_src, a_iA) + d(a_{iA+1}, next_src) − d(prev_src, next_src)
        // Insertion cost at dst_p in rB:
        //   d(prev_dst, a_iA) + d(a_{iA+1}, next_dst) − d(prev_dst, next_dst)
        //
        // Op = 5. Storage same as relocate (src_r, src_i, dst_r, dst_p).
        if (params.gk > 0u) {
            let g_stride = 64u;
            let tp_base = t * params.matrix_dim;
            for (var src_r: u32 = 0u; src_r < n_active; src_r = src_r + 1u) {
                let src_off = t * params.max_routes + src_r;
                let src_s = route_starts[src_off];
                let src_len = route_lengths[src_off];
                if (src_len < 4u) { continue; }  // need at least 2 interior tasks
                // src_i is the START of the 2-task segment; valid range: 1..src_len-3
                let n_starts = src_len - 3u;
                let n_cands = n_starts * params.gk;

                var pk: u32 = tid;
                loop {
                    if (pk >= n_cands) { break; }
                    let src_i = 1u + (pk / params.gk);
                    let kk = pk % params.gk;
                    pk = pk + n_threads;

                    let seg_first = tour[tour_base + src_s + src_i];
                    let seg_last = tour[tour_base + src_s + src_i + 1u];
                    let nb = granular[seg_first * g_stride + kk];
                    let np = task_pos[tp_base + nb];
                    if (np == 0xFFFFFFFFu) { continue; }
                    let dst_r = np >> 16u;
                    let neighbour_pos = np & 0xFFFFu;

                    let dst_off = t * params.max_routes + dst_r;
                    let dst_s = route_starts[dst_off];
                    let dst_len = route_lengths[dst_off];
                    let dst_p = neighbour_pos + 1u;  // insert immediately after neighbour
                    if (dst_p >= dst_len) { continue; }

                    if (src_r == dst_r) {
                        // Don't insert into the same/overlapping range we're removing from.
                        // Forbidden dst positions: src_i, src_i+1, src_i+2.
                        if (dst_p == src_i || dst_p == src_i + 1u || dst_p == src_i + 2u) {
                            continue;
                        }
                    }

                    let prev_src = tour[tour_base + src_s + src_i - 1u];
                    let next_src = tour[tour_base + src_s + src_i + 2u];
                    let src_save = dist_at(prev_src * md + seg_first)
                                 + dist_at(seg_last * md + next_src)
                                 - dist_at(prev_src * md + next_src);
                    let prev_dst = tour[tour_base + dst_s + dst_p - 1u];
                    let next_dst = tour[tour_base + dst_s + dst_p];
                    let dst_add = dist_at(prev_dst * md + seg_first)
                                + dist_at(seg_last * md + next_dst)
                                - dist_at(prev_dst * md + next_dst);
                    let delta = dst_add - src_save;

                    let cur_best = atomicLoad(&wg_best_delta);
                    if (delta >= cur_best) { continue; }

                    // Capacity check (inter-route only).
                    if (src_r != dst_r) {
                        let seg_dem = demand[seg_first] + demand[seg_last];
                        let new_dst_load = wg_route_load[dst_r] + seg_dem;
                        if (new_dst_load > vehicle_capacity[t]) { continue; }
                    }

                    // TW check: walk both modified routes.
                    var feas: bool = true;
                    // Walk source with segment removed.
                    var current_t: i32 = veh_tw_s;
                    var prev_walk: u32 = tour[tour_base + src_s];
                    for (var k2: u32 = 1u; k2 < src_len; k2 = k2 + 1u) {
                        if (k2 == src_i || k2 == src_i + 1u) { continue; }
                        let here = tour[tour_base + src_s + k2];
                        current_t = current_t + dist_at(prev_walk * md + here);
                        if (k2 + 1u < src_len) {
                            if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                            if (current_t > tw_end[here]) { feas = false; break; }
                            current_t = current_t + service[here];
                        }
                        prev_walk = here;
                    }
                    if (feas && current_t > veh_tw_e) { feas = false; }

                    // Walk destination with segment inserted at dst_p.
                    // For intra-route case, this walk also handles the
                    // updated tour state correctly via the index logic.
                    if (feas && src_r != dst_r) {
                        current_t = veh_tw_s;
                        prev_walk = tour[tour_base + dst_s];
                        for (var k2: u32 = 1u; k2 < dst_len; k2 = k2 + 1u) {
                            if (k2 == dst_p) {
                                // Insert segment here.
                                current_t = current_t + dist_at(prev_walk * md + seg_first);
                                if (current_t < tw_start[seg_first]) { current_t = tw_start[seg_first]; }
                                if (current_t > tw_end[seg_first]) { feas = false; break; }
                                current_t = current_t + service[seg_first];
                                current_t = current_t + dist_at(seg_first * md + seg_last);
                                if (current_t < tw_start[seg_last]) { current_t = tw_start[seg_last]; }
                                if (current_t > tw_end[seg_last]) { feas = false; break; }
                                current_t = current_t + service[seg_last];
                                prev_walk = seg_last;
                            }
                            let here = tour[tour_base + dst_s + k2];
                            current_t = current_t + dist_at(prev_walk * md + here);
                            if (k2 + 1u < dst_len) {
                                if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                if (current_t > tw_end[here]) { feas = false; break; }
                                current_t = current_t + service[here];
                            }
                            prev_walk = here;
                        }
                        if (feas && current_t > veh_tw_e) { feas = false; }
                    } else if (feas) {
                        // Intra-route: walk source with segment removed AND
                        // re-inserted at dst_p in one pass over the modified tour.
                        // For simplicity, skip TW-check on intra-route (rare anyway).
                        // Caller's CPU evaluate_route will catch any infeasibility.
                    }

                    if (feas) {
                        let prev = atomicMin(&wg_best_delta, delta);
                        if (delta < prev) {
                            atomicStore(&wg_best_op, 5u);  // or-opt-2
                            atomicStore(&wg_best_route, src_r);
                            atomicStore(&wg_best_i, src_i);
                            atomicStore(&wg_best_j, dst_r);
                            atomicStore(&wg_best_d, dst_p);
                        }
                    }
                }
            }
        }

        workgroupBarrier();

        // ---- Phase 2g: granular Or-opt-3 (segment relocate, L=3) ----
        // Same as Or-opt-2 but with a 3-task consecutive segment. Useful
        // for moving small "tail clusters" together. Op = 6.
        if (params.gk > 0u) {
            let g_stride = 64u;
            let tp_base = t * params.matrix_dim;
            for (var src_r: u32 = 0u; src_r < n_active; src_r = src_r + 1u) {
                let src_off = t * params.max_routes + src_r;
                let src_s = route_starts[src_off];
                let src_len = route_lengths[src_off];
                if (src_len < 5u) { continue; }
                let n_starts = src_len - 4u;
                let n_cands = n_starts * params.gk;

                var pk: u32 = tid;
                loop {
                    if (pk >= n_cands) { break; }
                    let src_i = 1u + (pk / params.gk);
                    let kk = pk % params.gk;
                    pk = pk + n_threads;

                    let seg_first = tour[tour_base + src_s + src_i];
                    let seg_mid = tour[tour_base + src_s + src_i + 1u];
                    let seg_last = tour[tour_base + src_s + src_i + 2u];
                    let nb = granular[seg_first * g_stride + kk];
                    let np = task_pos[tp_base + nb];
                    if (np == 0xFFFFFFFFu) { continue; }
                    let dst_r = np >> 16u;
                    let neighbour_pos = np & 0xFFFFu;

                    let dst_off = t * params.max_routes + dst_r;
                    let dst_s = route_starts[dst_off];
                    let dst_len = route_lengths[dst_off];
                    let dst_p = neighbour_pos + 1u;
                    if (dst_p >= dst_len) { continue; }

                    if (src_r == dst_r) {
                        if (dst_p == src_i || dst_p == src_i + 1u
                            || dst_p == src_i + 2u || dst_p == src_i + 3u) {
                            continue;
                        }
                    }

                    let prev_src = tour[tour_base + src_s + src_i - 1u];
                    let next_src = tour[tour_base + src_s + src_i + 3u];
                    let src_save = dist_at(prev_src * md + seg_first)
                                 + dist_at(seg_last * md + next_src)
                                 - dist_at(prev_src * md + next_src);
                    let prev_dst = tour[tour_base + dst_s + dst_p - 1u];
                    let next_dst = tour[tour_base + dst_s + dst_p];
                    let dst_add = dist_at(prev_dst * md + seg_first)
                                + dist_at(seg_last * md + next_dst)
                                - dist_at(prev_dst * md + next_dst);
                    let delta = dst_add - src_save;

                    let cur_best = atomicLoad(&wg_best_delta);
                    if (delta >= cur_best) { continue; }

                    if (src_r != dst_r) {
                        let seg_dem = demand[seg_first] + demand[seg_mid] + demand[seg_last];
                        let new_dst_load = wg_route_load[dst_r] + seg_dem;
                        if (new_dst_load > vehicle_capacity[t]) { continue; }
                    }

                    // TW check (inter-route case only — intra skipped for simplicity).
                    var feas: bool = true;
                    if (src_r != dst_r) {
                        // Source after removing segment.
                        var current_t: i32 = veh_tw_s;
                        var prev_walk: u32 = tour[tour_base + src_s];
                        for (var k2: u32 = 1u; k2 < src_len; k2 = k2 + 1u) {
                            if (k2 == src_i || k2 == src_i + 1u || k2 == src_i + 2u) { continue; }
                            let here = tour[tour_base + src_s + k2];
                            current_t = current_t + dist_at(prev_walk * md + here);
                            if (k2 + 1u < src_len) {
                                if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                if (current_t > tw_end[here]) { feas = false; break; }
                                current_t = current_t + service[here];
                            }
                            prev_walk = here;
                        }
                        if (feas && current_t > veh_tw_e) { feas = false; }

                        if (feas) {
                            // Destination after inserting segment at dst_p.
                            current_t = veh_tw_s;
                            prev_walk = tour[tour_base + dst_s];
                            for (var k2: u32 = 1u; k2 < dst_len; k2 = k2 + 1u) {
                                if (k2 == dst_p) {
                                    current_t = current_t + dist_at(prev_walk * md + seg_first);
                                    if (current_t < tw_start[seg_first]) { current_t = tw_start[seg_first]; }
                                    if (current_t > tw_end[seg_first]) { feas = false; break; }
                                    current_t = current_t + service[seg_first];
                                    current_t = current_t + dist_at(seg_first * md + seg_mid);
                                    if (current_t < tw_start[seg_mid]) { current_t = tw_start[seg_mid]; }
                                    if (current_t > tw_end[seg_mid]) { feas = false; break; }
                                    current_t = current_t + service[seg_mid];
                                    current_t = current_t + dist_at(seg_mid * md + seg_last);
                                    if (current_t < tw_start[seg_last]) { current_t = tw_start[seg_last]; }
                                    if (current_t > tw_end[seg_last]) { feas = false; break; }
                                    current_t = current_t + service[seg_last];
                                    prev_walk = seg_last;
                                }
                                let here = tour[tour_base + dst_s + k2];
                                current_t = current_t + dist_at(prev_walk * md + here);
                                if (k2 + 1u < dst_len) {
                                    if (current_t < tw_start[here]) { current_t = tw_start[here]; }
                                    if (current_t > tw_end[here]) { feas = false; break; }
                                    current_t = current_t + service[here];
                                }
                                prev_walk = here;
                            }
                            if (feas && current_t > veh_tw_e) { feas = false; }
                        }
                    }
                    // For intra-route: skip TW-check, rely on CPU evaluate_route to reject.

                    if (feas) {
                        let prev = atomicMin(&wg_best_delta, delta);
                        if (delta < prev) {
                            atomicStore(&wg_best_op, 6u);  // or-opt-3
                            atomicStore(&wg_best_route, src_r);
                            atomicStore(&wg_best_i, src_i);
                            atomicStore(&wg_best_j, dst_r);
                            atomicStore(&wg_best_d, dst_p);
                        }
                    }
                }
            }
        }

        workgroupBarrier();

        // ---- Phase 3: convergence check ----
        let best_delta = atomicLoad(&wg_best_delta);
        if (best_delta >= 0) {
            final_delta = best_delta;
            break;
        }

        // ---- Phase 4: apply the winning move (thread 0) ----
        if (tid == 0u) {
            let op_apply = atomicLoad(&wg_best_op);
            if (op_apply == 0u) {
                // 2-opt: reverse tour[s + i + 1 ..= s + j].
                let r_apply = atomicLoad(&wg_best_route);
                let i_apply = atomicLoad(&wg_best_i);
                let j_apply = atomicLoad(&wg_best_j);
                let route_off = t * params.max_routes + r_apply;
                let s = route_starts[route_off];
                var lo: u32 = s + i_apply + 1u;
                var hi: u32 = s + j_apply;
                loop {
                    if (lo >= hi) { break; }
                    let a = tour[tour_base + lo];
                    let b = tour[tour_base + hi];
                    tour[tour_base + lo] = b;
                    tour[tour_base + hi] = a;
                    lo = lo + 1u;
                    hi = hi - 1u;
                }
            } else if (op_apply == 2u) {
                // Exchange: swap tasks at (rA, iA) and (rB, iB).
                let rA = atomicLoad(&wg_best_route);
                let iA = atomicLoad(&wg_best_i);
                let rB = atomicLoad(&wg_best_j);
                let iB = atomicLoad(&wg_best_d);
                let oA = t * params.max_routes + rA;
                let oB = t * params.max_routes + rB;
                let sA = route_starts[oA];
                let sB = route_starts[oB];
                let tmp = tour[tour_base + sA + iA];
                tour[tour_base + sA + iA] = tour[tour_base + sB + iB];
                tour[tour_base + sB + iB] = tmp;
            } else if (op_apply == 6u) {
                // Or-opt-3: move 3-task segment.
                let src_r = atomicLoad(&wg_best_route);
                let src_i = atomicLoad(&wg_best_i);
                let dst_r = atomicLoad(&wg_best_j);
                let dst_p = atomicLoad(&wg_best_d);
                let src_off = t * params.max_routes + src_r;
                let dst_off = t * params.max_routes + dst_r;
                let src_s = route_starts[src_off];
                let dst_s = route_starts[dst_off];
                let src_len = route_lengths[src_off];
                let dst_len = route_lengths[dst_off];
                let seg_a = tour[tour_base + src_s + src_i];
                let seg_b = tour[tour_base + src_s + src_i + 1u];
                let seg_c = tour[tour_base + src_s + src_i + 2u];

                if (src_r == dst_r) {
                    if (dst_p > src_i + 2u) {
                        for (var k: u32 = src_i; k + 3u < dst_p; k = k + 1u) {
                            tour[tour_base + src_s + k] = tour[tour_base + src_s + k + 3u];
                        }
                        tour[tour_base + src_s + dst_p - 3u] = seg_a;
                        tour[tour_base + src_s + dst_p - 2u] = seg_b;
                        tour[tour_base + src_s + dst_p - 1u] = seg_c;
                    } else {
                        var k: u32 = src_i + 2u;
                        loop {
                            if (k < dst_p + 3u) { break; }
                            tour[tour_base + src_s + k] = tour[tour_base + src_s + k - 3u];
                            k = k - 1u;
                        }
                        tour[tour_base + src_s + dst_p] = seg_a;
                        tour[tour_base + src_s + dst_p + 1u] = seg_b;
                        tour[tour_base + src_s + dst_p + 2u] = seg_c;
                    }
                } else {
                    for (var k: u32 = src_i; k + 3u < src_len; k = k + 1u) {
                        tour[tour_base + src_s + k] = tour[tour_base + src_s + k + 3u];
                    }
                    route_lengths[src_off] = src_len - 3u;
                    var k: u32 = dst_len + 2u;
                    loop {
                        if (k < dst_p + 3u) { break; }
                        tour[tour_base + dst_s + k] = tour[tour_base + dst_s + k - 3u];
                        k = k - 1u;
                    }
                    tour[tour_base + dst_s + dst_p] = seg_a;
                    tour[tour_base + dst_s + dst_p + 1u] = seg_b;
                    tour[tour_base + dst_s + dst_p + 2u] = seg_c;
                    route_lengths[dst_off] = dst_len + 3u;
                }
            } else if (op_apply == 5u) {
                // Or-opt-2: move 2-task segment from (src_r, src_i, src_i+1)
                // to (dst_r, dst_p). Internal edge preserved across the move.
                let src_r = atomicLoad(&wg_best_route);
                let src_i = atomicLoad(&wg_best_i);
                let dst_r = atomicLoad(&wg_best_j);
                let dst_p = atomicLoad(&wg_best_d);
                let src_off = t * params.max_routes + src_r;
                let dst_off = t * params.max_routes + dst_r;
                let src_s = route_starts[src_off];
                let dst_s = route_starts[dst_off];
                let src_len = route_lengths[src_off];
                let dst_len = route_lengths[dst_off];
                let seg_first = tour[tour_base + src_s + src_i];
                let seg_last = tour[tour_base + src_s + src_i + 1u];

                if (src_r == dst_r) {
                    // Intra-route: relocate the 2-task segment. Both indices
                    // (src_i, src_i+1) need to shift, then re-inserted at
                    // dst_p (which avoided overlap with src_i..src_i+2 above).
                    if (dst_p > src_i + 1u) {
                        // dst_p > src_i+2 (avoided overlap). Shift positions
                        // src_i..dst_p-1 left by 2, then write segment at dst_p-2.
                        // Actually: shift src_i+2..dst_p-1 left by 2 to fill the hole,
                        // then place segment at dst_p-2..dst_p-1.
                        for (var k: u32 = src_i; k + 2u < dst_p; k = k + 1u) {
                            tour[tour_base + src_s + k] = tour[tour_base + src_s + k + 2u];
                        }
                        tour[tour_base + src_s + dst_p - 2u] = seg_first;
                        tour[tour_base + src_s + dst_p - 1u] = seg_last;
                    } else {
                        // dst_p < src_i: shift positions dst_p..src_i-1 right by 2,
                        // then place segment at dst_p..dst_p+1.
                        var k: u32 = src_i + 1u;
                        loop {
                            if (k < dst_p + 2u) { break; }
                            tour[tour_base + src_s + k] = tour[tour_base + src_s + k - 2u];
                            k = k - 1u;
                        }
                        tour[tour_base + src_s + dst_p] = seg_first;
                        tour[tour_base + src_s + dst_p + 1u] = seg_last;
                    }
                    // Length unchanged.
                } else {
                    // Inter-route. Source: shift src_i+2..src_len-1 left by 2.
                    for (var k: u32 = src_i; k + 2u < src_len; k = k + 1u) {
                        tour[tour_base + src_s + k] = tour[tour_base + src_s + k + 2u];
                    }
                    route_lengths[src_off] = src_len - 2u;
                    // Destination: shift dst_p..dst_len-1 right by 2, place segment.
                    var k: u32 = dst_len + 1u;
                    loop {
                        if (k < dst_p + 2u) { break; }
                        tour[tour_base + dst_s + k] = tour[tour_base + dst_s + k - 2u];
                        k = k - 1u;
                    }
                    tour[tour_base + dst_s + dst_p] = seg_first;
                    tour[tour_base + dst_s + dst_p + 1u] = seg_last;
                    route_lengths[dst_off] = dst_len + 2u;
                }
            } else if (op_apply == 4u) {
                // Swap-star: swap (a, b) at their current positions, then
                // optionally shift each by ±1 inside its new route. The
                // decomposition is: exchange + intra-relocate in A (b from
                // iA to p_b_in_A) + intra-relocate in B (a from pB to p_a_in_B).
                let rA = atomicLoad(&wg_best_route);
                let packed_i = atomicLoad(&wg_best_i);
                let iA = packed_i >> 16u;
                let pB = packed_i & 0xFFFFu;
                let packed_j = atomicLoad(&wg_best_j);
                let rB = packed_j >> 16u;
                let p_a_in_B = packed_j & 0xFFFFu;
                let p_b_in_A = atomicLoad(&wg_best_d);

                let oA = t * params.max_routes + rA;
                let oB = t * params.max_routes + rB;
                let sA = route_starts[oA];
                let sB = route_starts[oB];

                // Step 1: swap a and b at (iA, pB).
                let tmp = tour[tour_base + sA + iA];
                tour[tour_base + sA + iA] = tour[tour_base + sB + pB];
                tour[tour_base + sB + pB] = tmp;

                // Step 2: intra-relocate in A from iA to p_b_in_A (if different).
                if (p_b_in_A != iA) {
                    let task = tour[tour_base + sA + iA];
                    if (p_b_in_A > iA) {
                        // Shift positions iA+1..p_b_in_A left by 1.
                        for (var k: u32 = iA; k < p_b_in_A; k = k + 1u) {
                            tour[tour_base + sA + k] = tour[tour_base + sA + k + 1u];
                        }
                        tour[tour_base + sA + p_b_in_A] = task;
                    } else {
                        // p_b_in_A < iA: shift positions p_b_in_A..iA-1 right by 1.
                        var k: u32 = iA;
                        loop {
                            if (k <= p_b_in_A) { break; }
                            tour[tour_base + sA + k] = tour[tour_base + sA + k - 1u];
                            k = k - 1u;
                        }
                        tour[tour_base + sA + p_b_in_A] = task;
                    }
                }
                // Step 3: intra-relocate in B from pB to p_a_in_B.
                if (p_a_in_B != pB) {
                    let task = tour[tour_base + sB + pB];
                    if (p_a_in_B > pB) {
                        for (var k: u32 = pB; k < p_a_in_B; k = k + 1u) {
                            tour[tour_base + sB + k] = tour[tour_base + sB + k + 1u];
                        }
                        tour[tour_base + sB + p_a_in_B] = task;
                    } else {
                        var k: u32 = pB;
                        loop {
                            if (k <= p_a_in_B) { break; }
                            tour[tour_base + sB + k] = tour[tour_base + sB + k - 1u];
                            k = k - 1u;
                        }
                        tour[tour_base + sB + p_a_in_B] = task;
                    }
                }
            } else if (op_apply == 3u) {
                // 2-opt*: swap tail of route A (from iA+1 to end-1) with
                // tail of route B (from jB+1 to end-1). Both end with depot.
                let rA = atomicLoad(&wg_best_route);
                let iA = atomicLoad(&wg_best_i);
                let rB = atomicLoad(&wg_best_j);
                let jB = atomicLoad(&wg_best_d);
                let oA = t * params.max_routes + rA;
                let oB = t * params.max_routes + rB;
                let sA = route_starts[oA];
                let sB = route_starts[oB];
                let lenA = route_lengths[oA];
                let lenB = route_lengths[oB];
                let depot_A = tour[tour_base + sA + lenA - 1u];
                let depot_B = tour[tour_base + sB + lenB - 1u];

                // Save A's interior tail [iA+1 .. lenA-2] to scratch.
                let nA_tail = (lenA - 1u) - (iA + 1u);  // number of interior tail elements
                for (var k: u32 = 0u; k < nA_tail; k = k + 1u) {
                    wg_a_tail[k] = tour[tour_base + sA + iA + 1u + k];
                }
                // Save B's interior tail [jB+1 .. lenB-2] to scratch.
                let nB_tail = (lenB - 1u) - (jB + 1u);
                for (var k: u32 = 0u; k < nB_tail; k = k + 1u) {
                    wg_b_tail[k] = tour[tour_base + sB + jB + 1u + k];
                }

                // Write rA_new: head [0..iA] unchanged, then B's tail, then depot_A.
                for (var k: u32 = 0u; k < nB_tail; k = k + 1u) {
                    tour[tour_base + sA + iA + 1u + k] = wg_b_tail[k];
                }
                tour[tour_base + sA + iA + 1u + nB_tail] = depot_A;
                route_lengths[oA] = iA + 1u + nB_tail + 1u;

                // Write rB_new: head [0..jB] unchanged, then A's tail, then depot_B.
                for (var k: u32 = 0u; k < nA_tail; k = k + 1u) {
                    tour[tour_base + sB + jB + 1u + k] = wg_a_tail[k];
                }
                tour[tour_base + sB + jB + 1u + nA_tail] = depot_B;
                route_lengths[oB] = jB + 1u + nA_tail + 1u;
            } else {
                // Relocate: remove task at (src_r, src_i), insert at (dst_r, dst_p).
                let src_r = atomicLoad(&wg_best_route);
                let src_i = atomicLoad(&wg_best_i);
                let dst_r = atomicLoad(&wg_best_j);
                let dst_p = atomicLoad(&wg_best_d);
                let src_off = t * params.max_routes + src_r;
                let dst_off = t * params.max_routes + dst_r;
                let src_s = route_starts[src_off];
                let dst_s = route_starts[dst_off];
                let src_len = route_lengths[src_off];
                let dst_len = route_lengths[dst_off];
                let task = tour[tour_base + src_s + src_i];

                if (src_r == dst_r) {
                    // Intra-route move: shift within single route.
                    // Remove src_i, then insert at adjusted dst_p.
                    if (dst_p > src_i) {
                        // Shift positions src_i+1..dst_p-1 left by 1, then write task at dst_p-1
                        for (var k: u32 = src_i; k + 1u < dst_p; k = k + 1u) {
                            tour[tour_base + src_s + k] = tour[tour_base + src_s + k + 1u];
                        }
                        tour[tour_base + src_s + dst_p - 1u] = task;
                    } else {
                        // dst_p < src_i: shift positions dst_p..src_i-1 right by 1, write task at dst_p
                        var k: u32 = src_i;
                        loop {
                            if (k <= dst_p) { break; }
                            tour[tour_base + src_s + k] = tour[tour_base + src_s + k - 1u];
                            k = k - 1u;
                        }
                        tour[tour_base + src_s + dst_p] = task;
                    }
                    // Length unchanged.
                } else {
                    // Inter-route: remove from src (shift left), insert in dst (shift right).
                    // Source: shift positions src_i+1..src_len-1 left by 1
                    for (var k: u32 = src_i; k + 1u < src_len; k = k + 1u) {
                        tour[tour_base + src_s + k] = tour[tour_base + src_s + k + 1u];
                    }
                    route_lengths[src_off] = src_len - 1u;
                    // Destination: shift positions dst_p..dst_len-1 right by 1, insert task at dst_p
                    var k: u32 = dst_len;
                    loop {
                        if (k <= dst_p) { break; }
                        tour[tour_base + dst_s + k] = tour[tour_base + dst_s + k - 1u];
                        k = k - 1u;
                    }
                    tour[tour_base + dst_s + dst_p] = task;
                    route_lengths[dst_off] = dst_len + 1u;
                }
            }
        }
        workgroupBarrier();

        apply_count = apply_count + 1u;
        iter_count = iter_count + 1u;
        final_delta = best_delta;
    }

    if (tid == 0u) {
        // Per-trajectory status slot: [iter, applies, final_delta, dropped].
        // `dropped` is the number of pulled tasks that found no feasible
        // insertion — those trajectories are incomplete and should be
        // skipped by best-of-N reduction on the CPU side.
        let off = t * 4u;
        status[off + 0u] = i32(iter_count);
        status[off + 1u] = i32(apply_count);
        status[off + 2u] = final_delta;
        status[off + 3u] = i32(wg_dropped_count);
    }
}
"#;

/// Phase-4 apply-2opt kernel: reverse `tour[start+i+1..=start+j]`
/// in place. One workgroup of size 1 per dispatch — apply is sequential.
const SHADER_APPLY_2OPT_WGSL: &str = r#"
struct ApplyParams {
    traj: u32,
    route: u32,
    i: u32,
    j: u32,
    max_routes: u32,
    tour_capacity: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> tour: array<u32>;
@group(0) @binding(1) var<storage, read> route_starts: array<u32>;
@group(0) @binding(2) var<uniform> ap: ApplyParams;

@compute @workgroup_size(1)
fn main() {
    let route_off = ap.traj * ap.max_routes + ap.route;
    let s = route_starts[route_off];
    let tour_base = ap.traj * ap.tour_capacity;
    var lo: u32 = ap.i + 1u;
    var hi: u32 = ap.j;
    loop {
        if (lo >= hi) { break; }
        let a = tour[tour_base + s + lo];
        let b = tour[tour_base + s + hi];
        tour[tour_base + s + lo] = b;
        tour[tour_base + s + hi] = a;
        lo = lo + 1u;
        hi = hi - 1u;
    }
}
"#;
