//! Measure GPU "upload" cost in detail on Apple Silicon: separate
//! device init from each individual buffer upload, so we know whether
//! the 50ms "upload" is dominated by device setup or actual memcpy/blit.
//!
//! Run with:
//!   cargo run --release --example gpu_upload_bench

use std::time::Instant;

use bytemuck;
use wgpu::util::DeviceExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU upload micro-benchmark ===\n");

    // 1. Device init.
    let t0 = Instant::now();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).ok_or("no adapter")?;
    let info = adapter.get_info();
    println!("Adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("upload-bench"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))?;
    let init_t = t0.elapsed();
    println!("Device init: {:?}\n", init_t);

    // 2. Upload matrices of various sizes via three different paths and time each.
    for &n in &[100usize, 200, 500, 1000, 2000, 5000] {
        let bytes = n * n * 4;
        let mb = bytes as f64 / (1024.0 * 1024.0);
        println!("--- N = {n} ({:.2} MB matrix) ---", mb);

        // Random data (deterministic seed per N for reproducibility).
        let data: Vec<i32> = (0..n*n).map(|i| (i.wrapping_mul(2654435761)) as i32).collect();

        // Path A: create_buffer_init (what brooom currently uses).
        // wgpu-core internally allocates a Private buffer + Shared staging buffer
        // and queues a blit. Cost = malloc + memcpy + blit-schedule.
        for i in 0..3 {
            let t0 = Instant::now();
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrix-A"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let create_t = t0.elapsed();

            // Force the blit to actually run by submitting an empty queue.
            let t0 = Instant::now();
            queue.submit(std::iter::empty());
            device.poll(wgpu::Maintain::Wait);
            let submit_t = t0.elapsed();

            let total_us = (create_t.as_secs_f64() + submit_t.as_secs_f64()) * 1e6;
            let throughput_gbps = bytes as f64 / 1e9 / (total_us / 1e6);
            println!("  A run {i}: create_buffer_init  create={:>7.1} µs, submit+poll={:>6.1} µs, total={:>7.1} µs ({:.2} GB/s)",
                     create_t.as_secs_f64() * 1e6,
                     submit_t.as_secs_f64() * 1e6,
                     total_us,
                     throughput_gbps);
            drop(buf);
        }

        // Path B: pre-allocate (no contents) + write_buffer.
        // queue.write_buffer also uses a staging path internally on backends
        // without true direct-write support. Comparable to A.
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrix-B-empty"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for i in 0..3 {
            let t0 = Instant::now();
            queue.write_buffer(&buf_b, 0, bytemuck::cast_slice(&data));
            let write_t = t0.elapsed();
            let t0 = Instant::now();
            queue.submit(std::iter::empty());
            device.poll(wgpu::Maintain::Wait);
            let submit_t = t0.elapsed();
            let total_us = (write_t.as_secs_f64() + submit_t.as_secs_f64()) * 1e6;
            let throughput_gbps = bytes as f64 / 1e9 / (total_us / 1e6);
            println!("  B run {i}: pre-alloc+write       write ={:>7.1} µs, submit+poll={:>6.1} µs, total={:>7.1} µs ({:.2} GB/s)",
                     write_t.as_secs_f64() * 1e6,
                     submit_t.as_secs_f64() * 1e6,
                     total_us,
                     throughput_gbps);
        }
        drop(buf_b);

        // Path C: MAP_WRITE on a non-storage staging buffer + COPY_SRC.
        // wgpu disallows STORAGE | MAP_WRITE direct, but you CAN have a
        // mappable staging buffer that you write directly into and then
        // blit yourself in a single dispatch. Worth comparing to see if
        // user-controlled blit is faster than the wgpu-internal one.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrix-C-staging"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let target = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrix-C-target"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for i in 0..3 {
            let t0 = Instant::now();
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Write, move |r| { let _ = tx.send(r); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv()??;
            slice.get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&data));
            staging.unmap();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("user-blit"),
            });
            encoder.copy_buffer_to_buffer(&staging, 0, &target, 0, bytes as u64);
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
            let total_us = t0.elapsed().as_secs_f64() * 1e6;
            let throughput_gbps = bytes as f64 / 1e9 / (total_us / 1e6);
            println!("  C run {i}: user-blit (map+copy)   total={:>7.1} µs ({:.2} GB/s)",
                     total_us, throughput_gbps);
        }
        println!();
    }

    Ok(())
}
