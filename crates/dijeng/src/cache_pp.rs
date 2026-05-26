//! Cache for **fully preprocessed** graph:
//!   * Reordered CSR (BFS-order for cache-locality)
//!   * Edge partition (light/heavy split per vertex)
//!   * Reverse-CSR (for bidirectional Dijeng)
//!   * Vertex coordinates (for visualization / haversine)
//!   * `new_id` permutation (to map old IDs to new ones)
//!
//! All mmap-loaded in one syscall. Cold start = ~1 ms.
//!
//! Format (little-endian, alle 4-byte-aligned):
//! ```
//! 0:    magic        "SSSPP2A\0"        (8 bytes)
//! 8:    n            u64
//! 16:   m            u64
//! 24:   delta        f32
//! 28:   _flags       u32                (reserved)
//! 32:   head         (n+1) × u32
//! ...:  edge_to      m × u32
//! ...:  edge_w       m × f32
//! ...:  coords       n × (f32, f32)
//! ...:  light_count  n × u32
//! ...:  new_id       n × u32
//! ...:  rev_head     (n+1) × u32
//! ...:  rev_edge_to  m × u32
//! ...:  rev_edge_w   m × f32
//! ```

use crate::buffer::Buffer;
use crate::graph::CsrGraph;
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

/// `SSSPP2C\0` — adds a per-edge distance channel (parallel to `edge_w`)
/// after the existing layout. Earlier versions are rejected.
const MAGIC: &[u8; 8] = b"SSSPP2C\0";
const HEADER_BYTES: usize = 32;

pub struct PpFull {
    pub graph: CsrGraph,
    pub reverse: CsrGraph,
    /// Per-edge distance in metres (parallel to `graph.edge_w`).
    pub edge_dist: Buffer<f32>,
    /// Per-edge distance for `reverse`. Same edges, just permuted.
    pub rev_edge_dist: Buffer<f32>,
    pub light_count: Buffer<u32>,
    pub new_id: Buffer<u32>,
    pub coords: Buffer<(f32, f32)>,
    pub delta: f32,
}

pub fn save<P: AsRef<Path>>(
    path: P,
    g: &CsrGraph,
    reverse: &CsrGraph,
    light_count: &[u32],
    new_id: &[u32],
    coords: &[(f32, f32)],
    delta: f32,
    edge_dist: &[f32],
    rev_edge_dist: &[f32],
) -> std::io::Result<()> {
    assert_eq!(g.n, reverse.n);
    assert_eq!(g.m(), reverse.m());
    assert_eq!(light_count.len(), g.n);
    assert_eq!(new_id.len(), g.n);
    assert_eq!(coords.len(), g.n);
    assert_eq!(edge_dist.len(), g.m());
    assert_eq!(rev_edge_dist.len(), g.m());

    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);

    w.write_all(MAGIC)?;
    w.write_all(&(g.n as u64).to_le_bytes())?;
    w.write_all(&(g.m() as u64).to_le_bytes())?;
    w.write_all(&delta.to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?; // flags reserved

    w.write_all(slice_u8(&g.head[..]))?;
    w.write_all(slice_u8(&g.edge_to[..]))?;
    w.write_all(slice_u8(&g.edge_w[..]))?;
    w.write_all(slice_u8(coords))?;
    w.write_all(slice_u8(light_count))?;
    w.write_all(slice_u8(new_id))?;
    w.write_all(slice_u8(&reverse.head[..]))?;
    w.write_all(slice_u8(&reverse.edge_to[..]))?;
    w.write_all(slice_u8(&reverse.edge_w[..]))?;
    // SSSPP2C tail: distance channels, fwd then bwd.
    w.write_all(slice_u8(edge_dist))?;
    w.write_all(slice_u8(rev_edge_dist))?;
    w.flush()?;
    Ok(())
}

pub fn load_mmap<P: AsRef<Path>>(path: P) -> std::io::Result<PpFull> {
    let f = File::open(path)?;
    let mmap = unsafe { Mmap::map(&f)? };
    if mmap.len() < HEADER_BYTES || &mmap[..8] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ugyldig magic — preprocessing-cache korrupt eller annen versjon",
        ));
    }
    let n = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
    let m = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
    let delta = f32::from_le_bytes(mmap[24..28].try_into().unwrap());

    let expected = HEADER_BYTES
        + 4 * (n + 1)   // head
        + 4 * m         // edge_to
        + 4 * m         // edge_w
        + 8 * n         // coords
        + 4 * n         // light_count
        + 4 * n         // new_id
        + 4 * (n + 1)   // rev_head
        + 4 * m         // rev_edge_to
        + 4 * m         // rev_edge_w
        + 4 * m         // edge_dist (fwd)
        + 4 * m;        // rev_edge_dist
    if mmap.len() < expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("preprocessing-cache for liten: {} bytes, forventet {}", mmap.len(), expected),
        ));
    }

    let mmap_arc = Arc::new(mmap);

    let mut off = HEADER_BYTES;
    let head = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n + 1);
    off += 4 * (n + 1);
    let edge_to = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, m);
    off += 4 * m;
    let edge_w = Buffer::<f32>::from_mmap(mmap_arc.clone(), off, m);
    off += 4 * m;
    let coords = Buffer::<(f32, f32)>::from_mmap(mmap_arc.clone(), off, n);
    off += 8 * n;
    let light_count = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n);
    off += 4 * n;
    let new_id = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n);
    off += 4 * n;
    let rev_head = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n + 1);
    off += 4 * (n + 1);
    let rev_edge_to = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, m);
    off += 4 * m;
    let rev_edge_w = Buffer::<f32>::from_mmap(mmap_arc.clone(), off, m);
    off += 4 * m;
    let edge_dist = Buffer::<f32>::from_mmap(mmap_arc.clone(), off, m);
    off += 4 * m;
    let rev_edge_dist = Buffer::<f32>::from_mmap(mmap_arc, off, m);

    Ok(PpFull {
        graph: CsrGraph {
            n,
            head,
            edge_to,
            edge_w,
        },
        reverse: CsrGraph {
            n,
            head: rev_head,
            edge_to: rev_edge_to,
            edge_w: rev_edge_w,
        },
        edge_dist,
        rev_edge_dist,
        light_count,
        new_id,
        coords,
        delta,
    })
}

#[inline]
fn slice_u8<T>(s: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
    }
}
