//! Disk cache for a fully-built Contraction Hierarchy.
//!
//! Mirrors the mmap-friendly layout used in `cache_pp.rs`: a fixed header
//! followed by contiguous arrays. `load_mmap` returns the CH backed directly
//! by `Arc<Mmap>` slices, giving an instant cold start (~1 ms regardless
//! of CH size).
//!
//! Two on-disk versions:
//!
//!   * `SSSPCH1A` — vertex IDs in original (CSR) order. Layout benefits
//!     from CH but pages for hot (high-rank) vertices are scattered through
//!     the file.
//!   * `SSSPCH1B` — vertices renumbered so vertex 0 has the *highest* rank
//!     and vertex n-1 the lowest. Hot data clusters at the start of every
//!     section; on a tight memory budget the LRU naturally keeps it
//!     resident. The on-disk byte layout is identical; only the
//!     interpretation of vertex IDs changes.
//!
//! Layout (little-endian):
//!   0:    magic         "SSSPCH1A" or "SSSPCH1B"  (8 bytes)
//!   8:    n             u64
//!   16:   m_aug         u64
//!   24:   _reserved     u64                  (zero — alignment)
//!   32:   head_fwd      (n+1) × u32
//!   ...:  edge_to_fwd   m_aug × u32
//!   ...:  edge_w_fwd    m_aug × f32
//!   ...:  via_fwd       m_aug × u32
//!   ...:  up_count_fwd  n × u32
//!   ...:  head_bwd      (n+1) × u32
//!   ...:  edge_to_bwd   m_aug × u32
//!   ...:  edge_w_bwd    m_aug × f32
//!   ...:  via_bwd       m_aug × u32
//!   ...:  up_count_bwd  n × u32
//!   ...:  rank          n × u32
//!   ...:  perm          n × u32   (input-CSR-id → CH-id)
//!   ...:  edge_dist_fwd m_aug × f32  (SSSPCH1D — distance metres)
//!   ...:  edge_dist_bwd m_aug × f32  (SSSPCH1D)

use crate::buffer::Buffer;
use crate::ch::ContractionHierarchy;
use crate::graph::CsrGraph;
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

/// `SSSPCH1D` — rank-ordered, dual-channel. `edge_w` is duration (seconds);
/// `edge_dist_fwd` / `edge_dist_bwd` are parallel metres-arrays appended at
/// the end of the file. Earlier formats are rejected.
const MAGIC: &[u8; 8] = b"SSSPCH1D";
const HEADER_BYTES: usize = 32;

pub fn save<P: AsRef<Path>>(path: P, h: &ContractionHierarchy) -> std::io::Result<()> {
    let n = h.graph_fwd.n;
    let m_aug = h.graph_fwd.m();
    assert_eq!(h.graph_bwd.n, n);
    assert_eq!(h.graph_bwd.m(), m_aug);

    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);

    w.write_all(MAGIC)?;
    w.write_all(&(n as u64).to_le_bytes())?;
    w.write_all(&(m_aug as u64).to_le_bytes())?;
    w.write_all(&0u64.to_le_bytes())?; // reserved

    w.write_all(slice_u8(&h.graph_fwd.head[..]))?;
    w.write_all(slice_u8(&h.graph_fwd.edge_to[..]))?;
    w.write_all(slice_u8(&h.graph_fwd.edge_w[..]))?;
    w.write_all(slice_u8(&h.via_fwd[..]))?;
    w.write_all(slice_u8(&h.up_count_fwd[..]))?;
    w.write_all(slice_u8(&h.graph_bwd.head[..]))?;
    w.write_all(slice_u8(&h.graph_bwd.edge_to[..]))?;
    w.write_all(slice_u8(&h.graph_bwd.edge_w[..]))?;
    w.write_all(slice_u8(&h.via_bwd[..]))?;
    w.write_all(slice_u8(&h.up_count_bwd[..]))?;
    w.write_all(slice_u8(&h.rank[..]))?;
    w.write_all(slice_u8(&h.perm[..]))?;
    // SSSPCH1D dual-channel trailer: per-edge distance metres for fwd + bwd.
    w.write_all(slice_u8(&h.edge_dist_fwd[..]))?;
    w.write_all(slice_u8(&h.edge_dist_bwd[..]))?;
    w.flush()?;
    Ok(())
}

pub fn load_mmap<P: AsRef<Path>>(path: P) -> std::io::Result<ContractionHierarchy> {
    let f = File::open(path)?;
    let mmap = unsafe { Mmap::map(&f)? };
    if mmap.len() < HEADER_BYTES || &mmap[..8] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid magic — CH cache corrupt or wrong version (expected SSSPCH1C)",
        ));
    }
    let n = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
    let m_aug = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;

    let expected = HEADER_BYTES
        + 4 * (n + 1)   // head_fwd
        + 4 * m_aug     // edge_to_fwd
        + 4 * m_aug     // edge_w_fwd
        + 4 * m_aug     // via_fwd
        + 4 * n         // up_count_fwd
        + 4 * (n + 1)   // head_bwd
        + 4 * m_aug     // edge_to_bwd
        + 4 * m_aug     // edge_w_bwd
        + 4 * m_aug     // via_bwd
        + 4 * n         // up_count_bwd
        + 4 * n         // rank
        + 4 * n         // perm
        + 4 * m_aug     // edge_dist_fwd
        + 4 * m_aug;    // edge_dist_bwd
    if mmap.len() < expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "CH cache too small: {} bytes, expected {}",
                mmap.len(),
                expected
            ),
        ));
    }

    let mmap_arc = Arc::new(mmap);
    let mut off = HEADER_BYTES;

    let head_fwd = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n + 1);
    off += 4 * (n + 1);
    let edge_to_fwd = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, m_aug);
    off += 4 * m_aug;
    let edge_w_fwd = Buffer::<f32>::from_mmap(mmap_arc.clone(), off, m_aug);
    off += 4 * m_aug;
    let via_fwd = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, m_aug);
    off += 4 * m_aug;
    let up_count_fwd = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n);
    off += 4 * n;
    let head_bwd = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n + 1);
    off += 4 * (n + 1);
    let edge_to_bwd = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, m_aug);
    off += 4 * m_aug;
    let edge_w_bwd = Buffer::<f32>::from_mmap(mmap_arc.clone(), off, m_aug);
    off += 4 * m_aug;
    let via_bwd = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, m_aug);
    off += 4 * m_aug;
    let up_count_bwd = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n);
    off += 4 * n;
    let rank = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n);
    off += 4 * n;
    let perm = Buffer::<u32>::from_mmap(mmap_arc.clone(), off, n);
    off += 4 * n;
    let edge_dist_fwd = Buffer::<f32>::from_mmap(mmap_arc.clone(), off, m_aug);
    off += 4 * m_aug;
    let edge_dist_bwd = Buffer::<f32>::from_mmap(mmap_arc, off, m_aug);

    Ok(ContractionHierarchy {
        graph_fwd: CsrGraph {
            n,
            head: head_fwd,
            edge_to: edge_to_fwd,
            edge_w: edge_w_fwd,
        },
        graph_bwd: CsrGraph {
            n,
            head: head_bwd,
            edge_to: edge_to_bwd,
            edge_w: edge_w_bwd,
        },
        up_count_fwd,
        up_count_bwd,
        rank,
        via_fwd,
        via_bwd,
        edge_dist_fwd,
        edge_dist_bwd,
        perm,
    })
}

#[inline]
fn slice_u8<T>(s: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
    }
}
