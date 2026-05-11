//! Binær disk-cache for CSR-graf og koordinater.
//!
//! Format (little-endian, native f32/u32):
//!   magic:   8 bytes "SSSPCSR1"
//!   n:       u64
//!   m:       u64
//!   head:    (n+1) × u32
//!   edge_to: m × u32
//!   edge_w:  m × f32
//!   coords:  n × (f32, f32)
//!
//! To lasterier:
//!   * `load`     — leser hele fila inn i Vec via `read_exact`. ~4 ms for
//!                  London (cache i page cache).
//!   * `load_mmap`— mmap-er fila og returnerer en CsrGraph der head/edge_to/
//!                  edge_w peker direkte inn i page-cachen. ~1 ms uansett
//!                  graf-størrelse, og spørringer touch'er sider lazy via
//!                  page-fault.

use crate::buffer::Buffer;
use crate::graph::CsrGraph;
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

/// `SSSPCSR3` — `edge_w` is duration (seconds) and a parallel `edge_dist`
/// array (metres) is stored after `edge_w`. Earlier versions are rejected so
/// callers regenerate to pick up the new dual-channel layout.
const MAGIC: &[u8; 8] = b"SSSPCSR3";

/// Header bytes: magic(8) + n(8) + m(8) = 24
const HEADER_BYTES: usize = 24;

pub fn save<P: AsRef<Path>>(
    path: P,
    g: &CsrGraph,
    coords: &[(f32, f32)],
    edge_dist: &[f32],
) -> std::io::Result<()> {
    assert_eq!(coords.len(), g.n, "coords-lengde må matche n");
    assert_eq!(edge_dist.len(), g.m(), "edge_dist må matche m");
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);
    w.write_all(MAGIC)?;
    w.write_all(&(g.n as u64).to_le_bytes())?;
    w.write_all(&(g.m() as u64).to_le_bytes())?;
    w.write_all(slice_u8(&g.head[..]))?;
    w.write_all(slice_u8(&g.edge_to[..]))?;
    w.write_all(slice_u8(&g.edge_w[..]))?;
    w.write_all(slice_u8(coords))?;
    w.write_all(slice_u8(edge_dist))?;
    w.flush()?;
    Ok(())
}

/// Eager-load: kopier hele fila inn i Vecs. Trygg og garantert at all data
/// er i prosess-RAM når funksjonen returnerer.
pub fn load<P: AsRef<Path>>(
    path: P,
) -> std::io::Result<(CsrGraph, Vec<(f32, f32)>, Vec<f32>)> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ugyldig magic — cache-fil korrupt eller annen versjon",
        ));
    }
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8)?;
    let m = u64::from_le_bytes(buf8) as usize;

    let mut head = vec![0u32; n + 1];
    f.read_exact(slice_u8_mut(&mut head))?;

    let mut edge_to = vec![0u32; m];
    f.read_exact(slice_u8_mut(&mut edge_to))?;

    let mut edge_w = vec![0.0f32; m];
    f.read_exact(slice_u8_mut(&mut edge_w))?;

    let mut coords = vec![(0.0f32, 0.0f32); n];
    f.read_exact(slice_u8_mut(&mut coords))?;

    let mut edge_dist = vec![0.0f32; m];
    f.read_exact(slice_u8_mut(&mut edge_dist))?;

    Ok((
        CsrGraph {
            n,
            head: head.into(),
            edge_to: edge_to.into(),
            edge_w: edge_w.into(),
        },
        coords,
        edge_dist,
    ))
}

/// Mmap-load: åpner fila og returnerer CsrGraph der head/edge_to/edge_w er
/// pekere inn i mmappet. Faktisk lasting skjer lazy via page-fault. Fungerer
/// for grafer langt større enn RAM (OS dropper sider under press).
///
/// Coords returneres også som Buffer (mmap-backed). Bruk `.as_slice()` for
/// direkte tilgang.
pub fn load_mmap<P: AsRef<Path>>(
    path: P,
) -> std::io::Result<(CsrGraph, Buffer<(f32, f32)>, Buffer<f32>)> {
    let f = File::open(path)?;
    let mmap = unsafe { Mmap::map(&f)? };
    if mmap.len() < HEADER_BYTES || &mmap[..8] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ugyldig magic — cache-fil korrupt eller annen versjon",
        ));
    }
    let n = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
    let m = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;

    let expected = HEADER_BYTES
        + 4 * (n + 1)   // head
        + 4 * m         // edge_to
        + 4 * m         // edge_w (duration)
        + 8 * n         // coords
        + 4 * m;        // edge_dist (metres)
    if mmap.len() < expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cache-fil for liten: {} bytes, forventet {}", mmap.len(), expected),
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
    let edge_dist = Buffer::<f32>::from_mmap(mmap_arc, off, m);

    Ok((
        CsrGraph {
            n,
            head,
            edge_to,
            edge_w,
        },
        coords,
        edge_dist,
    ))
}

#[inline]
fn slice_u8<T>(s: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
    }
}
#[inline]
fn slice_u8_mut<T>(s: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s))
    }
}
