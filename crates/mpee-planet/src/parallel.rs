//! Chunked parallel blob driver.
//!
//! A planet pass must use all cores for zlib inflation, but it must not buffer
//! the whole output in RAM (pass 1 alone emits ~6 GB). So blobs are processed
//! in bounded chunks: inflate and transform a chunk in parallel, append the
//! per-blob byte buffers to disk **in blob order**, drop, repeat. Memory stays
//! O(chunk × blob size) and the output is deterministic.

use crate::pbf::{BlobDesc, BlobKind, Block, Inflater};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    static INFLATER: std::cell::RefCell<Inflater> = std::cell::RefCell::new(Inflater::default());
}

/// A raw pointer to a slice that many threads scatter into at **disjoint**
/// indices. Used for the dense coordinate array, where each node's slot is its
/// unique rank and no two threads can collide.
#[derive(Clone, Copy)]
pub struct Scatter<T>(pub *mut T, pub usize);
unsafe impl<T> Send for Scatter<T> {}
unsafe impl<T> Sync for Scatter<T> {}
impl<T> Scatter<T> {
    /// # Safety
    /// The caller guarantees no two concurrent writes share an index.
    #[inline]
    pub unsafe fn put(&self, i: usize, v: T) {
        debug_assert!(i < self.1);
        std::ptr::write(self.0.add(i), v);
    }
}

pub struct Progress {
    label: String,
    total: u64,
    done: AtomicU64,
    t0: std::time::Instant,
}

impl Progress {
    pub fn new(label: &str, total: u64) -> Self {
        Progress {
            label: label.to_string(),
            total,
            done: AtomicU64::new(0),
            t0: std::time::Instant::now(),
        }
    }
    pub fn add(&self, n: u64, extra: &str) {
        let d = self.done.fetch_add(n, Ordering::Relaxed) + n;
        let prev = d - n;
        // Report on every 2 % crossing, so output is bounded regardless of size.
        let step = (self.total / 50).max(1);
        if d / step != prev / step {
            let el = self.t0.elapsed().as_secs_f64();
            let eta = if d > 0 { el / d as f64 * (self.total - d) as f64 } else { 0.0 };
            eprintln!(
                "  [{}] {:.0} % — {:.0} s elapsed, ~{:.0} s left {extra}",
                self.label,
                d as f64 / self.total as f64 * 100.0,
                el,
                eta
            );
        }
    }
    pub fn finish(&self, extra: &str) {
        eprintln!(
            "  [{}] done in {:.1} s {extra}",
            self.label,
            self.t0.elapsed().as_secs_f64()
        );
    }
}

/// Run `f` over every data blob in `range`. `f` fills one byte buffer per
/// output stream; buffers are appended to the corresponding file in blob order.
///
/// The argument list is long because this is the pipeline's one driver: every
/// pass differs in source range, chunk size, shared state and output streams,
/// and bundling those into a struct would move the same parameters somewhere
/// less visible rather than removing them.
#[allow(clippy::too_many_arguments)]
pub fn scan_blobs<S: Sync>(
    path: &Path,
    blobs: &[BlobDesc],
    range: std::ops::Range<usize>,
    chunk: usize,
    shared: &S,
    outs: &[PathBuf],
    label: &str,
    f: impl Fn(&Block, &S, &mut Vec<Vec<u8>>) + Sync,
) -> io::Result<Vec<u64>> {
    let idx: Vec<usize> = range.filter(|&i| blobs[i].kind == BlobKind::Data).collect();
    let mut writers: Vec<BufWriter<File>> = outs
        .iter()
        .map(|p| File::create(p).map(|f| BufWriter::with_capacity(1 << 23, f)))
        .collect::<io::Result<_>>()?;
    let nout = outs.len();
    let mut written = vec![0u64; nout];
    let prog = Progress::new(label, idx.len() as u64);

    for group in idx.chunks(chunk) {
        let parts: Vec<Vec<Vec<u8>>> = group
            .par_iter()
            .map(|&bi| {
                INFLATER.with(|cell| {
                    let mut inf = cell.borrow_mut();
                    let file = File::open(path).expect("reopen pbf");
                    inf.load(&file, &blobs[bi]).expect("inflate blob");
                    let blk = Block::parse(&inf.out);
                    let mut bufs: Vec<Vec<u8>> = (0..nout).map(|_| Vec::new()).collect();
                    f(&blk, shared, &mut bufs);
                    bufs
                })
            })
            .collect();
        for bufs in &parts {
            for (i, b) in bufs.iter().enumerate() {
                writers[i].write_all(b)?;
                written[i] += b.len() as u64;
            }
        }
        let gb: String = if nout > 0 {
            format!("— {:.2} GB written", written.iter().sum::<u64>() as f64 / 1e9)
        } else {
            String::new()
        };
        prog.add(group.len() as u64, &gb);
    }
    for w in writers.iter_mut() {
        w.flush()?;
    }
    prog.finish("");
    Ok(written)
}
