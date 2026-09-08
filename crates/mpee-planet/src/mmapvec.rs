//! File-backed arrays.
//!
//! Every large intermediate lives in a file, not the heap: the planet's dense
//! coordinate table alone is ~16 GB against 36 GB of RAM, and it has to
//! coexist with two 2 GB id bitmaps. Mapping it lets the OS decide what stays
//! resident, and the same mapping is what the finished dataset is queried
//! through — a cold start is a page fault, not a load.

use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::io;
use std::path::Path;

/// Create (or truncate) `path` sized for `len` elements of `T` and map it
/// writable.
pub fn create<T>(path: &Path, len: usize) -> io::Result<MmapMut> {
    let bytes = len * std::mem::size_of::<T>();
    let f = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
    f.set_len(bytes.max(1) as u64)?;
    unsafe { MmapOptions::new().map_mut(&f) }
}

pub fn open(path: &Path) -> io::Result<Mmap> {
    let f = OpenOptions::new().read(true).open(path)?;
    unsafe { MmapOptions::new().map(&f) }
}

/// Reinterpret a mapping as a slice of `T`.
///
/// # Safety
/// The file must have been written as a dense array of `T` with the same
/// endianness and layout — which is true for every file this crate writes.
pub unsafe fn as_slice<T>(m: &[u8]) -> &[T] {
    std::slice::from_raw_parts(m.as_ptr() as *const T, m.len() / std::mem::size_of::<T>())
}

/// # Safety
/// See [`as_slice`]; additionally the caller must hold exclusive access.
pub unsafe fn as_mut_slice<T>(m: &mut [u8]) -> &mut [T] {
    std::slice::from_raw_parts_mut(m.as_mut_ptr() as *mut T, m.len() / std::mem::size_of::<T>())
}
