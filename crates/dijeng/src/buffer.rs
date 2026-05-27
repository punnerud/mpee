//! `Buffer<T>` — transparent slice owner that is either an owned `Vec<T>` or
//! a pointer into a memory-mapped region. All algorithms that take `&[T]`
//! or use indexing work unchanged thanks to `Deref<Target=[T]>`.
//!
//! On wasm (no `native` feature) only the `Owned` variant exists — caches are
//! loaded into memory via `load_bytes` rather than mmap'd.

use std::ops::Deref;

#[cfg(feature = "native")]
use memmap2::Mmap;
#[cfg(feature = "native")]
use std::sync::Arc;

pub struct Buffer<T> {
    inner: BufferInner<T>,
}

enum BufferInner<T> {
    Owned(Vec<T>),
    #[cfg(feature = "native")]
    Mapped {
        // `mmap` holds the actual file mapping alive. Multiple Buffers can
        // share the same Arc<Mmap> when the cache file backs head/edge_to/...
        _mmap: Arc<Mmap>,
        ptr: *const T,
        len: usize,
    },
}

// Mmap is Send + Sync. Raw pointers aren't auto-Send, so we add manual impls.
// Safety: We never expose interior mutability and the mmap is alive as long
// as any Buffer holding it.
unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}

impl<T> Buffer<T> {
    #[cfg(feature = "native")]
    pub fn from_mmap(mmap: Arc<Mmap>, byte_offset: usize, count: usize) -> Self {
        let ptr = unsafe { mmap.as_ptr().add(byte_offset) as *const T };
        Buffer {
            inner: BufferInner::Mapped {
                _mmap: mmap,
                ptr,
                len: count,
            },
        }
    }

    /// Build an **owned** buffer of `count` `T`s by copying `count *
    /// size_of::<T>()` bytes starting at `byte_offset` in `bytes`. The copy is
    /// alignment-safe (the destination `Vec<T>` is `T`-aligned; the source may
    /// be any alignment). Used by the wasm `load_bytes` cache loaders, where
    /// the cache is an in-memory `&[u8]` rather than an mmap. `T` must be a
    /// plain-old-data type (the on-disk format only stores `u32`/`f32`/tuples).
    pub fn from_bytes_copy(bytes: &[u8], byte_offset: usize, count: usize) -> Self {
        let need = count * std::mem::size_of::<T>();
        assert!(byte_offset + need <= bytes.len(), "from_bytes_copy out of range");
        let mut v: Vec<T> = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr().add(byte_offset),
                v.as_mut_ptr() as *mut u8,
                need,
            );
            v.set_len(count);
        }
        Buffer { inner: BufferInner::Owned(v) }
    }

    pub fn as_slice(&self) -> &[T] {
        match &self.inner {
            BufferInner::Owned(v) => v.as_slice(),
            #[cfg(feature = "native")]
            BufferInner::Mapped { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }

    pub fn len(&self) -> usize {
        match &self.inner {
            BufferInner::Owned(v) => v.len(),
            #[cfg(feature = "native")]
            BufferInner::Mapped { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    fn from(v: Vec<T>) -> Self {
        Buffer {
            inner: BufferInner::Owned(v),
        }
    }
}

impl<T: Clone> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        // Mmap-backed buffers clone to owned to keep semantics simple.
        Buffer {
            inner: BufferInner::Owned(self.as_slice().to_vec()),
        }
    }
}
