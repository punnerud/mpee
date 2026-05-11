//! `Buffer<T>` — transparent slice-eier som er enten en eid `Vec<T>` eller
//! en peker inn i et memory-mapped område. Alle algoritmer som tar `&[T]`
//! eller bruker indeksering fungerer uendret takket være `Deref<Target=[T]>`.

use memmap2::Mmap;
use std::ops::Deref;
use std::sync::Arc;

pub struct Buffer<T> {
    inner: BufferInner<T>,
}

enum BufferInner<T> {
    Owned(Vec<T>),
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

    pub fn as_slice(&self) -> &[T] {
        match &self.inner {
            BufferInner::Owned(v) => v.as_slice(),
            BufferInner::Mapped { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }

    pub fn len(&self) -> usize {
        match &self.inner {
            BufferInner::Owned(v) => v.len(),
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
