//! Page-granular LRU on top of an `Arc<Mmap>`.
//!
//! Algorithms work directly on the mmap'd cache file as before. After each
//! access we call `touch(byte_offset, byte_size)` which:
//!
//!   1. Identifies the OS pages covering that range (typically 4 KB or 16 KB).
//!   2. Marks them most-recently-used in our internal LRU.
//!   3. If total tracked bytes exceeds the configured budget, picks the
//!      least-recently-used pages and calls `madvise(MADV_DONTNEED)` on them
//!      so the OS releases physical RAM (the virtual mapping stays valid;
//!      next access faults back in from disk).
//!
//! This gives a configurable memory budget for routing structures that
//! scale beyond available RAM (e.g. continent-scale road networks):
//!
//!   * 50 MB budget on a 1.5 GB CH cache → keep ~3% in RAM
//!   * 500 MB budget → keep ~30% in RAM
//!
//! Cold (rarely-used) regions get evicted; hot (high-rank vertices in CH)
//! stay resident.
//!
//! ## Pinning
//!
//! `pin_range(off, len)` excludes pages from the LRU entirely. Pinned pages
//! are never evicted and never counted against the byte budget. Useful for
//! the topmost vertices of a contraction hierarchy: they're touched by every
//! query and pinning saves both LRU bookkeeping and madvise churn under
//! tight budgets.
//!
//! ## Thread safety
//!
//! `PagedMmap` is `Send + Sync`. Internal state is behind a `Mutex`; counters
//! are `AtomicU64`. Multiple threads can call `touch` concurrently (e.g. via
//! `rayon::par_iter` over a query workload).

use memmap2::Mmap;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;

const DEFAULT_PAGE_SIZE: usize = 16 * 1024;

/// Number of LRU shards. `touch` and `pin_range` route each page to a shard
/// by `page_id & (N_SHARDS-1)`, so contention under multi-threaded use scales
/// down by `N_SHARDS`. Must be a power of two.
const N_SHARDS: usize = 16;
const SHARD_MASK: u64 = (N_SHARDS as u64) - 1;

pub struct PagedMmap {
    mmap: Arc<Mmap>,
    pub page_size: usize,
    pub budget_bytes: usize,
    /// Per-shard byte budget; total is `budget_per_shard * N_SHARDS`.
    budget_per_shard: usize,
    shards: Vec<Mutex<Shard>>,
}

struct Shard {
    /// Page-id → unit. Insertion-/touch-order; oldest at front (lru),
    /// newest at back (mru).
    warm: lru::LruCache<u64, ()>,
    bytes_warm: usize,
    /// Pages explicitly held in memory; never tracked by `warm`, never evicted,
    /// never count against `bytes_warm`.
    pinned: HashSet<u64>,
    /// Per-shard counters (no cross-core atomic bouncing on the hot path).
    n_touches: u64,
    n_pages_loaded: u64,
    n_pages_evicted: u64,
}

#[inline]
fn shard_of(page_id: u64) -> usize {
    (page_id & SHARD_MASK) as usize
}

/// Thread-private buffer for staging touches before committing them under
/// shard locks. Reusable across queries (call `commit` to flush, then reuse).
pub struct TouchBuf {
    per_shard: Vec<Vec<u64>>,
}

impl TouchBuf {
    pub fn new() -> Self {
        let mut per_shard = Vec::with_capacity(N_SHARDS);
        for _ in 0..N_SHARDS {
            per_shard.push(Vec::with_capacity(32));
        }
        Self { per_shard }
    }
}

impl Default for TouchBuf {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub warm_bytes: usize,
    pub budget_bytes: usize,
    pub pinned_bytes: usize,
    pub n_pages_resident: usize,
    pub n_pages_pinned: usize,
    pub n_touches: u64,
    pub n_pages_loaded: u64,
    pub n_pages_evicted: u64,
}

impl PagedMmap {
    pub fn new(mmap: Arc<Mmap>, budget_bytes: usize) -> Self {
        Self::new_with_page_size(mmap, budget_bytes, default_page_size())
    }

    pub fn new_with_page_size(
        mmap: Arc<Mmap>,
        budget_bytes: usize,
        page_size: usize,
    ) -> Self {
        // Per-shard budget. Pages are routed by id mod N_SHARDS; on a uniform
        // page-id distribution each shard sees ~budget/N_SHARDS bytes.
        let budget_per_shard = (budget_bytes / N_SHARDS).max(page_size);
        let cap = NonZeroUsize::new((budget_per_shard / page_size).max(1) * 2).unwrap();
        let mut shards = Vec::with_capacity(N_SHARDS);
        for _ in 0..N_SHARDS {
            shards.push(Mutex::new(Shard {
                warm: lru::LruCache::new(cap),
                bytes_warm: 0,
                pinned: HashSet::new(),
                n_touches: 0,
                n_pages_loaded: 0,
                n_pages_evicted: 0,
            }));
        }
        Self {
            mmap,
            page_size,
            budget_bytes,
            budget_per_shard,
            shards,
        }
    }

    /// Stage a byte range into a thread-local buffer without taking any
    /// shard lock. Pages are committed to the LRU when `commit` is called
    /// (typically once per query). Reduces lock acquisitions from O(touches)
    /// to O(N_SHARDS) per query, which is the difference between parallel
    /// scaling and parallel slowdown.
    #[inline]
    pub fn stage(&self, buf: &mut TouchBuf, byte_off: usize, byte_size: usize) {
        if byte_size == 0 {
            return;
        }
        let pg = self.page_size as u64;
        let first = (byte_off as u64) / pg;
        let last = ((byte_off + byte_size - 1) as u64) / pg;
        for page in first..=last {
            buf.per_shard[shard_of(page)].push(page);
        }
    }

    /// Commit a buffer of staged touches: for each shard, take the lock once,
    /// promote all referenced pages to MRU, then evict if over budget.
    pub fn commit(&self, buf: &mut TouchBuf) {
        for s in 0..N_SHARDS {
            let pages = &mut buf.per_shard[s];
            if pages.is_empty() {
                continue;
            }
            let mut shard = self.shards[s].lock().unwrap();
            shard.n_touches += pages.len() as u64;
            for &page in pages.iter() {
                if shard.pinned.contains(&page) {
                    continue;
                }
                if shard.warm.put(page, ()).is_none() {
                    shard.bytes_warm += self.page_size;
                    shard.n_pages_loaded += 1;
                }
            }
            while shard.bytes_warm > self.budget_per_shard {
                match shard.warm.pop_lru() {
                    Some((page_id, _)) => {
                        shard.bytes_warm -= self.page_size;
                        shard.n_pages_evicted += 1;
                        self.madvise_dont_need(page_id);
                    }
                    None => break,
                }
            }
            pages.clear();
        }
    }

    /// Convenience: stage + commit in one call. Exists for callers that don't
    /// want to manage a buffer (single-shot lookups). Internally allocates;
    /// for hot paths use `stage` + `commit` with a reused buffer.
    pub fn touch(&self, byte_off: usize, byte_size: usize) {
        let mut buf = TouchBuf::new();
        self.stage(&mut buf, byte_off, byte_size);
        self.commit(&mut buf);
    }


    /// Pin a byte range — the covering pages are kept resident permanently
    /// (never evicted, never counted against the budget). Use for hot data
    /// like the top of a CH (highest-rank vertices).
    ///
    /// Idempotent. Pinning a page that was previously tracked as "warm"
    /// removes it from the LRU and reduces `bytes_warm` accordingly.
    pub fn pin_range(&self, byte_off: usize, byte_size: usize) {
        if byte_size == 0 {
            return;
        }
        let pg = self.page_size as u64;
        let first = (byte_off as u64) / pg;
        let last = ((byte_off + byte_size - 1) as u64) / pg;

        for page in first..=last {
            let mut shard = self.shards[shard_of(page)].lock().unwrap();
            if shard.pinned.insert(page) {
                if shard.warm.pop(&page).is_some() {
                    shard.bytes_warm -= self.page_size;
                }
            }
        }
    }

    fn madvise_dont_need(&self, page_id: u64) {
        let addr = self.mmap.as_ptr() as usize + (page_id as usize) * self.page_size;
        let mmap_end = self.mmap.as_ptr() as usize + self.mmap.len();
        if addr >= mmap_end {
            return;
        }
        let len = self.page_size.min(mmap_end - addr);
        unsafe {
            // Safe assuming `addr` and `len` are within the mapping.
            let _ = libc::madvise(addr as *mut _, len as libc::size_t, libc::MADV_DONTNEED);
        }
    }

    pub fn stats(&self) -> Stats {
        let mut warm_bytes = 0;
        let mut n_pages_resident = 0;
        let mut n_pages_pinned = 0;
        let mut n_touches = 0;
        let mut n_pages_loaded = 0;
        let mut n_pages_evicted = 0;
        for shard in &self.shards {
            let s = shard.lock().unwrap();
            warm_bytes += s.bytes_warm;
            n_pages_resident += s.warm.len();
            n_pages_pinned += s.pinned.len();
            n_touches += s.n_touches;
            n_pages_loaded += s.n_pages_loaded;
            n_pages_evicted += s.n_pages_evicted;
        }
        Stats {
            warm_bytes,
            budget_bytes: self.budget_bytes,
            pinned_bytes: n_pages_pinned * self.page_size,
            n_pages_resident,
            n_pages_pinned,
            n_touches,
            n_pages_loaded,
            n_pages_evicted,
        }
    }

    /// Reset stats (preserves resident pages and pin set).
    pub fn reset_stats(&self) {
        for shard in &self.shards {
            let mut s = shard.lock().unwrap();
            s.n_touches = 0;
            s.n_pages_loaded = 0;
            s.n_pages_evicted = 0;
        }
    }

    pub fn mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }
}

fn default_page_size() -> usize {
    // sysconf(_SC_PAGESIZE) gives the OS page size. Fall back to 16K (typical
    // for ARM macOS) or 4K (typical for x86).
    #[cfg(unix)]
    {
        unsafe {
            let v = libc::sysconf(libc::_SC_PAGESIZE);
            if v > 0 {
                return v as usize;
            }
        }
    }
    DEFAULT_PAGE_SIZE
}

/// Layout description of the CH cache file (cache_ch.rs format) — used to
/// translate "logical" CH accesses (vertex u, edge k) into byte offsets so
/// we can touch the right pages.
pub struct ChLayout {
    pub n: usize,
    pub m: usize,
    pub head_fwd_off: usize,
    pub edge_to_fwd_off: usize,
    pub edge_w_fwd_off: usize,
    pub via_fwd_off: usize,
    pub up_count_fwd_off: usize,
    pub head_bwd_off: usize,
    pub edge_to_bwd_off: usize,
    pub edge_w_bwd_off: usize,
    pub via_bwd_off: usize,
    pub up_count_bwd_off: usize,
    pub rank_off: usize,
}

impl ChLayout {
    /// Mirror of `cache_ch::load_mmap` — compute byte offsets without
    /// allocating `Buffer`s. The on-disk layout has been the same across
    /// SSSPCH1A/B/C; only the values stored in `edge_w` differ (B/C are
    /// rank-ordered, C is duration-weighted).
    pub fn from_cache_file(mmap: &Mmap) -> std::io::Result<Self> {
        const HEADER_BYTES: usize = 32;
        if mmap.len() < HEADER_BYTES
            || (&mmap[..8] != b"SSSPCH1A"
                && &mmap[..8] != b"SSSPCH1B"
                && &mmap[..8] != b"SSSPCH1C")
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic",
            ));
        }
        let n = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let m = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let mut off = HEADER_BYTES;
        let head_fwd_off = off;
        off += 4 * (n + 1);
        let edge_to_fwd_off = off;
        off += 4 * m;
        let edge_w_fwd_off = off;
        off += 4 * m;
        let via_fwd_off = off;
        off += 4 * m;
        let up_count_fwd_off = off;
        off += 4 * n;
        let head_bwd_off = off;
        off += 4 * (n + 1);
        let edge_to_bwd_off = off;
        off += 4 * m;
        let edge_w_bwd_off = off;
        off += 4 * m;
        let via_bwd_off = off;
        off += 4 * m;
        let up_count_bwd_off = off;
        off += 4 * n;
        let rank_off = off;

        Ok(Self {
            n,
            m,
            head_fwd_off,
            edge_to_fwd_off,
            edge_w_fwd_off,
            via_fwd_off,
            up_count_fwd_off,
            head_bwd_off,
            edge_to_bwd_off,
            edge_w_bwd_off,
            via_bwd_off,
            up_count_bwd_off,
            rank_off,
        })
    }
}

/// True iff the file at this mmap was written with rank-ordered vertex
/// numbering (SSSPCH1B/C). Callers can use this to pre-pin the top-K-ranked
/// vertex regions cheaply (they're at the start of each section).
pub fn is_rank_ordered(mmap: &Mmap) -> bool {
    mmap.len() >= 8 && (&mmap[..8] == b"SSSPCH1B" || &mmap[..8] == b"SSSPCH1C")
}
