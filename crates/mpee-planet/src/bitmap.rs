//! A planet-sized bitmap over raw OSM node ids, with rank support.
//!
//! The whole out-of-core design hangs off this structure. OSM node ids are a
//! sparse 64-bit space (max ≈ 1.3e10 today) but a car network only references
//! ~15 % of them, so we cannot key a hash map by id (≈ 450 GB) and we cannot
//! afford to sort 2.5 billion (id, coord) pairs either.
//!
//! Instead: one bit per possible id (2 GB for 1.6e10 ids), then a rank index.
//! `rank1(id)` is the number of set bits below `id`, which is exactly the
//! node's position in a *dense* array holding only the nodes we kept. Because
//! a sorted PBF emits nodes in ascending id order, pass 2 fills that dense
//! array with pure sequential writes, and every later lookup is O(1).
//!
//! Bits are set from many blob threads at once, so writes go through
//! `AtomicU64::fetch_or`. `fetch_or` also *reports* the previous value, which
//! is what gives us junction detection for free: the second way to touch a
//! node is the one that sees the bit already set.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Words per rank block. 512 bits keeps the index at 1/64th of the bitmap.
const BLOCK_WORDS: usize = 8;
const BLOCK_BITS: u64 = (BLOCK_WORDS * 64) as u64;

pub struct BitMap {
    words: Vec<AtomicU64>,
    n_bits: u64,
    /// Cumulative popcount at the start of each 512-bit block. Built by
    /// [`BitMap::build_rank`]; empty until then.
    rank: Vec<u64>,
}

impl BitMap {
    pub fn new(n_bits: u64) -> Self {
        let nw = n_bits.div_ceil(64) as usize;
        let mut words = Vec::with_capacity(nw);
        words.resize_with(nw, || AtomicU64::new(0));
        BitMap { words, n_bits, rank: Vec::new() }
    }

    pub fn bits(&self) -> u64 {
        self.n_bits
    }

    /// Set bit `i`, returning true if it was **already** set. Thread-safe.
    #[inline(always)]
    pub fn set(&self, i: u64) -> bool {
        debug_assert!(i < self.n_bits);
        let w = (i >> 6) as usize;
        let m = 1u64 << (i & 63);
        (self.words[w].fetch_or(m, Ordering::Relaxed) & m) != 0
    }

    /// Set without reporting the previous value (one instruction cheaper on
    /// the paths that do not care).
    #[inline(always)]
    pub fn set_quiet(&self, i: u64) {
        let w = (i >> 6) as usize;
        self.words[w].fetch_or(1u64 << (i & 63), Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn get(&self, i: u64) -> bool {
        if i >= self.n_bits {
            return false;
        }
        let w = (i >> 6) as usize;
        (self.words[w].load(Ordering::Relaxed) & (1u64 << (i & 63))) != 0
    }

    pub fn count_ones(&self) -> u64 {
        self.words.iter().map(|w| w.load(Ordering::Relaxed).count_ones() as u64).sum()
    }

    /// Build the rank index. Must be called after all bits are set and before
    /// any [`BitMap::rank1`] call.
    pub fn build_rank(&mut self) {
        let nblocks = self.words.len().div_ceil(BLOCK_WORDS);
        let mut r = Vec::with_capacity(nblocks + 1);
        let mut acc = 0u64;
        for b in 0..nblocks {
            r.push(acc);
            let s = b * BLOCK_WORDS;
            let e = (s + BLOCK_WORDS).min(self.words.len());
            for w in &self.words[s..e] {
                acc += w.load(Ordering::Relaxed).count_ones() as u64;
            }
        }
        r.push(acc);
        self.rank = r;
    }

    /// Number of set bits strictly below `i` — i.e. the dense index of `i`
    /// when `i` is set. O(1): one block lookup plus at most 8 popcounts.
    #[inline]
    pub fn rank1(&self, i: u64) -> u64 {
        debug_assert!(!self.rank.is_empty(), "build_rank() was not called");
        let block = (i / BLOCK_BITS) as usize;
        let mut acc = self.rank[block];
        let ws = block * BLOCK_WORDS;
        let wi = (i >> 6) as usize;
        for w in &self.words[ws..wi] {
            acc += w.load(Ordering::Relaxed).count_ones() as u64;
        }
        let rem = i & 63;
        if rem != 0 {
            let m = (1u64 << rem) - 1;
            acc += (self.words[wi].load(Ordering::Relaxed) & m).count_ones() as u64;
        }
        acc
    }

    /// Total set bits (available after [`BitMap::build_rank`]).
    pub fn total(&self) -> u64 {
        *self.rank.last().unwrap_or(&0)
    }

    /// Visit every set bit in ascending order, with its dense rank. The rank
    /// is carried along rather than recomputed, so a full sweep is O(words).
    pub fn for_each_set(&self, mut f: impl FnMut(u64, u64)) {
        let mut rank = 0u64;
        for (wi, w) in self.words.iter().enumerate() {
            let mut x = w.load(Ordering::Relaxed);
            let base = (wi as u64) << 6;
            while x != 0 {
                let b = x.trailing_zeros() as u64;
                f(base + b, rank);
                rank += 1;
                x &= x - 1;
            }
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut f = std::io::BufWriter::with_capacity(1 << 22, File::create(path)?);
        f.write_all(b"MPBM0001")?;
        f.write_all(&self.n_bits.to_le_bytes())?;
        for w in &self.words {
            f.write_all(&w.load(Ordering::Relaxed).to_le_bytes())?;
        }
        f.flush()
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let mut f = std::io::BufReader::with_capacity(1 << 22, File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"MPBM0001" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad bitmap magic"));
        }
        let mut b8 = [0u8; 8];
        f.read_exact(&mut b8)?;
        let n_bits = u64::from_le_bytes(b8);
        let nw = n_bits.div_ceil(64) as usize;
        let mut words = Vec::with_capacity(nw);
        let mut chunk = vec![0u8; 1 << 20];
        let mut left = nw;
        while left > 0 {
            let take = left.min(chunk.len() / 8);
            f.read_exact(&mut chunk[..take * 8])?;
            for k in 0..take {
                words.push(AtomicU64::new(u64::from_le_bytes(
                    chunk[k * 8..k * 8 + 8].try_into().unwrap(),
                )));
            }
            left -= take;
        }
        Ok(BitMap { words, n_bits, rank: Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_matches_linear_count() {
        let mut bm = BitMap::new(10_000);
        let ids: Vec<u64> = (0..10_000).filter(|i| i % 7 == 3 || i % 13 == 0).collect();
        for &i in &ids {
            bm.set(i);
        }
        bm.build_rank();
        assert_eq!(bm.total(), ids.len() as u64);
        for (dense, &i) in ids.iter().enumerate() {
            assert_eq!(bm.rank1(i), dense as u64, "rank1({i})");
        }
    }

    #[test]
    fn set_reports_previous_value() {
        let bm = BitMap::new(128);
        assert!(!bm.set(42), "first set reports not-previously-set");
        assert!(bm.set(42), "second set reports previously-set");
        assert!(bm.get(42));
        assert!(!bm.get(43));
    }

    #[test]
    fn for_each_set_agrees_with_rank() {
        let mut bm = BitMap::new(3000);
        for i in (0..3000u64).filter(|i| i % 11 == 5) {
            bm.set(i);
        }
        bm.build_rank();
        let mut seen = Vec::new();
        bm.for_each_set(|id, rank| {
            assert_eq!(rank, bm.rank1(id));
            seen.push(id);
        });
        assert_eq!(seen.len(), bm.total() as usize);
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "ascending");
    }

    #[test]
    fn rank_across_block_boundaries() {
        // 512 bits per rank block — check ids either side of several blocks.
        let mut bm = BitMap::new(5000);
        for i in (0..5000).step_by(3) {
            bm.set(i);
        }
        bm.build_rank();
        let mut expect = 0u64;
        for i in 0..5000u64 {
            assert_eq!(bm.rank1(i), expect, "rank1({i})");
            if i % 3 == 0 {
                expect += 1;
            }
        }
    }
}
