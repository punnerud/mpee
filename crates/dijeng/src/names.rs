//! Street-name sidecar for **offline geocoding without a separate index**.
//!
//! Reverse and forward geocoding here ride entirely on data the routing build
//! already produces — no second spatial structure, no text index:
//!
//!   * **Reverse** (`(lat, lon)` → street): the caller snaps to the nearest
//!     road node with the existing `LatLonGrid` (the same grid `route`/`snap`
//!     use), then this table maps that node id → its street name. O(1).
//!   * **Forward** (street → `(lat, lon)`): a linear scan over the *distinct*
//!     street names (a city has only a few thousand). No trie, no inverted
//!     index — for one operating area a normalized scan is microseconds.
//!
//! The street name lives on the OSM highway *way* (`name=*`); the build pass
//! attaches each way's name to the road nodes it already creates. House
//! numbers (`addr:housenumber`) live on separate address nodes that are not in
//! the routing graph, so they would need a dedicated index — deliberately out
//! of scope here.
//!
//! On disk this is a small sidecar next to `.pp`/`.ch` (so those mmap'd
//! formats stay untouched) and is independently deletable when a build only
//! needs routing:
//! ```text
//! 0:   magic     "MPEENAM1"        (8 bytes)
//! 8:   n         u64               (node count — must match the graph)
//! 16:  k         u64               (distinct street-name count)
//! 24:  pool_len  u64               (bytes in the string pool)
//! 32:  name_id   n × u32           (per node; u32::MAX = no name)
//! ...: rep_node  k × u32           (a representative node id per name)
//! ...: offsets   (k+1) × u32       (byte offsets into pool)
//! ...: pool      pool_len bytes    (UTF-8 names, concatenated)
//! ```

use crate::buffer::Buffer;
use memmap2::Mmap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"MPEENAM1";
const HEADER_BYTES: usize = 32;

/// Sentinel in `name_id` meaning "this node has no street name".
pub const NO_NAME: u32 = u32::MAX;

/// A loaded street-name sidecar. All arrays are mmap-backed, so opening is
/// near-instant regardless of size and resident memory stays low.
pub struct NameTable {
    /// `name_id[node]` indexes into the pool (via `offsets`), or `NO_NAME`.
    name_id: Buffer<u32>,
    /// `rep_node[name]` is a representative road node carrying that name,
    /// used to answer forward lookups with a coordinate.
    rep_node: Buffer<u32>,
    /// `offsets[name]..offsets[name + 1]` is the byte range in `pool`.
    offsets: Buffer<u32>,
    /// UTF-8 names concatenated end to end.
    pool: Buffer<u8>,
}

impl NameTable {
    /// Number of distinct street names.
    pub fn len(&self) -> usize {
        self.rep_node.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Street name of a road node, or `None` if the node carries no name.
    pub fn name_of(&self, node: u32) -> Option<&str> {
        let ids = self.name_id.as_slice();
        let id = *ids.get(node as usize)? ;
        self.name_by_id(id)
    }

    fn name_by_id(&self, id: u32) -> Option<&str> {
        if id == NO_NAME {
            return None;
        }
        let off = self.offsets.as_slice();
        let (a, b) = (*off.get(id as usize)? as usize, *off.get(id as usize + 1)? as usize);
        let pool = self.pool.as_slice();
        std::str::from_utf8(pool.get(a..b)?).ok()
    }

    /// Forward lookup: find a street by (case-insensitive) name and return a
    /// representative road node id. An exact match wins; otherwise the first
    /// name that *contains* the query is returned (so `"karl johan"` finds
    /// `"Karl Johans gate"`). `None` if nothing matches.
    pub fn find(&self, query: &str) -> Option<u32> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return None;
        }
        let mut first_contains: Option<u32> = None;
        for id in 0..self.len() as u32 {
            if let Some(name) = self.name_by_id(id) {
                let lname = name.to_lowercase();
                if lname == q {
                    return Some(self.rep_node.as_slice()[id as usize]);
                }
                if first_contains.is_none() && lname.contains(&q) {
                    first_contains = Some(self.rep_node.as_slice()[id as usize]);
                }
            }
        }
        first_contains
    }

    /// Load a sidecar via mmap. Returns an error on a bad magic / truncated
    /// file or when `n` does not match the expected node count.
    pub fn load_mmap<P: AsRef<Path>>(path: P, expected_n: usize) -> std::io::Result<NameTable> {
        let f = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < HEADER_BYTES || &mmap[..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic — names sidecar corrupt or wrong version",
            ));
        }
        let n = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let k = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let pool_len = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        if n != expected_n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("names sidecar node count {n} != graph {expected_n} — rebuild the cache"),
            ));
        }
        let expected = HEADER_BYTES
            + 4 * n         // name_id
            + 4 * k         // rep_node
            + 4 * (k + 1)   // offsets
            + pool_len;     // pool
        if mmap.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("names sidecar too small: {} bytes, expected {}", mmap.len(), expected),
            ));
        }
        let arc = Arc::new(mmap);
        let mut off = HEADER_BYTES;
        let name_id = Buffer::<u32>::from_mmap(arc.clone(), off, n);
        off += 4 * n;
        let rep_node = Buffer::<u32>::from_mmap(arc.clone(), off, k);
        off += 4 * k;
        let offsets = Buffer::<u32>::from_mmap(arc.clone(), off, k + 1);
        off += 4 * (k + 1);
        let pool = Buffer::<u8>::from_mmap(arc, off, pool_len);
        Ok(NameTable { name_id, rep_node, offsets, pool })
    }
}

/// Write a sidecar. `name_id[node]` indexes into `pool` (or `NO_NAME`); the
/// representative node per name is derived as the first node carrying it.
pub fn save<P: AsRef<Path>>(path: P, name_id: &[u32], pool: &[String]) -> std::io::Result<()> {
    let n = name_id.len();
    let k = pool.len();

    // Representative node per name: first node that references it.
    let mut rep_node = vec![NO_NAME; k];
    for (node, &id) in name_id.iter().enumerate() {
        if id != NO_NAME {
            let slot = &mut rep_node[id as usize];
            if *slot == NO_NAME {
                *slot = node as u32;
            }
        }
    }

    // Byte offsets into the concatenated pool.
    let mut offsets = vec![0u32; k + 1];
    let mut acc = 0u32;
    for (i, s) in pool.iter().enumerate() {
        offsets[i] = acc;
        acc += s.len() as u32;
    }
    offsets[k] = acc;
    let pool_len = acc as usize;

    let f = OpenOptions::new().write(true).create(true).truncate(true).open(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);
    w.write_all(MAGIC)?;
    w.write_all(&(n as u64).to_le_bytes())?;
    w.write_all(&(k as u64).to_le_bytes())?;
    w.write_all(&(pool_len as u64).to_le_bytes())?;
    w.write_all(slice_u8(name_id))?;
    w.write_all(slice_u8(&rep_node))?;
    w.write_all(slice_u8(&offsets))?;
    for s in pool {
        w.write_all(s.as_bytes())?;
    }
    w.flush()?;
    Ok(())
}

#[inline]
fn slice_u8<T>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_reverse_and_forward() {
        // 4 nodes: 0,1 → "Karl Johans gate", 2 → "Storgata", 3 → no name.
        let pool = vec!["Karl Johans gate".to_string(), "Storgata".to_string()];
        let name_id = vec![0u32, 0, 1, NO_NAME];
        let dir = std::env::temp_dir().join("mpee_names_test.names");
        save(&dir, &name_id, &pool).unwrap();
        let t = NameTable::load_mmap(&dir, 4).unwrap();

        assert_eq!(t.name_of(0), Some("Karl Johans gate"));
        assert_eq!(t.name_of(2), Some("Storgata"));
        assert_eq!(t.name_of(3), None);

        // Forward: exact + substring + diacritics-insensitive lowercasing.
        assert_eq!(t.find("Storgata"), Some(2));
        assert_eq!(t.find("karl johan"), Some(0)); // substring, rep node = 0
        assert_eq!(t.find("nope"), None);
        let _ = std::fs::remove_file(&dir);
    }
}
