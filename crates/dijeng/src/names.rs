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
//! Intersection search ("street A × street B") rides on the same idea: the
//! build records, per street, the road nodes it touches (a CSR). The crossing
//! is then the **set intersection** of the two streets' node lists — no
//! polyline geometry needed; an at-grade junction is a node both streets share.
//! Two streets can share several nodes (cross twice, or run together), so the
//! query returns every shared node for the caller to choose from.
//!
//! On disk this is a small sidecar next to `.pp`/`.ch` (so those mmap'd
//! formats stay untouched) and is independently deletable when a build only
//! needs routing:
//! ```text
//! 0:    magic       "MPEENAM2"      (8 bytes)
//! 8:    n           u64             (node count — must match the graph)
//! 16:   k           u64             (distinct street-name count)
//! 24:   pool_len    u64             (bytes in the string pool)
//! 32:   sn_total    u64             (total street→node memberships)
//! 40:   name_id     n × u32         (per node; u32::MAX = no name)
//! ...:  rep_node    k × u32         (a representative node id per name)
//! ...:  offsets     (k+1) × u32     (byte offsets into pool)
//! ...:  pool        pool_len bytes  (UTF-8 names, concatenated)
//! ...:  sn_offsets  (k+1) × u32     (CSR offsets into sn_nodes, per street)
//! ...:  sn_nodes    sn_total × u32  (sorted road-node ids each street touches)
//! ```

use crate::buffer::Buffer;
#[cfg(feature = "native")]
use memmap2::Mmap;
#[cfg(feature = "native")]
use std::fs::OpenOptions;
#[cfg(feature = "native")]
use std::io::{BufWriter, Write};
#[cfg(feature = "native")]
use std::path::Path;
#[cfg(feature = "native")]
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"MPEENAM2";
const HEADER_BYTES: usize = 40;

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
    /// CSR offsets into `street_nodes`, per street name.
    sn_offsets: Buffer<u32>,
    /// Sorted road-node ids each street touches (for intersection search).
    sn_nodes: Buffer<u32>,
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

    /// The canonical street name for a name id.
    pub fn name_by_id(&self, id: u32) -> Option<&str> {
        if id == NO_NAME {
            return None;
        }
        let off = self.offsets.as_slice();
        let (a, b) = (*off.get(id as usize)? as usize, *off.get(id as usize + 1)? as usize);
        let pool = self.pool.as_slice();
        std::str::from_utf8(pool.get(a..b)?).ok()
    }

    /// Resolve a street name to its name id (case-insensitive). An exact match
    /// wins; otherwise the first name that *contains* the query (so
    /// `"karl johan"` finds `"Karl Johans gate"`). `None` if nothing matches.
    pub fn find_id(&self, query: &str) -> Option<u32> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return None;
        }
        let mut first_contains: Option<u32> = None;
        for id in 0..self.len() as u32 {
            if let Some(name) = self.name_by_id(id) {
                let lname = name.to_lowercase();
                if lname == q {
                    return Some(id);
                }
                if first_contains.is_none() && lname.contains(&q) {
                    first_contains = Some(id);
                }
            }
        }
        first_contains
    }

    /// Forward lookup: find a street by name and return a representative road
    /// node id (see [`find_id`](Self::find_id) for matching rules).
    pub fn find(&self, query: &str) -> Option<u32> {
        self.find_id(query).map(|id| self.rep_node.as_slice()[id as usize])
    }

    /// Up to `limit` distinct street names matching `query` (case-insensitive),
    /// prefix matches first then substring matches — for type-ahead suggestions.
    pub fn suggest(&self, query: &str, limit: usize) -> Vec<String> {
        let q = query.trim().to_lowercase();
        if q.is_empty() || limit == 0 {
            return Vec::new();
        }
        let (mut prefix, mut contains): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
        for id in 0..self.len() as u32 {
            if prefix.len() >= limit {
                break;
            }
            if let Some(name) = self.name_by_id(id) {
                let l = name.to_lowercase();
                if l.starts_with(&q) {
                    prefix.push(name.to_string());
                } else if contains.len() < limit && l.contains(&q) {
                    contains.push(name.to_string());
                }
            }
        }
        let mut out = prefix;
        for c in contains {
            if out.len() >= limit {
                break;
            }
            out.push(c);
        }
        out
    }

    /// The road nodes a street touches (sorted), by name id.
    pub fn street_nodes(&self, id: u32) -> &[u32] {
        let off = self.sn_offsets.as_slice();
        match (off.get(id as usize), off.get(id as usize + 1)) {
            (Some(&a), Some(&b)) => &self.sn_nodes.as_slice()[a as usize..b as usize],
            _ => &[],
        }
    }

    /// Road nodes where two streets meet — the set intersection of their node
    /// lists (both sorted, so a linear merge). Empty if either name doesn't
    /// resolve or they share no node (e.g. a grade-separated crossing).
    pub fn intersections(&self, query_a: &str, query_b: &str) -> Vec<u32> {
        let (ia, ib) = match (self.find_id(query_a), self.find_id(query_b)) {
            (Some(a), Some(b)) if a != b => (a, b),
            _ => return Vec::new(),
        };
        let (na, nb) = (self.street_nodes(ia), self.street_nodes(ib));
        let (mut i, mut j) = (0usize, 0usize);
        let mut out = Vec::new();
        while i < na.len() && j < nb.len() {
            match na[i].cmp(&nb[j]) {
                std::cmp::Ordering::Equal => {
                    out.push(na[i]);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        out
    }

    /// Load a sidecar from an in-memory byte slice (wasm / no-mmap path).
    /// Copies arrays into owned buffers; same layout as [`load_mmap`].
    pub fn load_bytes(bytes: &[u8], expected_n: usize) -> std::io::Result<NameTable> {
        if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic — names sidecar corrupt or wrong version",
            ));
        }
        let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let k = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let pool_len = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        let sn_total = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
        if n != expected_n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("names sidecar node count {n} != graph {expected_n} — rebuild the cache"),
            ));
        }
        let expected = HEADER_BYTES
            + 4 * n + 4 * k + 4 * (k + 1) + pool_len + 4 * (k + 1) + 4 * sn_total;
        if bytes.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("names sidecar too small: {} bytes, expected {}", bytes.len(), expected),
            ));
        }
        let mut off = HEADER_BYTES;
        let name_id = Buffer::<u32>::from_bytes_copy(bytes, off, n);
        off += 4 * n;
        let rep_node = Buffer::<u32>::from_bytes_copy(bytes, off, k);
        off += 4 * k;
        let offsets = Buffer::<u32>::from_bytes_copy(bytes, off, k + 1);
        off += 4 * (k + 1);
        let pool = Buffer::<u8>::from_bytes_copy(bytes, off, pool_len);
        off += pool_len;
        let sn_offsets = Buffer::<u32>::from_bytes_copy(bytes, off, k + 1);
        off += 4 * (k + 1);
        let sn_nodes = Buffer::<u32>::from_bytes_copy(bytes, off, sn_total);
        Ok(NameTable { name_id, rep_node, offsets, pool, sn_offsets, sn_nodes })
    }

    /// Load a sidecar via mmap. Returns an error on a bad magic / truncated
    /// file or when `n` does not match the expected node count.
    #[cfg(feature = "native")]
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
        let sn_total = u64::from_le_bytes(mmap[32..40].try_into().unwrap()) as usize;
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
            + pool_len      // pool
            + 4 * (k + 1)   // sn_offsets
            + 4 * sn_total; // sn_nodes
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
        let pool = Buffer::<u8>::from_mmap(arc.clone(), off, pool_len);
        off += pool_len;
        let sn_offsets = Buffer::<u32>::from_mmap(arc.clone(), off, k + 1);
        off += 4 * (k + 1);
        let sn_nodes = Buffer::<u32>::from_mmap(arc, off, sn_total);
        Ok(NameTable { name_id, rep_node, offsets, pool, sn_offsets, sn_nodes })
    }
}

/// Write a sidecar. `name_id[node]` indexes into `pool` (or `NO_NAME`); the
/// representative node per name is derived as the first node carrying it.
/// `street_offsets` (len `k+1`) + `street_nodes` are the per-street node CSR
/// for intersection search (`street_offsets[id]..street_offsets[id+1]` slices
/// `street_nodes`, sorted).
#[cfg(feature = "native")]
pub fn save<P: AsRef<Path>>(
    path: P,
    name_id: &[u32],
    pool: &[String],
    street_offsets: &[u32],
    street_nodes: &[u32],
) -> std::io::Result<()> {
    let n = name_id.len();
    let k = pool.len();
    assert_eq!(street_offsets.len(), k + 1, "street_offsets must have k+1 entries");

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
    w.write_all(&(street_nodes.len() as u64).to_le_bytes())?;
    w.write_all(slice_u8(name_id))?;
    w.write_all(slice_u8(&rep_node))?;
    w.write_all(slice_u8(&offsets))?;
    for s in pool {
        w.write_all(s.as_bytes())?;
    }
    w.write_all(slice_u8(street_offsets))?;
    w.write_all(slice_u8(street_nodes))?;
    w.flush()?;
    Ok(())
}

#[cfg(feature = "native")]
#[inline]
fn slice_u8<T>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_reverse_forward_and_intersection() {
        // 5 nodes. Streets: 0="Karl Johans gate" (nodes 0,1,2),
        // 1="Kongens gate" (nodes 2,3). They share node 2 → the crossing.
        // 2="Storgata" (node 4). node_name keeps the first street per node.
        let pool = vec![
            "Karl Johans gate".to_string(),
            "Kongens gate".to_string(),
            "Storgata".to_string(),
        ];
        let name_id = vec![0u32, 0, 0, 1, 2];
        // Per-street node CSR (sorted): street 0 → [0,1,2], 1 → [2,3], 2 → [4].
        let street_offsets = vec![0u32, 3, 5, 6];
        let street_nodes = vec![0u32, 1, 2, 2, 3, 4];

        let dir = std::env::temp_dir().join("mpee_names_test.names");
        save(&dir, &name_id, &pool, &street_offsets, &street_nodes).unwrap();
        let t = NameTable::load_mmap(&dir, 5).unwrap();

        // Reverse.
        assert_eq!(t.name_of(0), Some("Karl Johans gate"));
        assert_eq!(t.name_of(4), Some("Storgata"));

        // Forward: exact + substring + diacritics-insensitive lowercasing.
        assert_eq!(t.find("Storgata"), Some(4));
        assert_eq!(t.find("karl johan"), Some(0)); // substring, rep node = 0
        assert_eq!(t.find("nope"), None);

        // Intersection: the two streets meet at node 2 (order-independent).
        assert_eq!(t.intersections("Karl Johans gate", "Kongens gate"), vec![2]);
        assert_eq!(t.intersections("kongens gate", "karl johans gate"), vec![2]);
        // No shared node → empty.
        assert!(t.intersections("Karl Johans gate", "Storgata").is_empty());
        let _ = std::fs::remove_file(&dir);
    }
}
