//! Reading OSM's replication diffs, so a refresh does not need a planet file.
//!
//! A planet PBF is 94.7 GB and takes two hours to fetch. The same four days of
//! edits are four `.osc.gz` files of about 95 MB each — 118x less data, and
//! sixty seconds instead of two hours. The format exists for exactly this, and
//! it hands over what `diff` otherwise has to derive: the ids that changed.
//!
//! So this is the cheap path and `diff` is the thorough one. `diff` re-hashes
//! every way in a new file and therefore catches *any* difference, including
//! one our own build got wrong. A change file only reports what the OSM editors
//! actually did. Both end in the same place — a set of segments, and from there
//! the regions to recompute.
//!
//! # What it can and cannot see
//!
//! A way that was created, modified or deleted is named outright, and that is
//! exact.
//!
//! A *node* that moved is harder, and the limit is in our own index: `way.topo`
//! stores a hash of a way's node list, not the list, so nothing here can ask
//! which ways referenced node N. What the change file does give is the node's
//! new position, and the road nearest that position is almost always the way
//! that moved — so a node change is placed geographically instead. That is
//! conservative in one direction and incomplete in the other: a node dragged a
//! long way invalidates its destination and not its origin. Storing node ids
//! per way would close it, at the cost of an index the size of the node list.

use crate::dataset::Dataset;
use crate::mmapvec;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

/// Which section of an OsmChange file an element sat in.
const CREATE: usize = 0;
const MODIFY: usize = 1;
const DELETE: usize = 2;

#[derive(Default, Debug)]
pub struct Touched {
    /// Way ids named by the file, sorted and deduplicated.
    pub ways: Vec<u64>,
    /// Of those, the ones whose *cost* changed — sorted, a subset of `ways`.
    ///
    /// A way can be edited without its cost moving: a corrected name, a new
    /// sidewalk tag, a `source` field. Only the attribute half decides what a
    /// table entry costs, and `way.hash` holds exactly that half — the node
    /// list lives in `way.topo` — so recomputing it from the change file's tags
    /// and comparing says whether a table could have moved at all.
    ///
    /// Empty when the comparison could not be made, which is not the same as
    /// "nothing changed": see `cost_known`.
    pub cost_changed: Vec<u64>,
    /// Whether `cost_changed` was actually computed. False when the live way
    /// index is missing, and then every caller must fall back to assuming any
    /// edit could have moved a cost.
    pub cost_known: bool,
    /// Ways the change file names that the index has never heard of. Mostly not
    /// roads at all — the index holds only routable ways, and most OSM edits are
    /// buildings and landuse — plus anything created since the dataset was built.
    /// Either way there are no segments here to invalidate.
    pub unknown: u64,
    /// Ways in the index whose attribute hash came out the same: edited, but not
    /// in a way that can move a cost.
    pub cost_same: u64,
    /// Ways in the index the change file no longer classifies as routable, so
    /// there is no hash to compare and they are counted as changed.
    pub cost_unroutable: u64,
    /// Positions of nodes it *modified*, for the ways it cannot name.
    ///
    /// Only the modified ones. A created node is reachable only through a way
    /// that references it, and that way is named in the same file; a node
    /// cannot be deleted while a way still references it, so a deletion implies
    /// its ways were modified too. Both are therefore already covered by
    /// `ways`, and snapping them would be 25 million lookups for nothing.
    pub nodes: Vec<(i32, i32)>,
    /// Ways seen per section: created, modified, deleted.
    pub way_counts: [u64; 3],
    /// Nodes seen per section.
    pub node_counts: [u64; 3],
    /// Nodes that carried no position, so could not be placed. A delete
    /// element often omits them, and then the edit is invisible to this path.
    pub unplaced: u64,
}

/// Parse a decimal degree string into 1e-7 degrees, without floating point.
///
/// `"59.9139"` is 599 139 000. Truncating rather than rounding keeps this
/// total: a coordinate is only ever used to find the nearest road.
fn deg_e7(s: &[u8]) -> Option<i32> {
    let (neg, s) = match s.first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let mut whole: i64 = 0;
    let mut frac: i64 = 0;
    let mut scale = 10_000_000i64;
    let mut seen_dot = false;
    let mut any = false;
    for &c in s {
        match c {
            b'.' if !seen_dot => seen_dot = true,
            b'0'..=b'9' => {
                any = true;
                if seen_dot {
                    if scale > 1 {
                        scale /= 10;
                        frac += (c - b'0') as i64 * scale;
                    }
                } else {
                    whole = whole.checked_mul(10)?.checked_add((c - b'0') as i64)?;
                }
            }
            _ => return None,
        }
    }
    if !any {
        return None;
    }
    let v = whole.checked_mul(10_000_000)?.checked_add(frac)?;
    let v = if neg { -v } else { v };
    i32::try_from(v).ok()
}

/// One attribute value out of a tag's attribute text.
///
/// Matches on `name="` rather than parsing attributes in order, because a
/// generator is free to order them how it likes. The values here are ids and
/// coordinates, which carry no entities to unescape.
fn attr<'a>(tag: &'a [u8], name: &str) -> Option<&'a [u8]> {
    let pat = name.as_bytes();
    let mut i = 0;
    while i + pat.len() + 2 <= tag.len() {
        // The name must start at a boundary, or `id` would match `uid`.
        let boundary = i == 0 || tag[i - 1] == b' ' || tag[i - 1] == b'\t';
        if boundary && &tag[i..i + pat.len()] == pat {
            let mut j = i + pat.len();
            while j < tag.len() && (tag[j] == b' ' || tag[j] == b'\t') {
                j += 1;
            }
            if j < tag.len() && tag[j] == b'=' {
                j += 1;
                while j < tag.len() && (tag[j] == b' ' || tag[j] == b'\t') {
                    j += 1;
                }
                if j < tag.len() && (tag[j] == b'"' || tag[j] == b'\'') {
                    let q = tag[j];
                    j += 1;
                    let start = j;
                    while j < tag.len() && tag[j] != q {
                        j += 1;
                    }
                    return Some(&tag[start..j]);
                }
            }
        }
        i += 1;
    }
    None
}

/// A way and the tags the change file gave it, as raw bytes.
///
/// Bytes rather than strings because the hash they feed is computed over the
/// bytes the PBF would have held, and a lossy conversion would change it.
type TaggedWay = (u64, Vec<(Vec<u8>, Vec<u8>)>, Vec<i64>);

/// XML entities, in the five forms a change file actually uses plus numeric.
///
/// The bytes have to come out exactly as the PBF would give them, or the hash
/// computed from them will not match the one the build stored and every way
/// will look like its cost moved. A street called `Rue de l&apos;Église` is
/// not a rare case.
fn unescape(v: &[u8]) -> Vec<u8> {
    if !v.contains(&b'&') {
        return v.to_vec();
    }
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0;
    while i < v.len() {
        if v[i] != b'&' {
            out.push(v[i]);
            i += 1;
            continue;
        }
        let end = match v[i..].iter().position(|&c| c == b';') {
            Some(e) if e <= 10 => i + e,
            // No terminator within reach: not an entity, keep the ampersand.
            _ => {
                out.push(v[i]);
                i += 1;
                continue;
            }
        };
        match &v[i + 1..end] {
            b"amp" => out.push(b'&'),
            b"lt" => out.push(b'<'),
            b"gt" => out.push(b'>'),
            b"quot" => out.push(b'"'),
            b"apos" => out.push(b'\''),
            e if e.first() == Some(&b'#') => {
                let txt = std::str::from_utf8(&e[1..]).unwrap_or("");
                let cp = if let Some(hex) = txt.strip_prefix('x').or(txt.strip_prefix('X')) {
                    u32::from_str_radix(hex, 16).ok()
                } else {
                    txt.parse::<u32>().ok()
                };
                match cp.and_then(char::from_u32) {
                    Some(c) => {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                    None => out.extend_from_slice(&v[i..=end]),
                }
            }
            // Something else entirely; keep it verbatim rather than guess.
            _ => out.extend_from_slice(&v[i..=end]),
        }
        i = end + 1;
    }
    out
}

/// The attribute hash of a way as the build would have computed it.
///
/// `way.hash` holds the attribute half — packed class, direction, speed, plus
/// name and ref — and `way.topo` holds the node list. So this says whether an
/// edit could have moved what a table entry costs, without needing the nodes,
/// which a change file gives only for the ways it rewrote whole.
/// The packed attribute and both hashes, as the build would have them.
pub fn way_state(
    id: u64,
    tags: &[(Vec<u8>, Vec<u8>)],
    refs: &[i64],
) -> Option<(u32, u64, u64)> {
    let pairs: Vec<(&[u8], &[u8])> =
        tags.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    let a = crate::profile::classify(pairs.iter().copied(), |_| crate::profile::NO_STR)?;
    let name = pairs.iter().find(|(k, _)| *k == b"name").map(|(_, v)| *v);
    let rf = pairs.iter().find(|(k, _)| *k == b"ref").map(|(_, v)| *v);
    let packed = crate::build::pack_attr(&a);
    let mut rec = Vec::with_capacity(256);
    crate::build::encode_way(&mut rec, id, packed, name.unwrap_or(b""), rf.unwrap_or(b""), refs);
    let (_, attr, topo) = crate::build::way_hashes(&rec);
    Some((packed, attr, topo))
}

fn cost_hash(id: u64, tags: &[(Vec<u8>, Vec<u8>)]) -> Option<u64> {
    let pairs: Vec<(&[u8], &[u8])> =
        tags.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    let a = crate::profile::classify(pairs.iter().copied(), |_| crate::profile::NO_STR)?;
    let name = pairs.iter().find(|(k, _)| *k == b"name").map(|(_, v)| *v);
    let rf = pairs.iter().find(|(k, _)| *k == b"ref").map(|(_, v)| *v);
    let mut rec = Vec::with_capacity(256);
    // The node list is not part of the attribute half, so an empty one gives
    // the same hash as the real one would.
    crate::build::encode_way(
        &mut rec,
        id,
        crate::build::pack_attr(&a),
        name.unwrap_or(b""),
        rf.unwrap_or(b""),
        &[],
    );
    Some(crate::build::way_hashes(&rec).1)
}

/// Read one or more `.osc` or `.osc.gz` change files.
///
/// Streamed, never held: a 95 MB change file is about 1.5 GB of XML, and the
/// machine this is built for does not have that to spare. The scanner carries
/// one tag at a time.
pub fn read(live: &Path, files: &[std::path::PathBuf]) -> io::Result<Touched> {
    let mut t = Touched::default();
    let mut tagged: Vec<TaggedWay> = Vec::new();
    for f in files {
        let file = std::fs::File::open(f)?;
        let gz = f.extension().is_some_and(|e| e == "gz");
        let rd: Box<dyn Read> = if gz {
            Box::new(flate2::read::GzDecoder::new(BufReader::with_capacity(1 << 20, file)))
        } else {
            Box::new(file)
        };
        scan(BufReader::with_capacity(1 << 20, rd), &mut t, &mut tagged)?;
    }
    t.ways.sort_unstable();
    t.ways.dedup();

    // Which of those edits could have moved a cost. Without the index this
    // cannot be answered, and saying so is the point: a caller that assumes
    // "none changed" because the list is empty would skip real work.
    if let Ok(m) = mmapvec::open(&live.join("way.hash")) {
        let whash: &[u64] = unsafe { mmapvec::as_slice(&m[..]) };
        let idm = mmapvec::open(&live.join("way.id"))?;
        let wid: &[u64] = unsafe { mmapvec::as_slice(&idm[..]) };
        // Last edit wins: a way touched on several of the nine days appears
        // once per file, and only its final state decides what it costs now.
        // Counting the duplicates as separate ways made `unknown` larger than
        // the number of ways there were.
        tagged.sort_by_key(|(id, _, _)| *id);
        tagged.dedup_by_key(|(id, _, _)| *id);
        for (id, tags, _) in &tagged {
            match wid.binary_search(id) {
                Ok(i) => match cost_hash(*id, tags) {
                    Some(h) if h == whash[i] => t.cost_same += 1,
                    // The change file no longer classifies it as routable —
                    // downgraded to a track, or its highway tag gone. No hash
                    // to compare, and it has certainly changed.
                    None => {
                        t.cost_unroutable += 1;
                        t.cost_changed.push(*id);
                    }
                    Some(_) => t.cost_changed.push(*id),
                },
                Err(_) => t.unknown += 1,
            }
        }
        t.cost_changed.sort_unstable();
        t.cost_changed.dedup();
        t.cost_known = true;
    }
    Ok(t)
}

fn scan<R: BufRead>(
    mut rd: R,
    t: &mut Touched,
    tagged: &mut Vec<TaggedWay>,
) -> io::Result<()> {
    let mut section = MODIFY; // a file without sections is read as a modify
    let mut tag: Vec<u8> = Vec::with_capacity(1 << 12);
    let mut cur_way: Option<u64> = None;
    let mut cur_tags: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut cur_refs: Vec<i64> = Vec::new();
    loop {
        // Everything outside a tag is whitespace or text we do not want.
        let mut skipped = Vec::new();
        let n = rd.read_until(b'<', &mut skipped)?;
        if n == 0 {
            return Ok(());
        }
        if skipped.last() != Some(&b'<') {
            return Ok(()); // end of input without another tag
        }
        tag.clear();
        let n = rd.read_until(b'>', &mut tag)?;
        if n == 0 {
            return Ok(());
        }
        // No closing delimiter means the input ended mid-element, and a
        // truncated element is worse than a missing one: `id="123` cut from
        // `id="1234` parses cleanly and names a different way. Drop it.
        if tag.last() != Some(&b'>') {
            return Ok(());
        }
        tag.pop();
        if tag.first() == Some(&b'?') || tag.first() == Some(&b'!') {
            continue;
        }
        if tag.first() == Some(&b'/') {
            if tag[1..].starts_with(b"way") {
                if let Some(id) = cur_way.take() {
                    tagged.push((
                        id,
                        std::mem::take(&mut cur_tags),
                        std::mem::take(&mut cur_refs),
                    ));
                }
            }
            continue;
        }
        let self_closing = tag.last() == Some(&b'/');
        if self_closing {
            tag.pop();
        }
        // The element name, then its attribute text.
        let end = tag.iter().position(|c| c.is_ascii_whitespace()).unwrap_or(tag.len());
        let (name, rest) = tag.split_at(end);
        match name {
            b"create" => section = CREATE,
            b"modify" => section = MODIFY,
            b"delete" => section = DELETE,
            b"way" => {
                t.way_counts[section] += 1;
                cur_way = attr(rest, "id")
                    .and_then(|v| std::str::from_utf8(v).ok())
                    .and_then(|v| v.trim().parse::<u64>().ok());
                if let Some(id) = cur_way {
                    t.ways.push(id);
                }
                cur_tags.clear();
                cur_refs.clear();
                // `<way .../>` closes itself, so there are no children to wait for.
                if self_closing {
                    if let Some(id) = cur_way.take() {
                        tagged.push((
                            id,
                            std::mem::take(&mut cur_tags),
                            std::mem::take(&mut cur_refs),
                        ));
                    }
                }
            }
            b"nd" => {
                if cur_way.is_some() {
                    if let Some(r) = attr(rest, "ref")
                        .and_then(|v| std::str::from_utf8(v).ok())
                        .and_then(|v| v.trim().parse::<i64>().ok())
                    {
                        cur_refs.push(r);
                    }
                }
            }
            b"tag" => {
                // Only the tags inside a way; a node's tags decide nothing here.
                if cur_way.is_some() {
                    if let (Some(k), Some(v)) = (attr(rest, "k"), attr(rest, "v")) {
                        cur_tags.push((unescape(k), unescape(v)));
                    }
                }
            }
            b"node" => {
                t.node_counts[section] += 1;
                if section == MODIFY {
                    match (attr(rest, "lat").and_then(deg_e7), attr(rest, "lon").and_then(deg_e7))
                    {
                        (Some(la), Some(lo)) => t.nodes.push((la, lo)),
                        _ => t.unplaced += 1,
                    }
                }
            }
            _ => {}
        }
    }
}

/// The segments a set of way ids owns, from the live way index.
///
/// A merge join, not a binary search each. Both sides are sorted — the change
/// file's ids are sorted on the way in, and the index is sorted by
/// construction — so this is one sequential pass. The binary-search version
/// worked and cost 3.1 GB resident: five million searches over a 1.34 GB array
/// is 140 million random probes, and the kernel ends up holding the whole
/// index. Reading it in order instead lets the pages go as soon as they are
/// past, which is the difference between a refresh that fits on a small machine
/// and one that does not.
/// `budget_bytes` caps the resident set. The index is 1.34 GB of way ids alone
/// and the join reads all of it, so without a cap the kernel keeps the lot —
/// harmless on this machine, fatal on the one this is built for. The pages are
/// read once and in order, so handing them back as the scan passes costs
/// nothing.
pub fn segments_of(
    live: &Path,
    ways: &[u64],
    mut cap: Option<&mut crate::cachecap::CacheCap>,
) -> io::Result<Vec<u32>> {
    let wid_map = mmapvec::open(&live.join("way.id"))?;
    let whead_map = mmapvec::open(&live.join("way.head"))?;
    let wseg_map = mmapvec::open(&live.join("way.seg"))?;
    let wid: &[u64] = unsafe { mmapvec::as_slice(&wid_map[..]) };
    let whead: &[u32] = unsafe { mmapvec::as_slice(&whead_map[..]) };
    let wseg: &[u32] = unsafe { mmapvec::as_slice(&wseg_map[..]) };
    // The caller's cap already holds the dataset and the overlay; these three
    // join it, so one budget covers everything the refresh touches.
    if let Some(c) = cap.as_deref_mut() {
        c.govern(std::slice::from_ref(&wid_map));
        c.govern(std::slice::from_ref(&whead_map));
        c.govern(std::slice::from_ref(&wseg_map));
    }
    debug_assert!(ways.windows(2).all(|w| w[0] <= w[1]), "a merge join needs sorted input");
    let mut out = Vec::new();
    let mut i = 0usize;
    for (n, &w) in ways.iter().enumerate() {
        // Often enough to hold the line, seldom enough that the check itself
        // does not show up next to a sequential read.
        if n % 65_536 == 0 {
            if let Some(c) = cap.as_deref_mut() {
                c.enforce();
            }
        }
        // Skip the index forward to this id. Never backwards, which is what
        // makes the whole thing one pass however many ids are asked for.
        while i < wid.len() && wid[i] < w {
            i += 1;
        }
        if i < wid.len() && wid[i] == w {
            let (a, b) = (whead[i] as usize, whead[i + 1] as usize);
            out.extend_from_slice(&wseg[a..b]);
            i += 1;
        }
        // No match means the way is new in the change file and has no segments
        // here yet — counted as added, nothing to invalidate.
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// The segment nearest each changed node position, within `radius_m`.
///
/// The stand-in for a node-to-way index we do not keep. See the module note:
/// this finds where the node now is, not where it was.
/// Costs one snap per node, and a planet's nine days hold 4.8 million modified
/// nodes — so this is opt-in rather than part of every refresh. The `limit`
/// stops it after that many lookups and says how far it got, because a refresh
/// that runs for half an hour without saying why is worse than one that admits
/// it sampled.
pub fn segments_near(
    ds: &Dataset,
    nodes: &[(i32, i32)],
    radius_m: f64,
    limit: usize,
) -> (Vec<u32>, usize) {
    let mut out = Vec::new();
    let n = nodes.len().min(limit);
    for &(la, lo) in &nodes[..n] {
        if let Some(s) = ds.snap(la, lo, radius_m) {
            out.push(s.seg);
        }
    }
    out.sort_unstable();
    out.dedup();
    (out, n)
}
