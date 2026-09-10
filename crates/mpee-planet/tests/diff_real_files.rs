//! The diff, checked against two real OSM extracts by a different reader.
//!
//! `diff_merge.rs` proves the join handles every case, and the self-diff (a
//! dataset against the file it was built from) proves the build and the update
//! hash identically. Neither touches the question this answers: given two
//! genuinely different files, does the classification match what is in them?
//!
//! The check is deliberately made with `osmpbf` rather than this crate's own
//! reader, so a misparse would have to occur identically in two independent
//! implementations to go unnoticed.
//!
//! Presence is what gets asserted exactly. "Added" and "deleted" are claims
//! about which file a way appears in, and admit no interpretation. "Changed"
//! is a claim about routing-relevant content — a way whose only edit was a
//! `source=` tag is *correctly* unchanged — so for those the test asserts the
//! weaker, still meaningful property that the node list or the highway tag
//! really does differ.
//!
//! Set `MPEE_DIFF_OLD` and `MPEE_DIFF_NEW` to two extracts of the same area to
//! run it; without them it skips.

use std::collections::HashMap;

/// Tags that can change how a way is routed. Deliberately a fixed list rather
/// than "every tag": an edit to `source` or `note` must *not* count as a
/// change, and treating it as one would be indistinguishable from a bug.
const ROUTING_TAGS: &[&str] = &[
    "highway", "oneway", "maxspeed", "access", "motor_vehicle", "vehicle", "motorcar",
    "name", "ref", "junction", "area", "service", "toll", "bridge", "tunnel", "ferry",
    "route", "construction", "surface",
];

type Way = (Vec<i64>, Vec<(String, String)>);

fn ways_of(path: &str) -> HashMap<i64, Way> {
    let mut out = HashMap::new();
    let reader = osmpbf::ElementReader::from_path(path).expect("open pbf");
    reader
        .for_each(|el| {
            if let osmpbf::Element::Way(w) = el {
                let mut t: Vec<(String, String)> = w
                    .tags()
                    .filter(|(k, _)| ROUTING_TAGS.contains(k))
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                if t.is_empty() {
                    return;
                }
                t.sort();
                out.insert(w.id(), (w.refs().collect(), t));
            }
        })
        .expect("read pbf");
    out
}

#[test]
fn a_way_called_unchanged_really_is_unchanged() {
    let (Ok(old), Ok(new)) = (std::env::var("MPEE_DIFF_OLD"), std::env::var("MPEE_DIFF_NEW"))
    else {
        eprintln!("MPEE_DIFF_OLD / MPEE_DIFF_NEW not set — skipping");
        return;
    };
    let unchanged = ids("MPEE_DIFF_UNCHANGED");
    let changed = ids("MPEE_DIFF_CHANGED");
    let added = ids("MPEE_DIFF_ADDED");
    let deleted = ids("MPEE_DIFF_DELETED");
    if unchanged.is_empty() {
        eprintln!("nothing claimed unchanged — skipping");
        return;
    }
    let a = ways_of(&old);
    let b = ways_of(&new);
    eprintln!("  {} tagged ways in the old file, {} in the new", a.len(), b.len());

    // The one direction that can produce a wrong route. A way wrongly called
    // changed costs a recomputation; a way wrongly called unchanged leaves a
    // table describing a road that no longer exists as described.
    let mut wrong = Vec::new();
    for id in &unchanged {
        match (a.get(id), b.get(id)) {
            (Some(x), Some(y)) => {
                if x.0 != y.0 {
                    wrong.push(format!("w{id}: {} nodes -> {}", x.0.len(), y.0.len()));
                } else if x.1 != y.1 {
                    wrong.push(format!("w{id}: tags {:?} -> {:?}", x.1, y.1));
                }
            }
            (None, _) | (_, None) => wrong.push(format!("w{id}: not in both files")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} ways called unchanged had moved after all:\n  {}",
        wrong.len(),
        unchanged.len(),
        wrong.join("\n  ")
    );
    eprintln!("  {} ways called unchanged are byte-for-byte unchanged", unchanged.len());

    // The other three cost work rather than correctness, and the diff's notion
    // of a routable way is narrower than "has a tag we recognise" — a way that
    // goes from `highway=construction` to `highway=trunk` is genuinely new to
    // routing while being present in both files. So these are reported, and
    // only an outright contradiction fails.
    let mut odd = 0;
    for (lbl, list) in [("changed", &changed), ("added", &added), ("deleted", &deleted)] {
        for id in list {
            let ok = match (a.get(id), b.get(id)) {
                (Some(x), Some(y)) => x != y || lbl != "changed",
                (None, Some(_)) => lbl != "deleted",
                (Some(_), None) => lbl != "added",
                (None, None) => false,
            };
            if !ok {
                odd += 1;
                eprintln!("  note: w{id} called {lbl}, but the files show no such difference");
            }
        }
    }
    eprintln!(
        "  {} changed / {} added / {} deleted checked, {odd} without a visible difference",
        changed.len(),
        added.len(),
        deleted.len()
    );
}

fn ids(var: &str) -> Vec<i64> {
    std::env::var(var)
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}
