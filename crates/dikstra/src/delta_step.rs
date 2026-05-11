//! Δ-stepping (Meyer & Sanders, 2003).
//!
//! Avoids a global priority queue. Vertices live in *buckets* of width Δ:
//! bucket B[i] contains vertices with tentative distance in [iΔ, (i+1)Δ).
//! Process the lowest non-empty bucket repeatedly:
//!   1. Relax all *light* edges (w ≤ Δ) from its vertices, possibly adding
//!      new vertices into the current or later buckets.
//!   2. Once that bucket is stable, relax all *heavy* edges (w > Δ) once.
//!
//! Why this matters here: it shares the *core idea* of Duan et al. — drop the
//! requirement of pulling vertices in strict sorted order. Instead, settle
//! vertices in *batches* whose distances are within Δ of each other. With a
//! good Δ, expected work is close to Dijkstra but with much cheaper queue
//! operations on sparse graphs with bounded weights.
//!
//! Δ choice: Δ ≈ 1 / max_degree gives a reasonable starting point for weights
//! in (0, 1]. We let the caller pick.

use crate::dijkstra::INF;
use crate::graph::CsrGraph;

/// Debug-versjon — sporer pushes og avslører hvor en bestemt vertex
/// "forsvinner" fra bucket-systemet.
pub fn delta_stepping_debug(g: &CsrGraph, src: u32, delta: f32, watch: u32) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;

    let bucket_of = |d: f32| -> usize {
        if !d.is_finite() {
            usize::MAX
        } else {
            (d / delta) as usize
        }
    };

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new()];
    buckets[0].push(src);

    let mut i = 0usize;
    let mut settled: Vec<u32> = Vec::new();
    let mut max_pushed_bucket = 0usize;
    let mut watch_pushed_at: Vec<usize> = Vec::new();
    let mut watch_settled_at: Vec<usize> = Vec::new();

    loop {
        while i < buckets.len() && buckets[i].is_empty() {
            i += 1;
        }
        if i >= buckets.len() {
            break;
        }

        settled.clear();
        loop {
            let current = std::mem::take(&mut buckets[i]);
            if current.is_empty() {
                break;
            }
            for &u in &current {
                let du = dist[u as usize];
                if bucket_of(du) != i {
                    continue;
                }
                if u == watch {
                    watch_settled_at.push(i);
                }
                settled.push(u);
                let s = g.head[u as usize] as usize;
                let e = g.head[u as usize + 1] as usize;
                for k in s..e {
                    let w = g.edge_w[k];
                    if w > delta {
                        continue;
                    }
                    let v = g.edge_to[k];
                    let nd = du + w;
                    if nd < dist[v as usize] {
                        dist[v as usize] = nd;
                        let nb = bucket_of(nd);
                        if nb > max_pushed_bucket {
                            max_pushed_bucket = nb;
                        }
                        if nb >= buckets.len() {
                            buckets.resize_with(nb + 1, Vec::new);
                        }
                        buckets[nb].push(v);
                        if v == watch {
                            watch_pushed_at.push(nb);
                            eprintln!(
                                "[watch] push v={watch} via light edge from u={u} du={du} w={w} nd={nd} nb={nb} (current i={i})"
                            );
                        }
                    }
                }
            }
        }

        for &u in &settled {
            let du = dist[u as usize];
            if bucket_of(du) != i {
                continue;
            }
            let s = g.head[u as usize] as usize;
            let e = g.head[u as usize + 1] as usize;
            for k in s..e {
                let w = g.edge_w[k];
                if w <= delta {
                    continue;
                }
                let v = g.edge_to[k];
                let nd = du + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    let nb = bucket_of(nd);
                    if nb > max_pushed_bucket {
                        max_pushed_bucket = nb;
                    }
                    if nb >= buckets.len() {
                        buckets.resize_with(nb + 1, Vec::new);
                    }
                    buckets[nb].push(v);
                    if v == watch {
                        watch_pushed_at.push(nb);
                        eprintln!(
                            "[watch] push v={watch} via HEAVY edge from u={u} du={du} w={w} nd={nd} nb={nb} (current i={i})"
                        );
                    }
                }
            }
        }

        i += 1;
    }

    eprintln!(
        "[watch] watch={watch} pushed at buckets {:?}, settled at {:?}, dist={}, max_bucket_pushed={}, buckets.len()={}",
        watch_pushed_at, watch_settled_at, dist[watch as usize], max_pushed_bucket, buckets.len()
    );

    dist
}

/// Δ-stepping på en pre-partisjonert CSR. `light_count[u]` angir hvor
/// mange av u sine kanter som er light (w ≤ delta). Sparer en branch per
/// kant ved å iterere bare over relevant range i hver phase.
pub fn delta_stepping_partitioned(
    g: &CsrGraph,
    light_count: &[u32],
    src: u32,
    delta: f32,
) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;

    let bucket_of = |d: f32| -> usize {
        if !d.is_finite() { usize::MAX } else { (d / delta) as usize }
    };

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new()];
    buckets[0].push(src);

    let mut i = 0usize;
    let mut settled: Vec<u32> = Vec::with_capacity(1024);
    loop {
        while i < buckets.len() && buckets[i].is_empty() {
            i += 1;
        }
        if i >= buckets.len() {
            break;
        }

        // Phase A: light-only relax to convergence.
        settled.clear();
        loop {
            let current = std::mem::take(&mut buckets[i]);
            if current.is_empty() {
                break;
            }
            for &u in &current {
                let du = dist[u as usize];
                if bucket_of(du) != i {
                    continue;
                }
                settled.push(u);
                let s = g.head[u as usize] as usize;
                let lc = light_count[u as usize] as usize;
                let light_end = s + lc;
                for k in s..light_end {
                    let v = g.edge_to[k];
                    let nd = du + g.edge_w[k];
                    if nd < dist[v as usize] {
                        dist[v as usize] = nd;
                        let nb = bucket_of(nd);
                        if nb >= buckets.len() {
                            buckets.resize_with(nb + 1, Vec::new);
                        }
                        buckets[nb].push(v);
                    }
                }
            }
        }

        // Phase B: heavy-only relax once.
        let mut needs_redo = false;
        for &u in &settled {
            let du = dist[u as usize];
            if bucket_of(du) != i {
                continue;
            }
            let lc = light_count[u as usize] as usize;
            let heavy_start = g.head[u as usize] as usize + lc;
            let heavy_end = g.head[u as usize + 1] as usize;
            for k in heavy_start..heavy_end {
                let v = g.edge_to[k];
                let nd = du + g.edge_w[k];
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    let nb = bucket_of(nd);
                    if nb <= i {
                        needs_redo = true;
                    }
                    if nb >= buckets.len() {
                        buckets.resize_with(nb + 1, Vec::new);
                    }
                    buckets[nb].push(v);
                }
            }
        }

        if !needs_redo {
            i += 1;
        }
    }
    dist
}

/// Robust delta-stepping. **Fixed**: heavy-relaksering kan pga f32-presisjon
/// ende opp i SAMME bucket som kilden (ikke forwarding). Vi sjekker det og
/// gjenkjører Phase A på samme bucket i stedet for å hoppe forbi.
pub fn delta_stepping_v2(g: &CsrGraph, src: u32, delta: f32) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;

    let bucket_of = |d: f32| -> usize {
        if !d.is_finite() {
            usize::MAX
        } else {
            (d / delta) as usize
        }
    };

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new()];
    buckets[0].push(src);

    let mut i = 0usize;
    let mut settled: Vec<u32> = Vec::with_capacity(1024);
    loop {
        while i < buckets.len() && buckets[i].is_empty() {
            i += 1;
        }
        if i >= buckets.len() {
            break;
        }

        // Phase A: relax light edges within bucket i, until stable.
        settled.clear();
        loop {
            // Take the contents (move out, leaving an empty Vec behind).
            let current = std::mem::take(&mut buckets[i]);
            if current.is_empty() {
                break;
            }
            for &u in &current {
                let du = dist[u as usize];
                if bucket_of(du) != i {
                    continue;
                }
                settled.push(u);
                let s = g.head[u as usize] as usize;
                let e = g.head[u as usize + 1] as usize;
                for k in s..e {
                    let w = g.edge_w[k];
                    if w > delta {
                        continue;
                    }
                    let v = g.edge_to[k];
                    let nd = du + w;
                    if nd < dist[v as usize] {
                        dist[v as usize] = nd;
                        let nb = bucket_of(nd);
                        if nb >= buckets.len() {
                            buckets.resize_with(nb + 1, Vec::new);
                        }
                        buckets[nb].push(v);
                    }
                }
            }
        }

        // Phase B: relax heavy edges from settled vertices.
        // FP-pitfall: en "heavy" kant (w > delta) gir nd = du + w. Vanligvis
        // havner det i en SENERE bucket. Men når delta ikke er eksakt
        // representerbar i f32 (f.eks. 0.3 = 0.30000001…), kan
        // floor(nd / delta) bli SAMME som bucket-id-en til du. Da må vi
        // re-prosessere bucket i før vi går videre.
        let mut needs_redo = false;
        for &u in &settled {
            let du = dist[u as usize];
            if bucket_of(du) != i {
                continue;
            }
            let s = g.head[u as usize] as usize;
            let e = g.head[u as usize + 1] as usize;
            for k in s..e {
                let w = g.edge_w[k];
                if w <= delta {
                    continue;
                }
                let v = g.edge_to[k];
                let nd = du + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    let nb = bucket_of(nd);
                    if nb <= i {
                        needs_redo = true;
                    }
                    if nb >= buckets.len() {
                        buckets.resize_with(nb + 1, Vec::new);
                    }
                    buckets[nb].push(v);
                }
            }
        }

        if !needs_redo {
            i += 1;
        }
    }
    dist
}


pub fn delta_stepping(g: &CsrGraph, src: u32, delta: f32) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;

    // Buckets: Vec<Vec<u32>> grown on demand.
    // Vi gjør (lett/tung)-distinkjsonen inline ved relaksering — å forhåndspartisjonere
    // CSR-arrayet i lett/tung ville spart en branch per kant, men det krever en
    // omstokking av hele `edge_to`/`edge_w` ved konstruksjon. På workloads målt
    // her dominerer cache-effekter, og branch-prediktoren håndterer w<=Δ fint.
    let mut buckets: Vec<Vec<u32>> = Vec::new();
    let bucket_of = |d: f32, delta: f32| -> usize { (d / delta) as usize };

    let push_bucket = |buckets: &mut Vec<Vec<u32>>, b: usize, v: u32| {
        if b >= buckets.len() {
            buckets.resize_with(b + 1, Vec::new);
        }
        buckets[b].push(v);
    };

    push_bucket(&mut buckets, 0, src);

    // Reusable scratch buffers.
    let mut current_bucket: Vec<u32> = Vec::new();
    let mut deferred_heavy: Vec<u32> = Vec::new();

    let mut i = 0usize;
    loop {
        // Find next non-empty bucket.
        while i < buckets.len() && buckets[i].is_empty() {
            i += 1;
        }
        if i >= buckets.len() {
            break;
        }

        // Phase A: settle bucket i via repeated light-edge relaxations.
        deferred_heavy.clear();
        loop {
            std::mem::swap(&mut current_bucket, &mut buckets[i]);
            if current_bucket.is_empty() {
                break;
            }
            // Snapshot vertices that are *finalized* in this bucket so we can
            // relax their heavy edges once at the end.
            for &u in &current_bucket {
                // Only process u if its dist actually places it in bucket i.
                let du = dist[u as usize];
                if bucket_of(du, delta) != i {
                    continue; // stale: u was moved elsewhere later
                }
                deferred_heavy.push(u);

                let s = g.head[u as usize] as usize;
                let e = g.head[u as usize + 1] as usize;
                for k in s..e {
                    let w = g.edge_w[k];
                    if w > delta {
                        continue;
                    }
                    let v = g.edge_to[k];
                    let nd = du + w;
                    if nd < dist[v as usize] {
                        let old_b = if dist[v as usize].is_finite() {
                            Some(bucket_of(dist[v as usize], delta))
                        } else {
                            None
                        };
                        dist[v as usize] = nd;
                        let new_b = bucket_of(nd, delta);
                        // We don't bother removing v from its old bucket; the
                        // staleness check at top of the loop filters it out.
                        let _ = old_b;
                        push_bucket(&mut buckets, new_b, v);
                    }
                }
            }
            current_bucket.clear();
        }

        // Phase B: relax heavy edges of vertices that finalized in bucket i.
        // FP-pitfall: w > delta garanterer ikke at floor((du+w)/delta) > i
        // når delta ikke er eksakt representerbar i f32 (f.eks. 0.3).
        // Hvis det skjer, må vi gjenkjøre Phase A på samme bucket.
        let mut needs_redo = false;
        for &u in &deferred_heavy {
            let du = dist[u as usize];
            if bucket_of(du, delta) != i {
                continue;
            }
            let s = g.head[u as usize] as usize;
            let e = g.head[u as usize + 1] as usize;
            for k in s..e {
                let w = g.edge_w[k];
                if w <= delta {
                    continue;
                }
                let v = g.edge_to[k];
                let nd = du + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    let new_b = bucket_of(nd, delta);
                    if new_b <= i {
                        needs_redo = true;
                    }
                    push_bucket(&mut buckets, new_b, v);
                }
            }
        }

        if !needs_redo {
            i += 1;
        }
    }

    dist
}
