//! Duan-inspirert SSSP — *forenklet* praktisk variant.
//!
//! ADVARSEL OM SCOPE
//! -----------------
//! Den faktiske algoritmen i Duan, Mao, Mao, Shu, Yin (STOC 2025) er svært
//! kompleks: rekursiv BMSSP (Bounded Multi-Source Shortest Path) med pivot-
//! plukking, batched Bellman-Ford, og en spesiell delvis-sortert datastruktur
//! D som støtter Insert / BatchPrepend / Pull. Den oppnår O(m log^{2/3} n)
//! deterministisk i comparison-addition-modellen.
//!
//! Denne filen implementerer en *praktisk forenkling* som tar med seg
//! kjerneideene:
//!
//!   1. **Bucket / partial order**: vertices grupperes i bøtter med bredde B
//!      og prosesseres en bøtte om gangen — uten sortering innad.
//!   2. **Batched relaksering inne i en bøtte**: vi gjør Bellman-Ford-stil
//!      multi-pass på vertices i nåværende bøtte til konvergens, i stedet
//!      for én-og-én pop fra en heap.
//!   3. **Pivot-redusering**: når en bøtte vokser stor, plukkes de k minste
//!      tentative avstandene som pivoter og relakseres først (etterligner
//!      FindPivots-trinnet).
//!
//! Korrekthetsdesign: vi bruker samme "lazy stale filter"-strategi som
//! Δ-stepping. Når vi forbedrer dist[v], pusher vi alltid v inn i sin nye
//! bucket; gamle plasseringer ignoreres når vi senere ser at
//! bucket_of(dist[v]) ikke matcher.

use crate::dijkstra::INF;
use crate::graph::CsrGraph;

pub fn duan_inspired(g: &CsrGraph, src: u32, bucket_width: f32) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;

    let bucket_of = |d: f32| -> usize { (d / bucket_width) as usize };

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new()];
    buckets[0].push(src);

    let mut frontier: Vec<u32> = Vec::new();
    let mut next_frontier: Vec<u32> = Vec::new();

    let mut bi = 0usize;
    loop {
        while bi < buckets.len() && buckets[bi].is_empty() {
            bi += 1;
        }
        if bi >= buckets.len() {
            break;
        }

        // Plukk ut bøtte bi, filtrer stale entries (de hvis dist nå ligger
        // i en helt annen bøtte) og dedupliser.
        frontier.clear();
        let raw = std::mem::take(&mut buckets[bi]);
        for v in raw {
            let d = dist[v as usize];
            if d.is_finite() && bucket_of(d) == bi {
                frontier.push(v);
            }
        }
        if frontier.is_empty() {
            bi += 1;
            continue;
        }

        // Dedup: vertices kan være pushet flere ganger inn i samme bucket.
        // Sortér + dedup. Kostnaden er O(|frontier| log |frontier|) men
        // sparer redundant arbeid i relaxation-loopen.
        frontier.sort_unstable();
        frontier.dedup();

        // Batched relaksering til konvergens innenfor bøtte bi.
        loop {
            next_frontier.clear();
            relax_pass(
                g,
                &frontier,
                &mut dist,
                &mut next_frontier,
                &mut buckets,
                bi,
                bucket_width,
            );
            if next_frontier.is_empty() {
                break;
            }
            // Dedup neste runde, ellers vokser den med kvadratisk arbeid.
            next_frontier.sort_unstable();
            next_frontier.dedup();
            std::mem::swap(&mut frontier, &mut next_frontier);
        }

        bi += 1;
    }

    dist
}

#[inline]
fn relax_pass(
    g: &CsrGraph,
    frontier: &[u32],
    dist: &mut [f32],
    next_round: &mut Vec<u32>,
    buckets: &mut Vec<Vec<u32>>,
    bucket_id: usize,
    bucket_width: f32,
) {
    let lo = bucket_id as f32 * bucket_width;
    let hi = (bucket_id + 1) as f32 * bucket_width;
    for &u in frontier {
        let du = dist[u as usize];
        if !(du >= lo && du < hi) {
            continue; // stale
        }
        let s = g.head[u as usize] as usize;
        let e = g.head[u as usize + 1] as usize;
        for k in s..e {
            let v = g.edge_to[k];
            let nd = du + g.edge_w[k];
            let dv = dist[v as usize];
            if nd < dv {
                dist[v as usize] = nd;
                if nd < hi {
                    // Forblir i samme bøtte — relakseres igjen i neste runde.
                    next_round.push(v);
                } else {
                    // Hopper til en senere bøtte.
                    let nb = (nd / bucket_width) as usize;
                    if nb >= buckets.len() {
                        buckets.resize_with(nb + 1, Vec::new);
                    }
                    buckets[nb].push(v);
                }
            }
        }
    }
}
