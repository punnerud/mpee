//! Two Dijeng implementations.
//!
//! `dijeng_binary` — textbook binary heap with lazy deletion (skip stale
//! entries on pop). Avoids the cost of supporting `decrease_key`.
//!
//! `dijeng_4ary` — 4-ary heap. Shallower (log_4 n vs log_2 n levels), and
//! sift-down processes 4 children per level which the branch predictor handles
//! well; in practice it usually beats binary by 10–25 % on sparse graphs.

use crate::graph::CsrGraph;

pub const INF: f32 = f32::INFINITY;

// -----------------------------------------------------------------------------
// Binary heap (min-heap of (dist, vertex)) with lazy deletion.
// -----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct HeapItem {
    dist: f32,
    v: u32,
}

pub fn dijeng_binary(g: &CsrGraph, src: u32) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;
    let mut heap: Vec<HeapItem> = Vec::with_capacity(n);
    heap.push(HeapItem { dist: 0.0, v: src });

    while let Some(HeapItem { dist: d, v: u }) = bin_pop(&mut heap) {
        // Lazy-deletion: skip entries that no longer reflect dist[u].
        if d > dist[u as usize] {
            continue;
        }
        let start = g.head[u as usize] as usize;
        let end = g.head[u as usize + 1] as usize;
        for i in start..end {
            let v = g.edge_to[i];
            let nd = d + g.edge_w[i];
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                bin_push(&mut heap, HeapItem { dist: nd, v });
            }
        }
    }
    dist
}

#[inline]
fn bin_push(heap: &mut Vec<HeapItem>, item: HeapItem) {
    let mut i = heap.len();
    heap.push(item);
    // Sift up.
    while i > 0 {
        let parent = (i - 1) >> 1;
        if heap[parent].dist <= heap[i].dist {
            break;
        }
        heap.swap(parent, i);
        i = parent;
    }
}

#[inline]
fn bin_pop(heap: &mut Vec<HeapItem>) -> Option<HeapItem> {
    let n = heap.len();
    if n == 0 {
        return None;
    }
    let top = heap[0];
    let last = heap.pop().unwrap();
    if n == 1 {
        return Some(top);
    }
    heap[0] = last;
    // Sift down.
    let mut i = 0usize;
    let len = heap.len();
    loop {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        let mut smallest = i;
        if l < len && heap[l].dist < heap[smallest].dist {
            smallest = l;
        }
        if r < len && heap[r].dist < heap[smallest].dist {
            smallest = r;
        }
        if smallest == i {
            break;
        }
        heap.swap(i, smallest);
        i = smallest;
    }
    Some(top)
}

// -----------------------------------------------------------------------------
// 4-ary heap. Same lazy-deletion strategy.
// -----------------------------------------------------------------------------

pub fn dijeng_4ary(g: &CsrGraph, src: u32) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;
    let mut heap: Vec<HeapItem> = Vec::with_capacity(n);
    heap.push(HeapItem { dist: 0.0, v: src });

    while let Some(HeapItem { dist: d, v: u }) = quad_pop(&mut heap) {
        if d > dist[u as usize] {
            continue;
        }
        let start = g.head[u as usize] as usize;
        let end = g.head[u as usize + 1] as usize;
        for i in start..end {
            let v = g.edge_to[i];
            let nd = d + g.edge_w[i];
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                quad_push(&mut heap, HeapItem { dist: nd, v });
            }
        }
    }
    dist
}

#[inline]
fn quad_push(heap: &mut Vec<HeapItem>, item: HeapItem) {
    let mut i = heap.len();
    heap.push(item);
    while i > 0 {
        let parent = (i - 1) >> 2;
        if heap[parent].dist <= heap[i].dist {
            break;
        }
        heap.swap(parent, i);
        i = parent;
    }
}

// -----------------------------------------------------------------------------
// 8-ary heap. Cache line on most modern CPUs holds ~16 floats, so 8 children
// fit nicely in one or two cache lines. Theoretical wins on push-heavy
// workloads (sparse graphs with low average degree).
// -----------------------------------------------------------------------------

pub fn dijeng_8ary(g: &CsrGraph, src: u32) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;
    let mut heap: Vec<HeapItem> = Vec::with_capacity(n);
    heap.push(HeapItem { dist: 0.0, v: src });

    while let Some(HeapItem { dist: d, v: u }) = oct_pop(&mut heap) {
        if d > dist[u as usize] {
            continue;
        }
        let start = g.head[u as usize] as usize;
        let end = g.head[u as usize + 1] as usize;
        for i in start..end {
            let v = g.edge_to[i];
            let nd = d + g.edge_w[i];
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                oct_push(&mut heap, HeapItem { dist: nd, v });
            }
        }
    }
    dist
}

#[inline]
fn oct_push(heap: &mut Vec<HeapItem>, item: HeapItem) {
    let mut i = heap.len();
    heap.push(item);
    while i > 0 {
        let parent = (i - 1) >> 3;
        if heap[parent].dist <= heap[i].dist {
            break;
        }
        heap.swap(parent, i);
        i = parent;
    }
}

#[inline]
fn oct_pop(heap: &mut Vec<HeapItem>) -> Option<HeapItem> {
    let n = heap.len();
    if n == 0 {
        return None;
    }
    let top = heap[0];
    let last = heap.pop().unwrap();
    if n == 1 {
        return Some(top);
    }
    heap[0] = last;
    let len = heap.len();
    let mut i = 0usize;
    loop {
        let first_child = 8 * i + 1;
        if first_child >= len {
            break;
        }
        let mut smallest = first_child;
        let mut smallest_d = heap[first_child].dist;
        let last_child = (first_child + 8).min(len);
        for c in (first_child + 1)..last_child {
            let dc = heap[c].dist;
            if dc < smallest_d {
                smallest = c;
                smallest_d = dc;
            }
        }
        if smallest_d >= heap[i].dist {
            break;
        }
        heap.swap(i, smallest);
        i = smallest;
    }
    Some(top)
}

#[inline]
fn quad_pop(heap: &mut Vec<HeapItem>) -> Option<HeapItem> {
    let n = heap.len();
    if n == 0 {
        return None;
    }
    let top = heap[0];
    let last = heap.pop().unwrap();
    if n == 1 {
        return Some(top);
    }
    heap[0] = last;
    let len = heap.len();
    let mut i = 0usize;
    loop {
        let first_child = 4 * i + 1;
        if first_child >= len {
            break;
        }
        // Find smallest of up to 4 children using direct unrolled compares.
        let mut smallest = first_child;
        let mut smallest_d = heap[first_child].dist;
        let last_child = (first_child + 4).min(len);
        // Loop is bounded to ≤3 iterations; LLVM unrolls this nicely.
        for c in (first_child + 1)..last_child {
            let dc = heap[c].dist;
            if dc < smallest_d {
                smallest = c;
                smallest_d = dc;
            }
        }
        if smallest_d >= heap[i].dist {
            break;
        }
        heap.swap(i, smallest);
        i = smallest;
    }
    Some(top)
}
