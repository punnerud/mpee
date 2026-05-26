//! Word ladder graph: each node is a word. Two words are neighbours if they
//! have the same length and differ in *exactly* one letter.
//!
//! Example: cat → bat → bag → big → pig
//!
//! Efficient neighbour computation: for each word, generate all "wildcard"
//! patterns (one letter replaced with '*'). Words sharing a wildcard are
//! neighbours.
//!
//! Weights: all = 1.0. SSSP gives BFS distance (number of letter swaps).

use crate::graph::CsrGraph;
use std::collections::HashMap;
use std::path::Path;

pub fn build_word_ladder<P: AsRef<Path>>(
    words_path: P,
    min_len: usize,
    max_len: usize,
) -> std::io::Result<(CsrGraph, Vec<String>)> {
    let s = std::fs::read_to_string(words_path)?;
    let mut words: Vec<String> = s
        .lines()
        .map(|l| l.trim().to_lowercase())
        .filter(|w| {
            w.len() >= min_len
                && w.len() <= max_len
                && w.chars().all(|c| c.is_ascii_lowercase())
        })
        .collect();
    words.sort();
    words.dedup();

    let n = words.len();
    println!(
        "[ladder] {} ord etter filter (len {}..{})",
        n, min_len, max_len
    );

    // Wildcard-bucket: "c*t", "ca*", "*at"  →  alle ord som matcher.
    let mut wildcards: HashMap<String, Vec<u32>> = HashMap::with_capacity(n * 8);
    let mut buf = String::with_capacity(32);
    for (i, w) in words.iter().enumerate() {
        for j in 0..w.len() {
            buf.clear();
            buf.push_str(&w[..j]);
            buf.push('*');
            buf.push_str(&w[j + 1..]);
            wildcards
                .entry(buf.clone())
                .or_insert_with(|| Vec::with_capacity(2))
                .push(i as u32);
        }
    }

    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(n * 4);
    for group in wildcards.values() {
        if group.len() < 2 {
            continue;
        }
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                edges.push((group[i], group[j], 1.0));
                edges.push((group[j], group[i], 1.0));
            }
        }
    }
    println!(
        "[ladder] {} kanter (begge retninger), avg_deg = {:.2}",
        edges.len(),
        edges.len() as f32 / n.max(1) as f32
    );

    let g = CsrGraph::from_edges(n, &edges);
    Ok((g, words))
}

/// Find the word with the highest index in the candidates that is valid (as
/// a reasonable `src` for SSSP). Returns (idx, word).
pub fn pick_word<'a>(words: &'a [String], target: &str) -> Option<(u32, &'a str)> {
    words
        .iter()
        .position(|w| w == target)
        .map(|i| (i as u32, words[i].as_str()))
}
