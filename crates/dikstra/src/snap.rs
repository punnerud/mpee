//! SNAP edge-list loader (Stanford Network Analysis Platform).
//!
//! Format: hver linje er enten en kommentar (begynner med `#`) eller en
//! kant `<from>\t<to>\n`. Node-IDer er u32 (passer for de fleste SNAP-
//! datasett opp til 100M noder).
//!
//! Vekter: alle = 1.0 (sosiale grafer er normalt uvektede).

use crate::graph::CsrGraph;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

pub fn load_snap_edge_list<P: AsRef<Path>>(
    path: P,
) -> std::io::Result<(CsrGraph, Vec<u32>)> {
    let path = path.as_ref();
    println!("[snap] åpner {}", path.display());
    let f = File::open(path)?;
    let reader: Box<dyn Read> = if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Box::new(GzDecoder::new(f))
    } else {
        Box::new(f)
    };
    let mut br = BufReader::with_capacity(1 << 20, reader);

    let t = std::time::Instant::now();
    let mut edges_u: Vec<u32> = Vec::with_capacity(100_000_000);
    let mut edges_v: Vec<u32> = Vec::with_capacity(100_000_000);
    let mut max_id: u32 = 0;
    let mut line = String::new();
    let mut total_lines = 0usize;
    let mut comments = 0usize;
    loop {
        line.clear();
        let n = br.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        total_lines += 1;
        if line.starts_with('#') {
            comments += 1;
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut it = trimmed.split_ascii_whitespace();
        let u_s = match it.next() {
            Some(s) => s,
            None => continue,
        };
        let v_s = match it.next() {
            Some(s) => s,
            None => continue,
        };
        let u: u32 = match u_s.parse() {
            Ok(x) => x,
            Err(_) => continue,
        };
        let v: u32 = match v_s.parse() {
            Ok(x) => x,
            Err(_) => continue,
        };
        if u > max_id {
            max_id = u;
        }
        if v > max_id {
            max_id = v;
        }
        edges_u.push(u);
        edges_v.push(v);
    }
    println!(
        "[snap] parse: {:.2} s — {} linjer ({} kommentarer), {} kanter, max_id = {}",
        t.elapsed().as_secs_f64(),
        total_lines,
        comments,
        edges_u.len(),
        max_id
    );

    // SNAP-kanter er DIRECTED. For å gjøre algoritmer som forutsetter
    // tilgjengelighet meningsfulle, gjør vi grafen *undirected* ved å legge
    // til reverse-kanter. Det matcher hvordan disse datasettene typisk
    // brukes for korteste-vei-eksperimenter (friend-of-friend osv).
    let n = (max_id as usize) + 1;
    let m = edges_u.len();
    let mut all: Vec<(u32, u32, f32)> = Vec::with_capacity(2 * m);
    for i in 0..m {
        let u = edges_u[i];
        let v = edges_v[i];
        if u == v {
            continue;
        }
        all.push((u, v, 1.0));
        all.push((v, u, 1.0));
    }
    println!(
        "[snap] symmetriserer: {} kanter (begge retninger), n = {}",
        all.len(),
        n
    );

    let g = CsrGraph::from_edges(n, &all);

    // Lag en list over node-IDer som faktisk har kanter, til kilde-utvelgelse.
    let mut nonempty: Vec<u32> = (0..n)
        .filter(|&u| g.head[u + 1] - g.head[u] > 0)
        .map(|u| u as u32)
        .collect();
    nonempty.shrink_to_fit();
    println!(
        "[snap] {} av {} noder har minst én kant",
        nonempty.len(),
        n
    );

    Ok((g, nonempty))
}
