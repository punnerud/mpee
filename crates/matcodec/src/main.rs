//! `matcodec` — lossless structural compression for distance/duration matrices.
//!
//!   matcodec compress   <in.json> <out.mtz> [--clusters K]
//!   matcodec decompress <in.mtz>  <out.json>
//!   matcodec roundtrip  <in.json> [--clusters K]     # verify lossless + ratio
//!
//! Input JSON may be a bare `[[...]]`, `{"durations": [[...]]}`, or a full
//! instance `{"matrices": {"<profile>": {"durations": [[...]]}}}`.

use serde_json::Value;
use std::fs;
use std::process::exit;

fn read_matrix(path: &str) -> (Vec<i32>, usize) {
    let txt = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("read {path}: {e}");
        exit(1)
    });
    let v: Value = serde_json::from_str(&txt).unwrap_or_else(|e| {
        eprintln!("parse {path}: {e}");
        exit(1)
    });
    let rows = if v.is_array() {
        v
    } else if let Some(d) = v.get("durations") {
        d.clone()
    } else if let Some(m) = v.get("matrices").and_then(|m| m.as_object()) {
        let first = m.values().next().unwrap_or_else(|| {
            eprintln!("empty matrices");
            exit(1)
        });
        first.get("durations").cloned().unwrap_or_else(|| {
            eprintln!("no durations in matrix");
            exit(1)
        })
    } else {
        eprintln!("no matrix found (expected array, .durations, or .matrices)");
        exit(1)
    };
    let arr = rows.as_array().unwrap();
    let n = arr.len();
    let mut d = vec![0i32; n * n];
    for (i, row) in arr.iter().enumerate() {
        let r = row.as_array().unwrap();
        if r.len() != n {
            eprintln!("matrix not square at row {i}");
            exit(1);
        }
        for (j, x) in r.iter().enumerate() {
            d[i * n + j] = x.as_i64().unwrap_or(0) as i32;
        }
    }
    (d, n)
}

fn write_matrix(path: &str, d: &[i32], n: usize) {
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(d[i * n..i * n + n].to_vec());
    }
    let v = serde_json::json!({ "durations": rows });
    fs::write(path, serde_json::to_string(&v).unwrap()).unwrap_or_else(|e| {
        eprintln!("write {path}: {e}");
        exit(1)
    });
}

fn parse_clusters(args: &[String], n: usize) -> usize {
    for w in args.windows(2) {
        if w[0] == "--clusters" {
            return w[1].parse().unwrap_or_else(|_| {
                eprintln!("bad --clusters");
                exit(1)
            });
        }
    }
    matcodec::default_k(n)
}

fn parse_landmarks(args: &[String], n: usize) -> usize {
    for w in args.windows(2) {
        if w[0] == "--landmarks" {
            return w[1].parse::<usize>().unwrap_or(32).min(n.saturating_sub(1)).max(1);
        }
    }
    32.min(n.saturating_sub(1)).max(1)
}

fn print_report(rep: &matcodec::ValidationReport) {
    let warns = rep.warnings();
    if warns.is_empty() {
        return;
    }
    eprintln!("matrix anomalies (rows checked: {}):", rep.rows_seen);
    for w in &warns {
        eprintln!("{w}");
    }
    for ex in rep.examples.iter().take(6) {
        eprintln!("    e.g. {ex}");
    }
    if !rep.metric_ok() {
        eprintln!(
            "  -> non-metric/asymmetric: triangle-inequality shortcuts (ALT pruning, bridge predictor) disabled; value-driven methods only."
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: matcodec <compress|decompress|roundtrip> ...");
        exit(2);
    }
    match args[1].as_str() {
        "compress" => {
            if args.len() < 4 {
                eprintln!(
                    "usage: matcodec compress <in.json> <out.mtz> [--clusters K] [--stream] [--landmarks L]"
                );
                exit(2);
            }
            let (d, n) = read_matrix(&args[2]);
            let rest = &args[4..];
            let raw = n * n * 4;
            // validate first and warn (anomalies surface before we trust the data)
            let rep = matcodec::validate(&d, n);
            print_report(&rep);
            if rest.iter().any(|a| a == "--stream") {
                let l = parse_landmarks(rest, n);
                let lm = matcodec::pick_landmarks(&d, n, l);
                let mut src = matcodec::SliceRows { d: &d, n };
                let mut f = std::io::BufWriter::new(fs::File::create(&args[3]).unwrap());
                matcodec::compress_stream(&mut src, &lm, &mut f).unwrap();
                drop(f);
                let sz = fs::metadata(&args[3]).map(|m| m.len()).unwrap_or(0);
                println!(
                    "compressed (stream, bridge L={l}) N={n}: {} -> {} bytes ({:.2}x)",
                    raw,
                    sz,
                    raw as f64 / sz.max(1) as f64
                );
            } else {
                let k = parse_clusters(rest, n);
                let comp = matcodec::compress(&d, n, k);
                fs::write(&args[3], &comp).unwrap();
                println!(
                    "compressed N={n} k={k}: {} -> {} bytes ({:.2}x)",
                    raw,
                    comp.len(),
                    raw as f64 / comp.len() as f64
                );
            }
        }
        "validate" => {
            if args.len() < 3 {
                eprintln!("usage: matcodec validate <in.json>");
                exit(2);
            }
            let (d, n) = read_matrix(&args[2]);
            let rep = matcodec::validate(&d, n);
            print_report(&rep);
            println!(
                "validated N={n}: metric_ok={} hard_error={}",
                rep.metric_ok(),
                rep.has_hard_error()
            );
            if rep.has_hard_error() {
                exit(1);
            }
        }
        "decompress" => {
            if args.len() < 4 {
                eprintln!("usage: matcodec decompress <in.mtz> <out.json>");
                exit(2);
            }
            let bytes = fs::read(&args[2]).unwrap();
            let (d, n) = matcodec::decompress(&bytes).unwrap_or_else(|e| {
                eprintln!("decompress: {e}");
                exit(1)
            });
            write_matrix(&args[3], &d, n);
            println!("decompressed N={n} -> {}", args[3]);
        }
        "roundtrip" => {
            if args.len() < 3 {
                eprintln!("usage: matcodec roundtrip <in.json> [--clusters K]");
                exit(2);
            }
            let (d, n) = read_matrix(&args[2]);
            let k = parse_clusters(&args[3..], n);
            let comp = matcodec::compress(&d, n, k);
            let (back, n2) = matcodec::decompress(&comp).unwrap();
            let lossless = n2 == n && back == d;
            let raw = n * n * 4;
            println!(
                "N={n} k={k}  raw={} bytes  codec={} bytes  ratio={:.2}x  lossless={}",
                raw,
                comp.len(),
                raw as f64 / comp.len() as f64,
                lossless
            );
            if !lossless {
                exit(1);
            }
        }
        other => {
            eprintln!("unknown command: {other}");
            exit(2);
        }
    }
}
