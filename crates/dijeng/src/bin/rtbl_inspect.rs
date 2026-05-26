//! Inspect a binary row-streamed matrix file (RTBL0001).
//!
//! Usage: rtbl_inspect <path> [rows_to_show]

use std::fs::File;
use std::io::BufReader;

use dijeng::binary_table::BinaryTableReader;

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: rtbl_inspect <path> [rows_to_show]");
    let n_show: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "5".to_string())
        .parse()
        .unwrap_or(5);

    let f = File::open(&path)?;
    let file_size = f.metadata()?.len();
    let mut r = BinaryTableReader::new(BufReader::new(f))?;
    println!(
        "n_src={} n_dst={} dual_channel={} dtype={:?} scale_exp={} pad_64={} crc32_footer={} symmetric_ut={}",
        r.n_src,
        r.n_dst,
        r.dual_channel,
        r.cell_dtype,
        r.scale_exp,
        r.pad_64,
        r.crc32_footer,
        r.symmetric_ut
    );
    println!("file size on disk: {} bytes", file_size);

    let mut row_indices: Vec<u32> = Vec::new();
    let mut finite_total = 0usize;
    let mut sum_dur = 0.0_f64;
    let mut sum_dist = 0.0_f64;
    let mut total_cells = 0usize;
    let mut rows_seen = 0u32;
    while let Some((idx, dur, dist)) = r.read_row()? {
        if rows_seen < n_show as u32 {
            let n_show_cells = 3usize.min(dur.len());
            println!(
                "row[{rows_seen}]: src_idx={idx}, dur[0..{n_show_cells}]={:?}, dist[0..{n_show_cells}]={:?}",
                &dur[..n_show_cells],
                dist.as_ref().map(|d| &d[..n_show_cells])
            );
        }
        row_indices.push(idx);
        for k in 0..dur.len() {
            total_cells += 1;
            if dur[k].is_finite() {
                finite_total += 1;
                sum_dur += dur[k] as f64;
                if let Some(d) = dist.as_ref() {
                    sum_dist += d[k] as f64;
                }
            }
        }
        rows_seen += 1;
    }
    println!(
        "rows read: {rows_seen}, total cells: {total_cells}, finite: {} ({:.1}%)",
        finite_total,
        100.0 * finite_total as f64 / total_cells as f64
    );
    println!(
        "avg dur over finite: {:.0} s, avg dist over finite: {:.0} m",
        sum_dur / finite_total.max(1) as f64,
        sum_dist / finite_total.max(1) as f64
    );

    // Permutation sanity: count unique row_indices.
    let mut sorted = row_indices.clone();
    sorted.sort();
    sorted.dedup();
    println!(
        "unique row_indices: {} (of {} rows)",
        sorted.len(),
        row_indices.len()
    );

    // Read + verify the CRC trailer if present.
    if r.crc32_footer {
        match r.read_trailer()? {
            Some(crc) => println!("crc32 trailer present: {:08x}", crc),
            None => println!("crc32 trailer missing!"),
        }
    }

    Ok(())
}
