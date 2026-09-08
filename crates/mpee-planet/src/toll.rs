//! Pass 6 — toll gates as a directed cost layer.
//!
//! A toll is not a property of a road, it is a property of *passing a point in
//! a direction*: the Oslo ring charges inbound and not outbound, and an
//! AutoPASS gantry on the E6 charges northbound traffic only. So a toll cannot
//! live on a segment — it lives on a **vertex transition**, `(from, gate, to)`.
//!
//! Pass 2 already forced every `barrier=toll_booth` / `highway=toll_gantry`
//! node to become a graph vertex, so the gate is guaranteed to be a point the
//! router actually passes *through* rather than something it drives past. Here
//! we resolve those OSM node ids to final vertex ids and keep their tags.
//!
//! Prices are deliberately kept out of the graph. OSM's `charge` tag is
//! patchy and prices change; the gate identity (OSM node id) is stable. So the
//! dataset stores identity + tags, and a separate tariff table — an operator's
//! own, or Norway's NVDB export — is joined at query time.

use crate::bitmap::BitMap;
use crate::build::Paths;
use crate::mmapvec;
use crate::varint::{get_bytes, get_u};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// A toll gate, as stored.
#[derive(Clone, Debug)]
pub struct Gate {
    pub vertex: u32,
    pub osm_id: u64,
    pub lat_e7: i32,
    pub lon_e7: i32,
    pub name: String,
    pub operator: String,
    /// `charge`/`toll:*` as tagged, verbatim — often absent.
    pub charge: String,
    /// OSM `direction`: "forward", "backward", or empty for both ways.
    pub direction: String,
}

pub fn pass6(paths: &Paths, junc: &BitMap) -> io::Result<u64> {
    let perm_map = mmapvec::open(&paths.f("vperm.bin"))?;
    let perm: &[u32] = unsafe { mmapvec::as_slice(&perm_map[..]) };
    let raw = std::fs::read(paths.f("tollnodes.bin"))?;
    let mut p = 0usize;
    let mut gates: Vec<Gate> = Vec::new();
    let mut off_network = 0u64;
    while p < raw.len() {
        let id = get_u(&raw, &mut p);
        let lat = i32::from_le_bytes(raw[p..p + 4].try_into().unwrap());
        p += 4;
        let lon = i32::from_le_bytes(raw[p..p + 4].try_into().unwrap());
        p += 4;
        let n = get_u(&raw, &mut p) as usize;
        let mut tags: HashMap<String, String> = HashMap::new();
        for _ in 0..n {
            let k = String::from_utf8_lossy(get_bytes(&raw, &mut p)).to_string();
            let v = String::from_utf8_lossy(get_bytes(&raw, &mut p)).to_string();
            tags.insert(k, v);
        }
        // A gate that no drivable way passes through is not on our network.
        if !junc.get(id) {
            off_network += 1;
            continue;
        }
        let old = junc.rank1(id) as usize;
        if old >= perm.len() {
            off_network += 1;
            continue;
        }
        let charge = ["charge", "toll:motorcar", "fee", "toll:price"]
            .iter()
            .find_map(|k| tags.get(*k).cloned())
            .unwrap_or_default();
        gates.push(Gate {
            vertex: perm[old],
            osm_id: id,
            lat_e7: lat,
            lon_e7: lon,
            name: tags.get("name").cloned().unwrap_or_default(),
            operator: tags.get("operator").cloned().unwrap_or_default(),
            charge,
            direction: tags.get("direction").cloned().unwrap_or_default(),
        });
    }
    gates.sort_by_key(|g| g.vertex);

    let mut w_v = BufWriter::with_capacity(1 << 18, File::create(paths.f("toll.vertex"))?);
    let mut w_m = BufWriter::with_capacity(1 << 20, File::create(paths.f("toll.meta"))?);
    for g in &gates {
        w_v.write_all(&g.vertex.to_le_bytes())?;
        writeln!(
            w_m,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            g.osm_id, g.lat_e7, g.lon_e7, g.name, g.operator, g.charge, g.direction
        )?;
    }
    w_v.flush()?;
    w_m.flush()?;
    eprintln!(
        "[pass6] {} toll gates on the network ({} tagged off-network, ignored)",
        gates.len(),
        off_network
    );
    let charged = gates.iter().filter(|g| !g.charge.is_empty()).count();
    eprintln!(
        "[pass6] {} carry a price in OSM ({:.0} %); the rest need a tariff table",
        charged,
        charged as f64 / gates.len().max(1) as f64 * 100.0
    );
    Ok(gates.len() as u64)
}

/// Query-side toll layer: vertex ids to test a route against, plus metadata.
pub struct TollIndex {
    pub vertex: Vec<u32>,
    pub gates: Vec<Gate>,
    /// Optional tariff, keyed by OSM node id, in minor currency units.
    pub tariff: HashMap<u64, Tariff>,
}

#[derive(Clone, Debug, Default)]
pub struct Tariff {
    pub name: String,
    pub currency: String,
    /// Price for a small car, in whole currency units.
    pub car: f64,
    /// Direction of collection as the operator states it.
    pub direction: String,
    /// The charging scheme this gate belongs to (a toll ring, a project).
    /// Gates in the same scheme share the hour rule.
    pub scheme: String,
    /// Minutes within which repeat passages in the same scheme are free.
    /// 0 disables the rule.
    pub hour_rule_min: u32,
    pub rush_car: f64,
}

impl TollIndex {
    pub fn open(dir: &Path) -> io::Result<TollIndex> {
        let vraw = std::fs::read(dir.join("toll.vertex")).unwrap_or_default();
        let vertex: Vec<u32> = vraw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let meta = std::fs::read_to_string(dir.join("toll.meta")).unwrap_or_default();
        let mut gates = Vec::with_capacity(vertex.len());
        for (i, line) in meta.lines().enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 7 {
                continue;
            }
            gates.push(Gate {
                vertex: *vertex.get(i).unwrap_or(&u32::MAX),
                osm_id: f[0].parse().unwrap_or(0),
                lat_e7: f[1].parse().unwrap_or(0),
                lon_e7: f[2].parse().unwrap_or(0),
                name: f[3].to_string(),
                operator: f[4].to_string(),
                charge: f[5].to_string(),
                direction: f[6].to_string(),
            });
        }
        let mut tariff = HashMap::new();
        // Optional sidecar, tab-separated:
        //   osm_id · name · currency · car · direction · scheme · hour_rule_min · rush_car
        // Everything after `car` is optional, so a two-column price list works.
        if let Ok(t) = std::fs::read_to_string(dir.join("toll.tariff.tsv")) {
            for line in t.lines() {
                if line.starts_with('#') || line.trim().is_empty() {
                    continue;
                }
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 4 {
                    continue;
                }
                let g = |i: usize| f.get(i).map(|s| s.to_string()).unwrap_or_default();
                if let Ok(id) = f[0].parse::<u64>() {
                    tariff.insert(
                        id,
                        Tariff {
                            name: f[1].to_string(),
                            currency: f[2].to_string(),
                            car: f[3].parse().unwrap_or(0.0),
                            direction: g(4),
                            scheme: g(5),
                            hour_rule_min: g(6).parse().unwrap_or(0),
                            rush_car: g(7).parse().unwrap_or(0.0),
                        },
                    );
                }
            }
        }
        Ok(TollIndex { vertex, gates, tariff })
    }

    pub fn is_gate(&self, v: u32) -> Option<usize> {
        self.vertex.binary_search(&v).ok()
    }

    /// Price for one gate, preferring the tariff table over the OSM tag.
    pub fn price(&self, i: usize) -> Option<(f64, String)> {
        let g = &self.gates[i];
        if let Some(t) = self.tariff.get(&g.osm_id) {
            return Some((t.car, t.currency.clone()));
        }
        parse_charge(&g.charge)
    }

    pub fn tariff_of(&self, i: usize) -> Option<&Tariff> {
        self.tariff.get(&self.gates[i].osm_id)
    }

    /// Total for a list of gates passed in order, applying each scheme's hour
    /// rule: inside a toll ring you pay once, not once per gantry.
    ///
    /// The rule is time-based in reality; we approximate it as "once per
    /// scheme per route", which is right for a through-trip and conservative
    /// for a long one that re-enters the same ring hours later.
    pub fn total(&self, hits: &[usize]) -> Vec<(String, f64, Vec<String>)> {
        let mut per_currency: HashMap<String, (f64, Vec<String>)> = HashMap::new();
        let mut charged_scheme: HashMap<String, f64> = HashMap::new();
        for &i in hits {
            let Some((price, cur)) = self.price(i) else { continue };
            let t = self.tariff_of(i);
            let scheme = t.map(|t| t.scheme.clone()).unwrap_or_default();
            let hour_rule = t.map(|t| t.hour_rule_min > 0).unwrap_or(false);
            let label = if self.gates[i].name.is_empty() {
                t.map(|t| t.name.clone()).unwrap_or_else(|| "(unnamed)".into())
            } else {
                self.gates[i].name.clone()
            };
            if hour_rule && !scheme.is_empty() {
                // Keep only the dearest passage within the scheme.
                let prev = charged_scheme.get(&scheme).copied().unwrap_or(0.0);
                if price <= prev {
                    continue;
                }
                let e = per_currency.entry(cur.clone()).or_default();
                e.0 += price - prev;
                e.1.push(format!("{label} ({scheme}, hour rule)"));
                charged_scheme.insert(scheme, price);
            } else {
                let e = per_currency.entry(cur.clone()).or_default();
                e.0 += price;
                e.1.push(label);
            }
        }
        per_currency.into_iter().map(|(c, (v, n))| (c, v, n)).collect()
    }
}

/// Parse an OSM `charge` value such as "25 NOK" or "2.50 EUR".
///
/// Deliberately strict. Surveying the planet's tags through the tariff
/// database turned up `BRL/motorcar`, a bare `€`, a price of 25000 with no
/// currency at all, and `RUR @ (class:1 AND 7:00-0:00)` — a conditional
/// expression this model has no way to honour. A loose parser turns each of
/// those into a confident wrong number; leaving them unparsed leaves the gate
/// visibly unpriced instead, which is the honest answer and the one a
/// dispatcher can act on.
pub fn parse_charge(s: &str) -> Option<(f64, String)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // `@` introduces conditional pricing (by vehicle class, by time of day).
    // That is a richer model than one number, so decline rather than guess.
    if s.contains('@') {
        return None;
    }
    // OSM separates variants with ';'; take the first clause.
    let first = s.split(';').next()?.trim();
    let num: String = first
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .collect();
    if num.is_empty() {
        return None;
    }
    let v: f64 = num.replace(',', ".").parse().ok()?;
    if !(v.is_finite() && v > 0.0) {
        return None;
    }
    // "4.50 BRL/motorcar" — the suffix after '/' qualifies the vehicle, not
    // the currency.
    let raw = first[num.len()..].trim();
    let cur = raw.split('/').next().unwrap_or("").trim();
    let cur = match cur {
        "€" => "EUR",
        "£" => "GBP",
        "¥" => "JPY",
        "₽" => "RUB",
        other => other,
    };
    // A currency we cannot name is a price we cannot use.
    if cur.len() != 3 || !cur.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    Some((v, cur.to_ascii_uppercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_parse_in_the_forms_osm_uses() {
        assert_eq!(parse_charge("25 NOK"), Some((25.0, "NOK".into())));
        assert_eq!(parse_charge("2.50 EUR"), Some((2.5, "EUR".into())));
        assert_eq!(parse_charge("2,50 EUR"), Some((2.5, "EUR".into())));
        assert_eq!(parse_charge("25 NOK; 50 NOK"), Some((25.0, "NOK".into())));
        assert_eq!(parse_charge(""), None);
        assert_eq!(parse_charge("yes"), None);
    }

    /// The shapes a planet-wide survey actually turned up. Each of these used
    /// to become a confident wrong number.
    #[test]
    fn malformed_charges_are_declined_rather_than_guessed() {
        // A vehicle qualifier is not part of the currency.
        assert_eq!(parse_charge("4.50 BRL/motorcar"), Some((4.5, "BRL".into())));
        // Symbols are normalised to their ISO code.
        assert_eq!(parse_charge("1.80 €"), Some((1.8, "EUR".into())));
        // Conditional pricing needs a model we do not have here.
        assert_eq!(parse_charge("50 RUR @ (class:1 AND 7:00-0:00)"), None);
        // A number with no currency is not a price.
        assert_eq!(parse_charge("25000"), None);
        // Neither is a free or negative one.
        assert_eq!(parse_charge("0 EUR"), None);
        // Nor a currency that is not a currency.
        assert_eq!(parse_charge("10 dollars"), None);
        assert_eq!(parse_charge("3 R$"), None);
    }
}
