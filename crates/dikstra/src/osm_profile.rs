//! Routing profiles: which OSM ways a vehicle can use, and how fast.
//!
//! A `Profile` answers two questions per OSM way:
//!
//!   1. **`accepts(highway)`** — is this kind of road/path usable at all?
//!   2. **`speed_kmh(highway, maxspeed)`** — what's the realistic speed?
//!
//! From these two, the OSM loader produces a `CsrGraph` with edge weight =
//! traversal time in seconds. The CH is then built on those weights, so a
//! query gives the fastest legal route for that profile.
//!
//! For an entirely non-OSM domain (Internet routing, supply networks,
//! social-graph distances, …), don't go through this module at all — build
//! your own `CsrGraph` with whatever weight you like and feed it to
//! `ch::build`. The CH library is graph-agnostic.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Standard private car / passenger vehicle.
    Car,
    /// Motorcycle: same network as car but ignores HOV/bus lane restrictions
    /// (we don't model those anyway) and is slightly faster on small roads.
    Motorcycle,
    /// Bicycle: avoids motorways, prefers cycleway/track, ignores `maxspeed`.
    Bicycle,
    /// Pedestrian: avoids motorways/trunks, includes pedestrian/footway,
    /// constant 5 km/h.
    Foot,
}

impl Profile {
    pub fn name(&self) -> &'static str {
        match self {
            Profile::Car => "car",
            Profile::Motorcycle => "motorcycle",
            Profile::Bicycle => "bicycle",
            Profile::Foot => "foot",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "car" | "driving" => Some(Profile::Car),
            "motorcycle" | "moto" => Some(Profile::Motorcycle),
            "bicycle" | "bike" | "cycling" => Some(Profile::Bicycle),
            "foot" | "walk" | "walking" | "pedestrian" => Some(Profile::Foot),
            _ => None,
        }
    }

    /// Whether the profile is allowed to use this `highway=*` value.
    pub fn accepts(&self, highway: &str) -> bool {
        match self {
            Profile::Car | Profile::Motorcycle => CAR_DRIVABLE.iter().any(|h| *h == highway),
            Profile::Bicycle => BIKE_RIDEABLE.iter().any(|h| *h == highway),
            Profile::Foot => FOOT_WALKABLE.iter().any(|h| *h == highway),
        }
    }

    /// Realistic traversal speed in km/h. `maxspeed_tag` is the parsed
    /// `maxspeed=*` value if any; profiles that don't legally have a speed
    /// (bike, foot) ignore it.
    pub fn speed_kmh(&self, highway: &str, maxspeed_tag: Option<f32>) -> f32 {
        match self {
            Profile::Car => maxspeed_tag.unwrap_or_else(|| car_speed(highway)),
            // Motorcycles obey speed limits but cap at the same class default
            // when the limit is missing. Slight bonus on small roads.
            Profile::Motorcycle => {
                let base = maxspeed_tag.unwrap_or_else(|| car_speed(highway));
                let bonus = match highway {
                    "tertiary" | "tertiary_link" | "unclassified" | "residential" => 1.05,
                    _ => 1.0,
                };
                base * bonus
            }
            // Bikes ignore maxspeed; we use a flat per-class cycling speed.
            Profile::Bicycle => bike_speed(highway),
            // Walkers go a constant ~5 km/h regardless of road class.
            Profile::Foot => 5.0,
        }
    }
}

/// Drivable for cars + motorcycles.
const CAR_DRIVABLE: &[&str] = &[
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
];

/// Rideable on a bicycle. No motorway/trunk; allows cycleway + path.
const BIKE_RIDEABLE: &[&str] = &[
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
    "cycleway",
    "path",
    "track",
    "bridleway",
];

/// Walkable on foot. Adds footway/pedestrian/steps; excludes high-speed roads.
const FOOT_WALKABLE: &[&str] = &[
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
    "footway",
    "path",
    "pedestrian",
    "steps",
    "track",
    "bridleway",
    "cycleway",
];

fn car_speed(highway: &str) -> f32 {
    match highway {
        "motorway" => 110.0,
        "motorway_link" => 70.0,
        "trunk" => 90.0,
        "trunk_link" => 60.0,
        "primary" => 75.0,
        "primary_link" => 50.0,
        "secondary" => 65.0,
        "secondary_link" => 45.0,
        "tertiary" => 55.0,
        "tertiary_link" => 40.0,
        "unclassified" => 40.0,
        "residential" => 30.0,
        "living_street" => 10.0,
        "service" => 20.0,
        "road" => 35.0,
        _ => 25.0,
    }
}

fn bike_speed(highway: &str) -> f32 {
    match highway {
        // Off-road dedicated cycling is fast and uninterrupted.
        "cycleway" => 22.0,
        // Mixed but quiet roads.
        "primary" | "primary_link" | "secondary" | "secondary_link" => 16.0,
        "tertiary" | "tertiary_link" | "unclassified" => 18.0,
        "residential" | "living_street" | "service" | "road" => 16.0,
        // Mixed surfaces are slower.
        "path" | "track" | "bridleway" => 12.0,
        _ => 14.0,
    }
}

// ============================================================================
// `maxspeed` tag parsing — kept here so the OSM parser has a single import.
// ============================================================================

/// Parse an OSM `maxspeed` tag value into km/h, if possible. Recognises bare
/// numbers (assumed km/h), `"<n> km/h"`/`"kmh"`/`"kph"`, `"<n> mph"`, and
/// `"<n> knots"`. Returns `None` for advisory values like `"signals"` or
/// `"none"`.
pub fn parse_maxspeed(value: &str) -> Option<f32> {
    let v = value.trim().to_ascii_lowercase();
    if v.is_empty() || v == "none" || v == "signals" || v == "walk" || v == "variable" {
        return None;
    }
    let (num_str, suffix) = split_number_unit(&v);
    let n: f32 = num_str.parse().ok()?;
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    match suffix {
        "" | "km/h" | "kmh" | "kph" => Some(n),
        "mph" => Some(n * 1.609_344),
        "knots" => Some(n * 1.852),
        _ => Some(n),
    }
}

fn split_number_unit(s: &str) -> (&str, &str) {
    let mut i = 0;
    for (idx, ch) in s.char_indices() {
        if ch.is_ascii_digit() || ch == '.' || ch == '-' {
            i = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    let num = &s[..i];
    let rest = s[i..].trim();
    (num, rest)
}

#[inline]
pub fn kmh_to_mps(kmh: f32) -> f32 {
    kmh / 3.6
}

// ============================================================================
// Backwards-compat shims — let existing callers keep using the old free
// functions until they're migrated to `Profile`.
// ============================================================================

#[deprecated = "use Profile::Car.speed_kmh(highway, None)"]
pub fn class_speed_kmh(highway: &str) -> f32 {
    Profile::Car.speed_kmh(highway, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_names_roundtrip() {
        for p in [Profile::Car, Profile::Motorcycle, Profile::Bicycle, Profile::Foot] {
            assert_eq!(Profile::from_name(p.name()), Some(p));
        }
        // Aliases.
        assert_eq!(Profile::from_name("driving"), Some(Profile::Car));
        assert_eq!(Profile::from_name("bike"), Some(Profile::Bicycle));
        assert_eq!(Profile::from_name("walking"), Some(Profile::Foot));
        assert_eq!(Profile::from_name("nonsense"), None);
    }

    #[test]
    fn car_excludes_no_pedestrian() {
        assert!(Profile::Car.accepts("residential"));
        assert!(!Profile::Car.accepts("footway"));
        assert!(!Profile::Car.accepts("cycleway"));
    }

    #[test]
    fn bike_excludes_motorway() {
        assert!(!Profile::Bicycle.accepts("motorway"));
        assert!(!Profile::Bicycle.accepts("trunk"));
        assert!(Profile::Bicycle.accepts("cycleway"));
        assert!(Profile::Bicycle.accepts("residential"));
    }

    #[test]
    fn foot_constant_speed() {
        assert_eq!(Profile::Foot.speed_kmh("residential", None), 5.0);
        assert_eq!(Profile::Foot.speed_kmh("residential", Some(50.0)), 5.0);
        assert_eq!(Profile::Foot.speed_kmh("footway", None), 5.0);
    }

    #[test]
    fn car_uses_maxspeed_when_present() {
        assert_eq!(Profile::Car.speed_kmh("residential", None), 30.0);
        assert_eq!(Profile::Car.speed_kmh("residential", Some(50.0)), 50.0);
    }

    #[test]
    fn maxspeed_parsing_unchanged() {
        assert_eq!(parse_maxspeed("50"), Some(50.0));
        assert_eq!(parse_maxspeed("none"), None);
        assert!((parse_maxspeed("30 mph").unwrap() - 48.28).abs() < 0.1);
    }
}
