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
// Custom profiles — the OSRM-Lua / Valhalla-costing equivalent.
// ============================================================================

/// A user-defined routing profile: per-class speeds, allow/block lists,
/// penalties and an optional programmable hook — everything OSRM's Lua
/// profiles are used for in practice, without a scripting runtime in the
/// engine. Loaded from a dependency-free line format:
///
/// ```text
/// # delivery_van.profile
/// name = delivery_van
/// base = car                  # inherit accepts/speeds from a builtin
/// speed motorway = 95         # km/h override per highway class
/// speed residential = 25
/// allow track = 15            # additionally usable class (with speed)
/// block living_street         # never use this class
/// penalty service = 1.5       # time multiplier on a class
/// respect_maxspeed = true     # cap class speeds at the OSM maxspeed tag
/// speed_factor = 0.9          # global realism multiplier on every speed
/// ```
///
/// The programmable path: embedders (the Python surface compiles a PySpell
/// expression, any Rust caller passes a closure) set [`CustomProfile::speed_fn`]
/// — called as `f(highway, maxspeed_kmh) -> Option<speed_kmh>` per way, with
/// `None` falling through to the declarative rules above. This is the hook the
/// engine exposes instead of embedding Lua.
#[derive(Clone, Default)]
pub struct CustomProfile {
    pub name: String,
    /// Builtin profile providing accepts/speed fallbacks for classes not
    /// mentioned in this file.
    pub base: Option<Profile>,
    /// Per-class speed overrides (km/h). Also implies the class is accepted.
    pub speeds: Vec<(String, f32)>,
    /// Extra accepted classes, optionally with a speed (else class default).
    pub allow: Vec<(String, Option<f32>)>,
    /// Classes never used, regardless of base/allow.
    pub block: Vec<String>,
    /// Per-class travel-time multipliers (> 1 = slower).
    pub penalties: Vec<(String, f32)>,
    /// Cap speeds at the way's `maxspeed` tag.
    pub respect_maxspeed: bool,
    /// Global multiplier applied to every resulting speed.
    pub speed_factor: f32,
    /// Programmable override: `f(highway, maxspeed_kmh)` → `Some(speed_kmh)`
    /// decides the way outright (`Some(0.0)` rejects it); `None` falls through
    /// to the declarative rules. This is where PySpell/closures plug in.
    pub speed_fn:
        Option<std::sync::Arc<dyn Fn(&str, Option<f32>) -> Option<f32> + Send + Sync>>,
}

impl std::fmt::Debug for CustomProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomProfile")
            .field("name", &self.name)
            .field("base", &self.base)
            .field("speeds", &self.speeds)
            .field("allow", &self.allow)
            .field("block", &self.block)
            .field("penalties", &self.penalties)
            .field("respect_maxspeed", &self.respect_maxspeed)
            .field("speed_factor", &self.speed_factor)
            .field("speed_fn", &self.speed_fn.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl CustomProfile {
    /// Parse the line format documented on the type. Unknown keys error so a
    /// typo can't silently produce a default profile.
    pub fn parse(text: &str, fallback_name: &str) -> Result<Self, String> {
        let mut p = CustomProfile {
            name: fallback_name.to_string(),
            speed_factor: 1.0,
            respect_maxspeed: true,
            ..Default::default()
        };
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let err = |msg: &str| format!("profile line {}: {} ({:?})", lineno + 1, msg, raw);
            // Forms: `key = value`, `speed <class> = <kmh>`,
            // `allow <class> [= kmh]`, `block <class>`, `penalty <class> = x`.
            let (head, value) = match line.split_once('=') {
                Some((h, v)) => (h.trim(), Some(v.trim())),
                None => (line, None),
            };
            let mut words = head.split_whitespace();
            let key = words.next().ok_or_else(|| err("empty key"))?;
            let arg = words.next();
            if words.next().is_some() {
                return Err(err("too many tokens before '='"));
            }
            match (key, arg) {
                ("name", None) => p.name = value.ok_or_else(|| err("name needs a value"))?.to_string(),
                ("base", None) => {
                    let v = value.ok_or_else(|| err("base needs a value"))?;
                    p.base = Some(
                        Profile::from_name(v).ok_or_else(|| err("unknown base profile"))?,
                    );
                }
                ("respect_maxspeed", None) => {
                    p.respect_maxspeed = value == Some("true") || value == Some("1");
                }
                ("speed_factor", None) => {
                    p.speed_factor = value
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| err("speed_factor needs a number"))?;
                }
                ("speed", Some(class)) => {
                    let v: f32 = value
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| err("speed needs `speed <class> = <kmh>`"))?;
                    p.speeds.push((class.to_string(), v));
                }
                ("allow", Some(class)) => {
                    let v: Option<f32> = match value {
                        Some(v) => Some(v.parse().map_err(|_| err("allow speed must be a number"))?),
                        None => None,
                    };
                    p.allow.push((class.to_string(), v));
                }
                ("block", Some(class)) => {
                    if value.is_some() {
                        return Err(err("block takes no value"));
                    }
                    p.block.push(class.to_string());
                }
                ("penalty", Some(class)) => {
                    let v: f32 = value
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| err("penalty needs `penalty <class> = <factor>`"))?;
                    if v <= 0.0 {
                        return Err(err("penalty must be > 0"));
                    }
                    p.penalties.push((class.to_string(), v));
                }
                _ => return Err(err("unknown directive")),
            }
        }
        if p.speed_factor <= 0.0 {
            return Err("speed_factor must be > 0".to_string());
        }
        Ok(p)
    }

    /// Load from a `.profile` file; the profile name defaults to the file stem.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "custom".to_string());
        Self::parse(&text, &stem)
    }

    fn lookup<'a, T>(list: &'a [(String, T)], class: &str) -> Option<&'a T> {
        list.iter().find(|(c, _)| c == class).map(|(_, v)| v)
    }

    pub fn accepts(&self, highway: &str) -> bool {
        if self.block.iter().any(|c| c == highway) {
            return false;
        }
        if Self::lookup(&self.speeds, highway).is_some()
            || self.allow.iter().any(|(c, _)| c == highway)
        {
            return true;
        }
        self.base.map_or(false, |b| b.accepts(highway))
    }

    pub fn speed_kmh(&self, highway: &str, maxspeed_tag: Option<f32>) -> f32 {
        if let Some(f) = &self.speed_fn {
            if let Some(v) = f(highway, maxspeed_tag) {
                return v * self.speed_factor;
            }
        }
        let class_speed = if let Some(&v) = Self::lookup(&self.speeds, highway) {
            v
        } else if let Some(v) = self.allow.iter().find(|(c, _)| c == highway).map(|(_, v)| *v) {
            v.unwrap_or_else(|| class_speed_kmh(highway))
        } else if let Some(b) = self.base {
            // Builtins already handle maxspeed their own way; avoid double-capping.
            return b.speed_kmh(highway, if self.respect_maxspeed { maxspeed_tag } else { None })
                * self.speed_factor
                / Self::lookup(&self.penalties, highway).copied().unwrap_or(1.0);
        } else {
            class_speed_kmh(highway)
        };
        let capped = match (self.respect_maxspeed, maxspeed_tag) {
            (true, Some(ms)) => class_speed.min(ms),
            _ => class_speed,
        };
        capped * self.speed_factor / Self::lookup(&self.penalties, highway).copied().unwrap_or(1.0)
    }
}

/// A routing profile argument: a builtin (`car`, `bicycle`, …) or a custom
/// `.profile` file. Everything that builds a graph takes `impl Into<ProfileSpec>`,
/// so existing `Profile::Car` call sites keep compiling.
#[derive(Debug, Clone)]
pub enum ProfileSpec {
    Builtin(Profile),
    Custom(std::sync::Arc<CustomProfile>),
}

impl From<Profile> for ProfileSpec {
    fn from(p: Profile) -> Self {
        ProfileSpec::Builtin(p)
    }
}

impl From<CustomProfile> for ProfileSpec {
    fn from(p: CustomProfile) -> Self {
        ProfileSpec::Custom(std::sync::Arc::new(p))
    }
}

impl ProfileSpec {
    /// Resolve a CLI/API profile argument: builtin name, or a path to a
    /// `.profile` file (recognised by extension or path separator).
    pub fn from_arg(arg: &str) -> Result<Self, String> {
        if let Some(p) = Profile::from_name(arg) {
            return Ok(ProfileSpec::Builtin(p));
        }
        if arg.ends_with(".profile") || arg.contains('/') || arg.contains('\\') {
            return CustomProfile::load(std::path::Path::new(arg)).map(Into::into);
        }
        Err(format!(
            "unknown profile '{arg}' (builtins: car, motorcycle, bicycle, foot — or a path to a .profile file)"
        ))
    }

    pub fn name(&self) -> &str {
        match self {
            ProfileSpec::Builtin(p) => p.name(),
            ProfileSpec::Custom(c) => &c.name,
        }
    }

    pub fn accepts(&self, highway: &str) -> bool {
        match self {
            ProfileSpec::Builtin(p) => p.accepts(highway),
            ProfileSpec::Custom(c) => c.accepts(highway),
        }
    }

    pub fn speed_kmh(&self, highway: &str, maxspeed_tag: Option<f32>) -> f32 {
        match self {
            ProfileSpec::Builtin(p) => p.speed_kmh(highway, maxspeed_tag),
            ProfileSpec::Custom(c) => c.speed_kmh(highway, maxspeed_tag),
        }
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

    #[test]
    fn custom_profile_parses_and_costs() {
        let p = CustomProfile::parse(
            "
            name = delivery_van
            base = car
            speed motorway = 95
            speed residential = 25
            allow track = 15
            block living_street
            penalty service = 2.0
            respect_maxspeed = true
            speed_factor = 0.9
            ",
            "fallback",
        )
        .expect("parses");
        assert_eq!(p.name, "delivery_van");
        // Blocked class loses even though base=car accepts it.
        assert!(!p.accepts("living_street"));
        // Allowed extra class.
        assert!(p.accepts("track"));
        assert!((p.speed_kmh("track", None) - 15.0 * 0.9).abs() < 1e-4);
        // Explicit speed, capped by maxspeed, scaled by factor.
        assert!((p.speed_kmh("motorway", None) - 95.0 * 0.9).abs() < 1e-4);
        assert!((p.speed_kmh("motorway", Some(80.0)) - 80.0 * 0.9).abs() < 1e-4);
        // Base fallback with penalty: car service = 20 km/h → /2 ×0.9.
        assert!(p.accepts("service"));
        assert!((p.speed_kmh("service", None) - 20.0 * 0.9 / 2.0).abs() < 1e-4);
        // Unknown class not in base list → rejected.
        assert!(!p.accepts("runway"));
    }

    #[test]
    fn custom_profile_speed_fn_overrides() {
        let mut p = CustomProfile::parse("base = car", "x").unwrap();
        p.speed_fn = Some(std::sync::Arc::new(|hw, _ms| {
            (hw == "residential").then_some(7.5)
        }));
        // Hook decides residential; everything else falls through to base.
        assert!((p.speed_kmh("residential", None) - 7.5).abs() < 1e-4);
        assert!((p.speed_kmh("motorway", None) - 110.0).abs() < 1e-4);
    }

    #[test]
    fn custom_profile_rejects_typos() {
        assert!(CustomProfile::parse("sped motorway = 95", "x").is_err());
        assert!(CustomProfile::parse("base = hovercraft", "x").is_err());
        assert!(CustomProfile::parse("speed_factor = -1", "x").is_err());
    }

    #[test]
    fn profile_spec_from_arg() {
        assert!(matches!(
            ProfileSpec::from_arg("car"),
            Ok(ProfileSpec::Builtin(Profile::Car))
        ));
        assert!(ProfileSpec::from_arg("warpdrive").is_err());
    }
}
