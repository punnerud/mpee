//! Car routing profile: which OSM ways are drivable, how fast, which way, and
//! what they cost.
//!
//! Kept deliberately tag-driven and allocation-free — this runs over ~1.6
//! billion way tags on a planet build, so every decision is made on `&[u8]`
//! slices straight out of the PBF string table.

/// Road class, 5 bits in the packed edge attribute. Order matters: lower is
/// more important, and the router's hierarchy filter compares against it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Class {
    Motorway = 0,
    MotorwayLink = 1,
    Trunk = 2,
    TrunkLink = 3,
    Primary = 4,
    PrimaryLink = 5,
    Secondary = 6,
    SecondaryLink = 7,
    Tertiary = 8,
    TertiaryLink = 9,
    Unclassified = 10,
    Residential = 11,
    LivingStreet = 12,
    Service = 13,
    Ferry = 14,
    Other = 15,
}

impl Class {
    #[inline]
    pub fn from_highway(v: &[u8]) -> Option<Class> {
        Some(match v {
            b"motorway" => Class::Motorway,
            b"motorway_link" => Class::MotorwayLink,
            b"trunk" => Class::Trunk,
            b"trunk_link" => Class::TrunkLink,
            b"primary" => Class::Primary,
            b"primary_link" => Class::PrimaryLink,
            b"secondary" => Class::Secondary,
            b"secondary_link" => Class::SecondaryLink,
            b"tertiary" => Class::Tertiary,
            b"tertiary_link" => Class::TertiaryLink,
            b"unclassified" => Class::Unclassified,
            b"residential" => Class::Residential,
            b"living_street" => Class::LivingStreet,
            b"service" => Class::Service,
            b"road" | b"busway" => Class::Other,
            _ => return None,
        })
    }

    /// Default free-flow speed in km/h when the way carries no `maxspeed`.
    #[inline]
    pub fn default_kmh(self) -> u16 {
        match self {
            Class::Motorway => 100,
            Class::MotorwayLink => 60,
            Class::Trunk => 85,
            Class::TrunkLink => 50,
            Class::Primary => 65,
            Class::PrimaryLink => 40,
            Class::Secondary => 55,
            Class::SecondaryLink => 35,
            Class::Tertiary => 40,
            Class::TertiaryLink => 30,
            Class::Unclassified => 30,
            Class::Residential => 25,
            Class::LivingStreet => 10,
            Class::Service => 15,
            Class::Ferry => 20,
            Class::Other => 25,
        }
    }

    /// Rank used by the hierarchy filter: 0 = long-distance backbone.
    #[inline]
    pub fn tier(self) -> u8 {
        match self {
            Class::Motorway | Class::MotorwayLink | Class::Trunk | Class::TrunkLink => 0,
            Class::Primary | Class::PrimaryLink | Class::Ferry => 1,
            Class::Secondary | Class::SecondaryLink => 2,
            Class::Tertiary | Class::TertiaryLink => 3,
            _ => 4,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OneWay {
    No,
    Forward,
    Backward,
}

/// Everything the builder needs from one accepted way.
#[derive(Clone, Copy, Debug)]
pub struct WayAttr {
    pub class: Class,
    pub oneway: OneWay,
    pub kmh: u16,
    pub toll: bool,
    pub bridge: bool,
    pub tunnel: bool,
    pub roundabout: bool,
    /// String-table index of `name=*`, or `u32::MAX`.
    pub name_idx: u32,
    /// String-table index of `ref=*` (road number, e.g. "E6"), or `u32::MAX`.
    pub ref_idx: u32,
}

pub const NO_STR: u32 = u32::MAX;

#[inline]
fn truthy(v: &[u8]) -> bool {
    matches!(v, b"yes" | b"true" | b"1" | b"designated" | b"permissive")
}

#[inline]
fn falsy(v: &[u8]) -> bool {
    matches!(v, b"no" | b"false" | b"0" | b"private" | b"agricultural" | b"forestry" | b"delivery" | b"customers")
}

/// Parse an OSM `maxspeed` value into km/h. Handles the plain number, `N mph`,
/// `walk`, `none`, and the `XX:zone` country defaults we can resolve cheaply.
pub fn parse_maxspeed(v: &[u8], class: Class) -> Option<u16> {
    let s = std::str::from_utf8(v).ok()?.trim();
    if s.is_empty() {
        return None;
    }
    let low = s.to_ascii_lowercase();
    match low.as_str() {
        "none" | "unlimited" => {
            return Some(if class == Class::Motorway { 130 } else { 90 })
        }
        "walk" => return Some(7),
        _ => {}
    }
    // "RO:urban", "DE:rural", "GB:nsl_single", ...
    if let Some(rest) = low.split(':').nth(1) {
        let z = match rest {
            "urban" => Some(50),
            "rural" => Some(80),
            "motorway" => Some(120),
            "living_street" => Some(7),
            "nsl_single" => Some(96),
            "nsl_dual" => Some(112),
            "trunk" => Some(90),
            _ => None,
        };
        if z.is_some() {
            return z;
        }
    }
    let num: String = low.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    if num.is_empty() {
        return None;
    }
    let n: f64 = num.parse().ok()?;
    if n <= 0.0 {
        return None;
    }
    let kmh = if low.contains("mph") { n * 1.609344 } else if low.contains("knots") { n * 1.852 } else { n };
    Some(kmh.round().clamp(1.0, 255.0) as u16)
}

/// Decide whether a way is drivable and, if so, collect its attributes.
///
/// `tags` yields raw `(key, value)` byte slices; `resolve` maps a value slice
/// back to its string-table index so names can be interned without copying.
pub fn classify<'a, I, F>(tags: I, mut resolve: F) -> Option<WayAttr>
where
    I: Iterator<Item = (&'a [u8], &'a [u8])>,
    F: FnMut(&'a [u8]) -> u32,
{
    let mut class: Option<Class> = None;
    let mut is_ferry = false;
    let mut maxspeed: Option<&[u8]> = None;
    let mut oneway = OneWay::No;
    let mut oneway_seen = false;
    let mut roundabout = false;
    let mut toll = false;
    let mut bridge = false;
    let mut tunnel = false;
    let mut name: Option<&'a [u8]> = None;
    let mut rref: Option<&'a [u8]> = None;
    let mut blocked = false;
    let mut motor_ok: Option<bool> = None;
    let mut area = false;

    for (k, v) in tags {
        match k {
            b"highway" => class = Class::from_highway(v),
            b"route" => {
                if v == b"ferry" {
                    is_ferry = true;
                }
            }
            b"maxspeed" => maxspeed = Some(v),
            b"name" => name = Some(v),
            b"ref" => rref = Some(v),
            b"area" => area = truthy(v),
            b"junction" => {
                if v == b"roundabout" || v == b"circular" {
                    roundabout = true;
                }
            }
            b"oneway" => {
                oneway_seen = true;
                oneway = match v {
                    b"yes" | b"true" | b"1" => OneWay::Forward,
                    b"-1" | b"reverse" => OneWay::Backward,
                    _ => OneWay::No,
                };
            }
            // A motorcar-specific direction overrides the generic one.
            b"oneway:motorcar" | b"oneway:motor_vehicle" => {
                oneway_seen = true;
                oneway = match v {
                    b"yes" | b"true" | b"1" => OneWay::Forward,
                    b"-1" | b"reverse" => OneWay::Backward,
                    _ => OneWay::No,
                };
            }
            b"toll" | b"toll:motorcar" => toll = truthy(v),
            b"bridge" => bridge = !falsy(v),
            b"tunnel" => tunnel = !falsy(v),
            b"access" => {
                if falsy(v) {
                    blocked = true;
                }
            }
            b"motor_vehicle" | b"motorcar" => motor_ok = Some(!falsy(v)),
            _ => {}
        }
    }

    let class = match (class, is_ferry) {
        (Some(c), _) => c,
        (None, true) => Class::Ferry,
        (None, false) => return None,
    };
    // `area=yes` on a highway is a pedestrian square, not a linear road.
    if area {
        return None;
    }
    // `access=no` is overridable by an explicit motorcar permission.
    if blocked && motor_ok != Some(true) {
        return None;
    }
    if motor_ok == Some(false) {
        return None;
    }

    // Motorways and roundabouts are directed unless tagged otherwise.
    if !oneway_seen && (matches!(class, Class::Motorway | Class::MotorwayLink) || roundabout) {
        oneway = OneWay::Forward;
    }

    let kmh = maxspeed
        .and_then(|v| parse_maxspeed(v, class))
        .unwrap_or_else(|| class.default_kmh());

    Some(WayAttr {
        class,
        oneway,
        kmh,
        toll,
        bridge,
        tunnel,
        roundabout,
        name_idx: name.map(&mut resolve).unwrap_or(NO_STR),
        ref_idx: rref.map(&mut resolve).unwrap_or(NO_STR),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cls(tags: &[(&'static [u8], &'static [u8])]) -> Option<WayAttr> {
        classify(tags.iter().copied(), |_| 0)
    }

    #[test]
    fn motorway_is_oneway_by_default_but_tag_wins() {
        let a = cls(&[(b"highway", b"motorway")]).unwrap();
        assert_eq!(a.oneway, OneWay::Forward);
        let b = cls(&[(b"highway", b"motorway"), (b"oneway", b"no")]).unwrap();
        assert_eq!(b.oneway, OneWay::No);
    }

    #[test]
    fn access_no_rejects_unless_motorcar_allowed() {
        assert!(cls(&[(b"highway", b"service"), (b"access", b"private")]).is_none());
        assert!(cls(&[
            (b"highway", b"service"),
            (b"access", b"private"),
            (b"motorcar", b"yes")
        ])
        .is_some());
    }

    #[test]
    fn footways_and_squares_are_not_drivable() {
        assert!(cls(&[(b"highway", b"footway")]).is_none());
        assert!(cls(&[(b"highway", b"pedestrian"), (b"area", b"yes")]).is_none());
    }

    #[test]
    fn ferries_are_routable_edges() {
        let a = cls(&[(b"route", b"ferry")]).unwrap();
        assert_eq!(a.class, Class::Ferry);
    }

    #[test]
    fn maxspeed_forms() {
        assert_eq!(parse_maxspeed(b"50", Class::Primary), Some(50));
        assert_eq!(parse_maxspeed(b"60 mph", Class::Primary), Some(97));
        assert_eq!(parse_maxspeed(b"none", Class::Motorway), Some(130));
        assert_eq!(parse_maxspeed(b"walk", Class::Residential), Some(7));
        assert_eq!(parse_maxspeed(b"DE:urban", Class::Residential), Some(50));
        assert_eq!(parse_maxspeed(b"signals", Class::Residential), None);
    }

    #[test]
    fn toll_flag_is_picked_up() {
        assert!(cls(&[(b"highway", b"motorway"), (b"toll", b"yes")]).unwrap().toll);
        assert!(!cls(&[(b"highway", b"motorway"), (b"toll", b"no")]).unwrap().toll);
    }
}
