# Example custom routing profile: a delivery van that avoids motorways,
# crawls residential streets, and pays a penalty on service roads.
#
# Build a cache with it:
#   mpee build map.osm.pbf docs/profiles/delivery_van.profile
# (caches land next to the PBF as <map>.delivery_van.pp/.ch)
name = delivery_van
base = car                  # inherit accepts/speeds from the builtin car profile
block motorway
block motorway_link
speed residential = 25      # km/h override
penalty service = 1.5       # 50% slower on service roads
respect_maxspeed = true
speed_factor = 0.92         # global realism multiplier
