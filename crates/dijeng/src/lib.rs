// Modules that need OS threads / mmap / OSM parsing are native-only. The
// remaining set (buffer, graph, geo_index, dijeng, ch, cache*, names, routing,
// osm_profile, polyline) compiles to wasm32 for the in-browser demo.
pub mod addresses;
#[cfg(feature = "native")]
pub mod auto;
#[cfg(feature = "native")]
pub mod bidir;
#[cfg(feature = "native")]
pub mod binary_table;
#[cfg(feature = "native")]
pub mod budget;
pub mod buffer;
#[cfg(feature = "native")]
pub mod build;
#[cfg(feature = "native")]
pub mod cache;
pub mod cache_ch;
pub mod cache_pp;
pub mod ch;
#[cfg(feature = "native")]
pub mod delta_step;
pub mod dijeng;
#[cfg(feature = "native")]
pub mod duan;
#[cfg(feature = "native")]
pub mod farthest_first;
pub mod geo_index;
pub mod graph;
#[cfg(feature = "native")]
pub mod knn;
pub mod names;
#[cfg(feature = "native")]
pub mod osm;
pub mod osm_profile;
#[cfg(feature = "native")]
pub mod paged;
pub mod polyline;
#[cfg(feature = "native")]
pub mod preprocess;
pub mod routing;
#[cfg(feature = "native")]
pub mod rubik;
#[cfg(feature = "native")]
pub mod snap;
#[cfg(feature = "native")]
pub mod synth;
#[cfg(feature = "native")]
pub mod varint;
#[cfg(feature = "native")]
pub mod elevation;
pub mod isochrone;
// matching rides on the native-gated parallel CH matrix (see routing.rs);
// trip stays buildable everywhere (tsp_order is pure), its service wrapper
// RoutingService::trip is native-gated.
#[cfg(feature = "native")]
pub mod matching;
pub mod trip;
pub mod wordladder;
