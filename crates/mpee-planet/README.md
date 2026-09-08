# mpee-planet

The whole world's road network as an offline routing dataset, on one disk.

**41.8 GB** for the planet — 270 173 819 junctions, 337 598 888 segments,
every address, and the toll gantries with their tariffs. Nothing is parsed or
decompressed when it opens: every array is a memory-mapped table of
fixed-width records, so the dataset is live the moment the mappings exist and
the OS pages in only what a query touches.

Distances are exact. Segment lengths are integrated along the full-resolution
WGS84 polyline and summed in integer centimetres, so a 500 km route is not
0.5 % out the way a haversine would be.

## It runs where memory is scarce

The design target is a machine where 512 MB is tight, and that claim is
measured rather than argued: `MPEE_CACHE_MB` caps the process's own resident
set and hands the mapped pages back to the kernel when it is exceeded, so a
query can be run as it would run on a small machine.

| route | search state | 512 MB | 128 MB | 64 MB |
|---|---:|---:|---:|---:|
| Oslo → Trondheim | 1.3 MB | 0.01 s | 0.01 s | **0.008 s** |
| Tokyo → Osaka | 12.8 MB | 0.9 s | 2.0 s | **3.1 s** |
| Lisboa → Warszawa | 84.2 MB | 6.8 s | 12.9 s | **16.9 s** |

Every cost is identical to the uncapped one. Dropping a clean page throws away
a cache and nothing else — a test pins that, with the cap set to one byte so
the pages are released at every check.

## What it does

- route from a coordinate or an address, with exact distances and toll costs
- geocode and reverse-geocode
- draw the result: `/api/roads` streams road geometry for the viewport from
  the same mmapped arrays the router uses, so the map background *is* the
  dataset — no tile server, nothing to be online for
- local overrides (a closed road, a corrected speed) applied without a rebuild,
  keyed on ground position rather than on segment ids that a rebuild renumbers

## Documentation

The full write-up — how it is built out of core, why the storage is raw rather
than compressed, the overlay ladder and what each rung measured, the toll
layer's split between flat files and MPEdb — is in
[`docs/planet.md`](../../docs/planet.md).

## Building

This crate is **excluded from the workspace** because it depends on MPEdb
through a path that is local to the author's machine:

```toml
mpedb = { path = "/Volumes/StorSSD/mpedb_src/crates/mpedb" }
```

Point that at your own MPEdb checkout in `crates/mpee-planet/Cargo.toml`
(two places: `[dependencies]` and `[dev-dependencies]`), then build the crate
on its own:

```bash
cd crates/mpee-planet && cargo build --release
```

Tests that need data are skipped unless you point them at a built dataset:

```bash
MPEE_TEST_DATA=/path/to/dataset MPEE_TEST_PBF=/path/to/area.osm.pbf cargo test --release
```
