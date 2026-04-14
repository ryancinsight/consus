# Consus — Backlog

## Phase 1: HDF5 MVP (Read + Write)

### P1.1 — HDF5 Read Path (Critical)
- [ ] Object header v1 parser (traverse header messages)
- [ ] Object header v2 parser (OHDR/OCHK chunk parsing)
- [ ] Datatype message parser (all 11 classes)
- [ ] Dataspace message parser ✅ (basic implementation exists)
- [ ] Data layout message parser (compact, contiguous, chunked)
- [ ] Filter pipeline message parser
- [ ] Symbol table message parser (v1 groups)
- [ ] Link message parser (v2 groups)
- [ ] B-tree v1 traversal (group + chunk index)
- [ ] B-tree v2 traversal (fractal heap integration)
- [ ] Local heap reader (group member names)
- [ ] Global heap reader (VL data)
- [ ] Contiguous dataset read
- [ ] Chunked dataset read (single chunk)
- [ ] Chunked dataset read (multi-chunk with filter pipeline)
- [ ] Hyperslab selection read (strided subarray)
- [ ] Point selection read
- [ ] Compound datatype read
- [ ] Variable-length datatype read
- [ ] Attribute read
- [ ] Superblock v0/v1/v2/v3 parsing ✅ (v0/v1/v2/v3 implemented)
- [ ] File open with validation ✅ (basic implementation exists)

### P1.2 — HDF5 Write Path
- [ ] Superblock v2 writer
- [ ] Object header v2 writer
- [ ] Datatype message writer
- [ ] Dataspace message writer
- [ ] Data layout message writer (contiguous)
- [ ] Data layout message writer (chunked)
- [ ] Filter pipeline message writer
- [ ] B-tree v2 writer (chunk index)
- [ ] Local heap writer
- [ ] Contiguous dataset write
- [ ] Chunked dataset write (with compression)
- [ ] Group creation
- [ ] Attribute write
- [ ] File creation (new file from scratch)
- [ ] File close with flush and checksum

### P1.3 — HDF5 Reference File Testing
- [ ] Download HDF5 reference test files from HDF Group
- [ ] Read tests against t_float.h5 (floating-point dataset)
- [ ] Read tests against t_int.h5 (integer dataset)
- [ ] Read tests against t_compound.h5 (compound datatype)
- [ ] Read tests against t_vlen.h5 (variable-length)
- [ ] Read tests against t_string.h5 (string dataset)
- [ ] Read tests against t_group.h5 (hierarchical groups)
- [ ] Read tests against t_chunk.h5 (chunked storage)
- [ ] Read tests against t_filter.h5 (compressed dataset)
- [ ] Write-then-read round-trip tests
- [ ] Comparison with h5dump output

### P1.4 — Benchmarks
- [ ] Criterion benchmarks: contiguous read throughput
- [ ] Criterion benchmarks: chunked read throughput
- [ ] Criterion benchmarks: compressed read (deflate, zstd, lz4)
- [ ] Comparison with HDF5 C library via hdf5-rs bindings
- [ ] Comparison with Python h5py

## Phase 2: Zarr + netCDF-4

### P2.1 — Zarr v2
- [ ] `.zarray` JSON metadata parser
- [ ] `.zattrs` JSON metadata parser
- [ ] `.zgroup` JSON metadata parser
- [ ] Directory store implementation
- [ ] Chunk read (single chunk)
- [ ] Chunk read (multi-chunk)
- [ ] Compression pipeline (Zarr codec chain)
- [ ] Full array read with selection
- [ ] Zarr v2 write path
- [ ] Round-trip tests against Python zarr library output

### P2.2 — Zarr v3
- [ ] `zarr.json` metadata parser
- [ ] Sharding codec
- [ ] v3 chunk key encoding
- [ ] v3 codec pipeline
- [ ] v3 write path
- [ ] Interop tests with zarr-python v3

### P2.3 — netCDF-4
- [ ] Dimension scale detection via HDF5 attributes
- [ ] Variable → HDF5 dataset mapping
- [ ] CF conventions attribute parsing
- [ ] Unlimited dimension handling
- [ ] netCDF-4 classic model read
- [ ] netCDF-4 enhanced model read (groups, user-defined types)
- [ ] netCDF-4 write path
- [ ] Comparison with Unidata netCDF-C reference files

## Phase 3: Parquet + Polish

### P3.1 — Parquet Interop
- [ ] Consus ↔ Parquet schema mapping
- [ ] Read Parquet files as Consus datasets
- [ ] Write Consus datasets to Parquet
- [ ] Hybrid mode: Parquet tables inside Consus containers
- [ ] Arrow array bridge (zero-copy)

### P3.2 — Performance & Production Readiness
- [ ] Parallel chunk I/O via Rayon
- [ ] Async I/O path via Tokio
- [ ] Memory-mapped I/O backend
- [ ] Large file (>4 GiB) regression tests
- [ ] Fuzz testing (cargo-fuzz / proptest)
- [ ] WASM target validation
- [ ] no_std smoke tests (thumbv7em-none-eabihf)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Documentation site
- [ ] crates.io publication