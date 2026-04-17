# Consus — Backlog

## Phase 1: HDF5 MVP

## Phase 1: HDF5 MVP

### P1.1 — HDF5 Read Path
- [x] Object header v1 parser
- [x] Object header v2 parser
- [x] Datatype message parser (all 11 classes)
- [x] Dataspace message parser
- [x] Data layout message parser (v3 and v4 variants currently implemented)
- [x] Filter pipeline message parser
- [x] Symbol table message parser (v1 groups)
- [x] Link message parser (v2 groups)
- [x] B-tree v1 traversal (group navigation)
- [x] B-tree v2 traversal (fractal heap integration)
- [x] Local heap reader
- [x] Global heap reader
- [x] Contiguous dataset read
- [x] Chunked dataset read (single chunk)
- [x] Chunked dataset read (multi-chunk with filter pipeline)
- [x] Hyperslab selection read
- [x] Point selection read
- [x] Compound datatype read
- [x] Variable-length datatype read
- [x] Attribute read
- [x] Dense group link enumeration
- [x] Dense attribute enumeration
- [x] Soft link resolution
- [x] Superblock v0/v1/v2/v3 parsing
- [x] File open with validation
- [x] Chunk index v4 B-tree v2 lookup
- [x] External link traversal beyond typed error reporting
- [x] Reference-file coverage against canonical HDF Group fixtures

## Phase 1: HDF5 MVP

### P1.1 — HDF5 Read Path
- [x] Object header v1 parser
- [x] Object header v2 parser
- [x] Datatype message parser (all 11 classes)
- [x] Dataspace message parser
- [x] Data layout message parser (v3 and v4 variants currently implemented)
- [x] Filter pipeline message parser
- [x] Symbol table message parser (v1 groups)
- [x] Link message parser (v2 groups)
- [x] B-tree v1 traversal (group navigation)
- [x] B-tree v2 traversal (fractal heap integration)
- [x] Local heap reader
- [x] Global heap reader
- [x] Contiguous dataset read
- [x] Chunked dataset read (single chunk)
- [x] Chunked dataset read (multi-chunk with filter pipeline)
- [x] Hyperslab selection read
- [x] Point selection read
- [x] Compound datatype read
- [x] Variable-length datatype read
- [x] Attribute read
- [x] Dense group link enumeration
- [x] Dense attribute enumeration
- [x] Soft link resolution
- [x] Superblock v0/v1/v2/v3 parsing
- [x] File open with validation
- [x] Chunk index v4 B-tree v2 lookup
- [x] External link traversal beyond typed error reporting
- [x] Reference-file coverage against canonical HDF Group fixtures

## Phase 1: HDF5 MVP (Read + Write)

### P1.1 — HDF5 Read Path
- [x] Object header v1 parser
- [x] Object header v2 parser
- [x] Datatype message parser (all 11 classes)
- [x] Dataspace message parser
- [x] Data layout message parser (v3 and v4 variants currently implemented)
- [x] Filter pipeline message parser
- [x] Symbol table message parser (v1 groups)
- [x] Link message parser (v2 groups)
- [x] B-tree v1 traversal (group navigation)
- [x] B-tree v2 traversal (fractal heap integration)
- [x] Local heap reader
- [x] Global heap reader
- [x] Contiguous dataset read
- [x] Chunked dataset read (single chunk)
- [x] Chunked dataset read (multi-chunk with filter pipeline)
- [x] Hyperslab selection read
- [x] Point selection read
- [x] Compound datatype read
- [x] Variable-length datatype read
- [x] Attribute read
- [x] Dense group link enumeration
- [x] Dense attribute enumeration
- [x] Soft link resolution
- [x] Superblock v0/v1/v2/v3 parsing
- [x] File open with validation
- [ ] Chunk index v4 B-tree v2 lookup
- [ ] External link traversal beyond typed error reporting
- [ ] Reference-file coverage against canonical HDF Group fixtures

### P1.2 — HDF5 Write Path
- [x] Superblock v2 writer
- [x] Object header v2 writer
- [x] Datatype message writer
- [x] Dataspace message writer
- [x] Data layout message writer (contiguous)
- [x] Data layout message writer (chunked with materialized v1 chunk index in current scope)
- [x] Contiguous dataset write
- [x] Group creation at root via link messages
- [x] Attribute write
- [x] File creation (new file from scratch)
- [x] File close with flush and checksum
- [x] Filter pipeline message writer
- [x] Chunk index writer for chunked datasets (v1 raw-data chunk B-tree leaf in current scope)
- [x] Chunked dataset write with persisted chunk index and end-to-end value roundtrip in current scope
- [ ] Chunked dataset compression roundtrip coverage
- [ ] Local heap writer for v1 group emission

### P1.3 — HDF5 Verification
- [x] In-memory round-trip tests for contiguous datasets
- [x] In-memory round-trip tests for chunked datasets (v3 layout, single-leaf v1 chunk B-tree scope)
- [x] In-memory round-trip tests for attributes
- [x] Reference-style tests against repository sample files
- [ ] Download and validate canonical HDF Group reference files
- [x] Read tests against `t_vlen.h5`
- [x] Read tests against `t_filter.h5`
- [x] Read tests against `t_compound.h5`
- [x] Read tests against `t_vlen.h5`
- [ ] Read tests against `t_string.h5`
- [ ] Read tests against `t_group.h5`
- [x] Read tests against `t_chunk.h5`
- [x] Read tests against `t_filter.h5`
- [ ] Comparison with `h5dump` output for verified fixtures

### P1.4 — Performance & Memory
- [x] Fill-value-aware undefined chunk reads
- [x] Parallel chunk I/O via Rayon
- [ ] Criterion benchmarks: contiguous read throughput
- [ ] Criterion benchmarks: chunked read throughput
- [ ] Criterion benchmarks: compressed read (deflate, zstd, lz4)
- [x] Allocation reduction in object-header and writer message assembly
- [ ] Comparison with HDF5 C library via `hdf5-rs`
- [ ] Comparison with Python `h5py`

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

## Phase 1.5 — Workspace Test Integrity
- [x] Restore compile-valid property integration suite against current stable APIs
- [x] Re-enable value-semantic property coverage for shape, selection, byte-order, datatype sizing, in-memory I/O, compression, Arrow schema conversion, and Parquet schema conversion
- [x] Align integration-test manifest with `consus-io` alloc-gated `MemCursor` support
- [x] Verified `cargo nextest run -p consus-hdf5 --test roundtrip_hdf5 --no-fail-fast`
- [x] Verified `cargo test -p consus-hdf5 --test reference_hdf_group`
- [x] Verified `cargo nextest run -p consus-hdf5 --test roundtrip_hdf5 --no-fail-fast`
- [x] Verified `cargo test -p consus-hdf5 --test reference_hdf_group`

## Phase 3: Parquet + Polish

### P3.1 — Parquet Interop
- [ ] Consus ↔ Parquet schema mapping
- [ ] Read Parquet files as Consus datasets
- [ ] Write Consus datasets to Parquet
- [ ] Hybrid mode: Parquet tables inside Consus containers
- [ ] Arrow array bridge (zero-copy)

### P3.2 — Production Readiness
- [x] CI/CD pipeline (GitHub Actions)
- [x] Async I/O path via Tokio
- [ ] Memory-mapped I/O backend
- [ ] Large file (>4 GiB) regression tests
- [ ] Fuzz testing (`cargo-fuzz` / `proptest`)
- [ ] WASM target validation
- [ ] `no_std` smoke tests (`thumbv7em-none-eabihf`)
- [ ] Documentation site
- [ ] crates.io publication