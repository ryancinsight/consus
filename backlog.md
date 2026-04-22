# Consus — Backlog

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
- [x] Chunk index v4 B-tree v2 lookup
- [x] External link traversal beyond typed error reporting
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
- [x] V4 layout message emission with B-tree v2 chunk index (layout_version=4)
- [x] Chunked dataset compression roundtrip coverage (deflate, fletcher32)
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
- [x] V4 B-tree v2 chunk index roundtrip tests (2D, 3D, single-chunk)
- [x] Compressed chunked dataset roundtrip tests (deflate, fletcher32, deflate+v4)
- [ ] Comparison with `h5dump` output for verified fixtures

### P1.4 — Performance & Memory
- [x] Fill-value-aware undefined chunk reads
- [x] Parallel chunk I/O via Rayon (serial + parallel paths both verified)
- [x] Criterion benchmarks: contiguous read throughput (1 MB dataset)
- [x] Criterion benchmarks: chunked read throughput (v3 + v4 B-tree v2)
- [x] Criterion benchmarks: compressed read (deflate); zstd/lz4 deferred to codec feature expansion
- [ ] Criterion benchmarks: zstd and lz4 compressed read (blocked on HDF5 test-time feature enablement)
- [x] Allocation reduction in object-header and writer message assembly
- [ ] Comparison with HDF5 C library via `hdf5-rs`
- [ ] Comparison with Python `h5py`

## Phase 2: Zarr + netCDF-4

### P2.1 — Zarr v2
- [x] `.zarray` JSON metadata parser
- [x] `.zattrs` JSON metadata parser
- [x] `.zgroup` JSON metadata parser
- [x] Directory store implementation
- [x] Chunk read (single chunk)
- [x] Chunk read (multi-chunk)
- [x] Compression pipeline (Zarr codec chain)
- [x] Full array read with selection
- [x] Partial selection read semantics across chunk boundaries
- [x] Partial selection write semantics across chunk boundaries
- [x] Zarr v2 write path
- [x] Chunk-grid bounds validation for `read_chunk` and `write_chunk`
- [x] Repository fixtures generated from Python zarr for v2 and v3 arrays
- [x] Integration tests against Python zarr-produced fixtures
- [x] Python v2 chunk-key interoperability against Python-generated filesystem stores
- [x] Python v2 gzip full-array interoperability against Python-generated filesystem stores
- [x] Python v3 default codec-chain interoperability against Python-generated filesystem stores
- [x] Full read/write interoperability against Python zarr library output

### P2.2 — Zarr v3
- [x] `zarr.json` metadata parser
- [x] Sharding codec
- [x] v3 chunk key encoding
- [x] v3 codec pipeline
- [x] v3 write path
- [x] Interop tests with zarr-python v3

### P2.3 — netCDF-4
- [x] Dimension scale detection via HDF5 attributes
- [x] Variable → HDF5 dataset mapping
- [x] CF conventions attribute parsing
- [x] Unlimited dimension handling
- [x] Full variable byte extraction for contiguous and chunked HDF5-backed netCDF variables
- [x] DIMENSION_LIST-based variable-to-dimension binding for HDF5-backed netCDF extraction
- [x] Nested-group dimension inheritance and nearest-scope shadowing validation
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

## Phase 2.5: Datatype Mapping Completion

### P2.5a — Arrow ↔ Core Nested-Type Conversion
- [x] `core_datatype_to_arrow_hint`: Compound → Struct with recursive fields
- [x] `core_datatype_to_arrow_hint`: Array → List with correct element type
- [x] `core_datatype_to_arrow_hint`: Complex → Struct with real/imaginary children
- [x] `arrow_datatype_to_core`: Struct → Compound with recursive field conversion
- [x] `arrow_datatype_to_core`: Map → Compound with key/value fields
- [x] `arrow_datatype_to_core`: Union → Compound with variant fields
- [x] Roundtrip tests: Compound → Struct → Compound preserves field names

### P2.5b — FITS Binary Table TFORM → Core Datatype
- [x] `BinaryFormatCode` enum for all 13 FITS Standard 4.0 format codes
- [x] `parse_binary_format` TFORM string parser (repeat + code)
- [x] `binary_format_to_datatype` per-code canonical mapping
- [x] `tform_to_datatype` high-level TFORM → Datatype conversion
- [x] `binary_format_element_size` byte-width lookup
- [x] Array wrapping for repeat > 1 scalar types
- [x] Crate-root re-exports for `BinaryFormatCode` and `tform_to_datatype`
- [x] Comprehensive value-semantic tests for all 13 format codes

### P2.5c — HDF5 Datatype Class Mapping
- [x] `map_string` (fixed/variable, ASCII/UTF-8)
- [x] `map_bitfield` (→ Opaque with HDF5_bitfield tag)
- [x] `map_opaque` (with optional tag)
- [x] `map_compound` (ordered fields + size)
- [x] `map_reference` (Object/Region, size-based default)
- [x] `map_enum` (structural envelope with empty members)
- [x] `map_variable_length` (→ VarLen with base type)
- [x] `map_array` (→ Array with base + dims)
- [x] `charset_from_flags` helper (ASCII/UTF-8/unknown→ASCII)
- [x] Coverage table in module documentation
- [x] Value-semantic tests for all mapping functions

### P2.5d — FITS Column Descriptor Datatype Integration
- [x] `FitsTableColumn` extended with `datatype: Datatype` and `byte_width: usize` fields
- [x] `FitsTableColumn::new()` updated to accept `datatype` and `byte_width` parameters
- [x] `FitsTableColumn::from_binary_tform()` constructor derives `datatype` and `byte_width` from TFORM
- [x] `datatype()` and `byte_width()` accessors on `FitsTableColumn`
- [x] `parse_column` dispatches binary→`from_binary_tform`, ASCII→`FixedString` with `parse_ascii_column_width`
- [x] `FitsBinaryTableDescriptor::from_header()` validates column byte widths sum to `NAXIS1`
- [x] 5 new value-semantic tests (Boolean/Int32/Float64, Array, NAXIS1 mismatch, ASCII FixedString, Complex/Compound)
- [x] Verified `cargo test -p consus-fits --lib` (128/128)

## Phase 3: Parquet + Polish

### P3.1 — Parquet Interop
- [x] Consus ↔ Parquet schema mapping
- [ ] Read Parquet files as Consus datasets
- [ ] Write Consus datasets to Parquet
- [ ] Hybrid mode: Parquet tables inside Consus containers
- [ ] Arrow array bridge (zero-copy)

### P3.2 — FITS Table Wiring
- [x] Wire `tform_to_datatype` into `FitsTableColumn` so parsed TFORMn produces canonical `Datatype`
- [x] Per-column byte-width computation and NAXIS1 validation for binary tables
- [ ] ASCII table column value decoding (numeric/string field extraction from raw row bytes)
- [ ] Binary table column value decoding (typed element extraction per Datatype)

### P3.3 — Production Readiness
- [x] CI/CD pipeline (GitHub Actions)
- [x] Async I/O path via Tokio
- [ ] Memory-mapped I/O backend
- [ ] Large file (>4 GiB) regression tests
- [ ] Fuzz testing (`cargo-fuzz` / `proptest`)
- [ ] WASM target validation
- [ ] `no_std` smoke tests (`thumbv7em-none-eabihf`)
- [ ] Documentation site
- [ ] crates.io publication