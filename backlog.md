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
  - [x] Canonical in-memory Parquet dataset descriptor model
  - [x] Top-level column descriptors with canonical `Datatype`, storage classification, and `[total_rows]` shape
  - [x] Row-group and column-chunk descriptor validation (`row_count > 0`, schema-order chunk alignment, exact chunk cardinality)
  - [x] Ordered projection API preserving source schema order for selected top-level columns
  - [x] Value-semantic tests for fixed-width, variable-width, nested-group, projection, and invalid row-group layouts
  - [x] Nested group canonicalization to `Datatype::Compound` with ordered child fields and analytically derived fixed-size offsets
  - [x] Repeated field canonicalization to `Datatype::VarLen` for scalar and group columns
  - [x] Value-semantic tests for repeated scalar columns and repeated group columns
  - [x] Byte-level footer trailer validation (`PAR1` magic, little-endian footer length, footer offset bounds)
  - [x] Canonical footer prelude and byte-range metadata descriptors (`FooterPrelude`, `RowGroupLocation`, `ColumnChunkLocation`, `ParquetFooterDescriptor`)
  - [x] Value-semantic tests for valid trailer parsing, short input rejection, invalid magic rejection, footer-length overflow rejection, overlapping chunk rejection, and footer-bound row-group rejection
  - [x] Minimal Thrift compact binary protocol decoder (`wire::thrift::ThriftReader`): zigzag varint, i16/i32/i64, string/binary, field header, list/set/map header, recursive skip
  - [x] Canonical Parquet wire metadata types: `FileMetadata`, `SchemaElement`, `RowGroupMetadata`, `ColumnChunkMetadata`, `ColumnMetadata`, `KeyValue`
  - [x] Footer payload extraction and Thrift decoding: `decode_file_metadata(bytes, prelude)` -> `FileMetadata`
  - [x] Value-semantic tests for FileMetadata decoding from hand-computed Thrift compact byte vectors (valid decode, missing-required-field rejection)
  - [x] Canonical Parquet page header types: `PageHeader`, `DataPageHeader`, `DictionaryPageHeader`, `DataPageHeaderV2`, `PageType`
  - [x] Page header Thrift decoder: `decode_page_header(bytes)` -> `(PageHeader, consumed)`
  - [x] Value-semantic tests for page header decoding (DATA_PAGE, DICTIONARY_PAGE with is_sorted bool, DataPageHeaderV2, empty-input rejection)
  - [x] Schema reconstruction bridge: `schema_elements_to_schema` rebuilds `SchemaDescriptor` from flat pre-order DFS `SchemaElement` list (recursive group support)
  - [x] Dataset materialization bridge: `dataset_from_file_metadata(meta)` -> `ParquetDatasetDescriptor`
  - [x] Value-semantic tests for schema reconstruction and dataset bridge
  - [x] Module decomposition: `wire/thrift.rs`, `wire/metadata.rs`, `wire/page.rs`, `dataset/mod.rs` (all files under 400-line constraint)
  - [x] Physical page payload decoding and level decoding (RLE/bit-packing hybrid, deprecated BIT_PACKED, definition/repetition levels)
  - [x] `encoding/levels.rs`: `decode_levels` (RLE/bit-packing hybrid), `decode_bit_packed_raw` (deprecated BIT_PACKED), `level_bit_width` — 14 value-semantic tests
  - [x] `encoding/plain.rs`: PLAIN encoding decoders for all Parquet physical types (BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY) — 14 value-semantic tests
  - [x] `encoding/rle_dict.rs`: `decode_rle_dict_indices` (RLE_DICTIONARY, encoding ID 8) — 5 value-semantic tests
  - [x] `wire/payload.rs`: `split_data_page_v1` and `split_data_page_v2` payload splitters with `PagePayload` struct — 6 value-semantic tests
  - [x] :  enum (8 variants covering all Parquet physical types), , ,  (PLAIN/PLAIN_DICTIONARY/RLE_DICTIONARY dispatch), ,  — 16 value-semantic tests
  - [x] :  and  mapping parquet.thrift Type enum discriminants 0–7 — 2 tests
  - [x] Typed column value extraction: compression pipeline (decompress values_bytes before decoding)
  - [ ] Real file-backed dataset read API (open file -> validate -> decode footer -> materialize dataset)
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

## Phase 2.6: MATLAB .mat Format Reader (consus-mat)

### P2.6a - MAT v4 (Binary)
- [x] V4Header::parse: type field decoding (M*1000+P*10+T), LE/BE byte-order, name extraction
- [x] read_v4_variable: numeric matrix (all 6 precisions), text matrix (f64->char), complex, LE/BE normalization
- [x] read_mat_v4: sequential record parsing until EOF
- [x] Positive: v4_double_array_shape_and_values (f64, shape [2,3], 6 exact column-major values)
- [x] Negative: v4_truncated_header_returns_error
- [x] Negative: v4_empty_slice_returns_error
- [x] v4 sparse matrix explicit permanent rejection policy with fixture coverage (v4_sparse_matrix_returns_unsupported_feature_error)

### P2.6b - MAT v5 (Structured Binary)
- [x] V5FileHeader::parse: LE/BE detection via endian indicator
- [x] read_tag: standard vs small element detection, all 15 miXXXX type codes
- [x] parse_matrix: mxDOUBLE..mxUINT64 numeric, mxCHAR, mxSPARSE, mxCELL, mxSTRUCT
- [x] Complex flag detection and imaginary sub-element extraction
- [x] Logical flag detection producing MatLogicalArray
- [x] Sparse invariants enforced: `ir.len() == nzmax`, `jc.len() == ncols + 1`
- [x] Expanded synthetic coverage for char, logical, complex, sparse, and cell decoding
- [x] Unknown top-level v5 element skipping with structural validation
- [x] `mxOBJECT_CLASS` explicit permanent rejection policy with integration coverage
- [x] Compressed `miCOMPRESSED` fixture coverage across enabled/disabled feature configurations

### P2.6c - MAT v7.3 (HDF5-backed)
- [x] Root-group traversal with MATLAB_class dispatch through `consus-hdf5`
- [x] Numeric, logical, char, cell, and scalar struct decoding
- [x] Deterministic numeric ordering for cell-group children named `"0"`, `"1"`, ...
- [x] Expanded synthetic HDF5-backed coverage for logical, char, cell, and struct decoding
- [x] Non-scalar struct array decoding with authoritative shape preservation
- [x] Character decoding from dataset datatype byte order instead of hardcoded LE
- [x] Sparse v7.3 decoding or explicit permanent rejection policy
- [x] Compact-layout rejection coverage
- [x] Virtual-layout rejection coverage (DatasetLayout::Virtual in HDF5 builder; rejection test passing)
- [x] Chunked-dataset fixture coverage: v73_chunked_double_array_roundtrip passing
- [x] v7.3 cell array group roundtrip: MATLAB_class="cell" group with decimal-named child datasets; value-semantic integration coverage
- [x] v7.3 struct array group roundtrip: MATLAB_class="struct" group with field-named child datasets; value-semantic integration coverage

### P2.6d - Model, Documentation, and Release Readiness
- [x] Canonical public model types for numeric, char, logical, sparse, cell, and struct arrays
- [x] Invariant-enforcing constructors added for cell, char, logical, sparse, and struct models
- [x] Crate-level documentation updated for feature flags, entry points, contracts, and unsupported cases
- [x] Remove redundant struct field-name storage: MatStructArray.fields removed; data keys are sole SSOT; new() signature changed to (shape, data); field_names() returns impl Iterator
- [x] Add crate README with usage examples, feature matrix, and version-specific behavior notes
- [x] Add CI coverage for `cargo test -p consus-mat` and feature-matrix verification, including `miCOMPRESSED` enabled/disabled configurations
- [x] miCOMPRESSED zlib decompression (compress feature)
- [x] read_mat_v5: sequential element parsing with miMATRIX and miCOMPRESSED dispatch
- [x] Positive: v5_double_array_roundtrip (f64, shape [1,3], 3 exact values)
- [x] Negative: v5_invalid_endian_indicator_returns_error, v5_truncated_header_returns_error
- [x] consus-hdf5 Hdf5FileBuilder extended: ChildDatasetSpec + add_group_with_attributes enables nested group authoring with attached attributes for v73 fixture coverage
- [x] Model unit tests added: 42 tests across all 6 model modules covering constructors, invariant enforcement, and accessor methods
- [x] MatError Display unit tests: 5 tests covering all Display impl variants
- [x] Multi-variable v5 integration test: v5_multiple_variables_roundtrip (2 named scalar doubles, value-semantic)
- [x] loadmat file I/O test: loadmat_from_reader_parses_test_fixture (std::fs::File + test_v5.mat roundtrip)
- [x] Doc test for loadmat_bytes: MAT v4 scalar double byte sequence, verifies variable count and name
- [x] Verified cargo test -p consus-mat: 71/71 pass (42 lib + 4 v4 + 1 v5-compressed + 14 v5 + 9 v73 + 1 doc)
- [x] Verified cargo test -p consus-mat --no-default-features --features std,alloc: 62/62 pass
- [x] Verified cargo test -p consus-hdf5: 321/321 pass
- [x] Verified cargo check --workspace: zero errors

### P2.6c - MAT v7.3 (HDF5-backed)
- [x] Version detection via HDF5 file signature at byte offset 0
- [x] read_mat_v73: root group traversal, MATLAB_class attribute dispatch via consus-hdf5
- [x] Numeric arrays: contiguous+chunked payload, shape reversal (C-order to Fortran-order)
- [x] Complex arrays: compound {real, imag} field deinterleaving
- [x] Logical arrays: uint8 payload decoded to Vec<bool>
- [x] Char arrays: uint16 payload decoded to UTF-8 String
- [x] Struct arrays: group children mapped to MatStructArray
- [x] Cell arrays: group children mapped to MatCellArray
- [x] Positive: v73_double_array_roundtrip (HDF5 + MATLAB_class attr, 3 exact f64 values)

### P2.6d - Version Detection and Entry Points
- [x] detect_version: HDF5 magic -> V73, MAT v5 endian indicator -> V5, fallback -> V4
- [x] loadmat_bytes: auto-detect and dispatch to version-specific parser
- [x] loadmat<R: Read + Seek>: std-feature convenience wrapper
- [x] 5 unit tests in detect::tests module

### P2.6e - Correctness Hardening and Coverage Expansion (this sprint)
- [x] Removed dead byteorder + consus-compression deps from Cargo.toml
- [x] Removed dead UnsupportedVersion variant; fixed lib.rs doc strings
- [x] v5 sparse: ir.len()==nzmax + jc.len()==ncols+1 invariants enforced
- [x] v73 cell group: children sorted by numeric name for deterministic element order
- [x] v5 vacuous truncated test replaced with value-asserting negative test
- [x] v5 synthetic test suite: char, logical, complex, sparse, cell, struct (7 new tests)
- [x] Verified cargo test -p consus-mat: 17/17 pass (3 v4, 10 v5, 4 v73)
