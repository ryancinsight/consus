# Consus — Implementation Checklist

## Current Sprint: Phase 2 — Zarr Chunk I/O Verification

### Milestone 1: Metadata and Store Foundation
- [x] `.zarray` JSON metadata parser
- [x] `.zattrs` JSON metadata parser
- [x] `.zgroup` JSON metadata parser
- [x] Canonical `ArrayMetadata` / `GroupMetadata` conversion
- [x] In-memory store implementation
- [x] Filesystem store implementation
- [x] Prefixed and split store wrappers

### Milestone 2: Codec and Chunk Addressing
- [x] Chunk key encoding for v2 dot and v3 slash forms
- [x] Codec pipeline execution for gzip, zstd, lz4, and bytes identity
- [x] Single-chunk read path with codec decompression
- [x] Single-chunk write path with codec compression
- [x] Public chunk API re-exports from crate root

### Milestone 3: Array-Level Chunk I/O
- [x] Fill-value expansion for typed element widths
- [x] Full-array write path over chunk grid
- [x] Full-array read path over chunk grid
- [x] Multi-chunk assembly for full-array reads
- [x] Uninitialized chunk handling via fill-value materialization
- [x] Partial selection read semantics across chunk boundaries
- [x] Partial selection write semantics beyond current full-array coverage

### Milestone 4: Verification
- [x] Library unit tests for chunk key generation
- [x] Library unit tests for chunk compression/decompression roundtrip
- [x] Library unit tests for full-array write/read roundtrip
- [x] Library unit tests for chunk-grid bounds rejection in `read_chunk` and `write_chunk`
- [x] Library unit tests for contiguous partial selection reads across chunk boundaries
- [x] Library unit tests for strided partial selection reads across chunk boundaries
- [x] Library unit tests for fill-value materialization in partial selection reads over uninitialized chunks
- [x] Library unit tests for contiguous partial selection writes across chunk boundaries
- [x] Library unit tests for strided partial selection writes across chunk boundaries
- [x] Library unit tests for fill-value materialization in partial selection writes over uninitialized chunks
- [x] Library unit tests for invalid partial selection write input length rejection
- [x] Verified `cargo test -p consus-zarr --lib`
- [x] Repository Python zarr fixtures generated for v2/v3 filesystem stores
- [x] Integration tests added against Python zarr-produced fixtures
- [x] Python fixture reads fully match Python output for Zarr v2 chunk-key layout
- [x] Python fixture-backed partial write interoperability preserves pre-existing Python chunk values for Zarr v2 stores
- [x] Python fixture reads fully match Python output for Zarr v2 gzip full-array assembly
- [x] Python fixture reads fully match Python output for the remaining Python-generated Zarr v3 default codec-chain variant
- [x] Zarr v3 sharding interop coverage through high-level array API

### Milestone 5: Artifact Synchronization
- [x] `backlog.md` updated to reflect verified P2.1 chunk I/O status
- [x] `gap_audit.md` updated for Python fixture verification status, resolved v2 chunk-key and v2 gzip interoperability, and the remaining Python v3 codec-chain mismatch
- [x] `README.md` phase wording synchronized to verified Zarr implementation state

### Milestone 6: Zarr v3 Write Path
- [x] `ZarrJson::from_array_metadata` conversion (canonical → wire format)
- [x] `ZarrJson::from_group_metadata` conversion (canonical → wire format)
- [x] `write_zarr_json<S: Store>` persists array zarr.json to store
- [x] `write_group_json<S: Store>` persists group zarr.json to store
- [x] Unit tests: `from_array_metadata_roundtrip`, `from_group_metadata_roundtrip`, `write_zarr_json_persists_to_store`
- [x] Integration test: `v3_write_metadata_and_data_roundtrip` (value-semantic end-to-end)

### Milestone 7: Zarr v3 Metadata Preservation and Fill Semantics
- [x] Canonical `ArrayMetadata` extended with optional `dimension_names`
- [x] `ZarrJson::to_array_canonical` preserves v3 `dimension_names`
- [x] `ZarrJson::from_array_metadata` preserves v3 `dimension_names`
- [x] `ZarrJson::from_group_metadata` preserves group attributes on the write path
- [x] Float fill-value expansion corrected for `f32` and `f64` element widths
- [x] Unit tests: `expand_fill_value_float32_one`, `expand_fill_value_float64_one`, dimension-name roundtrip, group-attribute roundtrip

### Milestone 8: netCDF-4 HDF5 Integration Phase 1
- [x] `src/hdf5/dimension_scale/mod.rs`: `is_dimension_scale`, `dimension_name_from_attrs` (7 unit tests)
- [x] `src/hdf5/variable/mod.rs`: `build_variable` (5 unit tests)
- [x] `src/hdf5/group/mod.rs`: `extract_group<R: ReadAt + Sync>` group traversal
- [x] `src/lib.rs`: `pub mod hdf5`, re-exports `extract_group`, `is_dimension_scale`
- [x] `tests/integration_netcdf_hdf5.rs`: 6 value-asserting integration tests
- [x] Verified `cargo nextest run -p consus-netcdf` (90/90)

### Milestone 9: netCDF-4 Semantic Model Enrichment
- [x] `NetcdfVariable` extended with decoded attributes and optional HDF5 object-header address
- [x] `NetcdfGroup` extended with group-level attributes and child-group lookup
- [x] HDF5 dataset attributes propagated into canonical netCDF variable descriptors
- [x] HDF5 group attributes propagated into canonical netCDF group descriptors
- [x] Unlimited HDF5 extents propagated into `NetcdfDimension::unlimited`
- [x] CF convention constant coverage expanded (`add_offset`, `scale_factor`, `missing_value`, `valid_range`, `valid_min`, `valid_max`, `calendar`, `positive`, `formula_terms`, `ancillary_variables`, `flag_values`, `flag_meanings`, `flag_masks`, `compress`)
- [x] Empty stale `src/core/` scaffolding removed from `consus-netcdf`
- [x] Verified `cargo test -p consus-netcdf` (91/91 including new semantic-model assertions)

### Milestone 10: netCDF-4 HDF5 Variable Byte-Read Bridge
- [x] `read_variable_bytes<R: ReadAt + Sync>` added under `src/hdf5/variable/mod.rs`
- [x] Contiguous HDF5-backed netCDF variables read through `Hdf5File::read_contiguous_dataset_bytes`
- [x] Chunked HDF5-backed netCDF variables read through `Hdf5File::read_chunked_dataset_all_bytes`
- [x] Missing `object_header_address` rejected with deterministic `InvalidFormat`
- [x] Compact and virtual HDF5 layouts rejected with explicit `UnsupportedFeature`
- [x] Crate-root and `hdf5` module re-exports added for the byte-read bridge
- [x] Unit tests added for contiguous reads, chunked reads, and missing-address rejection
- [x] Integration tests added for contiguous and chunked variable byte roundtrips with exact byte assertions
- [x] Verified `cargo test -p consus-netcdf` (97/97 including byte-read bridge coverage)

### Milestone 11: netCDF-4 DIMENSION_LIST Binding
- [x] `DIMENSION_LIST` object-reference decoding added under `src/hdf5/dimension_scale/mod.rs`
- [x] Variable axis binding resolves referenced dimension-scale names in declaration order when `DIMENSION_LIST` is present and valid
- [x] Conservative synthetic fallback (`d0`, `d1`, ...) preserved when `DIMENSION_LIST` is absent or invalid
- [x] Group extraction now builds an authoritative dimension-scale address → name mapping before variable construction
- [x] Unit tests added for object-reference decoding, missing-reference rejection, short-payload rejection, and axis-order preservation
- [x] Integration fallback coverage retained for variables without `DIMENSION_LIST`
- [x] Verified `cargo test -p consus-netcdf` (105/105 including DIMENSION_LIST binding coverage)

### Milestone 12: netCDF-4 Nested-Group Dimension Scope
- [x] `NetcdfGroup::validate` now accepts variable dimension references declared in ancestor scopes
- [x] Duplicate dimension names remain rejected within a single scope
- [x] Child-group local dimensions shadow ancestor dimensions during scoped resolution
- [x] Public `NetcdfGroup::resolve_dimension` added for deterministic nearest-scope lookup
- [x] Model tests added for inherited-dimension acceptance, missing inherited-dimension rejection, and local-shadowing preference
- [x] Reference tests added for nested-group inherited-dimension validation and shadowed-dimension resolution
- [x] Roundtrip tests added for nested-group dimension inheritance and shadowing
- [x] Verified `cargo test -p consus-netcdf` (113/113 including nested-group scope coverage)

### Milestone 13: Parquet Schema Mapping Verification
- [x] Canonical Parquet physical, logical, field, and hybrid schema modules remain the authoritative SSOT for `consus-parquet`
- [x] `cargo test -p consus-parquet --lib` passes with 31/31 tests
- [x] Core datatype mapping now covers `Boolean`, integer, float, complex, fixed string, variable string, opaque, compound, array, enum, varlen, and reference variants
- [x] Logical annotation mapping now covers `String`, `Enum`, `Decimal`, `Date`, `Time`, `Timestamp`, `Integer`, `UnsignedInteger`, `Json`, `Bson`, and `Uuid`
- [x] Compatibility analysis preserves field identity, nullability, and width-sensitive widening semantics
- [x] Exhaustive datatype coverage test added for compound, array, enum, and varlen mappings
- [x] Verified `cargo test -p consus-parquet --lib`

### Milestone 14: Parquet Interop Expansion
- [ ] Read Parquet files as Consus datasets
- [ ] Write Consus datasets to Parquet
- [ ] Hybrid mode: Parquet tables inside Consus containers
- [ ] Arrow array bridge (zero-copy)

### Milestone 15: Arrow ↔ Core Nested-Type Conversion

- [x] `core_datatype_to_arrow_hint`: Compound → Struct with recursive fields
- [x] `core_datatype_to_arrow_hint`: Array → List with correct element type
- [x] `core_datatype_to_arrow_hint`: Complex → Struct with real/imaginary children
- [x] `arrow_datatype_to_core`: Struct → Compound with recursive field conversion
- [x] `arrow_datatype_to_core`: Map → Compound with key/value fields
- [x] `arrow_datatype_to_core`: Union → Compound with variant fields
- [x] Roundtrip test: Compound → Struct → Compound preserves field names
- [x] Verified `cargo test -p consus-arrow --lib` (41/41)

### Milestone 16: FITS Binary Table TFORM → Core Datatype

- [x] `BinaryFormatCode` enum for all 13 FITS Standard 4.0 format codes
- [x] `parse_binary_format` TFORM string parser (repeat + code)
- [x] `binary_format_to_datatype` per-code canonical mapping
- [x] `tform_to_datatype` high-level TFORM → Datatype conversion
- [x] `binary_format_element_size` byte-width lookup
- [x] Array wrapping for repeat > 1 scalar types
- [x] Crate-root re-exports for `BinaryFormatCode` and `tform_to_datatype`
- [x] Comprehensive value-semantic tests for all 13 format codes
- [x] Verified `cargo test -p consus-fits --lib` (123/123)

### Milestone 17: HDF5 Datatype Class Mapping

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
- [x] Verified `cargo test -p consus-hdf5 --lib` (263/263)

### Milestone 18: Production Readiness
- [ ] Memory-mapped I/O backend
- [ ] Large file (>4 GiB) regression tests
- [ ] Fuzz testing (`cargo-fuzz` / `proptest`)
- [ ] WASM target validation
- [ ] `no_std` smoke tests (`thumbv7em-none-eabihf`)
- [ ] Documentation site
- [ ] crates.io publication

### Milestone 19: FITS Column Descriptor Datatype Integration
- [x] `FitsTableColumn` extended with `datatype: Datatype` and `byte_width: usize` fields
- [x] `FitsTableColumn::new()` updated to accept `datatype` and `byte_width` parameters
- [x] `FitsTableColumn::from_binary_tform()` constructor derives `datatype` and `byte_width` from TFORM via `tform_to_datatype` and `binary_format_element_size`
- [x] `datatype()` and `byte_width()` accessors added to `FitsTableColumn`
- [x] `parse_column` updated to accept `binary: bool` flag; binary columns use `from_binary_tform`, ASCII columns use `FixedString` with `parse_ascii_column_width`
- [x] `parse_ascii_column_width` helper extracts field width from ASCII table TFORM values (`Aw`, `Iw`, `Fw.d`, `Ew.d`, `Dw.d`)
- [x] `FitsBinaryTableDescriptor::from_header()` validates that per-column byte widths sum to `NAXIS1`, returning `InvalidFormat` on mismatch
- [x] Existing binary table test corrected: NAXIS1=8 (1J=4 + 1E=4), logical_data_len=48 (8×4 + 16)
- [x] Existing `FitsTableColumn::new()` call sites updated with `datatype` and `byte_width` parameters
- [x] New test: `binary_column_datatype_matches_tform` (1L→Boolean, 1J→Int32, 1D→Float64)
- [x] New test: `binary_column_array_tform_produces_array_datatype` (3E→Array{Float32,[3]})
- [x] New test: `binary_table_rejects_naxis1_mismatch` (NAXIS1=99 vs 1J=4)
- [x] New test: `ascii_column_datatype_is_fixed_string` (A8→FixedString{8,Ascii}, E16.7→FixedString{16,Ascii})
- [x] New test: `binary_column_complex_and_descriptor_types` (1C→Complex32, 1M→Complex64, 1P→Compound)
- [x] Verified `cargo test -p consus-fits --lib` (128/128)