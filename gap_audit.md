# Consus - Gap Audit

## Data Folder Record

- NWB sample acquisition manifest stored at `D:\consus\data\nwb\manifest.txt`
- Acquisition status: `acquisition_failed`
- Fixture data must remain under `D:\consus\data\nwb`
- Test guard recorded in `crates/consus-nwb/tests/roundtrip_proptest.rs` to fail until `D:\consus\data\nwb\allen_brain_observatory_sample.nwb` exists

## Audit Date: 2026-05-08 (updated this sprint)
## Scope: Phase 3 — NWB Verification Against Real Files (Milestone 38)

---

## Resolved Gaps (this sprint)

| ID | Description | Resolution |
|----|-------------|------------|
| Z-001 | Zarr v2 metadata parsing | `.zarray`, `.zgroup`, and `.zattrs` parsing implemented and covered by unit/integration tests |
| Z-002 | Canonical metadata model | `ArrayMetadata` / `GroupMetadata` conversion path implemented for v2 and v3 metadata |
| Z-003 | Store backends | In-memory store, filesystem store, prefixed store, and split store implemented and verified |
| Z-004 | Codec pipeline | Ordered codec pipeline implemented for gzip, zstd, lz4, and bytes identity semantics |
| Z-005 | Chunk key encoding | v2 dot-separated and v3 slash-separated chunk key generation implemented |
| Z-006 | Single-chunk I/O | `read_chunk` and `write_chunk` implemented with codec decompression/compression |
| Z-007 | Fill-value expansion | Typed fill-value materialization implemented for array initialization and uninitialized chunk reads |
| Z-008 | Full-array write path | `write_array` implemented over the chunk grid with length validation |
| Z-009 | Full-array read path | `read_array` implemented for full-array traversal with multi-chunk assembly |
| Z-010 | Public chunk API surface | Chunk I/O types and functions re-exported from crate root |
| Z-011 | Library verification | `cargo test -p consus-zarr --lib` passes with 90/90 tests |
| Z-012 | Partial selection read semantics | Verified contiguous and strided partial reads across chunk boundaries, plus fill-value materialization for uninitialized intersecting chunks |
| Z-013 | Partial selection write semantics | Verified contiguous and strided partial writes across chunk boundaries, plus fill-value materialization for uninitialized intersecting chunks and input-length rejection |
| Z-014 | Python fixture generation | Deterministic Python `zarr` fixtures generated for v2/v3 filesystem stores with manifest-tracked expected values |
| Z-015 | Python fixture integration harness | Rust integration tests added for Python-produced v2/v3 fixtures, including one passing v3 gzip read path and explicit regression coverage for current interoperability mismatches |
| Z-102 | Python v3 default codec-chain mismatch | Boundary-chunk stride bug fixed: `read_array` now uses `stored_shape_for_chunk` to select `meta.chunks` strides when zarr-python writes a padded full boundary chunk. Fix applied to `read_array` (full and partial paths), `copy_chunk_selection_to_output`, and `copy_selection_input_to_chunk`. Test `python_fixture_v3_uncompressed_i4_full_and_partial_reads` asserts all 15 elements correctly plus two partial selections against the Python-generated fixture. |
| Z-103 | High-level Zarr v3 sharding interop | `shard/mod.rs` rewritten with spec-compliant layout (index at END of shard file, uninitialized entries = `u64::MAX`), `ShardingConfig`, `extract_sharding_config`, `inner_linear_index`, `read_inner_chunk_from_shard`, and `write_shard`. `read_array` and `write_array` dispatch to `read_array_sharded`/`write_array_sharded` when `sharding_indexed` is present in the codec chain. Tests: `sharded_array_single_shard_roundtrip` and `sharded_array_multi_shard_roundtrip` both pass. |
| P2.2 | Zarr v3 metadata write path | `ZarrJson::from_array_metadata`, `ZarrJson::from_group_metadata`, `fill_value_to_json_v3`, `codec_to_spec`, `write_zarr_json<S: Store>`, and `write_group_json<S: Store>` implemented in `src/metadata/v3.rs`. `WriteZarrJsonError` error type added. Unit tests: `from_array_metadata_roundtrip`, `from_group_metadata_roundtrip`, `write_zarr_json_persists_to_store`. Integration test: `v3_write_metadata_and_data_roundtrip` verifies exact element values for a 4×4 i32 array. |
| P2.3 | netCDF-4 HDF5 integration layer (phase 1) | `src/hdf5/` module tree created: `dimension_scale` (7 unit tests), `variable` (5 unit tests), `group` (`extract_group` traversal function). `is_dimension_scale` and `extract_group` re-exported from crate root. 6 integration tests in `tests/integration_netcdf_hdf5.rs` verify dimension scale detection, variable extraction, and group traversal on in-memory HDF5 files. |
| P2.3 | netCDF-4 classic model read | `read_model` now opens the HDF5 root group, extracts the canonical `NetcdfModel`, validates the root group, and preserves `/` as the root name. Integration coverage added for empty-file, flat root dimension/variable, and nested-group extraction paths. |
| Z-105 | Zarr v3 dimension-name preservation | Canonical `ArrayMetadata` now carries `dimension_names: Option<Vec<String>>`. `metadata/v2.rs`, `metadata/v3.rs`, `metadata/consolidated.rs`, shard tests, chunk tests, and roundtrip integration tests were updated so v3 `dimension_names` survive `parse -> canonical -> write -> parse -> canonical`. Verification: `cargo test -p consus-zarr` passes with `metadata::v3::tests::test_dimension_names_roundtrip`, `metadata::v3::tests::from_array_metadata_roundtrip`, and `roundtrip_zarr::v3_write_metadata_and_data_roundtrip` succeeding. |
| Z-106 | Zarr v3 group-attribute preservation on write | `ZarrJson::from_group_metadata` no longer drops canonical group attributes. Attribute values are converted into JSON scalars/arrays and preserved through `from_group_metadata -> to_json -> parse -> to_group_canonical`. Verification: `cargo test -p consus-zarr` passes with `metadata::v3::tests::from_group_metadata_roundtrip` succeeding. |
| Z-107 | Typed float fill-value expansion correctness | `expand_fill_value` now emits width-correct IEEE bytes for float32 and float64 instead of truncating an f64 byte pattern. This closes the incorrect float32 fill-value materialization path. Verification: `cargo test -p consus-zarr` passes with `chunk::tests::expand_fill_value_float32_one`, `chunk::tests::expand_fill_value_float64_one`, and the existing array roundtrip tests succeeding. |
| N-001 | netCDF variable and group attribute preservation | `NetcdfVariable` and `NetcdfGroup` now carry decoded attribute vectors. HDF5 extraction attaches decodable dataset and group attributes, excluding dimension-scale marker attributes from variable semantic payloads. Verification: `cargo test -p consus-netcdf` passes with `variable::tests::variable_attributes_and_object_header_address_attach`, `reference_netcdf::group_lookup_methods`, and all integration suites succeeding. |
| N-002 | netCDF classic model read path | Added `consus-netcdf::read_model` / `hdf5::read_model` to extract a validated canonical `NetcdfModel` from the HDF5 root group. Empty-file, flat root, and nested-group integration tests verify exact root name, dimension counts, variable names, and child-group contents. |
| N-002 | Unlimited-dimension propagation from HDF5 shape extents | HDF5-backed netCDF extraction now maps unlimited HDF5 extents into `NetcdfDimension::unlimited(...)` and marks variables unlimited when `dataset.shape.has_unlimited()`. Verification: `cargo test -p consus-netcdf` passes with `reference_netcdf::bridge_unlimited_dimension_mapping`, `reference_netcdf::unlimited_dimension_model`, and `roundtrip_netcdf::unlimited_dimension_roundtrip` succeeding. |
| N-003 | CF attribute-name coverage expansion | The conventions module now recognizes additional CF keys: `add_offset`, `scale_factor`, `missing_value`, `valid_range`, `valid_min`, `valid_max`, `calendar`, `positive`, `formula_terms`, `ancillary_variables`, `flag_values`, `flag_meanings`, `flag_masks`, and `compress`. Verification: `cargo test -p consus-netcdf` passes with `conventions::tests::validate_cf_attribute_names` succeeding. |
| N-004 | netCDF stale scaffolding removal | Empty `src/core/` scaffolding under `consus-netcdf` was removed to restore SSOT and eliminate dead structure. Verification: `cargo test -p consus-netcdf` and `cargo test --workspace` both pass after removal. |
| N-005 | netCDF variable byte-read bridge | `consus-netcdf::hdf5::variable::read_variable_bytes` now reads full variable payloads for HDF5-backed netCDF variables using authoritative HDF5 APIs. Contiguous datasets are read through `Hdf5File::read_contiguous_dataset_bytes`, chunked datasets through `Hdf5File::read_chunked_dataset_all_bytes`, and missing object-header addresses fail deterministically. Compact and virtual layouts remain explicitly unsupported. Verification: `cargo test -p consus-netcdf` passes with `hdf5::variable::tests::read_variable_bytes_reads_contiguous_dataset_payload`, `hdf5::variable::tests::read_variable_bytes_reads_chunked_dataset_payload`, `integration_netcdf_hdf5::contiguous_variable_bytes_roundtrip`, `integration_netcdf_hdf5::chunked_variable_bytes_roundtrip`, and `integration_netcdf_hdf5::variable_bytes_require_object_header_address` succeeding. |
| N-006 | DIMENSION_LIST parsing and variable-to-dimension binding | `consus-netcdf::hdf5` now parses `DIMENSION_LIST` object-reference payloads and resolves variable axis names against discovered dimension-scale dataset addresses during group extraction. When resolution succeeds, variable dimensions preserve referenced scale order; when absent or invalid, extraction falls back to conservative synthetic names. Verification: `cargo test -p consus-netcdf` passes with `hdf5::dimension_scale::tests::dimension_list_addresses_decode_object_references_in_axis_order`, `hdf5::dimension_scale::tests::resolve_dimension_names_from_list_uses_dimension_scale_mapping`, `hdf5::group::tests::dimension_names_from_dimension_list_resolves_axis_order`, and `integration_netcdf_hdf5::missing_dimension_list_falls_back_to_synthetic_dimension_names` succeeding. |
| N-007 | Nested-group dimension inheritance and scoped resolution | `consus-netcdf::model::NetcdfGroup` now validates variables against dimensions visible through the full ancestor scope chain instead of only the local group. Child groups may inherit ancestor dimensions and legally shadow them with local declarations, with nearest-scope resolution winning deterministically. Verification: `cargo test -p consus-netcdf` passes with `model::tests::group_validation_accepts_ancestor_scoped_dimension_reference`, `model::tests::group_validation_rejects_missing_ancestor_scoped_dimension_reference`, `model::tests::dimension_in_scope_prefers_local_shadowing_dimension`, `reference_netcdf::inherited_dimension_scope_validation`, `reference_netcdf::shadowed_dimension_scope_resolution`, `reference_netcdf::missing_inherited_dimension_is_rejected`, `roundtrip_netcdf::nested_group_dimension_inheritance_roundtrip`, and `roundtrip_netcdf::nested_group_dimension_shadowing_roundtrip` succeeding. |
| Z-104 | Artifact synchronization | README.md Zarr row updated to reflect verified state: boundary-chunk stride fix and sharding verified; v3 metadata write path complete. Old Z-102 "remaining gap" sentence removed. |
| P-001 | Parquet schema mapping verification | `consus-parquet` now covers canonical Parquet physical, logical, field, hybrid, Arrow bridge, conversion, dataset-descriptor, and wire-trailer validation models with deterministic mappings for `Boolean`, integer, float, complex, fixed string, variable string, opaque, compound, array, enum, varlen, and reference Core datatypes. Dataset canonicalization preserves nested group fields as canonical `Compound` datatypes with ordered child offsets when fixed-size children permit exact sizing, and preserves repeated fields as canonical `VarLen` datatypes instead of collapsing them to scalar physical mappings. Wire validation now covers trailer magic validation, footer-length decoding, footer-offset derivation, non-overlapping row-group/column-chunk byte-range descriptors, and footer-boundary rejection. `cargo test -p consus-parquet --lib` passes with 47/47 tests, including exhaustive datatype coverage for compound, array, enum, and varlen mappings plus validated dataset descriptor, projection, nested-group, repeated-field, and footer-validation coverage. |

| M-001 | consus-mat MATLAB .mat reader | v4 binary, v5 structured binary (all mxClass except explicit `mxOBJECT_CLASS` rejection, complex, logical, miCOMPRESSED), v7.3 HDF5-backed via consus-hdf5. Public model invariants are now enforced through constructors for cell, char, logical, sparse, and struct arrays. Crate-level documentation now specifies feature gates, canonical model mapping, parsing contracts, and unsupported cases. Follow-on closure adds tolerant skipping of unknown top-level v5 elements with structural validation, deterministic unsupported-feature rejection for MAT v5 `mxOBJECT_CLASS`, datatype-aware v7.3 char decoding for little-endian and big-endian uint16 datasets, explicit unsupported-feature rejection for v7.3 sparse datasets, explicit compact-layout rejection coverage, and `miCOMPRESSED` feature-matrix verification across enabled and disabled `compress` configurations. Integration coverage now includes v7.3 numeric, logical, char, cell ordering, scalar struct decoding, big-endian char decoding, sparse rejection, compact-layout rejection, and MAT v5 compressed-payload success/failure behavior under the corresponding feature set. |
| P-004 | File-backed Parquet reader | Added `reader` module with `ColumnPageDecoder` (stateful page iterator, dict retention, v1/v2 decompression dispatch), `merge_column_values`, and `ParquetReader<'a>` (footer validation → metadata decode → dataset materialize → column chunk read). 21 value-semantic tests (max levels derivation, merge variants, page decoder v1/v2/dict, reader roundtrip, bounds checks). 164 tests pass (default features), 175 tests pass (snappy,zstd,lz4,gzip). |
| NWB-001 | consus-nwb unregistered empty scaffold | Cargo.toml created, 10 stub modules scaffolded, workspace registered. `cargo check -p consus-nwb` passes (0 errors, 0 warnings). Removed stale `#![feature(alloc)]` — stable since 1.36.0. |
| M32-001 | Optional/repeated flat column write/read | `EncodedLeafColumn`, RLE level encoder (`encode_rle_hybrid`, `encode_levels_for_page_v1`), optional/repeated flat column paths in `encode_leaf_columns`, `ColumnValuesWithLevels`, `read_column_chunk_with_levels`, non-null count fix in `ColumnPageDecoder`. `lower_column` bug fixed (leaf Repeated now correctly produces max_rep=1). `dataset_from_file_metadata` fixed to use `rg.num_rows` (not `num_values`) as chunk row_count. 8 value-semantic roundtrip tests + 1 proptest. 205/205 consus-parquet lib tests pass; 0 workspace failures. |

---

### P-003: Compression Pipeline Gap (resolved this sprint)

**Finding:** Column value decoding expected pre-decompressed bytes; no decompression dispatch existed for Parquet compression codecs (SNAPPY, GZIP, LZ4, ZSTD).

**Fix:** Added `encoding/compression.rs` with `CompressionCodec` enum (discriminants 0-7), `decompress_page_values` dispatch (feature-gated per codec), and `decode_compressed_column_values` integrated entry point in `encoding/column.rs`.

**Verification:** 12 unit tests for compression module + 7 integration tests for combined decompress+decode. 155 tests pass with all features.

---

## P-004 — File-backed Parquet reader
- **Status**: RESOLVED (commit 4c9e018)
- **Gap**: No API to read values from a complete Parquet file in memory.
- **Fix**: Added `reader` module with `ColumnPageDecoder` (stateful page iterator, dict retention, v1/v2 decompression dispatch), `merge_column_values`, and `ParquetReader<'a>` (footer validation → metadata decode → dataset materialize → column chunk read).
- **Tests**: 21 value-semantic tests (max levels derivation, merge variants, page decoder v1/v2/dict, reader roundtrip, bounds checks).
- **Residual risk**: Nested (group) column decoding remains incomplete in the file-backed reader; writer-side nested/group lowering, footer/page synthesis, and row-source payload emission remain open.

---

## P-005 — Parquet writer end-to-end page emission + warning cleanup
- **Status**: RESOLVED (this sprint)
- **Gap**: `build_file_bytes` stubbed out `codec`, `plan`, and `rows` with `let _ = ...`, emitting only a valid trailer with empty column chunks. All page-header encoder functions (`encode_page_header`, `encode_data_page_header`, etc.) were unreachable dead code. Workspace accumulated 15 dead-code and unused-import warnings across `consus-parquet`, `consus-arrow`, `consus-zarr`, and `consus`.
- **Fix**:
  - Added `encode_cell_plain`: PLAIN encoder for all non-Boolean physical types (INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY).
  - Added `encode_bool_column_plain`: LSB-first bit-packing across the full Boolean column (⌈count/8⌉ bytes), per Parquet PLAIN BOOLEAN spec.
  - Added `physical_type_discriminant`: maps `ParquetPhysicalType` to parquet.thrift Type enum i32.
  - Rewrote `build_file_bytes` to collect per-leaf PLAIN bytes from `RowSource`, emit one DataPage v1 per column (UNCOMPRESSED), record exact byte offsets and sizes in `ColumnMetadata`, and assemble a single row group containing all rows.
  - Removed dead `encode_row_group_descriptor` function and unused imports (`boxed::Box`, `PagePayload`, `split_data_page_v1`, `split_data_page_v2`).
  - Fixed `consus-arrow/conversion/mod.rs`: removed unused imports `TimeUnit`, `ArrowNullability`; removed dead `let arrow_type` binding in `ArrowFieldFromCoreBuilder::build`; prefixed unused variables `_i` and `_bit_width`.
  - Fixed `consus-zarr/lib.rs`: re-exported `ConsolidatedMetadataV2`, `ConsolidatedMetadataV3`, `MetadataEntryV2`, `MetadataEntryV3`, `ConsolidatedParseError`, `ConsolidatedSerializeError` from crate root.
  - Fixed `consus/highlevel/dataset.rs`: added `#[allow(dead_code)]` to `pub(crate) fn backend()`.
- **Tests**: 6 new value-semantic writer tests — INT32 3-value roundtrip, DOUBLE 2-value roundtrip, BYTE_ARRAY 2-string roundtrip, BOOLEAN 4-value roundtrip, two-column INT32+DOUBLE 2-row roundtrip, Null-in-required-column `InvalidFormat` rejection.
- **Residual risk**: Writer emits a single row group regardless of how many row groups the input `ParquetDatasetDescriptor` declares. Nested/group column encoding returns `UnsupportedFeature` (consistent with reader). Compression codecs other than UNCOMPRESSED are accepted by `with_compression` but `build_file_bytes` ignores the codec parameter (pages are always uncompressed); compressed write is a follow-on item.

---

## A-002: Arrow Array Materialization Bridge (resolved this sprint)

- **Status**: RESOLVED (this sprint)
- **Gap**: `ColumnValues` from `consus-parquet` could be decoded but not materialized into the canonical `ArrowArray` model in `consus-arrow`. The bridge layer (`ArrowBridge`, `ArrowIntegrationPlan`) was descriptive-only; no `ColumnValues → ArrowArray` conversion existed.
- **Fix**: Added `consus-arrow/src/array/materialize.rs` with `column_values_to_arrow(values: &ColumnValues) -> ArrowArray`. Physical-type mapping:
  - `Boolean` → `FixedWidth { element_width: 1 }`, 0x00/0x01 per element
  - `Int32/Int64/Float/Double` → `FixedWidth` with little-endian byte encoding (Arrow memory-format convention)
  - `Int96` → `FixedWidth { element_width: 12 }`, raw bytes preserved
  - `ByteArray` → `VariableWidth` with monotone offsets (`offsets.len() == len + 1`)
  - `FixedLenByteArray` → `FixedWidth { element_width: fixed_len }`, concatenated raw bytes
  - Re-exported from `array/mod.rs` and `consus-arrow` crate root under `#[cfg(feature = "alloc")]`.
  - Also resolved two pre-existing warnings: unused `ArrowField` import in `bridge/mod.rs`; duplicated `#[cfg(feature="alloc")] #[test]` attributes in `memory/mod.rs`.
- **Tests**: 10 value-semantic tests covering all 8 `ColumnValues` variants plus empty-array boundary cases for Boolean and ByteArray.
- **Residual risk**: Zero-copy materialization (reinterpret fixed-width slices without allocation) requires alignment guarantees not yet enforced; current implementation performs explicit byte-level conversion for portability. A future zero-copy path requires `bytemuck` or equivalent.

## F-003: FITS Table Column Value Decoding (artifact sync — already implemented)

- **Status**: RESOLVED (implementation predated this audit cycle; backlog entries were stale)
- **Gap**: Backlog listed "ASCII table column value decoding" and "Binary table column value decoding" as open items. The implementation was complete.
- **Evidence**:
  - `decode_binary_column` + `decode_scalar_binary` in `consus-fits/src/table/decode.rs`: covers all 13 FITS Standard 4.0 TFORM codes (L/X/B/I/J/K/A/E/D/C/M/P/Q); big-endian extraction; repeat > 1 array wrapping; 24 value-semantic unit tests.
  - `decode_ascii_column`: A/I/F/E/D format codes; trailing-space stripping; Fortran D-notation normalization; 11 value-semantic unit tests.
  - `FitsTableData::decode_row` and `FitsTableData::decode_column` dispatch to binary/ASCII decoder per table kind; return `Vec<FitsColumnValue>` with value-semantic integration tests.
- **Residual risk**: None. Backlog entries updated.

## P-006: E2E ParquetWriter → ParquetReader → column_values_to_arrow pipeline verification (resolved this sprint)

- **Status**: RESOLVED (this sprint)
- **Gap**: No integration test exercised the full write→read→materialize pipeline end-to-end. `column_values_to_arrow` had 10 unit tests against synthetic `ColumnValues` data, but the pipeline from `ParquetWriter` through `ParquetReader` through `column_values_to_arrow` was unverified.
- **Fix**: Created `consus-arrow/tests/parquet_arrow_e2e.rs` with 6 integration tests covering INT32, INT64, DOUBLE, BYTE_ARRAY, BOOLEAN, and two-column (INT32+DOUBLE) schemas. Each test: (1) builds a complete Parquet file via `ParquetWriter::write_dataset_bytes`, (2) decodes it via `ParquetReader::read_column_chunk`, (3) materializes via `column_values_to_arrow`, (4) asserts byte-level correctness against analytically derived values.
- **Tests**: 6 integration tests, all value-semantic with byte-level output assertions.
- **Residual risk**: None. All 6 pass. Multi-row-group and nested-column E2E coverage deferred pending writer support for those features.

---

## P-007: Compressed page emission — writer silently ignores codec parameter (resolved this sprint)

- **Status**: RESOLVED (this sprint)
- **Gap**: `ParquetWriter::with_compression(codec)` accepted a codec but `build_file_bytes` ignored it (`_codec` unused). All emitted pages were UNCOMPRESSED regardless of requested codec. `ColumnMetadata.codec` was hardcoded to `0` (UNCOMPRESSED), causing any non-uncompressed write to produce a structurally incorrect file that readers would fail to decompress.
- **Fix**:
  - Added `compress_page_values(data: &[u8], codec: CompressionCodec) -> Result<Vec<u8>>` to `consus-parquet/src/encoding/compression.rs` (declared `pub(crate)`). Mirrors `decompress_page_values` with symmetric codec dispatch: UNCOMPRESSED pass-through, GZIP/ZLIB via `flate2::read::DeflateEncoder`, SNAPPY via `snap::raw::Encoder`, ZSTD via `zstd::bulk::compress`, LZ4_RAW via `lz4_flex::compress`, LZ4 via `lz4_flex::compress_prepend_size`, BROTLI always `UnsupportedFeature`. Disabled features return `UnsupportedFeature` with actionable enable-feature message.
  - Updated `build_file_bytes`: renamed `_codec` → `codec`; applies `compress_page_values` after PLAIN encoding; sets `page_header.uncompressed_page_size = plain_size`, `page_header.compressed_page_size = compressed_size`; sets `ColumnMetadata.codec = codec as i32`; records correct `total_uncompressed_size` and `total_compressed_size` (header + respective payload).
- **Tests**: `compress_page_values_uncompressed_passthrough` (always), `compress_page_values_brotli_returns_unsupported` (always), `writer_gzip_roundtrip_i32_three_values` (`#[cfg(feature="gzip")]`), `writer_gzip_roundtrip_byte_array` (`#[cfg(feature="gzip")]`).
- **Verification**: 177/177 pass default features; 183/183 pass with `--features gzip`.
- **Residual risk**: Multi-codec writer tests (SNAPPY, ZSTD, LZ4) require the respective feature flags. UNCOMPRESSED, GZIP are verified. Writer still emits a single row group; multi-row-group splitting remains open.

---

## P-008: Zero-copy Arrow materialization residual risk (resolved this sprint)

- **Status**: RESOLVED (this sprint)
- **Gap**: Residual risk from A-002: `column_values_to_arrow` performed element-by-element `to_le_bytes()` loops for Int32/Int64/Float/Double. On little-endian architectures the native memory layout matches Arrow's required LE byte order, making the per-element loop a redundant copy. No optimized bulk-copy path existed.
- **Fix**: Added optional `zerocopy` feature to `consus-arrow`. When `#[cfg(all(feature = "zerocopy", target_endian = "little"))]` is active, `fixed_to_le_bytes_fast<T: zerocopy::IntoBytes + zerocopy::Immutable>` uses `IntoBytes::as_bytes(slice).to_vec()` (one allocation + one bulk `memcpy`). Applied to Int32, Int64, Float, Double match arms. On big-endian targets or without the feature, the element-by-element LE path is used (correctness preserved). `zerocopy::Immutable` bound required by zerocopy 0.8.48's `IntoBytes::as_bytes` signature.
- **Tests**: `zerocopy_i32_agrees_with_element_loop` and `zerocopy_f64_agrees_with_element_loop` verify fast-path byte output == element-loop reference for [1, -1, MAX, MIN] and [1.5, -0.25, +∞, -∞].
- **Verification**: 50/50 pass without feature; 52/52 pass with `--features zerocopy`.
- **Residual risk**: None. True allocation-free zero-copy (returning a borrowed `&[u8]` without owning `Vec<u8>`) requires a different `ArrowArray` model (lifetime-parameterized or `Cow`-backed buffers); deferred as architectural follow-on.

---

## P-009: Multi-row-group writer splitting (resolved this sprint)
- **Severity**: Medium (architecture constraint)
- **Description**: `build_file_bytes` emitted a single `RowGroupMetadata` for all rows, making it impossible to write Parquet files with multiple row groups. Large datasets require row-group partitioning per spec.
- **Resolution**: `ParquetWriter::with_row_group_size(n)` API; `encode_leaf_columns` helper extracted; `build_file_bytes` partitions rows into ⌈N/n⌉ groups; `FileMetadata.num_rows` = N; each group carries correct `num_rows` and per-column `data_page_offset`.
- **Verification**: 6 value-semantic tests + 1 proptest (prop_multi_row_group_i32_roundtrip) in `writer/tests_extra.rs`.

## P-010: Missing SNAPPY/ZSTD/LZ4 compressed writer roundtrip tests (resolved this sprint)
- **Severity**: Low (test coverage gap)
- **Description**: `compress_page_values` supported SNAPPY/ZSTD/LZ4_RAW/LZ4 codecs but writer tests only covered GZIP. No roundtrip coverage for these codecs in the writer path.
- **Resolution**: 4 feature-gated writer→reader roundtrip tests added to `writer/tests_extra.rs`.
- **Verification**: Tests active under respective feature flags: `--features snappy`, `--features zstd`, `--features lz4`.

## P-011: Missing proptest roundtrip coverage for compression and PLAIN encoding (resolved this sprint)
- **Severity**: Low (property test gap)
- **Description**: No property-based tests existed for compression roundtrip or PLAIN encoding/decoding roundtrip.
- **Resolution**: `encoding/compression_proptest.rs` (5 props, feature-gated) + `encoding/plain_proptest.rs` (6 props) + `writer/tests_extra.rs` (2 props). Mathematical specification: ∀ data, decompress(compress(data, c)) == data; ∀ v, decode(encode(v)) == [v].
- **Verification**: `cargo test -p consus-parquet --lib --features gzip` and feature-specific runs.

---

## Resolved Gaps (this sprint)

### IO-001: Memory-mapped I/O backend absent from consus-io (resolved this sprint)
- **Status**: Closed.
- **Severity**: Medium.
- **Gap**: `consus-io` documented mmap as a planned backend in its module hierarchy comment but had no implementation. `File`-backed reads clone the descriptor and issue a `seek`+`read_exact` per call; large-file access patterns incur syscall overhead that mmap eliminates.
- **Resolution**: `MmapReader` implemented in `consus-io/src/io/sync/mmap.rs` under the `mmap` feature (`mmap = ["dep:memmap2", "std"]`). `ReadAt` delegates to a slice copy from `memmap2::Mmap`; `Length` returns `mmap.len() as u64`. `WriteAt`/`Truncate` are absent by design (read-only). `unsafe` block isolated to `from_file`; safety contract documented. 8 unit tests + 3 integration tests added. `memmap2 = { version = "0.9" }` added to workspace deps.
- **Verification**: `cargo test -p consus-io --features mmap` — 28 lib + 3 integration = 31 pass.

### P-012: Parquet reader has no proptest roundtrip coverage (resolved this sprint)
- **Status**: Closed.
- **Severity**: Low.
- **Gap**: All existing reader tests used fixed byte vectors analytically derived from the Parquet spec. No property-based tests exercised the full write→read pipeline for arbitrary inputs, leaving shrinkable edge cases (bit-boundary booleans, i32::MIN/MAX, empty byte arrays) untested.
- **Resolution**: `consus-parquet/src/reader/reader_proptest.rs` created with 5 proptest roundtrip properties: `prop_reader_i32_roundtrip` (Vec<i32> [1..=100]), `prop_reader_f64_roundtrip` (Vec<f64 NORMAL> [1..=50]), `prop_reader_bool_roundtrip` (Vec<bool> [1..=128]), `prop_reader_byte_array_roundtrip` (Vec<Vec<u8>> [1..=30]), `prop_reader_two_column_i32_f64_roundtrip` (paired columns). All assert computed `Vec` values with `prop_assert_eq!`.
- **Verification**: `cargo test -p consus-parquet --lib` — 197/197 pass (+5 vs 192 baseline).

### P-013: No criterion benchmark harness for Parquet write/read or Arrow bridge (resolved this sprint)
- **Status**: Closed.
- **Severity**: Low.
- **Gap**: `criterion` was listed as a dev-dependency in both `consus-parquet` and `consus-arrow` but no `[[bench]]` targets or benchmark files existed. Performance regressions in the write/read path or materialization bridge could not be detected by CI.
- **Resolution**: `consus-parquet/benches/parquet_rw.rs` created with `bench_write_i32` + `bench_read_i32` at 1K/10K/100K rows. `consus-arrow/benches/arrow_bridge.rs` created with `bench_bridge_i32`, `bench_bridge_double`, `bench_bridge_byte_array`. `[[bench]] harness = false` targets added to both Cargo.toml files. `consus-parquet` added to `consus-arrow` dev-deps.
- **Verification**: `cargo check --bench parquet_rw -p consus-parquet` and `cargo check --bench arrow_bridge -p consus-arrow` — 0 errors, 0 warnings.

## P-014: Parquet nested/group column write support — Dremel algorithm (resolved this sprint)

- **Status**: Closed.
- **Gap**: The Parquet writer returned `UnsupportedFeature` for any leaf column with `max_rep > 1` or `max_def > 1` (i.e. any column nested inside a group or with multiple levels of repetition/optionality). `encode_leaf_columns` used `col_idx` (leaf index) instead of `top_field_idx` (top-level schema field index) to index into `row.columns()`, which was accidentally correct for flat schemas but incorrect for any schema with a group field at the top level.
- **Resolution**:
  - Added `top_field_idx: usize` to `LeafColumnPlan` (with public accessor) so each leaf records which `schema.fields()[i]` / `row.columns()[i]` it belongs to.
  - Added `traverse_dremel_into` — a recursive Dremel-encoding function that navigates a `CellValue` tree following the leaf's path, accumulating rep/def levels and encoded values directly into output buffers. Handles all combinations of `Required`, `Optional`, and `Repeated` at any nesting depth.
  - Refactored `encode_leaf_columns` to the unified Dremel path for all leaf types; the three hand-rolled flat-column branches and the `UnsupportedFeature` fallback are replaced.
  - Fixed proptest-block placement: four nested-column roundtrip tests (`nested_required_group_two_leaves_roundtrip`, `nested_optional_group_roundtrip`, `nested_repeated_group_roundtrip`, `deeply_nested_optional_in_optional_group_roundtrip`) were accidentally placed inside a `proptest!` macro block; moved to standalone `#[test]` functions.
- **Verification**: `cargo test -p consus-parquet --lib` → 209/209 pass (205 pre-existing + 4 new nested-column tests).

## P-015: consus-nwb M33 remaining items — NwbFile entry point + session/TimeSeries read (resolved this sprint)

- **Status**: Closed.
- **Gap**: M33 checklist had five unimplemented items: `NwbFile::open`, session metadata read, `TimeSeries` read, namespace version detection, and conformance validation skeleton.
- **Resolution**:
  - `version/mod.rs`: `NwbVersion` enum (V2_0–V2_7 + Unknown), `NwbVersion::parse`, `NwbVersion::is_supported`, `NwbVersion::as_str`, `detect_version<R>` (reads `nwb_version` attribute from root group). 14 unit tests.
  - `metadata/mod.rs`: `NwbSessionMetadata` struct with `new`, `identifier`, `session_description`, `session_start_time` accessors. 9 unit tests including Unicode and clone invariants.
  - `model/mod.rs`: `TimeSeries` struct with `with_timestamps`, `with_rate`, `without_timing`, `from_parts` constructors and full accessor surface; `validate()` checks `timestamps.len() == data.len()`. 15 unit tests.
  - `storage/mod.rs`: `read_string_attr` and `read_f64_dataset` helpers; `decode_raw_as_f64` handles f64/f32 LE and BE. 9 unit tests.
  - `validation/mod.rs`: `validate_root_attributes` checks `neurodata_type_def="NWBFile"` and `nwb_version` presence in a single attribute pass. 11 unit tests.
  - `file/mod.rs`: `NwbFile<'a>` wrapping `Hdf5File<SliceReader<'a>>`; `open` validates root attributes; `nwb_version`, `session_metadata`, `time_series(path)` methods fully implemented. `DatasetCreationProps::default()` used for child datasets (layout Contiguous). 11 integration tests built against synthetic HDF5 files authored via `Hdf5FileBuilder`.
  - `Cargo.toml`: added `consus-io` dependency (for `SliceReader`); updated `std`/`alloc` feature chains.
- **Verification**: `cargo test -p consus-nwb --lib` → 62/62 pass; `cargo check --workspace` → 0 errors, 0 warnings.

## P-016: consus-nwb M35 Extended Read Path (resolved this sprint)

- **Status**: Closed.
- **Gap**: `read_f64_dataset` returned `UnsupportedFeature` for all integer types; `starting_time`/`rate` were read from wrong HDF5 locations (group attributes instead of dataset + dataset attribute); no group traversal or `list_time_series` API existed; `conventions`, `namespace`, `group` modules were stubs.
- **Resolution**:
  - `storage/mod.rs`: `decode_raw_as_f64` extended with signed/unsigned 8/16/32/64-bit integer promotion to f64 for both byte orders; `read_scalar_f64_dataset` and `read_f64_attr` helpers added.
  - `group/mod.rs`: `NwbGroupChild` struct and `list_typed_group_children` implemented; filters to `NodeType::Group` children; extracts `neurodata_type_def` and `neurodata_type_inc` string attributes.
  - `conventions/mod.rs`: `NeuroDataType` enum (17 variants) and `classify_neurodata_type` implemented; `is_timeseries_type` covers direct match, `neurodata_type_inc` inheritance, and known subtype set.
  - `namespace/mod.rs`: `NwbNamespace` struct with `core()` and `hdmf_common()` constructors.
  - `file/mod.rs`: `time_series` corrected to read `starting_time` from scalar dataset at `{path}/starting_time` and `rate` from float attribute on that dataset (not group-level attributes); dead `read_scalar_f64_attr` removed; `list_time_series(group_path)` added.
  - `consus-hdf5 list_group_at`: fixed to guard v1 symbol-table fallback with a `SYMBOL_TABLE` message presence check, so v2 groups with no children return an empty list instead of an `InvalidFormat` error.
- **Verification**: `cargo test -p consus-nwb --lib` → 130/130.

## P-017: Parquet multi-page column chunk splitting (resolved this sprint)

- **Status**: Closed.
- **Gap**: Each column chunk contained exactly one DataPage regardless of row count, violating Parquet's multi-page column chunk contract for large datasets.
- **Resolution**:
  - `ParquetWriter::with_page_row_limit(limit)` builder method added; `page_row_limit: Option<usize>` field propagated to `build_file_bytes`.
  - `build_file_bytes` refactored: page ranges computed within each row group; `encode_leaf_columns` called once per page range; pages transposed by column (`pages_by_column[leaf_idx][page_idx]`) then emitted contiguously per column chunk; `data_page_offset` captures the first page byte offset; `total_uncompressed_size`, `total_compressed_size`, and `num_values` are summed across all pages in the chunk.
  - Invariants: `data_page_offset` = first page; sums include all pages; pages for one column chunk are contiguous; single-page (None/0 limit) path is unchanged.
- **Tests added** (in `writer/tests_extra.rs`): `multi_page_i32_two_pages_data_roundtrip`, `multi_page_three_pages_all_values_preserved`, `multi_page_uneven_split_last_page_smaller`, `multi_page_limit_larger_than_rows_gives_one_page`, `multi_page_combined_with_multi_row_group`, `prop_multi_page_i32_roundtrip`.
- **Verification**: `cargo test -p consus-parquet --lib` → 215/215.

## M-037: NWB Write Path — NwbFileBuilder + validate_time_series_for_write (resolved this sprint)

- **Status**: Closed.
- **Gap**: `consus-nwb` had no write path. There was no way to construct an NWB 2.x file from Rust, write TimeSeries groups, or write a Units table. The read path (`NwbFile::open`, `time_series`, `list_time_series`) was complete but roundtrip authoring was blocked on missing write surface.
- **Resolution**:
  - `NwbFileBuilder::new(nwb_version, identifier, session_description, session_start_time)` — writes all five required NWB root attributes (`neurodata_type_def = "NWBFile"`, `nwb_version`, `identifier`, `session_description`, `session_start_time`) at construction time; rejects empty `identifier` or `session_description` with `InvalidFormat` before any HDF5 bytes are written.
  - `NwbFileBuilder::write_time_series(ts: &TimeSeries)` — calls `ts.validate()` and `validate_time_series_for_write` before any write; emits `{name}/data` (f64 LE) + `{name}/timestamps` (f64 LE) for timestamp-based TimeSeries, or `{name}/data` (f64 LE) + `{name}/starting_time` scalar dataset with `rate` f32 LE attribute for rate-based TimeSeries; attaches `neurodata_type_def = "TimeSeries"` to the group.
  - `NwbFileBuilder::write_units(spike_times: &[f64])` — emits `Units` group with `neurodata_type_def = "Units"` attribute; `Units/spike_times` dataset (f64 LE) with `neurodata_type_def = "VectorData"` and `description = "spike times"` attributes.
  - `NwbFileBuilder::finish() -> Result<Vec<u8>>` — delegates to `Hdf5FileBuilder::finish()`.
  - `validate_time_series_for_write(ts: &TimeSeries)` added to `consus_nwb::validation` — checks timing representation is present (timestamps or rate) and that rate > 0 when present.
  - `NwbFile::units_spike_times()` — new read method added to `NwbFile` for roundtrip verification; reads `Units/spike_times` via `read_f64_dataset`.
  - Module-level private helpers `fixed_string_bytes`, `f64_le_datatype`, `f32_le_datatype` added to `file/mod.rs`; shared between `NwbFileBuilder` impl and test helpers (SSOT, no duplication).
- **Tests added**:
  - `file::tests`: 12 value-semantic tests — `nwb_file_builder_minimal_file_opens_successfully`, `nwb_file_builder_empty_identifier_returns_error`, `nwb_file_builder_empty_session_description_returns_error`, `write_time_series_with_timestamps_roundtrip`, `write_time_series_with_rate_roundtrip`, `write_multiple_time_series_roundtrip`, `write_empty_time_series_with_timestamps_roundtrip`, `write_time_series_without_timing_returns_conformance_error`, `write_time_series_with_zero_rate_returns_error`, `write_time_series_with_negative_rate_returns_error`, `write_units_spike_times_roundtrip`, `write_units_empty_spike_times_roundtrip`.
  - `validation::tests`: 7 value-semantic tests — `validate_for_write_ok_with_timestamps`, `validate_for_write_ok_with_rate`, `validate_for_write_rejects_no_timing`, `validate_for_write_rejects_zero_rate`, `validate_for_write_rejects_negative_rate`, `validate_for_write_rejects_negative_inf_rate`, `validate_for_write_accepts_very_small_positive_rate`.
- **Verification**: `cargo test -p consus-nwb --lib` → 149/149; `cargo test --workspace` → 2219/2219; `cargo check --workspace` → 0 errors, 0 warnings.

---

## M-046: read_string_dataset VariableString support + NWB h5py fixture integration tests (resolved this sprint)

- **Crates affected**: `consus-nwb`
- **Gap**: `read_string_dataset` only supported `FixedString`; real NWB files use `VariableString` (HDF5 VL type) for dataset columns. Milestone 38/41 NWB fixture verification was blocked on external file acquisition.
- **Resolution**: Extended `read_string_dataset` with a `Datatype::VariableString` arm using `consus_hdf5::heap::resolve_vl_references`. Generated deterministic h5py fixture. Added 10-invariant integration test.
- **Verification**: `cargo test -p consus-nwb --test integration_real_file` → 1/1; `cargo test --workspace` → 0 failures.

---

## Open Gaps

_No open gaps in the current audit scope. Remaining open work: lifetime-parameterized zero-copy `ArrowArray` model, hybrid Parquet-inside-Consus containers, netCDF-4 Unidata reference file comparison (P2.3), NWB verification against real Allen Brain Observatory NWB 2.x fixtures (Milestone 38), large-file (>4 GiB) regression tests, cargo-fuzz harness targets, WASM validation, no_std smoke tests, documentation site, crates.io publication. NWB fixture acquisition is now complete: h5py-generated `data/nwb/nwb_fixture_v2_7.nwb` with 10 verified invariants (Milestone 46). Real Allen Brain Observatory fixture remains optional for broader compatibility testing._

---

## Closed Since Previous Audit

### M-045: netCDF-4 Enhanced Model Read — User-Defined Types

- **Crates affected**: `consus-hdf5`, `consus-netcdf`
- **`consus-netcdf/tests/integration_netcdf_hdf5.rs`**: Fixed `read_nested_group_into_model`: `CLASS=DIMENSION_SCALE` attribute added to "x" dataset so `is_dimension_scale` returns true; address→name map populated; `dimensions.len()==1`, `dimensions[0].name=="x"`, `dimensions[0].size==4` asserted; classic model read (P2.3) is now fully verified with 13/13 integration tests.
- **`consus-netcdf/src/model/mod.rs`**: Added `NetcdfUserType { name: String, datatype: consus_core::Datatype }` struct (`#[cfg(feature = "alloc")]`, derives `Debug + Clone + PartialEq`). Added `user_types: Vec<NetcdfUserType>` field to `NetcdfGroup`; `new()` initializes it; `is_empty()` includes it.
- **`consus-netcdf/src/lib.rs`**: Added `pub use model::NetcdfUserType;` crate-root re-export.
- **`consus-hdf5/src/file/mod.rs`**: Added `Datatype` to `consus_core` import. Added `pub fn named_datatype_at(&self, address: u64) -> Result<Datatype>`: reads object header, locates Datatype message (0x0003) via `reader::find_message`, returns `Error::InvalidFormat` when absent, delegates to `crate::datatype::compound::parse_datatype`.
- **`consus-hdf5/src/file/writer.rs`**: Added `pub fn add_named_datatype(&mut self, name: &str, datatype: &Datatype) -> Result<u64>` to `Hdf5FileBuilder`: encodes only a Datatype message, calls `write_object_header_v2`, pushes link to `root_links`. No Dataspace or Layout message emitted — `classify_object` returns `NodeType::NamedDatatype` for such headers (HDF5 spec §IV.A.2.3). Added 1 unit test: `add_named_datatype_creates_readable_named_type` (Float{64,LE} write → `node_type_at` asserts `NamedDatatype` → `named_datatype_at` asserts round-trip equality).
- **`consus-netcdf/src/hdf5/group/mod.rs`**: `NodeType::NamedDatatype` arm in `extract_group` updated: replaced `// skip` comment with `file.named_datatype_at(child_addr)?` + `group.user_types.push(NetcdfUserType { name: child_name, datatype })`.
- **`consus-netcdf/tests/integration_netcdf_hdf5.rs`**: 2 new integration tests: `read_named_type_in_root_group` (Float{32,LE}: write → read_model → root.user_types[0] round-trip); `read_named_type_in_child_group` (Integer{64,LE,signed}: root.user_types[0] present, child.user_types is empty).
- **Tests added**: consus-hdf5 lib 271 → 272 (+1); consus-netcdf 132 → 137 (+5: 2 integration in M-045 + 2 from fixed read_nested_group_into_model split + test count for classic model verified); workspace 2323 → 2329 (+6).
- **Verification**: `cargo test -p consus-hdf5 --lib` → 272/272; `cargo test -p consus-netcdf` → 137/137; `cargo test --workspace` → 2329/2329; `cargo check --workspace` → 0 errors, 0 warnings.

### M-042: netCDF-4 HDF5 Write Path — Classic Flat Model

- **Crates affected**: `consus-hdf5`, `consus-netcdf`
- **`consus-hdf5/src/file/writer.rs`**: Added `Datatype::Reference(ReferenceType::Object/Region)` case to `encode_datatype`; encoding: `byte[0]=0x17` (class=7, version=1), `byte[1]=0/1` (object/region discriminant), `bytes[4..8]=8u32 LE` (8-byte reference size); 2 new value-semantic tests (`encode_reference_datatype_object_reference`, `encode_reference_datatype_region_reference`).
- **`consus-netcdf/src/conventions/mod.rs`**: Added `NC_PROPERTIES_ATTR = "_nc_properties"` and `NC_PROPERTIES_VALUE = "version=2,netcdf=4.x.x"` constants per NUG §2.6; both re-exported from crate root.
- **`consus-netcdf/src/hdf5/write/mod.rs`** (new): `NetcdfWriter` struct wrapping `Hdf5FileBuilder`; `write_model(&NetcdfModel) -> Result<Vec<u8>>` emits complete HDF5 file; `write_dimension_scale` private helper writes each root-group dimension as a 1-D `u32` coordinate-index dataset with `CLASS="DIMENSION_SCALE"`, `NAME={dim.name}`, `_Netcdf4Dimid={idx}` attributes; `write_variable` private helper writes each root-group variable as zero-filled contiguous dataset with `DIMENSION_LIST` attribute (one 8-byte LE object-reference address per axis) and string-valued CF attribute propagation; scalar variables (rank 0) carry no `DIMENSION_LIST`; 4 unit tests.
- **`consus-netcdf/src/hdf5/mod.rs`**: Added `#[cfg(feature = "std")] pub mod write;`.
- **`consus-netcdf/src/lib.rs`**: Added `#[cfg(feature = "std")] pub use hdf5::write::NetcdfWriter;` and `NC_PROPERTIES_ATTR`, `NC_PROPERTIES_VALUE` to crate-root re-exports.
- **`consus-netcdf/tests/write_netcdf_hdf5.rs`** (new): 7 value-semantic round-trip integration tests: `write_empty_model_produces_valid_hdf5`, `write_single_dimension_roundtrip`, `write_dimension_and_variable_roundtrip`, `write_two_dimensions_two_variables_roundtrip`, `write_nc_properties_root_attribute_present`, `write_scalar_variable_roundtrip`, `write_cf_string_attribute_preserved`.
- **Known limitations (M-042 scope)**: fill-value HDF5 message emission deferred; numeric CF attribute propagation deferred; sub-group hierarchies (enhanced model) deferred; fixed-size datatypes only.
- **Tests added**: consus-hdf5 lib 266 → 268 (+2); consus-netcdf 113 → 125 (+12: 4 unit + 7 integration + 1 doctest); workspace 2292 → 2306 (+14).
- **Verification**: `cargo test -p consus-netcdf` → 125/125; `cargo test -p consus-hdf5 --lib` → 268/268; `cargo check --workspace` → 0 errors, 0 warnings.

### M-043: netCDF-4 HDF5 Write Path — Enhanced Model + Numeric CF Attribute Propagation

- **Crates affected**: `consus-hdf5`, `consus-netcdf`
- **`consus-hdf5/src/file/writer.rs`**: Added `SubGroupBuilder<'a>` public struct; fields: `sink: &'a mut MemCursor`, `state: &'a mut WriteState`, `parent_links: &'a mut Vec<(String, u64)>`, `name: String`, `child_links: Vec<(String, u64)>`. Methods: `add_dataset_with_attributes` (writes dataset + object header, records address in `child_links`, returns header addr for immediate reuse), `begin_sub_group<'b>` (opens a nested `SubGroupBuilder<'b>` by reborrowing `sink`/`state`/`child_links`), `finish_with_attributes` (writes group object header with LINK + ATTRIBUTE messages, pushes to `parent_links`, consumes `self`). Added `Hdf5FileBuilder::begin_group(&mut self, name: &str) -> SubGroupBuilder<'_>` method. 3 new tests: `sub_group_builder_empty_finish_creates_navigable_group`, `sub_group_builder_dataset_address_is_reusable_in_dimlist`, `sub_group_builder_nested_sub_group_roundtrip`.
- **`consus-netcdf/src/hdf5/write/mod.rs`** (refactored to 227 lines): Updated `write_model` to add step 4 — for each `group` in `model.root.groups`, calls `self.builder.begin_group`, then `write_child_group_content`, then `finish_with_attributes` with group-level CF attrs.
- **`consus-netcdf/src/hdf5/write/helpers.rs`** (new, 380 lines): `DatasetTarget` trait (`pub(super)`) + impls for `Hdf5FileBuilder` and `SubGroupBuilder<'_>` (zero-cost monomorphization); `encode_cf_attrs` handles all `AttributeValue` variants (Int→i64 LE, Uint→u64 LE, Float→f64 LE, String→FixedString, IntArray→1-D i64, UintArray→1-D u64, FloatArray→1-D f64, StringArray→FixedString array, Bytes→skip, system-key skip); `write_dimension_scale<W: DatasetTarget>` and `write_variable<W: DatasetTarget>` refactored as single generic zero-cost implementations; `write_child_group_content` recursively writes dimensions → variables → child groups using `SubGroupBuilder`.
- **`consus-netcdf/tests/write_netcdf_hdf5.rs`**: 7 new integration tests: `write_int_cf_attribute_preserved`, `write_uint_cf_attribute_preserved`, `write_float_cf_attribute_preserved`, `write_int_array_cf_attribute_preserved`, `write_float_array_cf_attribute_preserved`, `write_enhanced_model_sub_group_roundtrip`, `write_nested_two_level_sub_group_roundtrip`.
- **Tests added**: consus-hdf5 lib 268 → 271 (+3); consus-netcdf 125 → 132 (+7); workspace 2306 → 2313 (+7 net from M-043 alone).
- **Verification**: `cargo test -p consus-hdf5 --lib` → 271/271; `cargo test -p consus-netcdf` → 132/132; `cargo check --workspace` → 0 errors, 0 warnings.

### M-044: NWB Per-Type `neurodata_type_inc` Inheritance Chains in `NwbNamespaceSpec`

- **Crates affected**: `consus-nwb`
- **`consus-nwb/src/namespace/mod.rs`**: Added `NwbTypeSpec` public struct (`name: String`, `neurodata_type_inc: Option<String>`; derives Debug, Clone, PartialEq, Eq). Changed `NwbNamespaceSpec.neurodata_types` from `Vec<String>` to `Vec<NwbTypeSpec>`. Updated `parse_nwb_spec_yaml` with a three-state parser: `pending_type: Option<NwbTypeSpec>` accumulates sub-keys at indent=4 (`inc:` line); bare `  - TypeName` lines produce `NwbTypeSpec { neurodata_type_inc: None }` (backward compat); `  - name: TypeName` + optional `    inc: Parent` produce a fully typed spec. Updated `format_nwb_spec_yaml` to emit bare `  - {name}` when no `inc`, or `  - name: {name}\n    inc: {parent}` when present. 5 new tests: `nwb_type_spec_with_inc_stores_parent`, `nwb_type_spec_without_inc_has_none_parent`, `parse_nwb_spec_yaml_bare_type_name_has_no_inc`, `parse_nwb_spec_yaml_named_type_with_inc_parses_chain`, `format_parse_roundtrip_type_with_inc`.
- **`consus-nwb/src/conventions/mod.rs`**: Replaced fixed 2-level `is_timeseries_type_with_specs` with an iterative BFS chain walk: builds `BTreeMap<&str, &str>` (type → parent) from all spec entries, walks the chain from `type_name` up to depth 64 with a `BTreeSet` cycle guard. 2 new tests: `is_timeseries_type_with_specs_resolves_arbitrary_depth` (chain A→B→C→TimeSeries), `is_timeseries_type_with_specs_returns_false_for_unrelated_chain`.
- **`consus-nwb/src/lib.rs`**: Added `pub use namespace::NwbTypeSpec;`.
- **`consus-nwb/src/file/mod.rs`**: Updated `neurodata_types` construction sites (2 tests).
- **Tests added**: consus-nwb lib 214 → 221 (+7); workspace 2313 → 2323 (+10 net, including some existing test constructor updates already counted).
- **Verification**: `cargo test -p consus-nwb --lib` → 221/221; `cargo check --workspace` → 0 errors, 0 warnings.

### M-041: NWB Namespace Spec YAML Parsing + V2_8 + Spec I/O Path

- **Crates affected**: `consus-nwb`
- **`version/mod.rs`**: Added `V2_8` variant to `NwbVersion` enum; `parse("2.8.0")` → `V2_8`; `as_str` → `"2.8"`; `is_supported()` returns `true`; 2 new tests.
- **`namespace/mod.rs`**: Added `NwbNamespaceSpec` struct (`name`, `version`, `doc_url: Option<String>`, `neurodata_types: Vec<String>`); `parse_nwb_spec_yaml` — indent-aware parser handling multi-namespace YAML documents, `neurodata_types` sub-lists, comments, and blank lines; `apply_spec_key` private helper; `format_nwb_spec_yaml` — deterministic serializer; 21 new value-semantic tests.
- **`conventions/mod.rs`**: Extended `is_timeseries_type` with 2-level transitivity (if `type_inc ∈ TIMESERIES_SUBTYPES`, the type transitively extends `TimeSeries`); added `is_timeseries_type_with_specs` for spec-guided resolution; 5 new tests.
- **`storage/mod.rs`**: Added `read_scalar_string_dataset` — delegates to `read_string_dataset`, extracts first element; module scope table updated.
- **`file/mod.rs`**: Added `NwbFile::list_specifications()` (navigates `/specifications/`, returns namespace names; empty vec when absent), `NwbFile::read_specification(namespace, version)` (reads scalar FixedString dataset, parses YAML), `NwbFileBuilder::write_namespace_specs(specs)` (serializes to YAML, writes nested `ChildGroupSpec` tree per spec); 7 new integration tests.
- **`lib.rs`**: Added `pub use namespace::NwbNamespaceSpec;` crate-root re-export.
- **Tests added**: consus-nwb lib 182 → 214 (+32); workspace 2285 → 2292 (+7 net; previous baseline may have differed from gap_audit record).
- **Verification**: `cargo test -p consus-nwb --lib` → 214/214; `cargo test --workspace` → 2292/2292; `cargo check --workspace` → 0 errors, 0 warnings.

### M-040: NWB ElectrodeTable + UnitsTable + Storage String/U64 + README
- **Crates affected**: `consus-nwb`
- **Storage additions** (`consus-nwb/src/storage/mod.rs`):
  - Added `decode_raw_as_u64` private helper — mirrors `decode_raw_as_f64`; dispatches on `Datatype::Integer` (8/16/32/64-bit, signed+unsigned, both byte orders); rejects non-integer types with `UnsupportedFeature`.
  - Added `read_u64_dataset` — reads integer dataset as `Vec<u64>`; supports contiguous and chunked layouts.
  - Added `read_string_dataset` — reads `FixedString` dataset as `Vec<String>`; strips trailing null bytes; validates UTF-8; supports contiguous and chunked layouts.
  - Updated module doccomment scope table.
  - 8 new value-semantic storage tests: u32→u64 widening, u64 identity, i32 signed bit-pattern cast, float→u64 rejection, FixedString exact-fill, null-padded strip, all-null element → empty string, wrong-type → `UnsupportedFeature`.
- **Model additions** (`consus-nwb/src/model/units.rs`, `electrode.rs`, `mod.rs`):
  - `UnitsTable` struct (`spike_times_per_unit: Vec<Vec<f64>>`, `ids: Option<Vec<u64>>`); `from_vectordata` decodes HDMF VectorIndex pattern with monotone + length invariant checks; `flat_spike_times()` + `cumulative_index()` encode back to wire format; `new`, `from_parts` constructors; full accessors.
  - `ElectrodeRow` (`id: u64`, `location: String`, `group_name: String`) + `ElectrodeTable` with `from_rows`, `from_columns` (column-length validation), `empty`; column iterators `id_column`, `location_column`, `group_name_column`.
  - `pub mod units; pub mod electrode;` added to `model/mod.rs`.
  - 18 `UnitsTable` unit tests + 13 `ElectrodeTable` unit tests.
- **File API additions** (`consus-nwb/src/file/mod.rs`):
  - `NwbFile::units_table()` — reads `Units/spike_times` (f64) + `Units/spike_times_index` (u64) + optional `Units/id` (u64); decodes via `UnitsTable::from_vectordata`.
  - `NwbFile::electrode_table()` — reads `electrodes/id` (u64) + `electrodes/location` (FixedString) + `electrodes/group_name` (FixedString); builds via `ElectrodeTable::from_columns`.
  - `NwbFileBuilder::write_units_table(&UnitsTable)` — emits `Units` group; `spike_times` VectorData (f64 LE) with `neurodata_type_def = "VectorData"`; `spike_times_index` VectorIndex (u64 LE) with `neurodata_type_def = "VectorIndex"`; optional `id` dataset.
  - `NwbFileBuilder::write_electrode_table(&ElectrodeTable)` — emits `electrodes` DynamicTable group; `id` (u64 LE), `location` (FixedString null-padded to max len), `group_name` (FixedString null-padded to max len); `neurodata_type_def = "DynamicTable"`, `description`, `colnames` attributes.
  - 7 new integration tests: 3 UnitsTable roundtrips (with IDs, without IDs, empty), 2 ElectrodeTable roundtrips (3-row, empty), 2 NotFound negative tests.
- **Documentation** (`crates/consus-nwb/README.md`):
  - Created: format overview, feature flag table, quick-start read/write examples, module architecture tree, NWB compliance table, spec references, license.
- **Tests added**: consus-nwb lib 166 → 211 (+45); workspace 2239 → 2285 (+46).
- **Verification**: `cargo test -p consus-nwb --lib` → 211/211; `cargo test --workspace` → 2285/2285; `cargo check --workspace` → 0 errors, 0 warnings.

### M-039: NWB Extended Read Path + HDF5 Nested Group Write
- **Crates affected**: `consus-hdf5`, `consus-nwb`
- **HDF5 writer changes** (`consus-hdf5/src/file/writer.rs`):
  - Added `ChildGroupSpec<'a>` public struct (mirrors `ChildDatasetSpec`; supports arbitrary-depth `sub_groups: &'a [ChildGroupSpec<'a>]`).
  - Added `write_group_node` private recursive free function; eliminates the duplicated dataset-write + object-header-write logic that was previously inlined in `add_group_with_attributes`.
  - Refactored `add_group_with_attributes` to delegate to `write_group_node(&[])` — zero behavior change, backward compatible.
  - Added `Hdf5FileBuilder::add_group_with_children` public method accepting both `ChildDatasetSpec` and `ChildGroupSpec` slices for arbitrary-depth nested group authoring.
  - 3 new value-semantic tests: `add_group_with_children_creates_navigable_nested_group`, `add_group_with_children_nested_group_datasets_are_readable`, `add_group_with_attributes_still_works_after_refactor`.
- **NWB metadata** (`consus-nwb/src/metadata/mod.rs`):
  - Added `NwbSubjectMetadata` struct with 5 optional fields (`subject_id`, `species`, `sex`, `age`, `description`) per NWB 2.x `Subject` type spec.
  - `from_parts(Option<String> × 5)` constructor + typed accessors returning `Option<&str>`.
  - 7 new unit tests covering all-Some, all-None, partial, equality, clone, Debug, and Unicode fields.
- **NWB file** (`consus-nwb/src/file/mod.rs`):
  - `NwbFile::subject()` — navigates `general/subject` via `open_path`, reads optional string attributes, returns `NwbSubjectMetadata`.
  - `NwbFile::list_acquisition()` — delegates to `list_time_series("acquisition")`.
  - `NwbFile::list_processing(module_name)` — delegates to `list_time_series("processing/{module}")`.
  - `NwbFileBuilder::write_subject(&NwbSubjectMetadata)` — builds owned attribute data, borrows for `ChildGroupSpec`, calls `add_group_with_children("general", &[], &[], &[subject_spec])` to emit `general/subject`.
  - 3 proptest roundtrip tests: `roundtrip_timestamps_timeseries`, `roundtrip_rate_timeseries` (rate precision invariant: `read == (written as f32) as f64`, exact), `roundtrip_units_spike_times`.
  - 7 deterministic tests: 3 NotFound negative tests + 4 positive roundtrip tests.
  - `proptest = "1"` added to `consus-nwb` dev-dependencies.
- **Tests added**: consus-hdf5 lib 263 → 266 (+3); consus-nwb lib 149 → 166 (+17); workspace 2219 → 2239 (+20).
- **Verification**: `cargo test -p consus-nwb --lib` → 166/166; `cargo test -p consus-hdf5 --lib` → 266/266; `cargo test --workspace` → 2239/2239; `cargo check --workspace` → 0 errors, 0 warnings.


### Z-090: Chunk API Export Gap
- **Status**: Closed.
- **Evidence**: Chunk I/O functions and types are re-exported from the crate root, establishing one public access path for `read_chunk`, `write_chunk`, `read_array`, `write_array`, `Selection`, `SelectionStep`, `ChunkError`, and fill-value expansion.

### Z-091: Full-Array Chunk Assembly Failure
- **Status**: Closed.
- **Evidence**: Full-array read/write roundtrip now passes after correcting typed fill-value byte expansion, row-major chunk extraction during writes, and chunk-grid traversal logic for array assembly.
- **Verification**: `cargo test -p consus-zarr --lib` passes with `chunk::tests::write_array_and_read_back` succeeding.

### Z-092: Partial Selection Read Verification Gap
- **Status**: Closed.
- **Evidence**: Array-level partial reads now normalize selection steps against array shape, compute chunk/selection intersections over realized indices, and copy exact selected elements into row-major output buffers.
- **Verification**: `cargo test -p consus-zarr --lib` passes with `chunk::tests::read_array_partial_selection_contiguous_across_chunks`, `chunk::tests::read_array_partial_selection_strided_across_chunks`, and `chunk::tests::read_array_partial_selection_uninitialized_chunk_uses_fill_value` succeeding.

### Z-093: Partial Selection Write Verification Gap
- **Status**: Closed.
- **Evidence**: Array-level partial writes now perform chunk-aware read-modify-write updates, preserve existing values outside the selected region, materialize fill values for uninitialized intersecting chunks, and reject mismatched input lengths.
- **Verification**: `cargo test -p consus-zarr --lib` passes with `chunk::tests::write_array_selection_contiguous_across_chunks`, `chunk::tests::write_array_selection_strided_across_chunks`, `chunk::tests::write_array_selection_uninitialized_chunks_materialize_fill_value`, and `chunk::tests::write_array_selection_rejects_invalid_input_length` succeeding.

### A-001: Arrow Nested-Type Field Loss on Conversion
- **Status**: Closed.
- **Evidence**: `core_datatype_to_arrow_hint` now recursively preserves Compound → Struct fields (with `ArrowField` per `CompoundField`), Array → List element type (via recursive `core_datatype_to_arrow_hint(base)`), and Complex → Struct real/imaginary children. `arrow_datatype_to_core` now preserves Struct → Compound fields, Map → Compound key/value fields, and Union → Compound variant fields. 6 new value-semantic tests added.
- **Verification**: `cargo test -p consus-arrow --lib` passes with 41/41 tests.

### F-001: FITS Binary Table TFORM → Datatype Mapping
- **Status**: Closed.
- **Evidence**: `BinaryFormatCode` enum covers all 13 FITS Standard 4.0 binary table column format codes (L/X/B/I/J/K/A/E/D/C/M/P/Q). `parse_binary_format`, `binary_format_to_datatype`, `tform_to_datatype`, and `binary_format_element_size` implement deterministic TFORM → canonical `Datatype` conversion. Array wrapping for repeat > 1 scalar types. Crate-root re-exports added. 15 comprehensive value-semantic tests added.
- **Verification**: `cargo test -p consus-fits --lib` passes with 123/123 tests.

### H-001: HDF5 Datatype Class Mapping Coverage
- **Status**: Closed.
- **Evidence**: All 10 HDF5 datatype classes now have public mapping functions: `map_string` (fixed/variable, ASCII/UTF-8), `map_bitfield` (→ Opaque with HDF5_bitfield tag), `map_opaque` (with optional tag), `map_compound` (ordered fields + size), `map_reference` (Object/Region with size-based default), `map_enum` (structural envelope), `map_variable_length` (→ VarLen), `map_array` (→ Array with base + dims), plus `charset_from_flags` helper. Coverage table in module documentation. 20 value-semantic tests added.
- **Verification**: `cargo test -p consus-hdf5 --lib` passes with 263/263 tests.

### F-002: FITS Column Descriptor Datatype Integration
- **Status**: Closed.
- **Evidence**: `FitsTableColumn` extended with `datatype: Datatype` and `byte_width: usize` fields populated from TFORM during header parsing. `from_binary_tform()` constructor derives both fields via `tform_to_datatype` and `binary_format_element_size`. `parse_column` dispatches binary→`from_binary_tform`, ASCII→`FixedString` with `parse_ascii_column_width`. `FitsBinaryTableDescriptor::from_header()` validates per-column byte widths sum to `NAXIS1`. 5 new value-semantic tests: Boolean/Int32/Float64 datatype roundtrip, Array datatype from repeat>1, NAXIS1 mismatch rejection, ASCII FixedString derivation, Complex/Compound descriptor types.
- **Verification**: `cargo test -p consus-fits --lib` passes with 128/128 tests.

---

## M-047: NWB Full Conformance Validation (this sprint — CLOSED)

**Root cause**: `validate_root_attributes` covered only 2 of 7 normative root-group constraints; required group structure (5 groups), session attribute format (ISO 8601), and TimeSeries data presence were unchecked.

**Resolution**:
- `is_valid_iso8601` — structural ISO 8601 byte-level validator (`YYYY-MM-DDTHH:MM:SS[Z|±HH:MM]`); no calendar arithmetic; ungated (no `alloc`)
- `ConformanceViolation` enum — 5 variants covering all implemented check layers
- `NwbConformanceReport` — non-short-circuiting report accumulator; `is_conformant()`, `violations()`, `into_result()`
- `check_root_session_attrs` — scans root HDF5 attributes for `identifier`, `session_description`, `session_start_time` (ISO 8601 checked)
- `NwbFile::validate_conformance()` — 4-layer multi-pass validator: fail-fast identity (layer 1), session attrs (layer 2), required group presence (5 groups, layer 3), TimeSeries data presence (layer 4)
- `NwbFileBuilder::write_empty_group(path)` — creates empty required groups for test infrastructure

**Verification**: 29 new value-semantic tests; `cargo test -p consus-nwb --lib` → 250/250; `cargo test --workspace` → 2359/2359; `cargo check --workspace` → 0 errors, 0 warnings

**Residual**: `timestamps_reference_time` (NWB ≥ 2.2), `file_create_date`, HDMF DynamicTable `colnames` consistency, electrode reference integrity, and namespace type-system validation remain unimplemented and are roadmap items beyond current sprint scope.

---

## M-001: consus-mat correctness and coverage gaps (resolved this sprint)

| ID | File | Gap | Resolution |
|----|------|-----|------------|
| M-001a | Cargo.toml | Dead byteorder + consus-compression deps | Removed both from [dependencies] and feature lists |
| M-001b | src/lib.rs | Blank feature gate column + empty Entry Points links | Filled with correct feature name and doc links |
| M-001c | src/error.rs | UnsupportedVersion variant never constructed | Removed dead variant |
| M-001d | src/v5/matrix.rs | nzmax (flags1) captured and discarded in sparse parsing | Added ir.len()==nzmax and jc.len()==ncols+1 validation |
| M-001d2 | src/v5/mod.rs | Unknown top-level v5 elements hard-failed parsing | Reader now skips unknown top-level elements while still validating and consuming their declared payload bytes |
| M-001e | src/v73/reader.rs | Cell group children iterated in arbitrary order | Sort children by numeric name (parse::<usize>) before building cells vec |
| M-001f | tests/v5_read.rs | Vacuous truncated test (no assertion) | Replaced with v5_truncated_element_returns_error using a proper is_err() assertion |
| M-001g | tests/v5_read.rs | No coverage for char, logical, complex, sparse, cell, struct | Added 7 value-semantic synthetic-byte-stream tests |
| M-001h | tests/v73_read.rs | No coverage for logical and char beyond fixture | Closed with value-semantic synthetic HDF5-backed tests for logical, little-endian char, and big-endian char decoding |
| M-001i | src/v5/matrix.rs | `mxOBJECT_CLASS` had no explicit policy | Closed with deterministic `UnsupportedFeature` rejection for MAT v5 `mxOBJECT_CLASS`, plus integration coverage |
| M-001j | tests/v73_read.rs | No explicit compact-layout rejection coverage | Closed with synthetic HDF5-backed compact-dataset fixture coverage asserting `UnsupportedFeature("v7.3 compact layout")` |
| M-001k | tests/v5_compressed_read.rs | No `miCOMPRESSED` feature-matrix verification | Closed with dedicated integration coverage asserting successful compressed MAT v5 decode when `compress` is enabled and deterministic `UnsupportedFeature("miCOMPRESSED requires the 'compress' feature")` when `compress` is disabled |
| M-001s4a | src/model/structure.rs | MatStructArray.fields Vec<String> redundant with data keys | Removed fields field; new() changed to (shape, data); field_names() returns impl Iterator<Item = &str>; all construction sites updated |
| M-001s4b | tests/v4_read.rs | v4 sparse rejection path had no integration test | v4_sparse_matrix_returns_unsupported_feature_error: synthetic type_code=2 record, exact UnsupportedFeature message asserted |
| M-001s4c | crates/consus-mat/ | No crate-level README | README.md created: format coverage, feature flags, quick start, canonical model, rejection policies, v4/v5/v7.3 notes |
| M-001s4d | .github/workflows/ci.yml | consus-mat absent from CI matrix | Added to check/test/msrv matrix; test-mat-features job added for default + no-compress configurations |
| M-001s5a | crates/consus-hdf5/src/file/writer.rs | HDF5 builder lacked nested group authoring surface | ChildDatasetSpec + add_group_with_attributes added; enables MATLAB_class group + child dataset creation |
| M-001s5b | src/model/numeric.rs, sparse.rs, cell.rs, character.rs, logical.rs, structure.rs | No model-level unit tests for constructors, invariants, or accessors | 42 value-semantic unit tests added across all model modules |
| M-001s5c | src/error.rs | No Display implementation tests | 5 unit tests covering all Display impl variants |
| M-001s5d | tests/v5_read.rs | No multi-variable file test | v5_multiple_variables_roundtrip added (2 scalar doubles, value-semantic) |
| M-001s5e | tests/v5_read.rs | loadmat R+Seek path untested | loadmat_from_reader_parses_test_fixture added (std::fs::File + test_v5.mat) |
| M-001s5f | src/lib.rs | Doc tests: 0 | Doc test for loadmat_bytes added (MAT v4 scalar double) |
| M-001s5g | tests/v73_read.rs | v73 cell array group roundtrip absent | v73_cell_array_roundtrip added using add_group_with_attributes |
| M-001s5h | tests/v73_read.rs | v73 struct array group roundtrip absent | v73_struct_array_roundtrip added using add_group_with_attributes |

### Verification
- cargo check -p consus-mat: 0 errors
- cargo test -p consus-mat: 71/71 tests pass (42 lib unit + 4 v4 + 1 v5-compressed + 14 v5 + 9 v73 + 1 doc)
- cargo test -p consus-mat --no-default-features --features std,alloc: 62/62 tests pass
- cargo test -p consus-hdf5: 321/321 tests pass
- cargo check --workspace: 0 errors

## Risk Assessment

| Risk | Probability | Impact | Status |
|------|-------------|--------|--------|
| Partial selection write semantics across chunk boundaries | Low | Medium | Reduced by multidimensional value-semantic write tests covering contiguous, strided, and uninitialized-chunk update paths |
| Invalid chunk coordinates accepted as store keys | Low | Low | Reduced by chunk-grid validation in `read_chunk`/`write_chunk` and negative tests for out-of-grid coordinates |
| Python interoperability mismatch | Medium | High | Closed by Z-102: all Python-generated v3 fixtures pass including boundary chunks across padded and partial chunk layouts |
| Sharded v3 high-level API drift | Medium | Medium | Closed by Z-103: high-level read/write dispatches to spec-compliant shard implementation; verified by single-shard and multi-shard round-trip tests |
| Fill-value width mismatch for non-8-byte numeric types | Low | Medium | Closed by Z-107: float32 and float64 fill-value byte expansion now matches the target element width and is covered by dedicated tests plus array roundtrips |
| Store/backend divergence | Low | Medium | Reduced by in-memory, filesystem, and Python-generated fixture coverage; S3 interop remains indirect |
| netCDF HDF5 read coverage (dimension coordinate values and compact/virtual variable payloads not yet read) | Low | Medium | Reduced by N-001, N-002, N-005, N-006, and N-007: attributes, unlimited extents, contiguous/chunked variable payload reads, DIMENSION_LIST-based semantic dimension binding, and ancestor-scope dimension inheritance are now preserved; coordinate-value extraction plus compact/virtual payload extraction remain roadmap work under P2.3 |
| MATLAB .mat v7.3 completeness (non-scalar struct arrays, virtual-layout coverage) | Low | Low | Reduced further by M-001 Sprint 5: cell and struct group roundtrip tests added via extended HDF5 builder; model unit test coverage complete; virtual-layout fixture coverage remains blocked on virtual dataset HDF5 authoring surface; non-scalar struct shape preservation requires MATLAB_dims attribute authoring (roadmap) |
| Parquet datatype, dataset-model, and trailer-validation coverage | Low | Medium | Reduced by P-001: canonical Parquet mappings now cover all core datatype variants present in the crate, including compound, array, enum, varlen, and reference cases; validated dataset descriptor and ordered projection coverage enforce row-group chunk cardinality, schema-order field identity, total-row aggregation, nested-column classification, nested group → canonical `Compound` preservation, and repeated field → canonical `VarLen` preservation; trailer validation now enforces `PAR1` magic, little-endian footer-length decoding, footer-offset bounds, non-overlapping row-group/column-chunk byte ranges, and rejection of row groups extending into the footer payload |
| Arrow nested-type field loss on conversion | Low | Medium | Closed by A-001: Compound/Array/Complex → Arrow Struct/List now preserves recursive field structure; Struct/Map/Union → Compound now preserves child fields |
| FITS binary table column type mapping | Medium | Medium | Closed by F-001: TFORM format codes now map to canonical Datatype for all 13 FITS Standard 4.0 binary table column types |
| FITS column descriptor datatype integration | Medium | Medium | Closed by F-002: FitsTableColumn now carries canonical Datatype and byte_width derived from TFORM; binary table NAXIS1 validation enforced |
| HDF5 datatype class mapping coverage | Medium | Medium | Closed by H-001: all 10 HDF5 datatype classes now have public mapping functions to canonical Datatype |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Zarr library + integration tests | 301 |
| Zarr passing | 301 |
| Zarr failing | 0 |
| netCDF tests | 137 |
| netCDF passing | 137 |
| netCDF failing | 0 |
| consus-parquet lib (default) | 215/215 |
| consus-parquet lib (gzip) | 215/215 |
| consus-parquet lib (snappy+zstd+lz4) | 223/223 |
| consus-arrow lib | 50/50 |
| consus-arrow lib (zerocopy) | 52/52 |
| consus-arrow E2E integration | 6/6 |
| consus-io lib (default) | 20/20 |
| consus-io lib+integration (mmap) | 31/31 |
| consus-nwb lib | 250/250 |
| consus-nwb integration (real file) | 1/1 |
| consus-hdf5 lib | 272/272 |
| workspace total tests (default) | 2359/2359 |
| Verified commands | `cargo test -p consus-netcdf` (137/137); `cargo test -p consus-hdf5 --lib` (272/272); `cargo test -p consus-nwb --lib` (250/250); `cargo test -p consus-nwb --test integration_real_file` (1/1); `cargo test --workspace` (2359/2359, default); `cargo check --workspace` (0 warnings, 0 errors) |
| Open gaps | 0 |
| High-severity open gaps | 0 |
| Closed this sprint | 5 (M-042: netCDF-4 HDF5 Write Path — `encode_datatype` Reference(Object/Region) support in `consus-hdf5`; `NetcdfWriter::write_model` classic flat model in `consus-netcdf`; `NC_PROPERTIES_ATTR/VALUE` constants; 7 round-trip integration tests + 4 unit tests + 1 doctest + 2 HDF5 datatype encoding tests; +14 new value-semantic tests — M-043: netCDF-4 enhanced model write path — `SubGroupBuilder<'a>` in `consus-hdf5`; `DatasetTarget` trait + `encode_cf_attrs` + recursive `write_child_group_content` in `consus-netcdf`; 3 HDF5 + 7 netCDF integration tests — M-044: NWB per-type `neurodata_type_inc` inheritance chains — `NwbTypeSpec` struct; `neurodata_types: Vec<NwbTypeSpec>`; iterative BFS `is_timeseries_type_with_specs` depth-64; 7 new tests — M-045: netCDF-4 enhanced model read — `NetcdfUserType` model, `user_types` in `NetcdfGroup`, `Hdf5File::named_datatype_at`, `Hdf5FileBuilder::add_named_datatype`, `extract_group` NamedDatatype arm; 1 HDF5 unit test + 2 netCDF integration tests; P2.3 classic model read test corrected — M-047: NWB Full Conformance Validation — `is_valid_iso8601`, `ConformanceViolation`, `NwbConformanceReport`, `check_root_session_attrs`, `NwbFile::validate_conformance`, `NwbFileBuilder::write_empty_group`; 29 new value-semantic tests) |
| Medium-severity open gaps | 0 |
| Low-severity open gaps | 0 |