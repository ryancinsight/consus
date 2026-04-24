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
- [x] Canonical `ParquetDatasetDescriptor` added for validated schema + row-group metadata
- [x] Canonical `ParquetColumnDescriptor` added with derived `Datatype`, `ColumnStorage`, and 1D row shape
- [x] Canonical `ColumnChunkDescriptor` and `RowGroupDescriptor` added with exact row-count and byte-length invariants
- [x] Ordered `ParquetProjection` / `ColumnProjection` API added for schema-subset views preserving source field order
- [x] Value-semantic tests added for total-row computation, fixed/variable/nested storage classification, projection ordering, and invalid row-group metadata rejection
- [x] Nested group columns now canonicalize to ordered `Datatype::Compound` descriptors with computed child offsets and fixed-size aggregation when derivable
- [x] Repeated scalar columns now canonicalize to `Datatype::VarLen` with variable-width storage classification
- [x] Repeated group columns now canonicalize to `Datatype::VarLen<Compound>` while preserving nested child ordering and offsets
- [x] Canonical `wire` module added for Parquet trailer validation and footer metadata envelopes
- [x] `FooterPrelude` validates trailer length, little-endian footer length, and `PAR1` magic invariants
- [x] `ColumnChunkLocation`, `RowGroupLocation`, and `ParquetFooterDescriptor` added with non-overlap and footer-boundary validation
- [x] Value-semantic tests added for valid trailer parsing, short input rejection, invalid magic rejection, footer-length overflow rejection, overlapping chunk rejection, and footer-boundary rejection
- [x] Verified `cargo test -p consus-parquet --lib` (116/116 after encoding + payload modules)
- [x] Verified `cargo check --workspace` (0 errors, workspace-wide)
- [x] `encoding/levels.rs`: `decode_levels` (RLE/bit-packing hybrid encoding ID 3), `decode_bit_packed_raw` (deprecated BIT_PACKED ID 4), `level_bit_width` -- 14 value-semantic tests
- [x] `encoding/plain.rs`: PLAIN decoders for BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY -- 14 value-semantic tests
- [x] `encoding/rle_dict.rs`: `decode_rle_dict_indices` for RLE_DICTIONARY (encoding ID 8) -- 5 value-semantic tests
- [x] `wire/payload.rs`: `PagePayload`, `split_data_page_v1`, `split_data_page_v2` -- 6 value-semantic tests
- [x] Minimal Thrift compact binary decoder (`wire/thrift.rs`): zigzag varint, i16/i32/i64, string/binary, field header, list/set/map header, recursive skip — 17 value-semantic tests
- [x] Parquet wire metadata types (`wire/metadata.rs`): `FileMetadata`, `SchemaElement`, `RowGroupMetadata`, `ColumnChunkMetadata`, `ColumnMetadata`, `KeyValue` with `decode_file_metadata`
- [x] Page header types (`wire/page.rs`): `PageHeader`, `DataPageHeader`, `DictionaryPageHeader`, `DataPageHeaderV2`, `PageType` with `decode_page_header`
- [x] Schema reconstruction bridge: `schema_elements_to_schema` (DFS traversal of flat schema list)
- [x] Dataset materialization bridge: `dataset_from_file_metadata` → `ParquetDatasetDescriptor`
- [x] Module decomposition: `wire` and `dataset` extracted from inline lib.rs blocks to external files
- [x] :  enum, ,  (PLAIN / PLAIN_DICTIONARY / RLE_DICTIONARY) -- 16 value-semantic tests
- [x] :  and  -- 2 value-semantic tests
- [x] Verified 
running 136 tests
test arrow_bridge::tests::arrow_hint_mapping_is_stable ... ok
test conversion::tests::core_to_parquet_roundtrip ... ok
test conversion::tests::field_to_core_conversion ... ok
test arrow_bridge::tests::bridge_counts_zero_copy_fields ... ok
test arrow_bridge::tests::integration_plan_reports_mode ... ok
test conversion::tests::schema_to_core_pairs ... ok
test conversion::tests::physical_type_mapping_is_correct ... ok
test dataset::tests::dataset_descriptor_rejects_chunk_field_order_mismatch ... ok
test dataset::tests::dataset_descriptor_computes_total_rows_and_columns ... ok
test conversion::tests::repetition_conversion ... ok
test dataset::tests::dataset_from_file_metadata_roundtrip ... ok
test conversion::tests::logical_type_refines_mapping ... ok
test dataset::tests::repeated_group_column_maps_to_varlen_compound_datatype ... ok
test dataset::tests::nested_group_column_maps_to_nested_storage ... ok
test conversion::tests::full_datatype_coverage_is_deterministic ... ok
test dataset::tests::projection_preserves_source_schema_order ... ok
test dataset::tests::dataset_descriptor_rejects_chunk_count_mismatch ... ok
test dataset::tests::repeated_scalar_column_maps_to_varlen_datatype_and_variable_storage ... ok
test dataset::tests::schema_elements_to_schema_flat ... ok
test encoding::column::tests::decode_column_plain_boolean_eight_values ... ok
test encoding::column::tests::decode_column_plain_byte_array_two_strings ... ok
test encoding::column::tests::column_values_len_is_empty_consistent ... ok
test encoding::column::tests::column_values_physical_type_all_variants ... ok
test encoding::column::tests::decode_column_plain_dict_i32_same_as_rle ... ok
test encoding::column::tests::decode_column_plain_f64_two_values ... ok
test encoding::column::tests::decode_column_plain_fixed_len_byte_array ... ok
test encoding::column::tests::decode_column_plain_i32_three_values ... ok
test encoding::column::tests::decode_column_plain_i64_two_values ... ok
test encoding::column::tests::decode_column_rle_dict_byte_array_three_values ... ok
test encoding::column::tests::decode_column_rle_dict_i32_bit_packed ... ok
test encoding::column::tests::decode_column_rle_dict_index_out_of_bounds_returns_error ... ok
test encoding::column::tests::decode_column_rle_dict_missing_dictionary_returns_error ... ok
test encoding::column::tests::decode_column_rle_dict_rle_run_five_copies ... ok
test encoding::column::tests::decode_column_rle_dict_type_mismatch_returns_error ... ok
test encoding::column::tests::decode_column_unsupported_encoding_returns_error ... ok
test encoding::column::tests::decode_dictionary_page_byte_array_two_strings ... ok
test encoding::column::tests::decode_dictionary_page_i32_three_values ... ok
test encoding::levels::tests::decode_bit_packed_raw_bit_width_1_ten_values ... ok
test encoding::levels::tests::decode_bit_packed_raw_bit_width_2_four_values ... ok
test encoding::levels::tests::decode_bit_packed_raw_truncated_errors ... ok
test encoding::levels::tests::decode_levels_bit_packed_bit_width_1_eight_values ... ok
test encoding::levels::tests::decode_levels_bit_packed_bit_width_2_eight_values ... ok
test encoding::levels::tests::decode_levels_bit_packed_partial_count ... ok
test encoding::levels::tests::decode_levels_empty_input_errors ... ok
test encoding::levels::tests::decode_levels_rle_bit_width_1_five_ones ... ok
test encoding::levels::tests::decode_levels_rle_bit_width_2_four_threes ... ok
test encoding::levels::tests::decode_levels_rle_run_truncated_to_count ... ok
test encoding::levels::tests::decode_levels_two_sequential_rle_runs ... ok
test encoding::levels::tests::decode_levels_zero_bit_width_returns_zeros ... ok
test encoding::levels::tests::decode_levels_zero_count_returns_empty ... ok
test encoding::levels::tests::level_bit_width_matches_spec ... ok
test encoding::plain::tests::decode_plain_boolean_ten_values ... ok
test encoding::plain::tests::decode_plain_boolean_zero_count ... ok
test encoding::plain::tests::decode_plain_byte_array_truncated_length_errors ... ok
test encoding::plain::tests::decode_plain_byte_array_two_values ... ok
test encoding::plain::tests::decode_plain_f32_two_values ... ok
test encoding::plain::tests::decode_plain_f64_two_values ... ok
test encoding::plain::tests::decode_plain_fixed_byte_array_two_values ... ok
test encoding::plain::tests::decode_plain_fixed_byte_array_zero_len ... ok
test encoding::plain::tests::decode_plain_i32_three_values ... ok
test encoding::plain::tests::decode_plain_i32_truncated_errors ... ok
test encoding::plain::tests::decode_plain_i32_zero_count ... ok
test encoding::plain::tests::decode_plain_i64_empty_errors ... ok
test encoding::plain::tests::decode_plain_i64_two_values ... ok
test encoding::plain::tests::decode_plain_i96_one_value ... ok
test encoding::rle_dict::tests::decode_rle_dict_indices_bit_packed_four_values ... ok
test encoding::rle_dict::tests::decode_rle_dict_indices_empty_input_errors ... ok
test encoding::rle_dict::tests::decode_rle_dict_indices_rle_run ... ok
test encoding::rle_dict::tests::decode_rle_dict_indices_two_rle_runs ... ok
test encoding::rle_dict::tests::decode_rle_dict_indices_zero_count_returns_empty ... ok
test hybrid::tests::default_descriptor_is_disabled ... ok
test hybrid::tests::embedded_descriptor_tracks_layout ... ok
test schema::arrow::tests::arrow_hint_mapping_is_stable ... ok
test schema::arrow::tests::bridge_counts_zero_copy_fields ... ok
test schema::arrow::tests::integration_plan_reports_mode ... ok
test schema::field::tests::field_id_roundtrip ... ok
test schema::field::tests::group_descriptor_validates ... ok
test schema::field::tests::scalar_descriptor_validates ... ok
test schema::field::tests::schema_find_field_works ... ok
test schema::field::tests::schema_rejects_duplicates ... ok
test schema::logical::tests::annotation_flags_are_consistent ... ok
test schema::logical::tests::compatibility_rules_match_expected_widths ... ok
test schema::logical::tests::numeric_types_are_detected ... ok
test schema::logical::tests::temporal_types_are_detected ... ok
test schema::physical::tests::classification_predicates_are_correct ... ok
test schema::physical::tests::from_parquet_type_i32_all_known_discriminants ... ok
test schema::physical::tests::from_parquet_type_with_length_fixed_len_byte_array ... ok
test schema::physical::tests::width_mapping_is_correct ... ok
test schema::tests::schema_descriptor_construction_works ... ok
test schema::tests::schema_module_exports_are_available ... ok
test tests::conversion_exports_are_available ... ok
test tests::dataset_exports_are_available ... ok
test tests::exports_hybrid_types ... ok
test tests::exports_schema_types ... ok
test tests::page_type_exports_are_available ... ok
test tests::wire_exports_are_available ... ok
test tests::wire_metadata_exports_are_available ... ok
test wire::metadata::tests::decode_file_metadata_minimal ... ok
test wire::metadata::tests::decode_file_metadata_rejects_missing_required_field ... ok
test wire::metadata::tests::decode_schema_element_group ... ok
test wire::metadata::tests::decode_schema_element_leaf ... ok
test wire::page::tests::decode_data_page_header_v2_minimal ... ok
test wire::page::tests::decode_dictionary_page_header_with_sorted ... ok
test wire::page::tests::decode_full_page_header_data_page ... ok
test wire::page::tests::decode_page_header_rejects_empty ... ok
test wire::page::tests::page_type_from_i32_covers_all_variants ... ok
test wire::payload::tests::split_v1_optional_column_rle_def_levels ... ok
test wire::payload::tests::split_v1_required_column_no_levels ... ok
test wire::payload::tests::split_v1_unsupported_level_encoding_errors ... ok
test wire::payload::tests::split_v2_optional_column_def_levels ... ok
test wire::payload::tests::split_v2_required_column_zero_level_lengths ... ok
test wire::payload::tests::split_v2_truncated_rep_level_section_errors ... ok
test wire::tests::footer_descriptor_computes_total_rows ... ok
test wire::tests::footer_descriptor_rejects_row_group_past_footer ... ok
test wire::tests::row_group_location_rejects_overlapping_columns ... ok
test wire::tests::validate_footer_prelude_accepts_valid_trailer ... ok
test wire::tests::validate_footer_prelude_rejects_footer_length_overflow ... ok
test wire::tests::validate_footer_prelude_rejects_invalid_magic ... ok
test wire::tests::validate_footer_prelude_rejects_short_input ... ok
test wire::thrift::tests::field_header_absolute ... ok
test wire::thrift::tests::field_header_stop_returns_none ... ok
test wire::thrift::tests::field_header_with_delta ... ok
test wire::thrift::tests::list_header_long_form ... ok
test wire::thrift::tests::list_header_short_form ... ok
test wire::thrift::tests::read_binary_works ... ok
test wire::thrift::tests::read_byte_exhausted_returns_error ... ok
test wire::thrift::tests::read_byte_works ... ok
test wire::thrift::tests::read_string_works ... ok
test wire::thrift::tests::skip_binary ... ok
test wire::thrift::tests::skip_i32 ... ok
test wire::thrift::tests::skip_struct ... ok
test wire::thrift::tests::varint_multi_byte ... ok
test wire::thrift::tests::varint_single_byte ... ok
test wire::thrift::tests::zigzag_i32_negative ... ok
test wire::thrift::tests::zigzag_i32_positive ... ok
test wire::thrift::tests::zigzag_i64_positive ... ok

test result: ok. 136 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s (136/136)
- [x] Verified 143 tests (default features), 155 tests (all compression features: snappy,zstd,lz4,gzip) — 0 failures
- [x] : integrated decompression + column decode, 7 integration tests (gzip/zstd/snappy/lz4 round-trips, brotli unsupported, malformed input error, uncompressed passthrough)
- [x] Verified  (0 errors)
- [x] Typed column value extraction: compression pipeline (decompress before PLAIN/dict decode)
- [x] : , , (UNCOMPRESSED/SNAPPY/GZIP/LZ4_RAW/ZSTD/LZ4/BROTLI/ZLIB), feature-gated codec dispatch, 12 value-semantic tests
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

### Milestone 20: MATLAB .mat Reader (consus-mat)
- [x] consus-mat crate created with features: std, alloc, v73 (HDF5), compress (zlib)
- [x] detect::detect_version: HDF5/v5/v4 auto-detection (5 unit tests)
- [x] v4 reader: V4Header::parse + read_v4_variable + read_mat_v4 (3 integration tests)
- [x] v5 reader: V5FileHeader + DataTag + parse_matrix (mxDOUBLE..mxUINT64, mxCHAR, mxSPARSE, mxCELL, mxSTRUCT, complex, logical, miCOMPRESSED) + read_mat_v5 (expanded value-semantic integration coverage)
- [x] v73 reader: Hdf5File-backed root group traversal, MATLAB_class dispatch, numeric/logical/char/struct/cell (expanded synthetic HDF5 integration coverage, datatype-aware char endianness, explicit sparse rejection, compact-layout rejection coverage)
- [x] loadmat_bytes + loadmat public API
- [x] Workspace Cargo.toml updated: consus-mat added to members and workspace.dependencies
- [x] Model invariants strengthened with constructors/validation helpers for cell, char, logical, sparse, and struct arrays
- [x] v7.3 cell ordering verified for numeric child names ("0", "1", ...)
- [x] v7.3 logical, char, cell, and struct decoding covered by value-semantic integration tests
- [x] v5 unknown top-level elements are skipped with structural validation instead of hard erroring
- [x] v7.3 char decoding honors dataset datatype byte order
- [x] v7.3 sparse datasets return explicit UnsupportedFeature errors
- [x] v5 `mxOBJECT_CLASS` returns explicit UnsupportedFeature errors
- [x] v7.3 compact-layout datasets return explicit UnsupportedFeature errors
- [x] v5 `miCOMPRESSED` roundtrip is covered when the `compress` feature is enabled
- [x] v5 `miCOMPRESSED` returns explicit UnsupportedFeature without the `compress` feature
- [x] Removed dead byteorder and consus-compression dependencies from Cargo.toml
- [x] Fixed lib.rs crate documentation (feature gate column + Entry Points links)
- [x] Removed dead UnsupportedVersion error variant from MatError
- [x] v5 sparse: nzmax invariant enforced (ir.len() == nzmax, jc.len() == ncols+1)
- [x] v73 cell group: child ordering fixed (sort by numeric name before building cells vec)
- [x] v5 synthetic test suite: char, logical, complex, sparse roundtrip, sparse nzmax mismatch, cell, struct (7 new value-semantic tests)
- [x] v5 vacuous truncated test replaced with v5_truncated_element_returns_error (proper Err assertion)
- [x] v5 compressed feature-matrix suite added as dedicated integration coverage
- [x] Verified cargo test -p consus-mat: 22/22 tests pass (3 v4 + 12 v5 + 7 v73)
- [x] Verified cargo check --workspace: zero errors
- [x] MatStructArray SSOT cleanup: removed fields: Vec<String>; data keys are sole SSOT; new(shape, data) signature; field_names() returns impl Iterator<Item = &str>
- [x] v4 sparse matrix explicit rejection covered: v4_sparse_matrix_returns_unsupported_feature_error (type_code=2 synthetic fixture, exact UnsupportedFeature message assertion)
- [x] crates/consus-mat/README.md created: format coverage table, feature flags, quick start, canonical model, rejection policies, version-specific notes
- [x] consus-mat added to CI check/test/msrv matrix jobs; test-mat-features job added for default + no-compress feature-matrix verification
- [x] Verified cargo test -p consus-mat: 29/29 pass (5 unit + 4 v4 + 1 v5-compressed + 12 v5 + 7 v73)
- [x] Verified cargo test -p consus-mat --no-default-features --features std,alloc: 22/22 pass
- [x] Verified cargo check --workspace: zero errors
- [x] consus-hdf5 Hdf5FileBuilder: ChildDatasetSpec struct + add_group_with_attributes method added (enables nested group authoring with MATLAB_class and child datasets)
- [x] Model unit tests (42): MatNumericClass element_size + as_str; MatNumericArray numel + is_complex; MatSparseArray nnz + is_complex + 6 invariant-enforcing rejection tests; MatCellArray new + numel + into_parts; MatCharArray new + row_vector + numel; MatLogicalArray new + numel; MatStructArray field_names + field + field_data + numel + duplicate rejection + element count mismatch rejection
- [x] MatError Display unit tests (5): InvalidFormat, UnsupportedFeature, InvalidClass, ShapeError, CompressionError
- [x] v5_multiple_variables_roundtrip: 2 named scalar doubles in one v5 file, value-semantic
- [x] loadmat_from_reader_parses_test_fixture: std::fs::File + test_v5.mat roundtrip via loadmat
- [x] Doc test for loadmat_bytes: inline MAT v4 scalar double, verifies variable count and name
- [x] v73_cell_array_roundtrip: MATLAB_class="cell" group, 2-element decimal-named children, value-semantic
- [x] v73_struct_array_roundtrip: MATLAB_class="struct" group, fields x=42.0 y=99.0, value-semantic field lookup
- [x] Verified cargo test -p consus-mat: 71/71 pass (42 lib + 4 v4 + 1 v5-compressed + 14 v5 + 9 v73 + 1 doc)
- [x] Verified cargo test -p consus-mat --no-default-features --features std,alloc: 62/62 pass
- [x] Verified cargo test -p consus-hdf5: 321/321 pass
- [x] Verified cargo check --workspace: zero errors
- [x] v73 MATLAB_dims attribute parsing: matlab_dims_attr() helper; struct group shape preserved from attribute
- [x] v73 split_field_elements: non-scalar numeric field datasets split into per-element scalars
- [x] v73 non-scalar struct roundtrip: v73_struct_array_non_scalar_roundtrip ([1,2] struct, 2 fields x 2 elements each)
- [x] HDF5 DatasetLayout::Virtual variant added to property_list enum; encode_layout_with_chunk_index emits [3,3]
- [x] HDF5 layout parser: virtual class 3 now returns StorageLayout::Virtual (typed Ok) instead of Err
- [x] v73 virtual-layout rejection test: v73_virtual_layout_returns_unsupported_feature_error
- [x] v73 chunked dataset fixture: v73_chunked_double_array_roundtrip (6-element, chunk_size=3)
- [x] Verified cargo test -p consus-mat: 74/74 pass (42 lib + 4 v4 + 1 v5-compressed + 14 v5 + 12 v73 + 1 doc)
- [x] Verified cargo test -p consus-hdf5: 321/321 pass
- [x] Verified cargo check --workspace: zero errors (Sprint 6)
