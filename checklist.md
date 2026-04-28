# Consus — Implementation Checklist

## Current Sprint: Phase 3 — Parquet Nested Column Support + NWB Foundation

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
- [x] Read Parquet files as Consus datasets
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
- [x] Real file-backed dataset read API: `ParquetReader::new(bytes)` validates footer, decodes FileMetadata, materializes dataset; `read_column_chunk(rg, col)` iterates pages via `ColumnPageDecoder`, handles DataPage v1 (full decompression then split), DataPage v2 (split first, optional value decompression), DictionaryPage (retain dict across pages); `merge_column_values` concatenates per-page results — 21 value-semantic tests
- [x] Write Consus datasets to Parquet
  - [x] Canonical writer-side planning over `SchemaDescriptor` trees with nested/group lowering to leaf paths
  - [x] Thrift compact footer encoder for `FileMetadata`, `SchemaElement`, `RowGroupMetadata`, `ColumnChunkMetadata`, and `ColumnMetadata`
  - [x] Page header encoder for `PageHeader`, `DataPageHeader`, `DataPageHeaderV2`, and `DictionaryPageHeader`
  - [x] Row-source to leaf-column value lowering for flat and nested/group schemas
  - [x] Complete file emission with trailer validation and `PAR1` footer assembly
  - [x] Footer metadata and trailer roundtrip verification against the existing reader
  - [x] `encode_cell_plain`: PLAIN encoder for INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY
  - [x] `encode_bool_column_plain`: LSB-first bit-packing across full Boolean column (⌈count/8⌉ bytes)
  - [x] `physical_type_discriminant`: `ParquetPhysicalType` → parquet.thrift Type enum i32
  - [x] `build_file_bytes` emits real DataPage v1 pages; `ColumnMetadata` records correct byte offsets and sizes
  - [x] End-to-end writer→reader roundtrip: INT32 3 values, DOUBLE 2 values, BYTE_ARRAY 2 strings, BOOLEAN 4 values, two-column INT32+DOUBLE 2 rows
  - [x] Negative: Null in required column returns `InvalidFormat`
  - [x] Verified `cargo test -p consus-parquet --lib`: 175/175 pass (default features)
  - [x] Verified `cargo check --workspace`: 0 warnings, 0 errors
- [ ] Hybrid mode: Parquet tables inside Consus containers
- [x] Arrow array bridge (zero-copy) — completed via Milestone 25 zerocopy optional feature (see Milestone 25 below)

### Milestone 14a: Warning Cleanup (this sprint)
- [x] `consus-parquet/writer/mod.rs`: removed unused imports (`boxed::Box`, `PagePayload`, `split_data_page_v1`, `split_data_page_v2`); removed dead `encode_row_group_descriptor`; all writer encoder functions now exercised by `build_file_bytes`
- [x] `consus-arrow/conversion/mod.rs`: removed unused imports `TimeUnit`, `ArrowNullability`; removed dead `let arrow_type` binding in `ArrowFieldFromCoreBuilder::build`; prefixed unused enumerate index `_i` and unused `_bit_width`
- [x] `consus-zarr/lib.rs`: re-exported `ConsolidatedMetadataV2`, `ConsolidatedMetadataV3`, `MetadataEntryV2`, `MetadataEntryV3`, `ConsolidatedParseError`, `ConsolidatedSerializeError` from crate root to eliminate dead_code warnings
- [x] `consus/highlevel/dataset.rs`: added `#[allow(dead_code)]` to `pub(crate) fn backend()` (intentional internal API, no external caller yet)
- [x] Verified `cargo check --workspace`: 0 warnings, 0 errors
- [x] Verified `cargo test --workspace --lib`: 1095/1095 pass, 0 failures

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

### Milestone 21: Arrow Array Materialization Bridge (this sprint)
- [x] `consus-arrow/src/array/materialize.rs` created: `column_values_to_arrow(values: &ColumnValues) -> ArrowArray`
- [x] Boolean → FixedWidth element_width=1, 0x00/0x01 per element
- [x] Int32 → FixedWidth element_width=4, little-endian bytes
- [x] Int64 → FixedWidth element_width=8, little-endian bytes
- [x] Int96 → FixedWidth element_width=12, raw bytes preserved
- [x] Float → FixedWidth element_width=4, little-endian bytes
- [x] Double → FixedWidth element_width=8, little-endian bytes
- [x] ByteArray → VariableWidth with monotone offsets (offsets.len() == len + 1)
- [x] FixedLenByteArray → FixedWidth element_width=fixed_len, concatenated raw bytes
- [x] `pub mod materialize` declared in `consus-arrow/src/array/mod.rs`; `column_values_to_arrow` re-exported
- [x] `column_values_to_arrow` exported from `consus-arrow` crate root under `#[cfg(feature = "alloc")]`
- [x] 10 value-semantic tests: boolean_false_and_true_map_to_zero_and_one, empty_boolean_array_produces_empty_fixed_width, int32_three_values_stored_little_endian, int64_two_values_stored_little_endian, int96_one_value_raw_twelve_bytes_preserved, float32_two_values_stored_little_endian, double_two_values_stored_little_endian, byte_array_two_entries_variable_width_offsets_and_payload, empty_byte_array_column_produces_singleton_offset, fixed_len_byte_array_two_values_concatenated
- [x] Resolved pre-existing warnings: removed unused `ArrowField` import in `bridge/mod.rs`; removed duplicated `#[cfg(feature="alloc")] #[test]` in `memory/mod.rs`
- [x] Verified cargo test -p consus-arrow --lib array::materialize: 10/10 pass
- [x] Verified cargo test --workspace --lib: 1104/1104 pass (1095 previous + 10 new - 1 deduped)
- [x] Verified cargo check --workspace: 0 warnings, 0 errors

### Milestone 23: E2E ParquetWriter → ParquetReader → column_values_to_arrow Integration Tests (this sprint)
- [x] `consus-arrow/tests/parquet_arrow_e2e.rs` created: 6 end-to-end integration tests exercising the full write→read→materialize pipeline
- [x] `single_column_dataset` helper mirrors private writer test pattern; `two_column_dataset` helper for multi-column case
- [x] `fixed_parts` / `var_parts` ArrayData extraction helpers for byte-level assertions
- [x] `e2e_i32_three_values_pipeline`: INT32 [10, 20, 30] → ArrowArray FixedWidth(element_width=4); bytes == `[10,0,0,0, 20,0,0,0, 30,0,0,0]`
- [x] `e2e_i64_two_values_pipeline`: INT64 [i64::MAX, -1] → ArrowArray FixedWidth(element_width=8); bytes == `i64::MAX.to_le_bytes()` + `(-1i64).to_le_bytes()`
- [x] `e2e_double_two_values_pipeline`: DOUBLE [1.5, -0.25] → ArrowArray FixedWidth(element_width=8); bytes == IEEE 754 LE per value
- [x] `e2e_byte_array_two_values_pipeline`: BYTE_ARRAY ["hello", "world"] → ArrowArray VariableWidth; offsets=[0,5,10], payload=b"helloworld"
- [x] `e2e_boolean_four_values_pipeline`: BOOLEAN [true, false, true, true] → Parquet PLAIN bit-packing → decode → materialize → bytes=[0x01,0x00,0x01,0x01]
- [x] `e2e_two_column_int32_double_pipeline`: 2-column (INT32+DOUBLE) 2-row file; both columns materialized; byte-level assertions on both
- [x] Each test uses an independent struct-level `RowSource` impl to avoid name collision
- [x] Verified `cargo test -p consus-arrow --test parquet_arrow_e2e`: 6/6 pass
- [x] Verified `cargo check --workspace`: 0 errors, 0 warnings

### Milestone 24: Compressed Page Emission — Parquet Writer (this sprint)
- [x] `compress_page_values(data: &[u8], codec: CompressionCodec) -> Result<Vec<u8>>` added to `consus-parquet/src/encoding/compression.rs` as `pub(crate)`
- [x] UNCOMPRESSED: pass-through (`data.to_vec()`)
- [x] GZIP/ZLIB: `flate2::read::DeflateEncoder` (raw deflate, `Compression::default()`) under `#[cfg(feature = "gzip")]`
- [x] SNAPPY: `snap::raw::Encoder::new().compress_vec(data)` under `#[cfg(feature = "snappy")]`
- [x] ZSTD: `zstd::bulk::compress(data, 3)` under `#[cfg(feature = "zstd")]`
- [x] LZ4_RAW: `lz4_flex::compress(data)` under `#[cfg(feature = "lz4")]`
- [x] LZ4: `lz4_flex::compress_prepend_size(data)` under `#[cfg(feature = "lz4")]`
- [x] BROTLI: always returns `Error::UnsupportedFeature { feature: "parquet compression codec BROTLI (6)" }`
- [x] Disabled features return `Error::UnsupportedFeature` with actionable enable-feature message
- [x] `build_file_bytes` in `writer/mod.rs`: `_codec` renamed to `codec` (now used); PLAIN bytes compressed via `compress_page_values`; `page_header.uncompressed_page_size = plain_size`, `page_header.compressed_page_size = compressed_size`; `ColumnMetadata.codec = codec as i32`; `total_uncompressed_size` and `total_compressed_size` reflect header + respective payload sizes
- [x] Existing UNCOMPRESSED roundtrip tests unaffected: UNCOMPRESSED compress = identity → no behavioral change
- [x] `compress_page_values_uncompressed_passthrough`: output bytes == input bytes (always-active)
- [x] `compress_page_values_brotli_returns_unsupported`: `UnsupportedFeature` returned (always-active)
- [x] `writer_gzip_roundtrip_i32_three_values` (`#[cfg(feature = "gzip")]`): INT32 [42, -1, 0] written GZIP, read back via `ParquetReader`, values exact
- [x] `writer_gzip_roundtrip_byte_array` (`#[cfg(feature = "gzip")]`): BYTE_ARRAY ["foo", "baz"] written GZIP, read back, values exact
- [x] Verified `cargo test -p consus-parquet --lib`: 177/177 pass (default features, +2 compress tests)
- [x] Verified `cargo test -p consus-parquet --lib --features gzip`: 183/183 pass (+4 gzip tests vs default)
- [x] Verified `cargo check --workspace`: 0 errors, 0 warnings

### Milestone 25: Zero-Copy Materialization via zerocopy (this sprint)
- [x] `consus-arrow/Cargo.toml`: added `zerocopy = ["dep:zerocopy"]` feature; added `zerocopy = { workspace = true, optional = true, features = ["derive"] }` dependency
- [x] `fixed_to_le_bytes_fast<T: zerocopy::IntoBytes + zerocopy::Immutable>(slice: &[T]) -> Vec<u8>` helper in `materialize.rs` under `#[cfg(all(feature = "alloc", feature = "zerocopy", target_endian = "little"))]`; calls `IntoBytes::as_bytes(slice).to_vec()` — one allocation + one bulk memcpy
- [x] `zerocopy::Immutable` added to bound: required by `as_bytes` in zerocopy 0.8.48 (`Self: Immutable` where-clause on `IntoBytes::as_bytes`)
- [x] Int32 match arm: `#[cfg(all(feature = "zerocopy", target_endian = "little"))]` fast path + `#[cfg(not(...))]` element-by-element fallback
- [x] Int64 match arm: same cfg-selected pattern, element_width=8
- [x] Float match arm: same cfg-selected pattern, element_width=4
- [x] Double match arm: same cfg-selected pattern, element_width=8
- [x] Boolean, Int96, ByteArray, FixedLenByteArray: unchanged (not applicable — non-numeric, raw, or variable-width)
- [x] `zerocopy_i32_agrees_with_element_loop`: verifies fast path bytes == `to_le_bytes()` reference for [1, -1, i32::MAX, i32::MIN]; asserts len=4, element_width=4, values.len()=16
- [x] `zerocopy_f64_agrees_with_element_loop`: verifies fast path bytes == `to_le_bytes()` reference for [1.5, -0.25, f64::INFINITY, f64::NEG_INFINITY]; asserts len=4, element_width=8, values.len()=32
- [x] Verified `cargo test -p consus-arrow --lib`: 50/50 pass (no zerocopy feature)
- [x] Verified `cargo test -p consus-arrow --lib --features zerocopy`: 52/52 pass (2 new zerocopy agreement tests active and passing)
- [x] Verified `cargo check --workspace`: 0 errors, 0 warnings

### Milestone 22: FITS Table Value Decoding Closure (artifact sync)
- [x] `decode_binary_column` + `decode_scalar_binary` in `consus-fits/src/table/decode.rs`: all 13 FITS Standard 4.0 TFORM codes (L/X/B/I/J/K/A/E/D/C/M/P/Q), big-endian extraction, repeat > 1 array wrapping, 24 value-semantic unit tests
- [x] `decode_ascii_column` in `consus-fits/src/table/decode.rs`: A/I/F/E/D format codes, trailing-space stripping, Fortran D-notation normalization, 11 value-semantic unit tests
- [x] `FitsTableData::decode_row` dispatches to binary/ASCII decoder per table kind; returns `Vec<FitsColumnValue>` in column order
- [x] `FitsTableData::decode_column` iterates all rows for a single column; returns `Vec<FitsColumnValue>`
- [x] `FitsColumnValue` enum covers all decoded physical types: Logical, Bits, UInt8, Int16, Int32, Int64, Chars, Float32, Float64, Complex32, Complex64, Descriptor32, Descriptor64, Array
- [x] Backlog entries P3.2 ASCII and Binary table column value decoding marked complete (were implemented but not recorded)

### Milestone 26: Multi-Row-Group Parquet Writer (this sprint)
- [x] `ParquetWriter` struct: `row_group_size: Option<usize>` field added; `new()` initializes to `None`
- [x] `with_row_group_size(n: usize) -> Self` builder method: `n=0` reverts to unlimited (single group)
- [x] `encode_leaf_columns(plan, rows, row_start, row_end) -> Result<Vec<Vec<u8>>>` private helper extracted from `build_file_bytes`
- [x] `build_file_bytes` refactored: accepts `row_group_size: Option<usize>`; partitions `[0, N)` into groups of `effective_group_size = row_group_size.unwrap_or(N.max(1))`; always emits ≥1 row group
- [x] Partition invariants: each group `g` spans `[g*n, min((g+1)*n, N))`; last group may be smaller; `FileMetadata.num_rows == N`; each `RowGroupMetadata.num_rows == group_end - group_start`; each `ColumnMetadata.data_page_offset` is the absolute file byte offset for that page
- [x] `write_dataset` call updated to pass `self.row_group_size` to `build_file_bytes`
- [x] `#[cfg(test)] mod tests_extra;` declaration added to `writer/mod.rs`; `writer/tests_extra.rs` created
- [x] `multi_row_group_even_split_ten_values_two_groups`: INT32 [1..10], rg_size=5 → 2 groups of 5; metadata.row_groups.len()==2; num_rows[0]==5; num_rows[1]==5; concat values exact
- [x] `multi_row_group_uneven_split_seven_values_three_groups`: [10..16], rg_size=3 → groups[3,3,1]; values exact
- [x] `multi_row_group_size_larger_than_row_count_gives_one_group`: rg_size=100, 3 values → 1 group
- [x] `multi_row_group_exact_multiple_of_group_size`: 6 values, rg_size=2 → 3 groups of 2
- [x] `default_writer_produces_single_row_group`: no rg_size → 1 group, 5 rows
- [x] `with_row_group_size_zero_produces_single_group`: rg_size=0 → 1 group (unlimited)
- [x] `prop_multi_row_group_i32_roundtrip`: proptest ∀ values (1..=50 i32s), m (1..=20) → roundtrip identity

### Milestone 27: Compressed Writer Roundtrip Tests — SNAPPY/ZSTD/LZ4 (this sprint)
- [x] `writer_snappy_roundtrip_i32_three_values` (`#[cfg(feature = "snappy")]`): INT32 [42, -1, 0] written SNAPPY, read back via `ParquetReader`, exact value equality
- [x] `writer_zstd_roundtrip_i32_three_values` (`#[cfg(feature = "zstd")]`): INT32 [1000, -9999, i32::MAX] written ZSTD, read back, exact
- [x] `writer_lz4_raw_roundtrip_i32_three_values` (`#[cfg(feature = "lz4")]`): INT32 [55, -55, 0] written LZ4_RAW, read back, exact
- [x] `writer_lz4_roundtrip_i32_three_values` (`#[cfg(feature = "lz4")]`): INT32 [1, 2, 3] written LZ4, read back, exact
- [x] All four tests in `writer/tests_extra.rs`; all feature-gated; existing UNCOMPRESSED and GZIP tests unaffected

### Milestone 28: proptest Roundtrip Suite (this sprint)
- [x] `#[cfg(test)] mod compression_proptest;` declared in `encoding/mod.rs`; `encoding/compression_proptest.rs` created
- [x] `prop_gzip_compress_decompress_identity`: ∀ data ∈ Vec<u8> [0..1024]: decompress(compress(data, Gzip), Gzip, |data|) == data (`#[cfg(feature="gzip")]`)
- [x] `prop_zlib_compress_decompress_identity`: same for Zlib (`#[cfg(feature="gzip")]`)
- [x] `prop_snappy_compress_decompress_identity` (`#[cfg(feature="snappy")]`)
- [x] `prop_zstd_compress_decompress_identity` (`#[cfg(feature="zstd")]`)
- [x] `prop_lz4_raw_compress_decompress_identity` (`#[cfg(feature="lz4")]`)
- [x] `prop_lz4_compress_decompress_identity` (`#[cfg(feature="lz4")]`)
- [x] `#[cfg(test)] mod plain_proptest;` declared in `encoding/mod.rs`; `encoding/plain_proptest.rs` created
- [x] `prop_i32_plain_roundtrip`: ∀ v ∈ i32: decode_plain_i32(v.to_le_bytes(), 1) == [v]
- [x] `prop_i64_plain_roundtrip`: ∀ v ∈ i64: decode_plain_i64(v.to_le_bytes(), 1) == [v]
- [x] `prop_f32_plain_roundtrip_bits`: ∀ bits ∈ u32: decode_plain_f32(f32::from_bits(bits).to_le_bytes(), 1)[0].to_bits() == bits (covers NaN)
- [x] `prop_f64_plain_roundtrip_bits`: ∀ bits ∈ u64: same for f64 (covers NaN, ±Inf)
- [x] `prop_i96_plain_roundtrip`: ∀ raw ∈ [u8; 12]: decode_plain_i96(raw, 1) == [raw]
- [x] `prop_byte_array_plain_roundtrip`: ∀ data ∈ Vec<u8> [0..256]: decode_plain_byte_array(4-byte-LE-len || data, 1) == [data]
- [x] `prop_fixed_len_byte_array_plain_roundtrip`: ∀ data ∈ Vec<u8> [1..16]: decode_plain_fixed_byte_array(data, 1, |data|) == [data]
- [x] `prop_bool_encode_decode_roundtrip` in `writer/tests_extra.rs`: ∀ bools (1..=64): decode_plain_boolean(encode_bool_column_plain(bools), |bools|) == bools

### Milestone 29: Memory-Mapped I/O Backend (this sprint)
- [x] `memmap2 = { version = "0.9" }` added to `[workspace.dependencies]` in root `Cargo.toml`
- [x] `mmap` feature added to `consus-io/Cargo.toml`: `mmap = ["dep:memmap2", "std"]`; `memmap2` optional dep wired
- [x] `tempfile` added to `consus-io` dev-dependencies (workspace)
- [x] `consus-io/src/io/sync/mmap.rs` created: `MmapReader` struct wrapping `memmap2::Mmap`
- [x] `MmapReader::open(path: impl AsRef<Path>) -> Result<Self>` — opens file + maps read-only
- [x] `MmapReader::from_file(file: &File) -> Result<Self>` — maps an already-open file
- [x] `MmapReader::as_slice() -> &[u8]` — borrows the mapped region
- [x] `ReadAt for MmapReader`: zero-length read succeeds unconditionally; out-of-bounds returns `Error::BufferTooSmall`; offset overflow returns `Error::Overflow`; in-bounds copies exact slice
- [x] `Length for MmapReader`: returns `mmap.len() as u64`
- [x] `MmapReader` is `Send + Sync` (inherits from `memmap2::Mmap`); `WriteAt` and `Truncate` intentionally absent (read-only; `RandomAccess` not implemented)
- [x] `unsafe` block isolated to `from_file`; safety contract (no concurrent truncation) documented in module and struct Rustdoc
- [x] 8 unit tests in `mmap.rs`: `mmap_reader_open_and_len`, `mmap_reader_read_at_beginning`, `mmap_reader_read_at_offset`, `mmap_reader_zero_len_read_succeeds`, `mmap_reader_out_of_bounds_returns_buffer_too_small`, `mmap_reader_from_file`, `mmap_reader_as_slice_equals_read_at`, `mmap_reader_is_send_sync`
- [x] `#[cfg(feature = "mmap")] pub mod mmap;` added to `sync/mod.rs`
- [x] `#[cfg(feature = "mmap")] pub use io::sync::mmap::MmapReader;` added to `lib.rs`
- [x] `[[test]] name = "unit_mmap" required-features = ["mmap"]` added to `consus-io/Cargo.toml`
- [x] `tests/unit_mmap.rs` created: 3 integration tests (`integration_mmap_read_large_payload` 64 KiB window, `integration_mmap_read_last_bytes`, `integration_mmap_length_matches_file_size`)
- [x] Verified `cargo test -p consus-io --lib`: 20/20 pass (default, no mmap)
- [x] Verified `cargo test -p consus-io --features mmap`: 28/28 lib + 3 integration = 31 pass
- [x] Verified `cargo check --workspace`: 0 errors, 0 warnings

### Milestone 30: Parquet Reader Proptest Suite (this sprint)
- [x] `consus-parquet/src/reader/reader_proptest.rs` created: 5 proptest roundtrip properties
- [x] `prop_reader_i32_roundtrip`: ∀ vals ∈ Vec<i32> [1..=100]: write INT32 column → read → `ColumnValues::Int32` exact match; asserts `metadata.num_rows == n` and `col.len() == n`
- [x] `prop_reader_f64_roundtrip`: ∀ vals ∈ Vec<f64 (NORMAL)> [1..=50]: write DOUBLE → read → exact match; uses `proptest::num::f64::NORMAL` to exclude NaN/Inf and preserve equality
- [x] `prop_reader_bool_roundtrip`: ∀ bools ∈ Vec<bool> [1..=128]: write BOOLEAN → read → exact match; covers LSB-first bit-packing across arbitrary lengths
- [x] `prop_reader_byte_array_roundtrip`: ∀ vals ∈ Vec<Vec<u8 [0..=16]>> [1..=30]: write BYTE_ARRAY → read → exact match; covers length-prefixed variable-width encoding
- [x] `prop_reader_two_column_i32_f64_roundtrip`: ∀ (ints, doubles) with n = min(|ints|, |doubles|) ∈ [1..=30]: write 2-column INT32+DOUBLE → read both columns → exact match on both
- [x] All 5 proptests use `prop_assert_eq!` on computed `Vec` values, not just `is_ok()` guards
- [x] `#[cfg(test)] mod reader_proptest;` declaration added to `reader/mod.rs`
- [x] Verified `cargo test -p consus-parquet --lib`: 197/197 pass (+5 vs 192 baseline)
- [x] Verified `cargo check --workspace`: 0 errors, 0 warnings

### Milestone 31: Criterion Benchmark Harness (this sprint)
- [x] `consus-parquet/benches/parquet_rw.rs` created: `bench_write_i32` + `bench_read_i32` at 1K/10K/100K rows using criterion 0.5.x `BenchmarkId::from_parameter` API
- [x] `[[bench]] name = "parquet_rw" harness = false` added to `consus-parquet/Cargo.toml`
- [x] `consus-arrow/benches/arrow_bridge.rs` created: `bench_bridge_i32`, `bench_bridge_double` at 1K/10K/100K rows; `bench_bridge_byte_array` at 1K/10K rows; all measure `column_values_to_arrow` throughput
- [x] `[[bench]] name = "arrow_bridge" harness = false` added to `consus-arrow/Cargo.toml`
- [x] `consus-parquet` added to `consus-arrow` dev-dependencies (provides `ColumnValues` for bench)
- [x] Verified `cargo check --bench parquet_rw -p consus-parquet`: 0 errors, 0 warnings
- [x] Verified `cargo check --bench arrow_bridge -p consus-arrow`: 0 errors, 0 warnings
- [x] Verified `cargo check --workspace`: 0 errors, 0 warnings

### Milestone 32: Optional and Repeated Flat Column Write/Read Support
- [x] `EncodedLeafColumn` struct separating payload from metadata
- [x] `encode_rle_hybrid` pure-RLE writer for DataPage v1 level sections
- [x] `encode_levels_for_page_v1` 4-byte LE length prefix wrapper
- [x] Optional flat column write: `CellValue::Null` emits def_level=0; leaf emits def_level=1
- [x] Repeated flat column write: empty list emits (rep=0,def=0); items emit (rep=0/1,def=1)
- [x] `DataPageHeader.definition_level_encoding` set to 3 (RLE) for optional/repeated
- [x] `DataPageHeader.num_values` = total count including null positions
- [x] `ColumnValuesWithLevels` struct with values + rep/def levels
- [x] `ColumnPageDecoder::decode_pages_from_chunk_bytes` uses non-null count from def_levels
- [x] `ColumnPageDecoder::decode_pages_with_levels` returns `ColumnValuesWithLevels`
- [x] `ParquetReader::read_column_chunk_with_levels` public API
- [x] Writer roundtrip test: optional i32 with nulls (value-semantic)
- [x] Writer roundtrip test: repeated i32 (value-semantic)
- [x] Property-based test: optional i32 roundtrip (proptest)

### Milestone 33: consus-nwb Foundation
- [x] `Cargo.toml` with consus-core + consus-hdf5 dependencies
- [x] `src/lib.rs` with 10 module declarations and architecture docs
- [x] 10 stub `mod.rs` files (conventions, file, group, io, metadata, model, namespace, storage, validation, version)
- [x] Registered in workspace `Cargo.toml`
- [x] `cargo check -p consus-nwb` passes
- [x] `NwbFile::open(bytes)` entry point with HDF5 validation
- [x] Session metadata read (identifier, session_description, session_start_time)
- [x] TimeSeries read (data dataset + timestamps/rate)
- [x] Namespace version detection (`NwbVersion::parse`, `detect_version`)
- [x] Conformance validation skeleton (`validate_root_attributes`)

### Milestone 34: Parquet Nested Column Write Support — Dremel Algorithm (this sprint)
- [x] `top_field_idx: usize` field added to `LeafColumnPlan` with public accessor
- [x] `lower_column` accepts and propagates `top_field_idx` from `plan()` loop index
- [x] `traverse_dremel_into` — recursive Dremel traversal for arbitrary nesting depth
  - [x] Required leaf: emit (rep, def_above, value); Null → error
  - [x] Optional leaf: Null → (rep, def_above); value → (rep, def_above+1, value)
  - [x] Repeated leaf: empty → (rep, def_above); items → (rep/this_rep, def_above+1, item)
  - [x] Required group: navigate children by positional index; Null → error
  - [x] Optional group: Null → one null entry at def_above; Group → recurse def_above+1
  - [x] Repeated group: empty → one null at def_above; items → recurse with rep tracking
- [x] `encode_leaf_columns` refactored to unified Dremel path (replaces 3 flat branches + UnsupportedFeature fallback)
- [x] `proptest!` block placement fixed: nested-column tests moved to standalone `#[test]`
- [x] Roundtrip test: required group two leaves (`nested_required_group_two_leaves_roundtrip`)
- [x] Roundtrip test: optional group with required leaf (`nested_optional_group_roundtrip`)
- [x] Roundtrip test: repeated group with required leaf (`nested_repeated_group_roundtrip`)
- [x] Roundtrip test: deeply nested optional-in-optional group (`deeply_nested_optional_in_optional_group_roundtrip`)

### Milestone 35: consus-nwb Extended Read Path (this sprint — CLOSED)
- [x] Integer dataset promotion to `f64` in `read_f64_dataset` — all signed/unsigned 8/16/32/64-bit widths, both byte orders
- [x] `read_scalar_f64_dataset` helper — wraps `read_f64_dataset` for scalar (single-element) datasets
- [x] `read_f64_attr` helper — finds float/int/uint attribute by name and returns as f64
- [x] `starting_time` + `rate` read from `{path}/starting_time` scalar dataset and its `rate` attribute (not group attrs)
- [x] Dead `read_scalar_f64_attr` private method removed (replaced by storage helpers)
- [x] `group/mod.rs` — `NwbGroupChild` struct + `list_typed_group_children` (filters to NodeType::Group, extracts `neurodata_type_def`/`neurodata_type_inc`)
- [x] `conventions/mod.rs` — `NeuroDataType` enum (17 variants), `classify_neurodata_type`, `is_timeseries_type` (def match + inc inheritance + known subtype set)
- [x] `namespace/mod.rs` — `NwbNamespace` struct with `core()` and `hdmf_common()` constructors and `CORE_NAME` constant
- [x] `NwbFile::list_time_series(group_path)` — enumerates TimeSeries children at any group path; `""` scans root
- [x] `consus-hdf5 list_group_at` fix: guards v1 symbol-table fallback with `SYMBOL_TABLE` message presence check; v2 groups with no children now return empty list instead of `InvalidFormat`
- [x] Tests: 8 integer promotion + scalar dataset + f64 attr tests (storage); 5 group traversal tests; 18 classification + 10 is_timeseries_type tests (conventions); 13 namespace tests; 7 file-level tests (rate timing, list_time_series filtering, subtype detection, empty group, not-found)
- [x] Verified `cargo test -p consus-nwb --lib` → 130/130
- [x] Verified `cargo test --workspace` → 2199/2199; `cargo check --workspace` → 0 errors, 0 warnings

### Milestone 36: Parquet Multi-Page Column Chunk Splitting (this sprint — CLOSED)
- [x] `page_row_limit: Option<usize>` field added to `ParquetWriter`
- [x] `with_page_row_limit(limit: usize) -> Self` builder method: `0` → `None` (unlimited/single page)
- [x] `build_file_bytes` accepts `page_row_limit: Option<usize>`; `write_dataset` passes `self.page_row_limit`
- [x] Page range computation: `effective_page_rows = limit.max(1)`; `ceil(group_rows / effective_page_rows)` pages per column chunk; last page trimmed to `group_end`
- [x] Per-page encoding: `encode_leaf_columns(plan, rows, page_start, page_end)` called once per page range
- [x] Transpose: `pages_by_column[leaf_idx][page_idx]` built before any file writes
- [x] Contiguous emission: all pages of one column chunk emitted before the next column chunk begins
- [x] `data_page_offset` = byte offset of first page header (snapshotted before first write)
- [x] `total_uncompressed_size` = Σ(page_header_bytes + uncompressed_payload) over all pages
- [x] `total_compressed_size` = Σ(page_header_bytes + compressed_payload) over all pages
- [x] `num_values` = Σ(enc.num_values) over all pages
- [x] Zero-row guard: `group_rows == 0` forced to single empty page (preserves prior behavior)
- [x] Single-page path unchanged when `page_row_limit = None`
- [x] Tests: `multi_page_i32_two_pages_data_roundtrip`, `multi_page_three_pages_all_values_preserved`, `multi_page_uneven_split_last_page_smaller`, `multi_page_limit_larger_than_rows_gives_one_page`, `multi_page_combined_with_multi_row_group`, `prop_multi_page_i32_roundtrip`
- [x] Verified `cargo test -p consus-parquet --lib` → 215/215

### Milestone 37: NWB Write Path (this sprint — CLOSED)
- [x] `NwbFileBuilder` — construct root HDF5 group with required NWB metadata attributes
- [x] Required root attributes: `neurodata_type_def = "NWBFile"`, `nwb_version`, `identifier`, `session_description`, `session_start_time`
- [x] `write_time_series(ts: &TimeSeries)` — write group with `data` dataset + `timestamps` or `starting_time` dataset
- [x] `TimeSeries` with `timestamps`: emit `data` (f64 array) + `timestamps` (f64 array) datasets
- [x] `TimeSeries` with rate: emit `data` (f64 array) + `starting_time` scalar dataset with `rate` float32 attribute
- [x] `neurodata_type_def = "TimeSeries"` attribute on each written TimeSeries group
- [x] Units table write: `Units` group with `spike_times` VectorData dataset
- [x] `NwbFile::units_spike_times()` — read path added for Units roundtrip verification
- [x] Roundtrip tests: write then re-open with `NwbFile::open` and verify all fields (12 tests)
- [x] Namespace conformance validation before write (`validate_time_series_for_write` — 7 tests)
- [x] `NwbFileBuilder::new` rejects empty `identifier` / `session_description` before any HDF5 bytes written
- [x] `validate_time_series_for_write` rejects no-timing, zero-rate, negative-rate, and `-∞`-rate
- [x] Verified `cargo test -p consus-nwb --lib` → 149/149
- [x] Verified `cargo test --workspace` → 2219/2219; `cargo check --workspace` → 0 errors, 0 warnings

### Milestone 38: NWB Verification Against Real Files (deferred — requires external fixtures)
- [ ] Download and test against Allen Brain Observatory NWB 2.x sample file
- [ ] Test `session_metadata`, `list_time_series`, `time_series` against real file
- [ ] Verify integer dataset promotion (i16 neural data) via real file

### Milestone 40: NWB ElectrodeTable + UnitsTable + Storage String/U64 + README — CLOSED
- [x] `read_string_dataset` in `consus-nwb::storage` — decode `FixedString` dataset → `Vec<String>`; trailing null bytes stripped; UTF-8 validated
- [x] `read_u64_dataset` in `consus-nwb::storage` — decode integer dataset → `Vec<u64>`; all 8/16/32/64-bit signed and unsigned widths; both byte orders
- [x] `decode_raw_as_u64` private helper — mirrors `decode_raw_as_f64`; rejects non-integer datatypes with `UnsupportedFeature`
- [x] 8 new value-semantic storage tests: u32→u64 widening, u64 identity, i32 signed bit-pattern cast, float→u64 rejection, FixedString exact-fill, null-padded strip, all-null element → empty string, wrong-type → `UnsupportedFeature`
- [x] `UnitsTable` model in `consus-nwb::model::units` — fields `spike_times_per_unit: Vec<Vec<f64>>`, `ids: Option<Vec<u64>>`; `#[derive(Debug, Clone, PartialEq)]`
- [x] `UnitsTable::new` — no-IDs constructor
- [x] `UnitsTable::from_parts` — constructor with optional IDs; validates `ids.len() == spike_times_per_unit.len()`
- [x] `UnitsTable::from_vectordata` — HDMF VectorIndex decode; validates monotone index, last index == flat_times.len(), ids length
- [x] `UnitsTable::flat_spike_times()` — flatten per-unit arrays into wire-format contiguous array
- [x] `UnitsTable::cumulative_index()` — compute cumulative end-offset array for `spike_times_index` dataset
- [x] `UnitsTable::num_units()`, `is_empty()`, `spike_times_per_unit()`, `ids()` accessors
- [x] 18 value-semantic `UnitsTable` unit tests: construction, VectorIndex decode (2-unit, 3-unit-with-ids, empty, zero-spike unit), error paths (last-index mismatch, non-monotone, ids-length mismatch), flat+index roundtrip, Clone/PartialEq
- [x] `ElectrodeRow` in `consus-nwb::model::electrode` — fields `id: u64`, `location: String`, `group_name: String`; `#[derive(Debug, Clone, PartialEq, Eq)]`
- [x] `ElectrodeTable` in `consus-nwb::model::electrode` — `rows: Vec<ElectrodeRow>`; `#[derive(Debug, Clone, PartialEq, Eq)]`
- [x] `ElectrodeTable::from_rows`, `from_columns` (validates equal column lengths), `empty`
- [x] `ElectrodeTable::len()`, `is_empty()`, `rows()`, `get(i)` accessors
- [x] `ElectrodeTable::id_column()`, `location_column()`, `group_name_column()` iterators
- [x] 13 value-semantic `ElectrodeTable` unit tests: from_rows, empty, from_columns, length-mismatch rejection, get bounds, column iterators, Clone/PartialEq, Debug
- [x] `pub mod units; pub mod electrode;` added to `consus-nwb::model::mod`
- [x] `NwbFile::units_table()` — reads `Units/spike_times` (f64) + `Units/spike_times_index` (u64) + optional `Units/id` (u64); decodes via `UnitsTable::from_vectordata`
- [x] `NwbFile::electrode_table()` — reads `electrodes/id` (u64) + `electrodes/location` (FixedString) + `electrodes/group_name` (FixedString); builds via `ElectrodeTable::from_columns`
- [x] `NwbFileBuilder::write_units_table(&UnitsTable)` — emits `Units` group; `spike_times` VectorData (f64 LE) + `spike_times_index` VectorIndex (u64 LE); optional `id` dataset; correct `neurodata_type_def` + `description` attributes per HDMF spec
- [x] `NwbFileBuilder::write_electrode_table(&ElectrodeTable)` — emits `electrodes` DynamicTable group; `id` (u64 LE), `location` (FixedString, null-padded to max len), `group_name` (FixedString, null-padded to max len); `neurodata_type_def = "DynamicTable"`, `description`, `colnames` attributes
- [x] 7 new file integration tests: `write_units_table_with_ids_roundtrip`, `write_units_table_no_ids_roundtrip`, `write_units_table_empty_roundtrip`, `write_electrode_table_roundtrip` (3 rows), `write_electrode_table_empty_roundtrip`, `units_table_missing_returns_not_found`, `electrode_table_missing_returns_not_found`
- [x] `crates/consus-nwb/README.md` created — format overview, feature flag table, quick-start read/write examples, module architecture tree, NWB compliance table (implemented / not-yet-implemented), spec references, license
- [x] Verified `cargo test -p consus-nwb --lib` → 211/211
- [x] Verified `cargo test --workspace` → 2285/2285; `cargo check --workspace` → 0 errors, 0 warnings

### Milestone 39: NWB Extended Read Path + HDF5 Nested Group Write — CLOSED
- [x] `ChildGroupSpec<'a>` — new public struct in `consus-hdf5::file::writer` for specifying sub-group children
- [x] `write_group_node` — private recursive helper in `consus-hdf5::file::writer`; extracts and generalises the dataset-write + object-header-write logic from `add_group_with_attributes`
- [x] `add_group_with_attributes` refactored to delegate to `write_group_node` (zero behavior change, backward compatible)
- [x] `Hdf5FileBuilder::add_group_with_children` — new public method accepting both `ChildDatasetSpec` and `ChildGroupSpec` for arbitrary-depth nested group authoring
- [x] 3 new value-semantic HDF5 writer tests: navigability, dataset integrity, backward compat
- [x] `NwbSubjectMetadata` — new struct in `consus-nwb::metadata` with 5 optional fields (`subject_id`, `species`, `sex`, `age`, `description`); `from_parts` constructor and accessors
- [x] 7 new `NwbSubjectMetadata` unit tests (all-Some, all-None, partial, equality, clone, debug, Unicode)
- [x] `NwbFile::subject()` — reads `general/subject` group attributes as `NwbSubjectMetadata`
- [x] `NwbFile::list_acquisition()` — thin delegation over `list_time_series("acquisition")`
- [x] `NwbFile::list_processing(module_name)` — thin delegation over `list_time_series("processing/{module}")`
- [x] `NwbFileBuilder::write_subject(&NwbSubjectMetadata)` — writes `general/subject` nested group using `add_group_with_children`
- [x] Negative tests: `subject()` NotFound, `list_acquisition()` NotFound, `list_processing()` NotFound
- [x] Positive roundtrip tests: `write_subject` all-fields, `write_subject` partial-fields, `list_acquisition`, `list_processing`
- [x] Proptest roundtrips: `roundtrip_timestamps_timeseries`, `roundtrip_rate_timeseries`, `roundtrip_units_spike_times`
- [x] `proptest = "1"` added to `consus-nwb` dev-dependencies
- [x] Verified `cargo test -p consus-nwb --lib` → 166/166
- [x] Verified `cargo test --workspace` → 2239/2239; `cargo check --workspace` → 0 errors, 0 warnings
