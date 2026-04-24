# Consus - Gap Audit

## Audit Date: 2026-04-22
## Scope: Phase 2 - Zarr Chunk I/O, Zarr v3 Metadata Correctness, and netCDF-4 Semantic Extraction Audit

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
| Z-105 | Zarr v3 dimension-name preservation | Canonical `ArrayMetadata` now carries `dimension_names: Option<Vec<String>>`. `metadata/v2.rs`, `metadata/v3.rs`, `metadata/consolidated.rs`, shard tests, chunk tests, and roundtrip integration tests were updated so v3 `dimension_names` survive `parse -> canonical -> write -> parse -> canonical`. Verification: `cargo test -p consus-zarr` passes with `metadata::v3::tests::test_dimension_names_roundtrip`, `metadata::v3::tests::from_array_metadata_roundtrip`, and `roundtrip_zarr::v3_write_metadata_and_data_roundtrip` succeeding. |
| Z-106 | Zarr v3 group-attribute preservation on write | `ZarrJson::from_group_metadata` no longer drops canonical group attributes. Attribute values are converted into JSON scalars/arrays and preserved through `from_group_metadata -> to_json -> parse -> to_group_canonical`. Verification: `cargo test -p consus-zarr` passes with `metadata::v3::tests::from_group_metadata_roundtrip` succeeding. |
| Z-107 | Typed float fill-value expansion correctness | `expand_fill_value` now emits width-correct IEEE bytes for float32 and float64 instead of truncating an f64 byte pattern. This closes the incorrect float32 fill-value materialization path. Verification: `cargo test -p consus-zarr` passes with `chunk::tests::expand_fill_value_float32_one`, `chunk::tests::expand_fill_value_float64_one`, and the existing array roundtrip tests succeeding. |
| N-001 | netCDF variable and group attribute preservation | `NetcdfVariable` and `NetcdfGroup` now carry decoded attribute vectors. HDF5 extraction attaches decodable dataset and group attributes, excluding dimension-scale marker attributes from variable semantic payloads. Verification: `cargo test -p consus-netcdf` passes with `variable::tests::variable_attributes_and_object_header_address_attach`, `reference_netcdf::group_lookup_methods`, and all integration suites succeeding. |
| N-002 | Unlimited-dimension propagation from HDF5 shape extents | HDF5-backed netCDF extraction now maps unlimited HDF5 extents into `NetcdfDimension::unlimited(...)` and marks variables unlimited when `dataset.shape.has_unlimited()`. Verification: `cargo test -p consus-netcdf` passes with `reference_netcdf::bridge_unlimited_dimension_mapping`, `reference_netcdf::unlimited_dimension_model`, and `roundtrip_netcdf::unlimited_dimension_roundtrip` succeeding. |
| N-003 | CF attribute-name coverage expansion | The conventions module now recognizes additional CF keys: `add_offset`, `scale_factor`, `missing_value`, `valid_range`, `valid_min`, `valid_max`, `calendar`, `positive`, `formula_terms`, `ancillary_variables`, `flag_values`, `flag_meanings`, `flag_masks`, and `compress`. Verification: `cargo test -p consus-netcdf` passes with `conventions::tests::validate_cf_attribute_names` succeeding. |
| N-004 | netCDF stale scaffolding removal | Empty `src/core/` scaffolding under `consus-netcdf` was removed to restore SSOT and eliminate dead structure. Verification: `cargo test -p consus-netcdf` and `cargo test --workspace` both pass after removal. |
| N-005 | netCDF variable byte-read bridge | `consus-netcdf::hdf5::variable::read_variable_bytes` now reads full variable payloads for HDF5-backed netCDF variables using authoritative HDF5 APIs. Contiguous datasets are read through `Hdf5File::read_contiguous_dataset_bytes`, chunked datasets through `Hdf5File::read_chunked_dataset_all_bytes`, and missing object-header addresses fail deterministically. Compact and virtual layouts remain explicitly unsupported. Verification: `cargo test -p consus-netcdf` passes with `hdf5::variable::tests::read_variable_bytes_reads_contiguous_dataset_payload`, `hdf5::variable::tests::read_variable_bytes_reads_chunked_dataset_payload`, `integration_netcdf_hdf5::contiguous_variable_bytes_roundtrip`, `integration_netcdf_hdf5::chunked_variable_bytes_roundtrip`, and `integration_netcdf_hdf5::variable_bytes_require_object_header_address` succeeding. |
| N-006 | DIMENSION_LIST parsing and variable-to-dimension binding | `consus-netcdf::hdf5` now parses `DIMENSION_LIST` object-reference payloads and resolves variable axis names against discovered dimension-scale dataset addresses during group extraction. When resolution succeeds, variable dimensions preserve referenced scale order; when absent or invalid, extraction falls back to conservative synthetic names. Verification: `cargo test -p consus-netcdf` passes with `hdf5::dimension_scale::tests::dimension_list_addresses_decode_object_references_in_axis_order`, `hdf5::dimension_scale::tests::resolve_dimension_names_from_list_uses_dimension_scale_mapping`, `hdf5::group::tests::dimension_names_from_dimension_list_resolves_axis_order`, and `integration_netcdf_hdf5::missing_dimension_list_falls_back_to_synthetic_dimension_names` succeeding. |
| N-007 | Nested-group dimension inheritance and scoped resolution | `consus-netcdf::model::NetcdfGroup` now validates variables against dimensions visible through the full ancestor scope chain instead of only the local group. Child groups may inherit ancestor dimensions and legally shadow them with local declarations, with nearest-scope resolution winning deterministically. Verification: `cargo test -p consus-netcdf` passes with `model::tests::group_validation_accepts_ancestor_scoped_dimension_reference`, `model::tests::group_validation_rejects_missing_ancestor_scoped_dimension_reference`, `model::tests::dimension_in_scope_prefers_local_shadowing_dimension`, `reference_netcdf::inherited_dimension_scope_validation`, `reference_netcdf::shadowed_dimension_scope_resolution`, `reference_netcdf::missing_inherited_dimension_is_rejected`, `roundtrip_netcdf::nested_group_dimension_inheritance_roundtrip`, and `roundtrip_netcdf::nested_group_dimension_shadowing_roundtrip` succeeding. |
| Z-104 | Artifact synchronization | README.md Zarr row updated to reflect verified state: boundary-chunk stride fix and sharding verified; v3 metadata write path complete. Old Z-102 "remaining gap" sentence removed. |
| P-001 | Parquet schema mapping verification | `consus-parquet` now covers canonical Parquet physical, logical, field, hybrid, Arrow bridge, conversion, dataset-descriptor, and wire-trailer validation models with deterministic mappings for `Boolean`, integer, float, complex, fixed string, variable string, opaque, compound, array, enum, varlen, and reference Core datatypes. Dataset canonicalization preserves nested group fields as canonical `Compound` datatypes with ordered child offsets when fixed-size children permit exact sizing, and preserves repeated fields as canonical `VarLen` datatypes instead of collapsing them to scalar physical mappings. Wire validation now covers trailer magic validation, footer-length decoding, footer-offset derivation, non-overlapping row-group/column-chunk byte-range descriptors, and footer-boundary rejection. `cargo test -p consus-parquet --lib` passes with 47/47 tests, including exhaustive datatype coverage for compound, array, enum, and varlen mappings plus validated dataset descriptor, projection, nested-group, repeated-field, and footer-validation coverage. |

| M-001 | consus-mat MATLAB .mat reader | v4 binary, v5 structured binary (all mxClass except explicit `mxOBJECT_CLASS` rejection, complex, logical, miCOMPRESSED), v7.3 HDF5-backed via consus-hdf5. Public model invariants are now enforced through constructors for cell, char, logical, sparse, and struct arrays. Crate-level documentation now specifies feature gates, canonical model mapping, parsing contracts, and unsupported cases. Follow-on closure adds tolerant skipping of unknown top-level v5 elements with structural validation, deterministic unsupported-feature rejection for MAT v5 `mxOBJECT_CLASS`, datatype-aware v7.3 char decoding for little-endian and big-endian uint16 datasets, explicit unsupported-feature rejection for v7.3 sparse datasets, explicit compact-layout rejection coverage, and `miCOMPRESSED` feature-matrix verification across enabled and disabled `compress` configurations. Integration coverage now includes v7.3 numeric, logical, char, cell ordering, scalar struct decoding, big-endian char decoding, sparse rejection, compact-layout rejection, and MAT v5 compressed-payload success/failure behavior under the corresponding feature set. |

---

### P-003: Compression Pipeline Gap (resolved this sprint)

**Finding:** Column value decoding expected pre-decompressed bytes; no decompression dispatch existed for Parquet compression codecs (SNAPPY, GZIP, LZ4, ZSTD).

**Fix:** Added `encoding/compression.rs` with `CompressionCodec` enum (discriminants 0-7), `decompress_page_values` dispatch (feature-gated per codec), and `decode_compressed_column_values` integrated entry point in `encoding/column.rs`.

**Verification:** 12 unit tests for compression module + 7 integration tests for combined decompress+decode. 155 tests pass with all features.

---

## Open Gaps

_No open gaps in the current audit scope. Parquet Phase 3 now covers: Thrift compact binary decoder, canonical wire metadata types, footer payload extraction, page header decoding, schema reconstruction bridge, dataset materialization bridge, physical page payload decoding (RLE/bit-packing hybrid levels + PLAIN decoders + RLE_DICTIONARY index decoder + DataPage v1/v2 payload splitter). Remaining Parquet roadmap items (typed column value extraction and file-backed read API) are planned next increments under P3.1._

---

## Closed Since Previous Audit

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
| netCDF tests | 113 |
| netCDF passing | 113 |
| netCDF failing | 0 |
| Verified commands | `cargo test -p consus-zarr` (301/301); `cargo test -p consus-netcdf` (113/113); `cargo test --workspace` (all pass); `cargo test -p consus-parquet --lib` (136/136); `cargo test -p consus-arrow --lib` (41/41); `cargo test -p consus-fits --lib` (128/128); `cargo test -p consus-hdf5 --lib` (263/263); `cargo test -p consus-hdf5` (321/321); `cargo test -p consus-mat` (74/74); `cargo test -p consus-mat --no-default-features --features std,alloc` (62/62) |
| Open gaps | 0 |
| High-severity open gaps | 0 |
| Closed this sprint | 1 (P-002: Parquet Thrift compact decoder + wire metadata types + page header decoder + schema reconstruction bridge + dataset materialization bridge) |
| Medium-severity open gaps | 0 |
| Low-severity open gaps | 0 |