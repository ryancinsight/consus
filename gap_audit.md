# Consus — Gap Audit

## Audit Date: 2026-04-17
## Scope: Phase 1 — HDF5 MVP (synchronized with current implementation)

---

## Resolved Gaps (this sprint)

| ID | Description | Resolution |
|----|-------------|------------|
| G-001 | Object header v1 parser | ✅ Full parser with continuation messages |
| G-001b | Object header v2 parser | ✅ OHDR/OCHK chunked layout, CRC-32 validation, creation-order tracking |
| G-002 | B-tree v1 traversal | ✅ Leaf/internal node traversal, group + chunk index |
| G-002b | B-tree v2 traversal | ✅ Header, internal nodes, leaf nodes, collect_all_records |
| G-003 | Local heap reader | ✅ Signature/version validation, name resolution by offset |
| G-003b | Global heap reader | ✅ Collection parse, object get/get_data |
| G-004 | Datatype message parser | ✅ All 11 classes: fixed-point, float, string, opaque, compound, reference, enum, VL, array, bitfield, time |
| G-005a | Data layout message parser | ✅ Version 3 (compact, contiguous, chunked) + version 4 (single chunk, implicit, fixed array, extensible array, B-tree v2) |
| G-005b | Filter pipeline message parser | ✅ Version 1 and version 2, all standard IDs |
| G-006 | Link message parser | ✅ Hard, soft, external link types; creation order; encoding flags |
| G-007 | Symbol table message parser | ✅ SNOD parsing, SymbolTableEntry cache types |
| G-008 | Attribute message parser | ✅ Versions 1, 2, 3; name encoding; creation order |
| G-009 | Attribute info message parser | ✅ Fractal heap address + name/order B-tree addresses |
| G-010 | Fractal heap header + ID | ✅ Header parse; managed/tiny/huge ID decode; direct-block object read |
| G-011 | Contiguous dataset read | ✅ read_contiguous_raw, read_fill_value |
| G-012 | Chunk I/O | ✅ read_chunk_raw / write_chunk_raw with full filter pipeline |
| G-013 | Hyperslab selection decomposition | ✅ Contiguous + strided hyperslabs mapped to per-chunk slices |
| G-014 | Point selection decomposition | ✅ Per-point ChunkSlice with 1-D output indexing |
| G-015 | Superblock v2 writer | ✅ write_superblock, update_superblock_eof with CRC-32 |
| G-016 | Object header v2 writer | ✅ write_object_header_v2 (generalised message list) |
| G-017 | Datatype / dataspace / layout encoding | ✅ encode_datatype, encode_dataspace, encode_layout |
| G-018 | Link encoding | ✅ encode_hard_link, encode_soft_link |
| G-019 | Attribute message encoding | ✅ encode_attribute (version 3 format) |
| G-020 | High-level file builder | ✅ Hdf5FileBuilder: add_dataset, add_dataset_with_attributes, add_root_attribute, finish |
| G-021 | Path navigation | ✅ open_path (slash-delimited, v1 + v2 groups), list_group_at, node_type_at |
| G-022 | decode_value() on Hdf5Attribute | ✅ Integer (signed/unsigned), float (f32/f64), fixed-string, boolean; array variants |
| G-023 | B-tree v2 bit-width formula | ✅ Fixed usize::BITS / u32 mixing bug |
| G-024 | Scalar chunk uncompressed size | ✅ Corrected empty-product to 1 (one scalar element) |
| G-025 | Integration tests | ✅ Integration coverage exists across `tests/integration.rs`, `tests/reference_hdf_group.rs`, and `tests/roundtrip_hdf5.rs` |
| G-103 | Variable-Length Datatype Read | ✅ `resolve_vl_references` in `src/heap/global.rs`; offset-size-aware undefined sentinel; 8 unit tests; re-exported from `src/heap/mod.rs` |

---

## Open Gaps

### G-102: Chunked Dataset Write Path (Severity: RESOLVED IN CURRENT WRITER SCOPE)
- **Previous state**: `encode_layout` emitted `UNDEFINED_ADDRESS` for the version-3 chunk index address, and `Hdf5FileBuilder::add_dataset` always wrote one contiguous data block even when `DatasetCreationProps.layout == Chunked`.
- **Resolution**: The writer now materializes per-chunk payloads, serializes a version-3 raw-data chunk B-tree v1 leaf index, emits the resolved chunk index address in the layout message, and includes filter pipeline metadata when chunk filters/compression are configured. The roundtrip suite now verifies chunked dataset values, not metadata only.
- **Verification**: `cargo test -p consus-hdf5 --test roundtrip_hdf5` passes with `chunked_dataset_value_roundtrip` asserting exact value preservation.
- **Residual scope**: The implemented writer path is verified for the current builder scope using version-3 chunked layout with a single-leaf raw-data chunk B-tree. Broader interoperability for other chunk index forms remains tracked separately under `G-104`.

### G-104: Chunk Index v4 B-tree v2 Lookup (Severity: MEDIUM)
- **Current state**: `read_dataset_metadata` parses v4 layout `chunk_index_address`, but the read path does not resolve B-tree v2 chunk index record types for dataset chunk lookup.
- **Impact**: Datasets written by HDF5 library ≥ 1.10 with v4 layout and B-tree v2 chunk indexing cannot be read through the chunked path.
- **Required**: Implement chunk record parsing for B-tree v2 record types 3 (non-filtered) and 4 (filtered), and route chunk lookup through `DataLayout.chunk_index_type == BTREE_V2`.
- **Effort**: ~200 lines.

### G-107: Reference File Compatibility Tests (Severity: MEDIUM)
- **Current state**: Reference coverage exists, including `tests/reference_hdf_group.rs`, but the canonical HDF Group compatibility matrix in the backlog is not yet present.
- **Impact**: Parser correctness is validated against in-repo samples and round-trip images, but not yet against the broader set of authoritative HDF5 reference fixtures.
- **Required**: Add canonical test files such as `t_float.h5`, `t_compound.h5`, `t_vlen.h5`, `t_chunk.h5`, and `t_filter.h5`, then assert decoded values against known outputs.
- **Effort**: ~300 lines of test code + test data files.

### G-108: Benchmarks (Severity: LOW)
- **Current state**: No Criterion benchmarks are present for `consus-hdf5`.
- **Required**: Add benchmarks for contiguous read throughput, chunked read throughput, and compressed read throughput (deflate/zstd/lz4).
- **Effort**: ~150 lines.

### G-110: Async I/O Path (Severity: LOW)
- **Current state**: All HDF5 I/O remains synchronous through `ReadAt`/`WriteAt`. The async capability in `consus-io` is not yet integrated into `consus-hdf5`.
- **Required**: Implement async read variants for the file traversal and dataset read path once the async trait boundary is finalized.
- **Effort**: ~400 lines. Blocked on final async trait design in `consus-io`.

### G-111: Parallel Chunk I/O (Severity: LOW)
- **Current state**: Chunked reads decompose selections correctly, but chunk processing remains serial.
- **Required**: Parallelize independent chunk reads and decompression with Rayon while preserving deterministic output assembly.
- **Effort**: ~80 lines.

### G-112: Workspace Integration-Test API Drift (Severity: RESOLVED)
- **Previous state**: The workspace integration-test package resolved as a workspace member, but `tests/property_integration.rs` targeted obsolete APIs and invalid helper constructors.
- **Resolution**: Rewrote `tests/property_integration.rs` against the current stable workspace API surface, enabled the required `consus-io` `alloc` feature in `tests/Cargo.toml`, and replaced obsolete assumptions with value-semantic property tests covering shape semantics, hyperslab construction, byte-order roundtrips, datatype size invariants, `MemCursor` positioned I/O, compression roundtrips, and Arrow/Parquet schema cardinality preservation.
- **Verification**: `cargo test -p consus-integration-tests --test property_integration --features compression,arrow,parquet` passes with 17/17 tests passing.
- **Residual scope**: `tests/end_to_end_reference.rs` remains separately tracked by fixture and backend coverage gaps, not by API drift in the property suite.

---

## Closed Since Previous Audit

### G-100: Dense Group Link Storage
- **Status**: ✅ Closed.
- **Evidence**: `list_group_v2` traverses dense storage through `collect_dense_links`, using fractal heap objects and B-tree v2 records.

### G-101: Dense Attribute Storage
- **Status**: ✅ Closed.
- **Evidence**: `read_attributes` resolves `ATTRIBUTE_INFO` and enumerates dense attributes through `collect_dense_attributes`.

### G-105: Soft and External Link Resolution
- **Status**: ✅ Closed.
- **Evidence**: `open_path` resolves soft links recursively and returns a typed unsupported-feature error for external links.

### G-106: Fill Value Application to Reads
- **Status**: ✅ Closed.
- **Evidence**: `dataset::chunk::read_chunk_raw` fills undefined chunks from the declared fill value pattern when provided.

### G-109: CI/CD Pipeline
- **Status**: ✅ Closed.
- **Evidence**: `.github/workflows/ci.yml` runs formatting, clippy with warnings denied, cargo check, cargo test, and MSRV validation.

---

## Risk Assessment

| Risk | Probability | Impact | Status |
|------|-------------|--------|--------|
| Chunked write metadata/data mismatch | Low | Medium | Reduced by verified v3 chunked write-path implementation; continue validating broader interoperability via G-104 |
| HDF5 v4 chunk index interoperability | Medium | High | Open until G-104 is implemented |
| Endianness edge cases | Low | Medium | `read_int_le` / `read_uint_le` handle both orders |
| Large file (>4 GiB) addressing | Low | High | All offsets use `u64`; no truncation observed |
| Checksum validation failures | Low | Medium | CRC-32 validated on v2 OHDR/OCHK chunks and superblock v2 writer path |
| VL type heap exhaustion | Medium | Medium | Managed by current parsing logic; continue validating against external fixtures |
| B-tree v2 depth overflow | Low | Medium | Depth and continuation guards exist in parser paths |
| Free-space manager interaction | Low | Low | Append-only write path avoids free-space entirely |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total unit tests | Audit artifact requires refresh from current test run output |
| Integration tests | Audit artifact requires refresh from current test run output |
| Total passing | Audit artifact requires refresh from current test run output |
| Total failing | 0 at last recorded audit snapshot |
| Compile warnings | 0 at last recorded audit snapshot |
| Open gaps | 5 |
| High-severity open gaps | 0 |
| Medium-severity open gaps | 2 (G-104, G-107) |
| Low-severity open gaps | 3 (G-108, G-110, G-111) |