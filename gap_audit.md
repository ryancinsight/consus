# Consus — Gap Audit

## Audit Date: 2025-07-14
## Scope: Phase 1 — HDF5 MVP (post-sprint update)

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
| G-025 | Integration tests | ✅ 13 integration tests; 213 unit tests — 226 total, 0 failures |

---

## Open Gaps

### G-100: Dense Group Link Storage (Severity: HIGH)
- **Current state**: `list_group_v2` reads compact links (direct link messages) only. Dense storage (fractal heap + B-tree v2 with record type 5/6) is not traversed.
- **Impact**: Groups with more than `max_compact_links` (default 8) links report only 0 children when using dense storage.
- **Required**: Traverse B-tree v2 (record type 5: link name index) and decode link objects from the fractal heap.
- **Effort**: ~200 lines. `FractalHeapHeader::parse` and `read_managed_object` already exist.

### G-101: Dense Attribute Storage (Severity: HIGH)
- **Current state**: `read_attributes` only reads inline attribute messages (type 0x000C). When `AttributeInfo` (type 0x0015) is present with a fractal heap address, dense attributes are silently skipped.
- **Impact**: Objects with more than `max_compact_attrs` (default 8) attributes return an incomplete attribute list.
- **Required**: When `AttributeInfo.fractal_heap_address` is defined, traverse the name-index B-tree v2 (record type 8) and decode attribute objects from the fractal heap.
- **Effort**: ~250 lines.

### G-102: Chunked Dataset Write Path (Severity: HIGH)
- **Current state**: `encode_layout` emits a placeholder `UNDEFINED_ADDRESS` for the B-tree v1 address in chunked layout messages. `Hdf5FileBuilder` does not support chunked datasets.
- **Impact**: Cannot create compressed or chunked datasets in the write path.
- **Required**: Allocate and write a B-tree v1 (or B-tree v2 for v4 layout) chunk index after writing chunk data blocks.
- **Effort**: ~300 lines.

### G-103: Variable-Length Datatype Read (Severity: HIGH)
- **Current state**: `Datatype::VarLen` is parsed correctly but `read_chunk_raw` and the contiguous read path return raw VL heap references (8-byte global heap object IDs) without resolving them through the global heap.
- **Impact**: VL string and VL sequence datasets return raw bytes instead of the actual variable-length values.
- **Required**: After reading raw data, detect `Datatype::VarLen`, resolve each 8-byte GHEAP reference via `GlobalHeapCollection::parse` and `get_data`.
- **Effort**: ~150 lines.

### G-104: Chunk Index v4 B-tree v2 Lookup (Severity: MEDIUM)
- **Current state**: `read_dataset_metadata` parses v4 layout `chunk_index_address` but no reader resolves it. `collect_btree_v1_leaves` is specific to group B-trees (record type 0) and does not handle chunk record types (3, 4, 10, 11).
- **Impact**: Datasets written by HDF5 library ≥ 1.10 with v4 layout and B-tree v2 chunk indexing cannot be read.
- **Required**: Implement chunk record parsing for B-tree v2 record types 3 (non-filtered) and 4 (filtered); extend reader to use these when `DataLayout.chunk_index_type == BTREE_V2`.
- **Effort**: ~200 lines.

### G-105: Soft and External Link Resolution (Severity: MEDIUM)
- **Current state**: `open_path` follows only hard links. Soft and external links stored in `list_group_v2` return `UNDEFINED_ADDRESS` and `open_path` treats them as dead ends.
- **Impact**: Files with soft-linked groups or cross-file references cannot be fully traversed.
- **Required**: In `open_path`, detect `LinkType::Soft` and recursively resolve the target path. For external links, return a typed error pointing to the external file.
- **Effort**: ~80 lines.

### G-106: Fill Value Application to Reads (Severity: LOW)
- **Current state**: `read_fill_value` parses fill value messages but the read path never uses the result. Uninitialized regions (e.g., edge chunks) return zero bytes instead of the declared fill value.
- **Required**: After reading and decompressing a chunk, if the chunk address is `UNDEFINED_ADDRESS`, fill the output buffer with the decoded fill value.
- **Effort**: ~50 lines.

### G-107: Reference File Compatibility Tests (Severity: MEDIUM)
- **Current state**: All tests use hand-crafted in-memory HDF5 images. No tests validate against files produced by the HDF5 C library or h5py.
- **Required**: Download canonical test files (t_float.h5, t_compound.h5, t_vlen.h5, t_chunk.h5, t_filter.h5) from the HDF Group; add read tests comparing output against h5dump.
- **Effort**: ~300 lines of test code + test data files.

### G-108: Benchmarks (Severity: LOW)
- **Current state**: No Criterion benchmarks exist.
- **Required**: Criterion benchmarks for contiguous read throughput, chunked read throughput, compressed read (deflate/zstd/lz4).
- **Effort**: ~150 lines.

### G-109: CI/CD Pipeline (Severity: MEDIUM)
- **Current state**: No automated quality gates.
- **Required**: GitHub Actions workflow: `cargo check`, `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`, MSRV (1.85), no_std smoke test.
- **Effort**: ~60 lines of YAML.

### G-110: Async I/O Path (Severity: LOW)
- **Current state**: All I/O is synchronous through `ReadAt`/`WriteAt`. The `async-io` feature in `consus-io` is present but unused in `consus-hdf5`.
- **Required**: Implement `AsyncReadAt`-backed variants of the key read functions.
- **Effort**: ~400 lines. Blocked on finalising the async trait design in `consus-io`.

### G-111: Parallel Chunk I/O (Severity: LOW)
- **Current state**: Chunked reads decompose the selection but process one chunk at a time.
- **Required**: Rayon-parallel chunk decompression for multi-chunk selections.
- **Effort**: ~80 lines (Rayon `.par_iter()` over `ChunkSlice` list).

---

## Risk Assessment

| Risk | Probability | Impact | Status |
|------|-------------|--------|--------|
| HDF5 spec ambiguity in dense storage | Medium | High | Mitigated by fractal heap + B-tree v2 parsers already in place |
| Endianness edge cases | Low | Medium | `read_int_le` / `read_uint_le` handle both orders |
| Large file (>4 GiB) addressing | Low | High | All offsets use u64; no truncation observed |
| Checksum validation failures | Low | Medium | CRC-32 validated on every v2 OHDR/OCHK chunk |
| VL type heap exhaustion | Medium | Medium | No bounds on GHEAP traversal yet (G-103) |
| B-tree v2 depth overflow | Low | Medium | MAX_CONTINUATION_DEPTH guard in v2 parser |
| Free-space manager interaction | Low | Low | Append-only write path avoids free-space entirely |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total unit tests | 213 |
| Integration tests | 13 |
| Total passing | 226 |
| Total failing | 0 |
| Compile warnings | 0 |
| Open gaps | 11 |
| High-severity open gaps | 4 (G-100, G-101, G-102, G-103) |
| Medium-severity open gaps | 4 (G-104, G-105, G-107, G-109) |
| Low-severity open gaps | 3 (G-106, G-108, G-110, G-111) |