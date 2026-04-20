# Consus - Gap Audit

## Audit Date: 2026-04-20
## Scope: Phase 2 - Zarr Chunk I/O (synchronized with current implementation and partial selection read verification)

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
| Z-011 | Library verification | `cargo test -p consus-zarr --lib` passes with 86/86 tests |
| Z-012 | Partial selection read semantics | Verified contiguous and strided partial reads across chunk boundaries, plus fill-value materialization for uninitialized intersecting chunks |
---

## Open Gaps

### Z-100: Partial Selection Write Semantics (Severity: MEDIUM)
- **Current state**: Partial selection read semantics are now verified for contiguous and strided reads across chunk boundaries, including fill-value materialization when intersecting chunks are uninitialized. Partial selection write semantics remain unimplemented.
- **Impact**: Hyperslab-style partial writes are not yet established as correct across chunk boundaries, strides, and edge chunks.
- **Required closure**: Define write-side selection invariants, implement chunk/selection intersection and in-place update semantics for N-dimensional partial regions, and add value-semantic tests that assert exact persisted bytes.
- **Verification target**: Library and integration tests covering contiguous, strided, boundary, and sparse partial writes.



### Z-102: Python Zarr Interoperability Fixtures (Severity: MEDIUM)
- **Current state**: Internal metadata, codec, store, and chunk tests pass, but no verified roundtrip against Python-generated Zarr fixtures is recorded.
- **Impact**: Spec conformance remains internal-only; cross-implementation compatibility is not yet demonstrated.
- **Required closure**: Add repository fixtures or generated test assets from Python zarr for v2 and v3 arrays, then assert metadata and chunk-value equivalence.
- **Verification target**: Integration tests against Python-produced stores for gzip and uncompressed arrays, then extend to additional codecs.

### Z-103: High-Level Zarr v3 Sharding Interop (Severity: MEDIUM)
- **Current state**: Shard support exists in the crate, but the high-level chunk/array API is not yet verified end-to-end against sharded v3 arrays.
- **Impact**: The public array API does not yet provide verified coverage for a major v3 storage mode.
- **Required closure**: Integrate shard-aware chunk resolution into the high-level read/write path or define a separate authoritative abstraction boundary.
- **Verification target**: End-to-end tests for shard index lookup, chunk extraction, and full-array reconstruction.

### Z-104: Artifact Synchronization (Severity: LOW)
- **Current state**: `backlog.md` and `checklist.md` reflect the Phase 2 transition, but this audit update is the remaining synchronization step.
- **Impact**: Sprint artifacts can diverge from verified implementation state if not updated together.
- **Required closure**: Keep `README.md`, `backlog.md`, `checklist.md`, and `gap_audit.md` aligned after each verified Zarr milestone.
- **Verification target**: Manual artifact audit during each sprint closure.
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

---

## Risk Assessment

| Risk | Probability | Impact | Status |
|------|-------------|--------|--------|
| Partial selection write semantics across chunk boundaries | Medium | High | Open until Z-100 is closed with multidimensional value-semantic write tests |
| Invalid chunk coordinates accepted as store keys | Low | Low | Reduced by chunk-grid validation in `read_chunk`/`write_chunk` and negative tests for out-of-grid coordinates |
| Python interoperability mismatch | Medium | High | Open until Z-102 fixture-based verification is added |
| Sharded v3 high-level API drift | Medium | Medium | Open until Z-103 is verified end-to-end |
| Fill-value width mismatch for non-8-byte numeric types | Low | Medium | Reduced by typed fill-value expansion fix and passing library roundtrip tests |
| Store/backend divergence | Low | Medium | Reduced by in-memory and filesystem store coverage; S3 interop remains indirect |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Zarr library tests | 86 |
| Passing | 86 |
| Failing | 0 |
| Verified command | `cargo test -p consus-zarr --lib` |
| Open gaps | 4 |
| High-severity open gaps | 0 |
| Closed this sprint | 2 (`Z-101`, `Z-092`) |
| Medium-severity open gaps | 3 |
| Low-severity open gaps | 1 |