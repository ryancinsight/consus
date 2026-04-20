# Consus - Gap Audit

## Audit Date: 2026-04-20
## Scope: Phase 2 - Zarr Chunk I/O and Python Fixture Interoperability Audit

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
---

## Open Gaps

### Z-102: Python Zarr Interoperability Fixtures (Severity: MEDIUM)
- **Current state**: Repository fixtures generated from Python `zarr` now exist for v2/v3 filesystem stores, and Rust integration tests execute against them. Verified interoperability now includes Python v2 uncompressed fixture reads, Python v2 partial-write preservation over Python-produced stores, Python v2 gzip full-array reads, and Python v3 gzip fixture reads. The remaining mismatch is limited to the Python-generated v3 default codec-chain variant observed in one uncompressed v3 fixture, where the decoded full-array result still contains an incorrect zero in place of one expected value.
- **Impact**: Cross-implementation verification is artifact-backed and mostly passing, but full Python interoperability is still incomplete because one v3 default codec-chain path does not yet match Python-produced data.
- **Required closure**: Support the Python-generated v3 default codec-chain variant observed in fixture metadata and eliminate the remaining incorrect decoded value in the v3 uncompressed fixture path.
- **Verification target**: Promote the remaining mismatch-documenting fixture test into a value-semantic passing interoperability test for the v3 uncompressed store, while preserving the already passing v2 uncompressed, v2 partial-write, v2 gzip, and v3 gzip fixture coverage.



### Z-103: High-Level Zarr v3 Sharding Interop (Severity: MEDIUM)
- **Current state**: Shard support exists in the crate, but the high-level chunk/array API is not yet verified end-to-end against sharded v3 arrays.
- **Impact**: The public array API does not yet provide verified coverage for a major v3 storage mode.
- **Required closure**: Integrate shard-aware chunk resolution into the high-level read/write path or define a separate authoritative abstraction boundary.
- **Verification target**: End-to-end tests for shard index lookup, chunk extraction, and full-array reconstruction.

### Z-104: Artifact Synchronization (Severity: LOW)
- **Current state**: `backlog.md` and `checklist.md` reflect resolved Python v2 chunk-key and v2 gzip interoperability plus the remaining Python v3 codec-chain mismatch; `README.md` still needs synchronization to distinguish verified fixture coverage from that remaining v3 gap.
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

### Z-093: Partial Selection Write Verification Gap
- **Status**: Closed.
- **Evidence**: Array-level partial writes now perform chunk-aware read-modify-write updates, preserve existing values outside the selected region, materialize fill values for uninitialized intersecting chunks, and reject mismatched input lengths.
- **Verification**: `cargo test -p consus-zarr --lib` passes with `chunk::tests::write_array_selection_contiguous_across_chunks`, `chunk::tests::write_array_selection_strided_across_chunks`, `chunk::tests::write_array_selection_uninitialized_chunks_materialize_fill_value`, and `chunk::tests::write_array_selection_rejects_invalid_input_length` succeeding.

---

## Risk Assessment

| Risk | Probability | Impact | Status |
|------|-------------|--------|--------|
| Partial selection write semantics across chunk boundaries | Low | Medium | Reduced by multidimensional value-semantic write tests covering contiguous, strided, and uninitialized-chunk update paths |
| Invalid chunk coordinates accepted as store keys | Low | Low | Reduced by chunk-grid validation in `read_chunk`/`write_chunk` and negative tests for out-of-grid coordinates |
| Python interoperability mismatch | Medium | High | Reduced from unverified status to fixture-backed, mostly passing interoperability; remains open until the remaining Python-generated v3 default codec-chain mismatch is resolved |
| Sharded v3 high-level API drift | Medium | Medium | Open until Z-103 is verified end-to-end |
| Fill-value width mismatch for non-8-byte numeric types | Low | Medium | Reduced by typed fill-value expansion fix and passing library roundtrip tests |
| Store/backend divergence | Low | Medium | Reduced by in-memory, filesystem, and Python-generated fixture coverage; S3 interop remains indirect |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Zarr library tests | 90 |
| Passing | 90 |
| Failing | 0 |
| Verified commands | `cargo test -p consus-zarr --lib`; `cargo test -p consus-zarr --test roundtrip_zarr` |
| Open gaps | 3 |
| High-severity open gaps | 0 |
| Closed this sprint | 4 (`Z-014`, `Z-015`, Python v2 chunk-key interoperability, Python v2 gzip full-array interoperability) |
| Medium-severity open gaps | 2 |
| Low-severity open gaps | 1 |