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
- [ ] Integration tests against Python zarr-produced fixtures
- [ ] Zarr v3 sharding interop coverage through high-level array API

### Milestone 5: Artifact Synchronization
- [x] `backlog.md` updated to reflect verified P2.1 chunk I/O status
- [ ] `gap_audit.md` updated for Phase 2 Zarr scope
- [ ] `README.md` phase wording synchronized to verified Zarr implementation state