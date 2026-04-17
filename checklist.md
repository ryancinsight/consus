# Consus — Implementation Checklist

## Current Sprint: Phase 1 — HDF5 MVP Closure

### Milestone 1: Object Header Parsing
- [x] Implement v1 object header parser
  - Parse version, message count, reference count, header size
  - Iterate header messages with type, size, flags, data
  - Handle continuation messages (type 0x0010)
- [x] Implement v2 object header parser
  - Parse OHDR signature, version, flags
  - Handle optional timestamp fields
  - Parse chunk-based message layout
  - Handle OCHK continuation chunks
  - Validate checksums
- [x] Unit tests with hand-crafted binary headers
- [x] Integration test: parse root group object header from minimal file

### Milestone 2: Group Navigation
- [x] Symbol table message parser (type 0x0011) → B-tree + local heap addresses
- [x] B-tree v1 traversal for group nodes
  - Parse TREE signature, type, level, entries
  - Leaf node: extract symbol table entries
  - Internal node: recursively follow child pointers
- [x] Local heap reader
  - Parse HEAP signature, data segment size, free list, data address
  - Resolve member names by offset into heap data segment
- [x] Link info message parser (type 0x0002) → B-tree v2 address
- [x] Link message parser (type 0x0006) → direct links in v2 groups
- [x] B-tree v2 traversal for dense link and attribute indices
- [x] Group member listing
- [x] Recursive hierarchy walk
- [x] Soft link resolution in `open_path`
- [ ] External link traversal beyond typed unsupported-feature reporting

### Milestone 3: Dataset Read
- [x] Data layout message parser (type 0x0008)
  - Compact layout: extract inline data
  - Contiguous layout: extract data address + size
  - Chunked layout: extract chunk dimensions + B-tree address
  - Version 4 chunk index metadata: parse single chunk, implicit, fixed array, extensible array, and B-tree v2 descriptors
- [x] Datatype message parser (type 0x0003)
  - Class extraction from 4-byte header
  - Fixed-point (class 0) → canonical Integer
  - Floating-point (class 1) → canonical Float
  - String (class 3) → canonical FixedString/VariableString
  - Opaque (class 5) → canonical Opaque
  - Compound (class 6) → canonical Compound
  - Reference (class 7) → canonical Reference
  - Enum (class 8) → canonical Enum
  - Variable-length (class 9) → canonical VarLen
  - Array (class 10) → canonical Array
  - Bitfield and time classes
- [x] Filter pipeline message parser (type 0x000B)
  - Number of filters, per-filter: ID, name, flags, parameters
  - Map filter IDs to codec registry
- [x] Contiguous read: seek to data address, read element_size × num_elements
- [x] Chunked read primitives
  - Read chunk bytes by location
  - Apply reverse filter pipeline
  - Support undefined chunk address with fill-value tiling
- [x] Hyperslab selection decomposition
- [x] Point selection decomposition
- [x] Fill value message parsing
- [x] End-to-end chunk index lookup for chunked dataset reads written with real v3 chunk index structures
- [ ] B-tree v2 chunk index record resolution for v4 chunked layouts

### Milestone 4: Write Path
- [x] Superblock v2 writer (48 bytes + checksum)
- [x] Object header v2 writer
- [x] Datatype/dataspace/layout message serialization
- [x] Contiguous dataset writer
- [x] Chunked dataset writer with materialized chunk index (v3 layout, v1 raw-data chunk B-tree leaf path)
- [x] Group writer (link messages)
- [x] Attribute writer
- [x] File-level create → write → close cycle
- [x] Round-trip tests for contiguous datasets and attributes
- [x] Round-trip test for chunked dataset values, not metadata only

### Milestone 5: Validation, Performance, and Documentation
- [x] Property tests present in workspace test strategy
- [x] Workspace property integration tests restored to current stable APIs (`consus-core`, `consus-io`, `consus-compression`, Arrow, Parquet)
- [ ] HDF5-specific property tests covering round-trip invariants for all supported datatype classes
- [ ] Reference file compatibility tests against broader canonical HDF5 fixture set
- [ ] Criterion benchmarks: contiguous read throughput
- [ ] Criterion benchmarks: chunked read throughput
- [ ] Criterion benchmarks: compressed read throughput
- [ ] Memory profile: peak allocation during chunked reads
- [ ] Performance optimization pass for multi-chunk read parallelism and allocation reduction
- [ ] Artifact synchronization: `README.md`, `backlog.md`, and `gap_audit.md` aligned to verified implementation state