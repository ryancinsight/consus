# Consus — Implementation Checklist

## Current Sprint: Phase 1 — HDF5 MVP

### Milestone 1: Object Header Parsing
- [ ] Implement v1 object header parser
  - Parse version, message count, reference count, header size
  - Iterate header messages with type, size, flags, data
  - Handle continuation messages (type 0x0010)
- [ ] Implement v2 object header parser
  - Parse OHDR signature, version, flags
  - Handle optional timestamp fields
  - Parse chunk-based message layout
  - Handle OCHK continuation chunks
  - Validate checksums
- [ ] Unit tests with hand-crafted binary headers
- [ ] Integration test: parse root group object header from minimal file

### Milestone 2: Group Navigation
- [ ] Symbol table message parser (type 0x0011) → B-tree + local heap addresses
- [ ] B-tree v1 traversal for group nodes
  - Parse TREE signature, type, level, entries
  - Leaf node: extract symbol table entries
  - Internal node: recursively follow child pointers
- [ ] Local heap reader
  - Parse HEAP signature, data segment size, free list, data address
  - Resolve member names by offset into heap data segment
- [ ] Link info message parser (type 0x0002) → B-tree v2 address
- [ ] Link message parser (type 0x0006) → direct links in v2 groups
- [ ] B-tree v2 traversal (if link info present)
- [ ] Group member listing: return `Vec<(String, NodeType, u64)>`
- [ ] Recursive hierarchy walk

### Milestone 3: Dataset Read
- [ ] Data layout message parser (type 0x0008)
  - Compact layout: extract inline data
  - Contiguous layout: extract data address + size
  - Chunked layout: extract chunk dimensions + B-tree address
- [ ] Datatype message parser (type 0x0003)
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
- [ ] Filter pipeline message parser (type 0x000B)
  - Number of filters, per-filter: ID, name, flags, parameters
  - Map filter IDs to codec registry
- [ ] Contiguous read: seek to data address, read element_size × num_elements
- [ ] Chunked read:
  - Determine which chunks overlap the selection
  - For each chunk: read compressed bytes, decompress, extract selection
  - Assemble output buffer
- [ ] Hyperslab read: apply strided selection to output buffer

### Milestone 4: Write Path
- [ ] Superblock v2 writer (48 bytes + checksum)
- [ ] Object header v2 writer
- [ ] Datatype/dataspace/layout message serialization
- [ ] Contiguous dataset writer
- [ ] Chunked dataset writer (single-level B-tree v2)
- [ ] Group writer (link messages)
- [ ] Attribute writer
- [ ] File-level create → write → close cycle
- [ ] Round-trip test: create file with consus, read back, verify values

### Milestone 5: Validation & Benchmarks
- [ ] Property tests (proptest): round-trip for all datatype classes
- [ ] Reference file compatibility tests
- [ ] Criterion benchmarks: read throughput by layout and compression
- [ ] Memory profile: peak allocation during chunked reads