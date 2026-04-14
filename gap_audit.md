# Consus — Gap Audit

## Audit Date: 2025-01-01
## Scope: Phase 1 Foundation

### Summary

| Category | Items | Complete | Gap |
|----------|-------|----------|-----|
| Core types | 8 modules | 8 | 0 |
| I/O abstraction | 3 modules | 3 | 0 |
| Compression | 3 codecs + registry | 4 | 0 |
| HDF5 structure | 10 modules | 10 (skeleton) | High: parsing logic incomplete |
| Tests | 12 total | 12 pass | Need 50+ for Phase 1 |
| Benchmarks | 0 | 0 | Blocked on read path |
| CI/CD | 0 | 0 | Not started |
| Documentation | Rustdoc + README | Partial | API examples needed |

### Critical Gaps

#### G-001: Object Header Parsing (Severity: BLOCKER)
- **Current state**: Struct definitions only. No parsing logic.
- **Required**: Full v1 and v2 object header parsers that extract header messages.
- **Impact**: Blocks all group navigation and dataset reads.
- **Effort**: ~400 lines of parsing code + tests.

#### G-002: B-tree Traversal (Severity: BLOCKER)
- **Current state**: Header structs defined. No traversal logic.
- **Required**: B-tree v1 traversal for groups and chunk indexes.
- **Impact**: Blocks group member listing and chunked dataset reads.
- **Effort**: ~300 lines + tests.

#### G-003: Local/Global Heap Readers (Severity: BLOCKER)
- **Current state**: Struct definitions only.
- **Required**: Parse heap headers, resolve offsets to string data.
- **Impact**: Blocks group member name resolution.
- **Effort**: ~150 lines + tests.

#### G-004: Datatype Message Parser (Severity: HIGH)
- **Current state**: Mapping functions for fixed-point and float only.
- **Required**: Full parser covering all 11 datatype classes.
- **Impact**: Blocks reading datasets with compound, VL, enum, array types.
- **Effort**: ~350 lines + tests.

#### G-005: Data Layout + Filter Pipeline (Severity: HIGH)
- **Current state**: Enum and struct definitions only.
- **Required**: Parsers for layout message (type 0x0008) and filter pipeline (type 0x000B).
- **Impact**: Blocks all dataset reads.
- **Effort**: ~200 lines + tests.

#### G-006: CI/CD Pipeline (Severity: MEDIUM)
- **Current state**: Not configured.
- **Required**: GitHub Actions workflow for check, test, clippy, fmt, MSRV.
- **Impact**: No automated quality gates.
- **Effort**: ~50 lines of YAML.

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HDF5 spec ambiguity | Medium | High | Cross-reference h5dump output, HDF5 C source, and hdf5-rs |
| Endianness edge cases | Low | Medium | Test on BE targets (s390x via QEMU) |
| Large file (>4 GiB) addressing | Low | High | Ensure all offset arithmetic uses u64; regression test with 5 GiB file |
| Superblock v0/v1 field alignment | Medium | Medium | Validated parser handles variable-size fields based on offset_size |
| Free-space manager interaction | Low | Low | Read path does not need free-space; write path can use simple append strategy |
| Checksum validation (v2 headers) | Medium | Medium | Implement after basic parsing works; checksum errors → Error::Corrupted |