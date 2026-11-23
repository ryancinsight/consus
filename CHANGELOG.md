# Changelog

## [consus-raster 0.5.1] - Unreleased

### Fixed

- [patch] Accept version 1 grayscale `LJIF` APP0 metadata with zero flags
  and mode in single-component Huffman lossless JPEG. The marker leaves
  predictive samples unchanged.

## [consus-hdf5 0.6.0] - 2026-11-21

### Added

- Opt-in exact IEEE binary16, binary32, and binary64 dataset admission in
  either byte order. The checked lookup compares the original HDF5 datatype
  fields before returning ordinary dataset metadata. See [ADR 0011](docs/adr/0011-exact-ieee-dataset-admission.md).
- This release covers `consus-hdf5`. Its Consus sibling dependencies continue
  to resolve from their published 0.5 series.

## [0.5.0] - Unreleased

### Breaking

- [major] Parquet column codec numbers now follow the specification.
  `CompressionCodec::Zlib` is removed, `Lzo` is added, and discriminants
  change. Exhaustive downstream matches must be updated. GZIP pages now
  use RFC 1952 rather than raw DEFLATE. Deprecated LZ4 (5) is unsupported;
  use LZ4_RAW (7) for new data. Existing Consus files using GZIP, ZSTD,
  LZ4_RAW, legacy LZ4 or ZLIB require provenance-aware migration;
  UNCOMPRESSED and SNAPPY files retain their codec interpretation.
  See [compatibility and migration](docs/parquet-compatibility.md) and ADR0009.
- [major] GZIP decoding reads all members and rejects trailing non-member
  bytes, including in Zarr chunks. Parquet readers reject mismatched decoded
  page lengths and V2 headers smaller than their level sections.
- [major] Public HDF5 async bounds use Moirai 0.12. Migrate custom
  positioned readers from the published HDF5 0.4.0/Moirai 0.10 identity.
- Provider coherence, normalized package verification, registry ownership
  and release approvals still gate publication.

### Added

- [minor] Select a named MATLAB Level-5 variable without decoding the values
  of unrelated matrices. `MatNumericClass::datatype` exposes the normalized
  scalar type for the shared Consus decoder.

### Fixed

- [patch] Decode packed MATLAB Level-5 double payloads using the stored
  scalar type, preserving the declared class and column-major order.
  Accept unpadded compressed elements emitted by SciPy and restore the
  alloc-only MAT build.

- [patch] Honor the configured Zarr `zlib` compression level when writing
  chunks. Keep default level 6, valid gzip levels, and framing unchanged.
  Clamp oversized unsigned gzip/zlib levels to 9 before signed conversion
  rather than wrapping negative. Existing release and registry holds still apply.

- [patch] Parse scalar NPY header dictionaries by structure. Accept either
  key quote delimiter and arbitrary key order, reject malformed boolean/tuple
  tokens, and consume the complete dictionary before reading payload values.
  Preserve parenthesized literals, comments and adjacent strings across all
  Python physical line endings. NPZ member reads use the same boundary.

## [0.4.0] - Unreleased

### Breaking

- [major] Public HDF5 async reader bounds use Moirai 0.10's nominal
  AsyncReadAt/AsyncLength traits. Published 0.3 used 0.9.3; callers crossing
  this boundary migrate provider identity. Workspace and internal versions
  advance together to 0.4. ADR0006 records the published baseline.
- [major] `consus_npy::Error::Zip` wraps `consus_zip::Error` instead of
  `zip::result::ZipError`, and `NpzWriter` needs only `Write`, not
  `Write + Seek`. ADR0007 records the first-party container.

### Added

- [minor] `consus-zip`: a bounded ZIP container for stored and DEFLATE
  members, with CRC-32 and ZIP64. No filesystem extraction exists. Every
  declared offset and length is checked against the region that holds it, and
  DEFLATE output against the 1032:1 expansion bound, before any read or
  allocation. A fuzz target covers malformed archives.
- [minor] `consus_compression::codec::raw_deflate::RawDeflateCodec`: unframed
  DEFLATE (RFC 1951), ZIP compression method 8.
- [minor] Support CPython 3.14 with PyO3/build-config 0.30.1, preserving the
  Python names, values and cloned class extraction contract. Retain the older
  wheel interpreters and run the actual format interoperability suite in CI.

### Fixed

- [patch] Refresh the provider lock onto the current cohort: Apollo DCT/DST
  core 0.17.2, Eunomia 0.12, Aequitas 0.3.2, Hephaestus 0.28, Moirai 0.11.3,
  and Hermes 0.9.2 at their fetched default heads. The previous lock pinned
  Apollo 0.17.1, whose Eunomia requirement excluded 0.12 and broke fresh
  resolution.
- [patch] Retain symbols in optimized release artifacts: Apple dyld rejects the
  stripped extension's misaligned LINKEDIT string pool. Preserve thin LTO,
  opt-level 3 and one codegen unit; CI now imports and tests the release wheel.
- [patch] Forward the caller-supplied variable element width through chunked
  HDF5 reads. Keep fixed-width dispatch and missing-width rejection; verify
  actual plain/compressed global-heap string references without weakening the
  existing HDMF string-column oracle.
- [patch] Remove the third-party `zip` crate (RUSTSEC-2026-0367/0368/0369):
  NPZ reads and writes through `consus-zip`. Refresh compatible root/fuzz
  dependencies through normal resolution.
- [patch] Compare the async provider regression with published HDF5 0.3 and
  Moirai 0.9.3, retaining the identical caller and two E0277 error assertions.

## [0.3.0] - 2026-10-15

### Breaking

- [major] Public HDF5 async readers require Moirai 0.9 positioned-reader
  traits. Published Consus HDF5 0.2.0 uses Moirai 0.7; identical trait text
  does not preserve identity across those package versions. Callers migrate
  their AsyncReadAt/AsyncLength bounds and reader implementations to the same
  Moirai 0.9 provider identity. ADR0005 records the uniform 0.3.0 boundary.
- The intermediate 0.2.1 patch-class claim and first-upload handoff are
  withdrawn. Its publisher is cancelled before publishing job steps execute.
  A comparison with current main alone did not cover the published baseline.

### Fixed

- Preserve Apollo core 0.17 admission, Moirai 0.9.3, real JPEG reconstruction,
  Rust 1.95, all existing production Rust/features and test budgets.
- [patch] Add the executed public generic-consumer regression: published
  0.2.0/Moirai0.7 passes; unchanged caller on current Consus fails exactly two
  E0277 errors; migrated Moirai0.9 caller passes. No adapter or source patch.

## [0.2.1] - 2026-10-15

### Fixed

- [patch] Admit Apollo DCT/DST core 0.17 in the private JPEG transform. The
  former 0.16 requirement rejects current Apollo main before compilation.
- [patch] Complete the approved Moirai update with 0.9.3 and canonical Git
  locks. Preserve every registry identity, public API, feature, and Rust 1.95
  minimum. The async public boundary already uses the Moirai 0.9 contract on
  main; this patch retains that contract.
- [patch] Add an overlay-free fresh-consumer regression that exercises real
  JPEG encode/decode, exact constant samples, and image metadata.
