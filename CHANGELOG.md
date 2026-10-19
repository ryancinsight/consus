# Changelog

## [0.4.0] - Unreleased

### Breaking

- [major] Public HDF5 async reader bounds use Moirai 0.10's nominal
  AsyncReadAt/AsyncLength traits. Published 0.3 used 0.9.3; callers crossing
  this boundary migrate provider identity. Workspace and internal versions
  advance together to 0.4. ADR0006 records the published baseline.

### Added

- [minor] Support CPython 3.14 with PyO3/build-config 0.29.3, preserving the
  Python names, values and cloned class extraction contract. Retain the older
  wheel interpreters and run the actual format interoperability suite in CI.

### Fixed

- [patch] Forward the caller-supplied variable element width through chunked
  HDF5 reads. Keep fixed-width dispatch and missing-width rejection; verify
  actual plain/compressed global-heap string references without weakening the
  existing HDMF string-column oracle.
- [patch] Admit patched ZIP 9.0.5 with the existing format/compression features,
  and refresh all compatible root/fuzz dependencies through normal resolution.
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
