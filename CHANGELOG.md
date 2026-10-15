# Changelog

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
