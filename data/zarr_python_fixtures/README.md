# Python Zarr Interoperability Fixtures

This directory contains deterministic filesystem-backed fixtures generated with Python `zarr` and `numpy` for cross-implementation verification of `consus-zarr`.

## Purpose

These fixtures close the Phase 2 interoperability gap by validating that the Rust implementation can read stores produced by Python Zarr for:

- Zarr v2 metadata and chunk layout
- Zarr v3 metadata and chunk layout
- uncompressed chunk payloads
- gzip-compressed chunk payloads
- full-array reads
- partial contiguous reads across chunk boundaries
- partial strided reads across chunk boundaries

The fixtures are intentionally small so they remain suitable for repository storage and fast integration testing.

## Layout

```text
data/zarr_python_fixtures/
├── README.md
├── generate_fixtures.py
└── generated/
    ├── manifest.json
    ├── v2_uncompressed_i4/
    ├── v2_gzip_f8/
    ├── v3_uncompressed_i4/
    └── v3_gzip_f8/
```

## Fixture Set

### `v2_uncompressed_i4`

- Zarr format: `2`
- dtype: `"<i4"`
- shape: `[4, 6]`
- chunks: `[2, 3]`
- compressor: none
- fill value: `-1`

Coverage:
- v2 `.zarray` parsing
- dot-separated chunk keys
- uncompressed chunk decoding
- full-array read
- contiguous partial read
- strided partial read

### `v2_gzip_f8`

- Zarr format: `2`
- dtype: `"<f8"`
- shape: `[5, 4]`
- chunks: `[2, 2]`
- compressor: `gzip(level=1)`
- fill value: `0.0`

Coverage:
- v2 compressed chunk decoding
- floating-point payload reconstruction
- strided partial read over compressed chunks

### `v3_uncompressed_i4`

- Zarr format: `3`
- dtype: `"int32"`
- shape: `[3, 5]`
- chunks: `[2, 2]`
- codecs: Python-generated v3 default chain for this configuration
- fill value: `0`

Coverage:
- `zarr.json` parsing
- slash-separated `c/...` chunk keys
- v3 chunk traversal
- full-array read
- contiguous partial read
- strided partial read

### `v3_gzip_f8`

- Zarr format: `3`
- dtype: `"float64"`
- shape: `[4, 4]`
- chunks: `[2, 2]`
- codecs: `bytes(endian=little)` then `gzip(level=1)`
- fill value: `0.0`

Coverage:
- ordered v3 codec chain decoding
- floating-point full-array reconstruction
- contiguous and strided partial reads

## Manifest

`generated/manifest.json` is the authoritative summary of generated fixtures. It records:

- generator script name
- Python `zarr` version
- `numpy` version
- fixture metadata
- analytically derived expected values used by Rust-side tests

The manifest exists to keep fixture intent and expected semantics synchronized with the generated stores.

## Regeneration Workflow

### Prerequisites

Use a Python environment with:

- `python`
- `zarr`
- `numpy`
- `numcodecs`

Example:

```text
py -m pip install zarr numpy numcodecs
```

### Generate Fixtures

From the repository root:

```text
py -3.13 data/zarr_python_fixtures/generate_fixtures.py
```

This script:

1. deletes `data/zarr_python_fixtures/generated/`
2. recreates all fixture stores from scratch
3. writes `manifest.json`
4. prints the generated fixture names

## Determinism Requirements

The generator must preserve these invariants:

- no randomness
- fixed shapes, chunk shapes, dtypes, and codec parameters
- analytically constructed array values
- stable expected-value lists in `manifest.json`
- complete regeneration from source script alone

If fixture content changes, the corresponding Rust integration tests and sprint artifacts must be updated in the same change.

## Notes

- Python Zarr may emit additional v3 metadata fields such as `attributes` or `storage_transformers`. Rust-side tests should validate semantically relevant fields and tolerate spec-valid extra fields when appropriate.
- Python Zarr v3 may choose a default codec chain for some arrays. The generated fixture files, not assumptions, are the source of truth.
- These fixtures are verification assets, not benchmarks. Keep them small and representative.

## Update Policy

Regenerate fixtures when one of the following changes:

- Python Zarr metadata conventions relevant to supported features
- codec configuration under test
- fixture shapes or expected selections
- Rust interoperability scope for `consus-zarr`

When regenerating:

1. run the generator
2. inspect `generated/manifest.json`
3. run the Rust Zarr integration tests
4. update `README.md`, `backlog.md`, `checklist.md`, and `gap_audit.md` if verification scope changes