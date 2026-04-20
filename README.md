# Consus

**Pure-Rust, `no_std`-compatible, memory-safe reimplementation of hierarchical and array-oriented storage formats for scientific computing.**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

## Overview

Consus replaces C-dependent bindings (hdf5-rs, netCDF-sys, etc.) with a native Rust implementation providing:

- **Zero-copy I/O** with hyperslab and selection reads
- **Full compression support** (zlib, gzip, zstd, lz4, blosc, szip)
- **Thread-safe parallel I/O** via Rayon and Tokio
- **WASM and embedded targets** (`no_std` compatible core)
- **Pluggable backend architecture** for format interoperability
- **Performance parity or better** than HDF5 C library with Rust safety guarantees

## Supported Formats

| Format | Status | Spec Compliance |
|--------|--------|-----------------|
| HDF5 | Phase 1 (in progress) | Read path implemented (v1/v2 groups, all datatype classes, multi-chunk with filter pipeline); Write path implemented (contiguous + chunked with v3 layout); async I/O path implemented |
| FITS | Phase 2 – Complete | Full read/write for primary images, IMAGE extensions, ASCII tables, and binary tables |
| Zarr v2/v3 | Phase 2 (in progress) | Metadata parsing, codec pipeline, chunk read/write, full-array read/write, and verified partial read/write selection support |
| netCDF-4 | Phase 2 (planned) | Classic + Enhanced |
| Apache Parquet | Phase 3 (planned) | Columnar interop |

## Architecture

```text
consus (facade)
├── consus-core        # Core types, traits, error types (no_std)
├── consus-io          # Sync/async I/O abstractions
├── consus-compression # Codec registry (zlib, zstd, lz4, etc.)
├── consus-hdf5        # HDF5 format implementation
├── consus-zarr        # Zarr v2/v3 implementation
├── consus-netcdf      # netCDF-4 implementation
└── consus-parquet     # Parquet interop layer
```

## Quick Start

```rust,ignore
use consus::File;

fn main() -> consus::Result<()> {
    // Create a new HDF5 file
    let file = File::create("experiment.h5")?;
    let group = file.create_group("/simulations/run_001")?;

    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    group.create_dataset("temperature")
        .shape(&[2, 2])
        .write(&data)?;

    Ok(())
}
```

## Current HDF5 Verification Status

Current repository verification indicates:

- HDF5 read support covers superblocks, object headers, datatype parsing, dataspace parsing, link traversal, attribute parsing, contiguous dataset reads, chunk metadata parsing, dense link and dense attribute enumeration, and soft-link path resolution.
- HDF5 write support currently covers superblock v2 writing, object header v2 writing, datatype/dataspace/layout encoding, contiguous dataset data blocks, hard-link encoding, soft-link encoding, and attribute encoding.
- Chunked dataset write is implemented: the v3 data layout message, v1 raw-data chunk B-tree leaf index, resolved chunk index address, and filter pipeline metadata are all serialized correctly. End-to-end value roundtrip is verified by `chunked_dataset_value_roundtrip`.
- Compressed chunked writes are tracked under the filter pipeline; full compression roundtrip coverage is in progress.


## Target Users

- Research labs and simulation teams
- ML pipelines requiring reproducible data storage
- Climate, geospatial, and bioinformatics workflows
- Enterprise R&D requiring long-term data preservation
- Embedded and WASM deployments needing portable I/O

## Strategic Advantages

- **Portability**: Single `cargo add consus` — no system libraries, no CMake, no pkg-config
- **Auditability**: Pure Rust source, no C FFI boundary to audit
- **Supply-chain security**: Minimal, auditable dependency tree
- **Cross-compilation**: Targets WASM, ARM, RISC-V without toolchain pain
- **Memory safety**: Zero unsafe in format logic; ownership-driven resource management
- **Astronomy pipeline interoperability**: FITS support enables direct ingestion of telescope image products, calibration frames, and catalog tables into the same unified API used for other scientific formats
- **Spectral and survey R&D continuity**: One storage facade now spans astronomy archive interchange and downstream analysis workflows, reducing format-specific glue code in spectral reduction, sky survey processing, and observatory data engineering

## Minimum Supported Rust Version

1.85.0 (edition 2024)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.