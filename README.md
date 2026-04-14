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
| HDF5 | Phase 1 (in progress) | 100% target |
| Zarr v2/v3 | Phase 2 (planned) | Full read/write |
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
        .chunks(&[1, 2])
        .compression(consus::Compression::Zstd { level: 3 })
        .write(&data)?;

    Ok(())
}
```

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

## Minimum Supported Rust Version

1.85.0 (edition 2024)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.