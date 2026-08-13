# consus

Unified facade for the Consus scientific storage formats.

Consus is a pure-Rust, `no_std`-compatible reimplementation of hierarchical and
array-oriented scientific storage formats. This crate re-exports the format,
compression, and I/O crates behind one feature-controlled API, and adds the
high-level `File`/`Group`/`Dataset` builders and the synchronous parallel I/O
helpers.

Applications that need only one format can depend on that format crate directly
(`consus-hdf5`, `consus-zarr`, `consus-fits`, …) and skip this facade.

```toml
[dependencies]
consus = "0.1"
```

```rust,ignore
use consus::File;

let file = File::create("experiment.h5")?;
let group = file.create_group("/simulations/run_001")?;
group.create_dataset("temperature").shape(&[2, 2]).write(&[1.0f64, 2.0, 3.0, 4.0])?;
```

## Re-exports

| Module | Crate | Feature |
| --- | --- | --- |
| `consus::core` | `consus-core` | always |
| `consus::io` | `consus-io` | always |
| `consus::compression` | `consus-compression` | always |
| `consus::hdf5` | `consus-hdf5` | `hdf5` (default) |
| `consus::fits` | `consus-fits` | `fits` (default) |
| `consus::zarr` | `consus-zarr` | `zarr` |
| `consus::netcdf` | `consus-netcdf` | `netcdf` |
| `consus::parquet` | `consus-parquet` | `parquet` |
| `consus::arrow` | `consus-arrow` | `arrow` |

## Parallel I/O

`consus::sync::par_read_ranges` executes disjoint positioned reads through
Moirai's parallel-slice API. `Parallelism::default()` sizes the partition count
from the Themis CPU topology when the default `atlas-themis` feature is on,
falling back to `std::thread::available_parallelism()` otherwise.

## Copy behavior

Hyperslab and selection reads materialize owned buffers. The `ZeroCopyRead`
trait in `consus::sync` is reserved surface, not a capability: its blanket
`impl<T: ReadAt>` always returns `ByteView::Owned`. To read a mapped file
without a copy, borrow from `consus::io::MmapReader::as_slice`.

- Documentation: <https://docs.rs/consus>
- Repository: <https://github.com/ryancinsight/consus>

Licensed under MIT OR Apache-2.0.
