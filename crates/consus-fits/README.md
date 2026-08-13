# consus-fits

Pure-Rust FITS (Flexible Image Transport System) implementation for the
[Consus](https://github.com/ryancinsight/consus) scientific storage library.

FITS is the standard interchange format for astronomical images, calibration
frames, and catalog tables. This crate reads and writes it without CFITSIO.

```toml
[dependencies]
consus-fits = { version = "0.1", default-features = false }
```

## Coverage

Read and write for primary images, `IMAGE` extensions, ASCII tables, and binary
tables, including header-card parsing and the 2880-byte block padding rules.

Through the `consus` facade, FITS products land in the same `File`/`Group`/
`Dataset` API as HDF5, Zarr, and netCDF data, so a pipeline can ingest telescope
output alongside simulation output without format-specific glue.

- Documentation: <https://docs.rs/consus-fits>

Licensed under MIT OR Apache-2.0.
