# consus-netcdf

Pure-Rust netCDF-4 implementation for the
[Consus](https://github.com/ryancinsight/consus) scientific storage library.

netCDF-4 is a semantic layer over HDF5. This crate implements that mapping on
top of `consus-hdf5`, so no `netcdf-sys` or C library is required.

```toml
[dependencies]
consus-netcdf = { version = "0.1", default-features = false }
```

## Coverage

HDF5-backed semantic extraction for dimension scales, variables, groups, decoded
attributes, unlimited-dimension propagation, ancestor-scope dimension resolution
for nested groups, and `DIMENSION_LIST`-based variable-to-dimension binding.
Both the classic and enhanced models have read and write paths.

- Documentation: <https://docs.rs/consus-netcdf>

Licensed under MIT OR Apache-2.0.
