# Data Formats

Consus provides a unified facade over multiple scientific data formats. Each
format is a separate crate that can be used independently or through the
`consus` facade crate.

## Format Backends

| Crate | Format | Use Cases |
|-------|--------|-----------|
| `consus-hdf5` | HDF5 | Hierarchical scientific data, medical imaging (DICOM), neuroscience |
| `consus-zarr` | Zarr v2/v3 | Cloud-native chunked arrays, large-scale imaging |
| `consus-netcdf` | netCDF-4 | Climate, oceanographic, atmospheric data |
| `consus-parquet` | Apache Parquet | Columnar analytics, ML datasets |
| `consus-arrow` | Apache Arrow | In-memory columnar data interchange |
| `consus-fits` | FITS | Astronomical imaging and spectral data |
| `consus-npy` | NumPy NPY/NPZ | Python array interchange |
| `consus-mat` | MATLAB MAT | MATLAB workspace files |
| `consus-nwb` | NWB | Neurodata Without Borders (neuroscience) |
| `consus-onnx` | ONNX | ML model interchange |

## Canonical Datatype System

All backends map their native types to the common `Datatype` enum defined in
`consus-core`. The canonicalization invariant ensures that two format-specific
types representing the same logical type produce identical `Datatype` values.

Key type groups:

| Group | Variants |
|-------|---------|
| Integers | `Int8`, `Int16`, `Int32`, `Int64`, `Uint8`, `Uint16`, `Uint32`, `Uint64` |
| Floats | `Float32`, `Float64` |
| Complex | `Complex64`, `Complex128` |
| Strings | `FixedString(n)`, `VariableString(enc)` |
| Structured | `Compound(Vec<CompoundField>)`, `Enum { base, members }` |
| N-D | `Array { datatype, shape }` |

`ByteOrder` (`LittleEndian` / `BigEndian`) qualifies all multi-byte scalar types.
`element_size()` returns `Some(n)` for fixed-size types and `None` for
variable-length types.

## Compression Codecs

The `consus-compression` crate registers codecs that are applied transparently
during chunked I/O:

| Codec | Feature |
|-------|---------|
| DEFLATE (zlib/gzip) | `deflate` (default) |
| Zstandard | `zstd` |
| Blosc | `blosc` |
| LZ4 | `lz4` |
| BZip2 | `bzip2` |
| SZIP | `szip` |

Codecs are selected per-dataset at creation time via `DatasetConfig`.

## Facade Feature Flags

```toml
[dependencies]
consus = { version = "0.1.0", features = ["hdf5", "zarr", "netcdf"] }
```

| Feature | Default | Enables |
|---------|---------|---------|
| `std` | yes | std I/O integration |
| `hdf5` | yes | HDF5 backend |
| `fits` | yes | FITS backend |
| `deflate` | yes | DEFLATE compression |
| `zarr` | no | Zarr v2/v3 backend |
| `netcdf` | no | netCDF-4 backend |
| `parquet` | no | Parquet backend |
| `arrow` | no | Arrow backend |

## Python Interoperability

Zarr metadata (`.zarray`, `.zgroup`, `.zattrs`, `zarr.json`, `.zmetadata`) is
byte-for-byte compatible with `zarr-python`. Chunk keys follow the canonical
Zarr convention. This allows Python tools to read arrays written by Consus and
vice versa without conversion.
