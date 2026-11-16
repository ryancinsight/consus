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
| `consus-zip` | ZIP container | NPZ and 3MF member storage |
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
consus = { version = "0.4", features = ["hdf5", "zarr", "netcdf"] }
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

Chunk compression follows the codec id in the metadata:

| Zarr codec id | Framing | Written by | Read by |
|---------------|---------|------------|---------|
| `zlib` | zlib, RFC 1950 (first byte `78`) | numcodecs `Zlib`, Consus | numcodecs `Zlib`, Consus |
| `gzip` | gzip, RFC 1952 (first bytes `1f 8b`) | numcodecs `GZip`, Consus | numcodecs `GZip`, Consus |

Earlier Consus versions registered the name `zlib` against the gzip codec, so
a caller that stored chunks through `CodecId::Name("zlib")` holds gzip streams
under that name; the Zarr pipeline itself looks codecs up by their own name and
could not resolve `zlib` at all. Consus keeps the gzip streams readable: when a
`zlib` chunk begins with the gzip magic `1f 8b`, it is decoded as gzip. No valid
zlib stream starts with `1f` (its low nibble `15` is the reserved compression
method of RFC 1950 section 2.2), so the test is unambiguous. The fallback is
decode-only; every chunk Consus writes under `zlib` is RFC 1950, which numcodecs
reads. A gzip-framed store is rewritten to be readable by numcodecs.
