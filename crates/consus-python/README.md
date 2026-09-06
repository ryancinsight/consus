# consus-python

Pure-Rust readers and writers for scientific storage formats, exposed to
Python by [Consus](https://github.com/ryancinsight/consus): HDF5, Zarr,
Parquet, netCDF-4, MATLAB `.mat`, and HDMF dynamic tables.

No C library is involved. There is no `libhdf5` to install, no version to
match against your interpreter, and nothing to build — the formats are
implemented in Rust and the wheel is self-contained.

## Install

```sh
pip install consus-python
```

The distribution is `consus-python`; the module you import is `consus`.
Wheels are built for CPython 3.9 through 3.13 on Linux, Windows, and macOS,
and have no runtime dependencies.

## Use

```python
import consus

# Readers take the file's bytes, not a path — despite the method name.
with open("measurements.h5", "rb") as f:
    h5 = consus.Hdf5File.open_path(f.read())

print(h5.list_root_group())          # ['pressure', 'temperature']

info = h5.dataset_at("/pressure")
print(info.dtype, info.shape)        # '<f8' [1024, 768]
print(info.layout, info.chunk_shape) # 'chunked' [128, 128]
print(info.filters)                  # filter pipeline ids

payload = h5.read_dataset("/pressure")   # raw bytes, decoded per info
```

MATLAB files have a one-call entry point:

```python
mat = consus.loadmat_bytes(open("run.mat", "rb").read())
```

## What is exposed

| Format | Types |
| --- | --- |
| HDF5 | `Hdf5File`, `FileBuilder`, `DatasetInfo` |
| Zarr | `ZarrArray` |
| Parquet | `ParquetFile`, `ParquetBuilder` |
| netCDF-4 | `NetcdfFile`, `NetcdfWriter` |
| MATLAB `.mat` | `MatFile`, `MatVariable`, `loadmat_bytes` |
| HDMF | `DynamicTable`, `HdmfFileBuilder`, `read_dynamic_table_bytes` |

Reads are in-memory: you hand over the bytes and the reader parses them.
That keeps the boundary explicit — nothing opens a file handle behind your
back, and the same call works on bytes that never touched a filesystem.

## Why the format logic is not here

This package converts Python values, maps Rust failures onto Python
exceptions, and registers the module. Every format implementation lives in
the Consus Rust crates it wraps, so there is one implementation and one place
to verify it against the published format specifications.

## Links

- [Source and issues](https://github.com/ryancinsight/consus)

## Licence

MIT or Apache-2.0, at your option.
