# consus-hdf5

Pure-Rust HDF5 format implementation for the
[Consus](https://github.com/ryancinsight/consus) scientific storage library.

No `hdf5-sys`, no C library, no CMake or pkg-config step. The reader and writer
are implemented directly against the HDF5 file-format specification and operate
over the `consus-io` synchronous positioned-read traits. Optional asynchronous
format reading uses Moirai's native `moirai_async::io::{AsyncReadAt,
AsyncLength}` contracts and executor; Consus does not define a second async
runtime or trait family.

```toml
[dependencies]
consus-hdf5 = { version = "0.1", default-features = false }
```

## Coverage

Read: superblocks v1/v2/v3, object headers, datatype and dataspace parsing, link
traversal (including soft-link path resolution), attribute parsing, contiguous
dataset reads, chunk metadata parsing, and dense link/attribute enumeration.

Write: superblock v2, object header v2, datatype/dataspace/layout encoding,
contiguous dataset data blocks, hard- and soft-link encoding, attribute
encoding, and chunked datasets (data layout v3 message, v1 raw-data chunk
B-tree leaf index, resolved chunk index address, filter pipeline metadata).
End-to-end value roundtrip is covered by `chunked_dataset_value_roundtrip`.

Compressed chunked writes are tracked under the filter pipeline; full
compression roundtrip coverage is in progress.

Chunk reads can be executed in parallel through Moirai under the
`parallel-io` feature. Async HDF5 reads are enabled with the `async` feature
and run on Moirai's native executor.

- Documentation: <https://docs.rs/consus-hdf5>

Licensed under MIT OR Apache-2.0.
