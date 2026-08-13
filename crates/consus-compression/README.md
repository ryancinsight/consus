# consus-compression

Compression codec registry and checksum utilities for the
[Consus](https://github.com/ryancinsight/consus) scientific storage library.

Format backends delegate compression and decompression through this crate's
trait-based abstraction instead of depending on codec crates directly, so the
codec set is a dependency decision made in one place.

```toml
[dependencies]
consus-compression = { version = "0.1", default-features = false }
```

## Contents

- `codec` — the `Codec` trait, `CodecId`, and `CompressionLevel`, with
  feature-gated implementations for deflate/zlib, gzip (Zarr), zstd, and lz4.
- `checksum` — CRC-32 (IEEE 802.3), Fletcher-32 (HDF5 filter ID 3), and Jenkins
  lookup3 (HDF5 v2 metadata checksums).
- A runtime registry mapping codec identifiers to implementations, used by the
  HDF5 filter pipeline and the Zarr codec pipeline.

Every codec is a pure-Rust implementation; no C compression library is linked.

- Documentation: <https://docs.rs/consus-compression>

Licensed under MIT OR Apache-2.0.
