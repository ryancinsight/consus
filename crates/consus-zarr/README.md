# consus-zarr

Pure-Rust Zarr v2 and v3 implementation for the
[Consus](https://github.com/ryancinsight/consus) scientific storage library.

```toml
[dependencies]
consus-zarr = { version = "0.1", default-features = false }
```

## Coverage

Metadata parsing, codec pipeline, chunk read/write, full-array read/write, and
partial selection read/write for both v2 and v3 stores. Boundary-chunk stride
handling and the sharding codec are verified against Python-generated fixtures.
The v3 metadata write path preserves dimension names and group attributes.

## Stores

The `Store` trait abstracts chunk placement. A filesystem store is available by
default; the `s3-moirai` feature adds `S3MoiraiStore`, an object-store backend
over the native `moirai-http` S3 client with SigV4 signing (no tokio, no AWS
SDK).

- Documentation: <https://docs.rs/consus-zarr>

Licensed under MIT OR Apache-2.0.
