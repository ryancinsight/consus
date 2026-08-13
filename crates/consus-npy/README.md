# consus-npy

Typed pure-Rust NPY and NPZ array storage.

Reads and writes NumPy's `.npy` array format and `.npz` ZIP archives without a
Python runtime and without an `ndarray` dependency.

```toml
[dependencies]
consus-npy = "0.1"
```

## Coverage

Typed little-endian `f32`, `f64`, `i32`, and `i64` arrays, with validated
shape and header parsing. `.npz` archives expose their named arrays
individually.

The crate declares `#![forbid(unsafe_code)]`.

Part of the [Consus](https://github.com/ryancinsight/consus) scientific storage
library; usable standalone.

- Documentation: <https://docs.rs/consus-npy>

Licensed under MIT OR Apache-2.0.
