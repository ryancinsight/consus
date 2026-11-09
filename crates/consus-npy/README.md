# consus-npy

Typed pure-Rust NPY and NPZ array storage.

Reads and writes NumPy's `.npy` array format and `.npz` ZIP archives without a
Python runtime and without an `ndarray` dependency.

```toml
[dependencies]
consus-npy = "0.4"
```

## Coverage

Typed little-endian `f32`, `f64`, `i32`, and `i64` arrays, with validated
shape and header parsing. `.npz` archives expose their named arrays
individually.

Scalar headers accept either quote delimiter and arbitrary dictionary-key
order. The reader requires complete dictionary and tuple boundaries, rejects
unknown or missing fields, and preserves the stored Fortran-order flag.
Parenthesized dictionaries, comments between tokens and adjacent strings are
accepted, including LF, CRLF and CR physical lines and explicit line joining
between tokens. Comment markers inside quoted strings remain data, and line
joining cannot fuse separate integer tokens. Header metadata is parsed as
literals, never executed as Python code.

The crate declares `#![forbid(unsafe_code)]`.

Part of the [Consus](https://github.com/ryancinsight/consus) scientific storage
library; usable standalone.

- Documentation: <https://docs.rs/consus-npy>

Licensed under MIT OR Apache-2.0.
