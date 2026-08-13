# consus-hdmf

Hierarchical Data Modeling Framework (HDMF) `DynamicTable` read and write
support for the [Consus](https://github.com/ryancinsight/consus) scientific
storage library.

HDMF is the container and table specification underlying NWB. This crate
implements the `DynamicTable` model over `consus-hdf5`, compatible with HDMF
Python 4.x and NWB 2.x files:

| HDMF type | Rust representation |
| --- | --- |
| `DynamicTable` | `DynamicTable` |
| `VectorData` (`f64`/`i64`/`u64`/`bool`/`str`) | `ColumnData` variants |
| `VectorIndex` (ragged columns) | `Column::index` |
| `ElementIdentifiers` | `DynamicTable::id` |

```toml
[dependencies]
consus-hdmf = { version = "0.1", default-features = false }
```

It is used by `consus-nwb`, and is available directly for other HDMF-based
specifications.

The crate declares `#![forbid(unsafe_code)]`.

- Documentation: <https://docs.rs/consus-hdmf>

Licensed under MIT OR Apache-2.0.
