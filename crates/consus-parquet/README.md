# consus-parquet

Apache Parquet interoperability layer for the
[Consus](https://github.com/ryancinsight/consus) scientific storage library.

```toml
[dependencies]
consus-parquet = { version = "0.1", default-features = false }
```

## Coverage

Canonical schema mapping between Parquet and the `consus-core` type model:
nested group fields map to canonical compound datatypes and repeated fields to
canonical variable-length datatypes. Row-group metadata, an ordered
column-projection model, Thrift footer decoding, a file-backed reader, and the
wire-level write path are implemented and covered by value-semantic tests.

An Arrow bridge planner reports which columns can be transferred without a
representation change.

- Documentation: <https://docs.rs/consus-parquet>

Licensed under MIT OR Apache-2.0.
