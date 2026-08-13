# consus-arrow

Apache Arrow interoperability layer for the
[Consus](https://github.com/ryancinsight/consus) scientific storage library.

This crate maps between the Consus canonical type model (`consus-core`) and the
Arrow columnar model: schema and field translation, an IPC record-batch
description layer, and a bridge planner that reports which fields can be handed
over without a representation change (`is_zero_copy_eligible`).

```toml
[dependencies]
consus-arrow = { version = "0.1", default-features = false }
```

This is a semantic and planning layer. It does not depend on the `arrow` crate
and does not execute Arrow compute kernels; it describes the mapping so a
consumer can perform the transfer.

- Documentation: <https://docs.rs/consus-arrow>

Licensed under MIT OR Apache-2.0.
