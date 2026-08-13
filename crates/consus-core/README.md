# consus-core

Core types, traits, and error definitions for the [Consus](https://github.com/ryancinsight/consus)
scientific storage library.

This crate holds the abstract storage model every Consus format backend depends
on. It is `no_std`-compatible by default; enable `std` for `std::io` integration
and `std::error::Error` implementations.

```toml
[dependencies]
consus-core = { version = "0.1", default-features = false }
```

## Contents

- `core::traits` — `File`, `Group`, `Dataset`, `Attribute`, `Link`, and
  `Selection` abstractions that format crates implement.
- `core::error` — the shared error hierarchy and `Result` alias.
- `types` — canonical definitions (single source of truth) for `Datatype`,
  `ByteOrder`, `StringEncoding`, `CompoundField`, `EnumMember`, `Extent`,
  `Shape`, `ChunkShape`, `Layout`, and `Selection`.

Format crates depend on these traits rather than on each other, so a backend can
be added or swapped without touching the shared model.

- Documentation: <https://docs.rs/consus-core>

Licensed under MIT OR Apache-2.0.
