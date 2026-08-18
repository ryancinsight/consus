# consus-io

Sync and async I/O abstractions for the [Consus](https://github.com/ryancinsight/consus)
scientific storage library.

This crate decouples format logic from physical I/O by defining position-aware
read/write traits. Format backends operate on `ReadAt`/`WriteAt` rather than on
`std::fs::File`, which makes in-memory buffers, memory maps, object stores, and
custom `no_std` transports interchangeable.

```toml
[dependencies]
consus-io = { version = "0.1", default-features = false }
```

## Contents

- `io::traits` — `ReadAt`, `WriteAt`, `Length`, `Truncate`, `Seekable`,
  `SeekFrom`, `RandomAccess`.
- `io::sync` — `MemCursor` (in-memory), `SliceReader` (`&[u8]`), `StreamReader`
  (sequential), the `std::fs::File` implementation, and `MmapReader`.
- `io::async_io` — asynchronous in-memory adapters over the runtime-agnostic
  `async-traits` surface; network and object-storage integrations are outside
  this crate.

## Copy behavior

`ReadAt::read_at` fills a caller-provided buffer. `MmapReader::as_slice` returns
the mapping itself, so callers that want to avoid a copy can borrow from it
directly.

`MmapReader::from_file` contains one of the two `unsafe` blocks in the
workspace: `memmap2::Mmap::map` is unsafe because the mapping is invalidated if
the backing file is truncated while the reader is alive. The safety contract is
documented at the call site and in the module docs.

- Documentation: <https://docs.rs/consus-io>

Licensed under MIT OR Apache-2.0.
