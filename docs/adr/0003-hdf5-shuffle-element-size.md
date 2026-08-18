# ADR-0003: Carry HDF5 shuffle element size through chunk reads

- Status: Accepted
- Date: 2026-08-18
- Board item: `ATLAS-CONSUS-SHUFFLE-038`
- Evidence: `crates/consus-hdf5/src/dataset/chunk.rs`,
  `crates/consus-hdf5/src/file/mod.rs`,
  `crates/consus-hdf5/src/file/async_file.rs`, and the shuffle regression
  tests in `crates/consus-hdf5/src/dataset/chunk.rs`

## Context

HDF5 filter ID 2 is a byte transposition over fixed-width dataset elements.
The chunk reader had no element-width input, so it returned shuffled bytes
unchanged. The writer had the same gap and encoded the original bytes while
claiming the shuffle filter had run. A high-level reader cannot recover the
width after the low-level pipeline has discarded it.

## Decision

`read_chunk_raw` and `async_read_chunk_raw` receive the validated dataset
element width beside the expected uncompressed byte count. The width is
carried through `ChunkTask` into both serial and parallel chunk readers. Filter
ID 2 delegates to the provider-owned `consus_compression::ShuffleFilter` in
both directions; no HDF5-local permutation is added.

The public low-level signatures change together with every in-repository
caller. Callers that do not use shuffle pass their actual dataset width; test
and benchmark byte payloads use width one. A zero width or byte count not
divisible by the width returns `Error::InvalidFormat`. The deterministic
built-in shuffle error is propagated by the writer instead of being converted
to a filter-mask bypass.

## Alternatives rejected

- Passing shuffle through to a higher-level reader: no such reader owns the
  complete filter pipeline, so this preserves silent corruption.
- Duplicating the permutation in `consus-hdf5`: this forks the canonical
  compression implementation and permits forward/reverse drift.
- Treating malformed shuffle input as an optional filter failure: that writes
  data without the requested transform and masks a format error.

## Verification

- `apply_forward_filter` and `apply_reverse_filter` round-trip element widths
  1, 2, 4, and 8.
- A write/read chunk round-trip proves that the on-disk bytes are shuffled and
  the restored bytes match the input.
- Zero-width and misaligned input return `Error::InvalidFormat`.
- The existing h5py shuffle+deflate integration test remains the end-to-end
  value oracle.
