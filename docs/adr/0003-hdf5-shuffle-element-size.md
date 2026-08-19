# ADR 0003: Carry HDF5 shuffle element size through chunk filters

- Status: Accepted
- Date: 2026-08-19
- Board item: `ATLAS-CONSUS-SHUFFLE-038`
- Evidence: `ef439b2` and the shuffle regression tests in
  `crates/consus-hdf5/src/dataset/chunk.rs`

## Context

HDF5 filter ID 2 is a byte-plane transposition over fixed-width dataset
elements. The chunk pipeline previously returned shuffled bytes unchanged, so
shuffle-plus-deflate datasets produced plausible but incorrect values. The
forward path had the same identity behavior, and the low-level pipeline needs
the dataset element width to reverse or apply the permutation.

## Decision

The synchronous and asynchronous chunk readers carry `element_size` from the
dataset reader into the filter pipeline. Filter ID 2 delegates the permutation
to the provider-owned `consus_compression::ShuffleFilter` in both directions.
The HDF5 wrapper transposes only the whole-element prefix and copies a trailing
partial element unchanged, matching the HDF5 filter framing contract. Widths
zero and one are identity cases; all other widths use the canonical provider
filter. Existing callers migrate to the changed low-level signatures together.

## Alternatives rejected

- Deferring the inverse transform to a higher-level reader preserves silent
  corruption because no such reader owns the complete filter pipeline.
- Reimplementing the permutation in `consus-hdf5` duplicates the canonical
  compression operation and permits forward/reverse drift.
- Treating a shuffle transform failure as an absent optional filter hides a
  format or pipeline defect behind plausible output.

## Verification

- The independent analytical shuffle oracle covers element sizes 1, 2, 4,
  and 8, including trailing partial elements.
- The write path is checked against the analytical byte-plane layout and the
  read path is checked as its inverse.
- The existing h5py shuffle-plus-deflate integration case verifies the
  end-to-end decoded values rather than only successful execution.
- Provider commit `ef439b2` reports workspace Nextest 2759/2759, strict
  Clippy, and formatting passing.
