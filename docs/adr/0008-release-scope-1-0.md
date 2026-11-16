# ADR 0008: Consus release scope for 1.0

- Status: Accepted
- Date: 2026-11-10
- Class: `[arch]` `[minor]`

## Context

Consus (workspace 0.4.0) provides pure-Rust hierarchical and array-oriented
storage format implementations for the Atlas stack. The backlog has open
provider-coherence, compression, and Python-binding items, and no milestone
ladder exists to 1.0.

## Decision

Adopt a three-milestone ladder to 1.0 recorded in `backlog/milestones.md`:

- **M-core-formats** (0.5.0): users read and write core hierarchical and array
  formats (HDF5, Zarr, netCDF-4) through Consus crates.
- **M-interop-contracts** (0.6.0): users access cross-format, document, and
  compression contracts with verified behavior.
- **M-1.0** (1.0.0): stable storage-format surface (authority-gated) with
  documented contracts and release gates satisfied.

## Alternatives

- Plan 1.0 as a single step: rejected because open provider-coherence and
  interop work remains.
- Keep progress implicit in format status tables: rejected because stack policy
  requires runnable capability-oriented milestone exit checks.

## Consequences

- Open items carry `milestone:` assignments and feature items name the exit
  check they serve.
- Milestones close only when exit checks pass on fetched default commits; 1.0
  release/publication remains authority-gated.
- This ADR is rewritten in place as scope evidence changes.

Revision 2026-11-15: the M-interop-contracts and M-1.0 exits named items (`CONSUS-PARQUET-RAW-DEFLATE`, `ATLAS-PR530-PROVIDER-COHERENCE`, `CONSUS-PYTHON-FREE-THREADING`), which the ladder rule forbids; none was a runnable check. `CONSUS-PYTHON-FREE-THREADING` becomes the free-threaded run of `scripts/concurrent_bindings.py` on `python3.14t` (PRs 195 and 200). The other two exits are removed, not replaced. `CONSUS-PARQUET-RAW-DEFLATE` (e09e063, PR 204) is a refactor onto `RawDeflateCodec` with identical output: the earlier code called flate2's `DeflateEncoder` at `Compression::default()`, level 6 in the locked flate2 1.1.11 (`src/lib.rs:264-267`), so no check fails without the delivery, and its test stays inside the `cargo test -p consus-parquet` exit. `ATLAS-PR530-PROVIDER-COHERENCE` is a `correctness` item, so the standing condition "no open correctness item" keeps M-1.0 open until it closes. No capability a user can exercise leaves the ladder.
