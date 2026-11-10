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
