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

Revision 2026-11-15: the M-interop-contracts and M-1.0 exit checks named the delivered items `CONSUS-PARQUET-RAW-DEFLATE` and `CONSUS-PYTHON-FREE-THREADING`, which the ladder rule forbids; they now name the runnable checks those deliveries added, `compress_gzip_and_zlib_emit_level_6_raw_deflate` (PR 204) and `scripts/concurrent_bindings.py` on `python3.14t` (PRs 195 and 200). No check is removed or weakened; `ATLAS-PR530-PROVIDER-COHERENCE` stays until its item has a runnable check.
