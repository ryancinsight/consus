# 0007 — First-party ZIP container

Status: Accepted

Date: 2026-10-20

Change class: [major] for `consus-npy`, whose error type changes; [minor] for the
new `consus-zip` crate and `RawDeflateCodec`.

## Context

NPZ archives (`consus-npy`) and 3MF meshes (gaia) are ZIP containers of
in-memory members. Both depended on the third-party `zip` crate, which shipped
three extraction advisories in one cohort: RUSTSEC-2026-0367, 0368 and 0369.
Those advisories concern SUID bits, symlink targets and permission application
during filesystem extraction. Neither consumer extracts to a filesystem, yet each
advisory still failed every downstream `cargo deny` gate, kwavers included, until
the crate moved. The owner chose to own the container instead of tracking the
crate's release cadence.

## Decision

Add `consus-zip`, a bounded reader and writer for stored (method 0) and DEFLATE
(method 8) members. It provides CRC-32 checks, ZIP64 sizes, offsets and entry
counts, and typed errors. It offers no filesystem extraction, symlink or
permission surface: members are returned as bytes.

- **Placement.** The two consumers share no existing crate, so the container is a
  leaf crate at their common ancestor, following the workspace convention of one
  crate per format. gaia is to depend on it by Git (its 3MF writer moves in
  the gaia 0.6.10 change of the atlas `ATLAS-ZIP9-COHORT-SWEEP` item); consus
  does not depend on gaia.
- **DEFLATE.** Raw DEFLATE comes from the new `consus-compression`
  `RawDeflateCodec`. Decoding goes through the shared bounded flate path.
- **Bounds.** Every declared offset and length is checked against the region that
  must hold it before any read or allocation. A DEFLATE member's declared size is
  checked against the 1032:1 expansion bound (zlib technical details, "Maximum
  Compression Factor"). The initial reservation is at most the compressed size.
  Hostile-input lints (`indexing_slicing`, `arithmetic_side_effects`) are denied
  crate-wide.
- **Determinism.** Written members carry the fixed date 1980-01-01, so equal
  inputs produce equal archives.
- **Migration.** `consus-npy` wraps `consus_zip::Error`, and its writer needs only
  `Write`.

## Rejected alternatives

- **Admit zip 9.** Fixes the current advisories but keeps the extraction surface
  and the advisory cadence that caused this cohort.
- **A module inside `consus-compression`.** It would have avoided the first-publish
  API token a new crate needs. The rejection reason is that it puts a container
  format in the codec crate, choosing structure by operational cost.
- **Encryption, other methods, multi-disk archives.** No consumer needs them, so
  they are rejected with typed errors.

## Consequences

- The consus graph carries no `zip` crate. gaia still does until its 3MF
  writer moves onto `consus-zip` under `ATLAS-ZIP9-COHORT-SWEEP`.
- The first crates.io publish of `consus-zip` needs an owner API token. After that,
  trusted publishing applies, and `consus-npy` 0.4.0 publishes once `consus-zip`
  exists on the registry.
- The `fuzz_zip_reader` target runs in the scheduled fuzz matrix.

## Evidence

The executable tests decode archives written by the `zip` crate 9.0.5 and by
NumPy 2.5.2 (`savez`, `savez_compressed`).

Independent review of PR #180 additionally reported the following. These
checks are not committed tests:

- Python's `zipfile` and NumPy read archives written by consus-zip, up to
  65,536 entries and a 4 GiB member.
- 28 crafted hostile inputs and 300,000 mutations produced typed errors only.

## Overturning evidence

Revisit this decision if a consumer needs encryption, streaming of members
larger than memory, or filesystem extraction.
