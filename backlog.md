# Consus — Backlog

<a id="CONSUS-RASTER-001"></a>
## CONSUS-RASTER-001 — Shared raster codecs [arch] [minor]

- Status: done; integrator: root; [PR 76](https://github.com/ryancinsight/consus/pull/76).
- Driver: [METIS-ASSETS-001](../metis/backlog.md#METIS-ASSETS-001).
- Outcome: one bounded JPEG decoder serves desktop assets and medical samples.
- Scope: `consus-raster`, JPEG sequential/progressive/lossless decoding and EXIF;
  RITK retains modality conversion, Metis retains scoped access and rendering.
- Acceptance: exact lossless samples, independent progressive fixtures, all eight
  orientations, malformed/truncated/budget rejection, no downstream decoder copy.
- Decision: ADR 0004; no reverse dependency on RITK or Metis.
- Verification: configured focused nextest, Clippy, docs, consumer gates and V06.
- Provider: focused debug/release tests and Clippy pass; consumer integration remains tracked by the driver.

## CONSUS-CONFORMANCE-RATCHET-2026-08-31 [patch] — implementation complete; merge pending

- Outcome: restore the Atlas debt ratchet at the exact pinned Consus revision
  without raising its committed baseline.
- Scope: bound the Python publishing job; classify the repository-standard
  `.git-blame-ignore-revs` false positive in the owning Atlas scanner.
- Acceptance: Consus `workflow_missing_timeout` and `root_sprawl` both return
  to zero; workflow syntax and the Atlas detector regression suite pass.
- Evidence: provider source `a6cf113`; Atlas detector source `e421540a6`; the
  27-case detector suite passes and the combined exact scan reports
  `root_sprawl=0`, `workflow_missing_timeout=0`.
- Integrator: Codex; lease: none. Provider and Atlas review/merge remain.
- Last update: 2026-08-31.

## ATLAS-CONSUS-UNWRAP-099 — Close parser-test ratchet delta [patch, complete]

**Owner:** Atlas session; scope is the three bare unwraps introduced by the
feature-qualified parser regression tests in `consus-mat` and
`consus-parquet`. The root classifier still needs an independent fix for its
`cfg(all(test, ...))` test-region detection; this item does not edit that
peer-owned root script or its baseline.

**Acceptance:** replace the three test unwraps with invariant-bearing,
value-semantic assertions; the provider scan returns `unwrap_production=383`
without a baseline edit; focused parser tests, strict Clippy, formatting, and
locked provider gates pass.

**Outcome:** `a9a56ad` replaces both numeric match-guard unwraps with bound
`if let` guards and serializes the NWB inheritance value through one optional
binding. The scan returns `unwrap_production=383`; no baseline edit is made.
Locked strict Clippy passes, default Nextest run
`6ad69fd9-39d8-4c96-a442-f02054bc9c97` passes 2553/2553, no-default Nextest
run `bb1c85da-f113-40a6-9eaf-7209c23e67` passes 2031/2031, and workspace
doctests pass. Hosted exact-head CI `32020339446`, Documentation
`32020339452`, and Pages `32020338335` all pass at `a9a56ad`.

The root classifier's `cfg(all(test, ...))` test-region defect remains a
separate peer-owned Atlas item; this provider change closes the committed
ratchet without hiding that residual.


## ATLAS-ORPHAN-MODULES-096-CONSUS — Remove unreachable source duplicates [patch, complete]

**Owner:** Atlas session; scope is the six current `orphan_modules` findings
and these provider-local PM records. No parser, format, lockfile, or peer
scope is included.

**Finding:** the Atlas module-graph detector reports
`consus-fits/src/card/mod.rs`, `consus-fits/src/fits/format.rs`,
`consus-parquet/src/schema/hybrid/mod.rs`,
`consus-parquet/src/wire/{metadata_writer,page_writer}.rs`, and
`consus-zarr/src/tests/integration.rs`. None is reached by a Cargo target
root or a `mod`/`#[path]` declaration. The live tree already owns the FITS
header card API, Parquet hybrid/wire APIs, and Zarr external test targets.

**Acceptance:** delete the unreachable files, reduce Consus's orphan count
from six to zero, preserve the live public APIs, and pass standalone locked
format/check/Clippy/Nextest/doctest gates without lockfile churn.

**Outcome:** all six files are deleted; the direct detector reports
`orphan_modules=0`, `git diff --check` passes, and locked metadata resolution
passes. The committed lock graph is refreshed from stale `zstd` entries to the
manifest's `zrip` graph, adding four `zrip` packages and removing stale
`jobserver`/`zstd` entries. Default locked check, warning-denied Clippy,
Nextest `2553/2553`, and doctests pass; no-default check, Clippy, Nextest
`2031/2031`, and doctests also pass. The deletion touches no reachable module,
so the compiled API surface is unchanged.


## ATLAS-CONSUS-PARSE-LIMITS-036 — Bound remaining HDF5 heap and dataset allocations [minor] — verified already bounded 2026-08-15

- Scope: `consus-hdf5`. The sweep during -037 re-verified every site named in
  the original brief against the current tree.
- Premise correction (same class as -035): all named sites are already
  bounded at the implementation head: `heap/global.rs` `collection_size` →
  `ParseBudget::checked_bytes` + `checked_sub` + `budget.zeroed`;
  Fixed-Array `nelmts` → `checked_elements`; dataspace products →
  `checked_*_size` via `checked_elements` (footprint bounded); fractal-heap
  `length` → `read_bounded_bytes` → `budget.zeroed`; v2 chunk sizes →
  `MAX_CHUNK_BYTES`; filter counts are single-byte (≤255) and symbol counts
  `u16` (≤65535), both inherently bounded.
- Acceptance: a fresh exhaustive allocation sweep (all production
  `vec!`/`with_capacity` sites in consus-hdf5) found no unbound
  attacker-chosen length; no code change required. Closed as verified.

## REL-001 — Python release wheels [patch] — blocked

- Owner: Codex `/root`; scope: `consus-python` package metadata, the Python
  release workflow, distribution documentation, committed Nextest budgets,
  the CI-blocking compression-test import, the touched compression package's
  warning floor, obsolete HDF5 property-test scaffolding, the adjacent
  large-file regression warning, cross-platform Arrow/IO/Zarr test build
  defects, deterministic S3 differential credentials, the committed workspace
  dependency lock, the native-test CI runner and supply-chain pins, and this
  owner-keyed PM entry.
  Python binding behavior and other Consus crate behavior are non-goals.
- Reopen trigger: the `consus-python` PyPI pending trusted publisher is
  registered and release authority is granted for the first tagged
  publication. Implementation and hosted verification are complete; no
  release or deployment is authorized by this repository state.
- Acceptance: a GitHub Release tagged `consus-python-v<version>` builds locked
  Linux, Windows, and universal macOS wheels for every supported CPython,
  installs and imports each wheel, validates metadata against the tag, attests
  and attaches the exact artifacts to the GitHub Release, then publishes the
  same wheels to the `consus-python` PyPI project through OIDC.
- Current evidence: actionlint, locked Cargo metadata, package check,
  warning-denied Clippy, and a production CPython 3.13 wheel build pass. The
  wheel installs as `consus-python` version `0.1.0`, imports as `consus`, and
  exposes the expected format classes. The GitHub environment `pypi` accepts
  only `consus-python-v*` tags. Hosted CI and PyPI pending-publisher
  registration are pending. The first hosted matrix exposed an unconditional
  unused `CodecId`
  test import; removing it and resolving the touched package's two range-loop
  diagnostics restores warning-denied all-target Clippy and all 357
  all-feature compression tests under committed 30/60-second Nextest budgets.
  The corrected head then exposed an empty HDF5 property-test artifact and one
  unused large-file-test local on macOS; both are removed at their source.
  Focused warning-denied Clippy and all 415 all-feature HDF5 tests pass.
  The same matrix exposed one unused Arrow setup value, two unused IO imports,
  one ambiguous empty-slice assertion, a missing `futures` test-only dependency
  caused by an unnecessary `join_all`, an unused Zarr test registry, and eight
  unused Zarr property-test strategies. Each is removed or expressed directly
  without a new dependency. Current-toolchain all-target Clippy also exposed an
  IO match guard and mechanical Zarr test representations; both packages now
  pass warning-denied all-target checks. The Zarr chunk-count property now uses
  independently counted chunk starts instead of comparing one formula to
  itself. Arrow passes 81 tests, IO passes 246 tests, and Zarr passes 314 tests
  under Nextest. The MinIO lane's 403 was a test race: the
  in-process differential overwrote process-global AWS credentials while the
  live-endpoint test read them. Rusoto now receives a static provider directly;
  both live tests pass concurrently against MinIO and Moirai `91c802e2`. The
  workspace now commits `Cargo.lock`, so fresh release runners can honor the
  Maturin `--locked` contract. CI now runs native tests exclusively through
  cargo-nextest `0.9.140`, pins every third-party action, pins MinIO by image
  digest, and checksum-verifies the versioned MinIO client. The live test puts a
  deterministic nontrivial byte pattern and verifies the ranged result against
  those source bytes; both S3 tests pass concurrently under Nextest. Workspace
  formatting runs once as the package-check prerequisite, and Clippy covers all
  package targets without repeating the format pass eleven times. Hosted run
  `29795739435` then exercised Rust `1.97.1` across all 57 jobs and isolated six
  package failures to new test-only Clippy diagnostics in Core, FITS, Arrow,
  HDF5, Parquet, and NetCDF. The corrective patch uses `BuildHasher::hash_one`,
  direct enum constructors, exactly representable numeric fixtures, an exact
  HDF5 reference oracle, non-vacuous object-header validation, array-backed
  FITS cards, and test modules after production items; exact-toolchain local
  verification and the replacement hosted head remain pending. Replacement
  run `29796739510` passed 42 jobs before its HDF5 check found one remaining
  approximate-constant assertion in a committed reference fixture. That
  assertion now uses the fixture's exact `157 / 50` value without a tolerance.
  Run `29797375813` then passed 54 jobs and exposed two final cross-platform
  defects: two more HDF5 test fixtures resembled approximate constants, and
  timestamp-derived mmap test paths collided under parallel macOS Nextest.
  The HDF5 oracles now use exact binary/rational values, while both mmap test
  layers use OS-unique `NamedTempFile` instances. Exact code head `a558e79`
  passes all 58 jobs in hosted run `29797846759`, including the HDF5 Clippy
  and macOS IO lanes that failed on the preceding head. PR #2 merged the
  corrected `consus-python` distribution contract to `main` as `e07c2b1`.
  Follow-up PR #3 excludes the Python `cdylib` from workspace Rustdoc output,
  preventing its `consus` filename from colliding with the Rust facade while
  leaving the built extension and Rust documentation owners unchanged.
- [x] [minor] M-053: Own bounded zero-copy ONNX protobuf document parsing for
  RITK so format inspection no longer pulls a tensor-runtime parser. The
  `consus-onnx` crate decodes graph topology, tensor metadata, operator sets,
  and borrowed raw initializer payloads under explicit document/field/node/
  value/dimension limits. Evidence: nextest 3/3, warning-denied Clippy, and
  Rustdoc.

- [x] [patch] Preserve `consus-io`'s bounded-capacity and unsized-reader
  consumer contracts across the ONNX provider revision. Evidence: exact cap
  laws and a `dyn Read` value-semantic regression.

- [x] [minor] Own bounded exact streaming reads for hostile format-declared
  lengths so RITK format crates can remove their legacy core dependency.

- [x] [minor] Own typed NPY/NPZ storage so simulation consumers can remove
  `ndarray-npy` without implementing format parsing downstream.
