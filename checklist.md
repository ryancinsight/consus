# Consus — Implementation Checklist

## ATLAS-CONSUS-PARSE-LIMITS-035 — Bound remaining untrusted length/depth sites

- [x] Reconcile the stale brief against the current provider default and keep
      the scope limited to the v1 HDF5 descent and FITS table count/repeat
      boundaries.
- [x] Verify the shared `ParseBudget::descend` ceiling, exact FITS header-card
      bound, and materialized-cell repeat bound with adversarial value/error
      evidence; no compatibility path or fallback is required.
- [x] Confirm the current provider default `0ed341c` passes hosted CI
      `31919441650`, Documentation `31919441619`, and Pages `31919441097`.
      This PM-only closure does not change provider source.

## ATLAS-CONSUS-PARQUET-058 — Consolidate PLAIN scalar decoders

- [x] Claim only `consus-parquet::encoding::plain` and its direct exports,
      callers, tests, and ADR/PM records; preserve peer-owned provider work.
- [x] Replace the four duplicated scalar decoders with one sealed const-width
      `PlainValue` seam and remove the old public names and re-exports.
- [x] Preserve value semantics and ParseBudget/bounds behavior with generic
      tests for INT32, INT64, FLOAT, and DOUBLE.
- [x] Pass the exact-head hosted CI before Atlas integration; local focused
      tests, strict Clippy, formatting, doctests, semver analysis, and provider
      scan are green.

  Hosted matrix `31880062463` passed at implementation head `e99a73a`; merged
  provider head `b20d419` passed CI `31880314888`, Documentation `31880314874`,
  and Pages `31880314709`. Atlas root commit `1b225ea` advances the Consus
  gitlink to that exact verified head.

## ATLAS-CONSUS-TYPES-057 — Consolidate endian scalar reads

- [x] Claim only the `consus-core` endian-reading seam and direct HDMF/NWB
      consumers; preserve peer-owned FITS/HDF5 work and unrelated residuals.
- [x] Replace the six type-named readers with one generic const-sized seam and
      migrate every direct caller without compatibility aliases.
- [x] Add value-semantic little/big-endian coverage for all supported widths,
      then pass focused tests, strict Clippy, formatting, doctests, and exact
      provider hosted CI before Atlas integration.

## ATLAS-CONSUS-UNWRAP-056 — Harden decode test diagnostics

- [x] Claim only `consus-core::decode` test diagnostics; preserve peer-owned
      FITS/HDF5 work and the separate type-suffix residual.
- [x] Replace bare unwraps with invariant-bearing `expect` messages without
      changing the decode implementation or weakening value assertions.
- [x] Pass focused core tests, strict Clippy, formatting, diff checks, and
      exact provider hosted CI before Atlas integration.

## ATLAS-CONSUS-HIERARCHY-055 — Isolate Arrow datatype descriptors

- [x] Claim the provider-owned `consus-arrow::datatype` scope; do not touch
      peer-owned FITS/HDF5/parser or generated lockfile changes.
- [x] Move descriptor families into a named vertical child module while
      preserving public exports, feature gates, and value semantics.
- [x] Pass focused Arrow tests 81/81 plus no-default 2/2, strict Clippy,
      formatting, doctests, and warning-denied Rustdoc. Hosted exact-head CI
      and Atlas integration remain the delivery gate for this provider slice.

## CONSUS-NODEF-ARROW-PARQUET-002 — Close Arrow/Parquet no-default cfg boundary

- [x] Gate alloc-only `consus-parquet` schema, bridge, conversion, wire, and
      hybrid modules and re-exports at their ownership boundaries.
- [x] Gate alloc-only `consus-arrow` facade modules and re-exports while
      retaining the no-alloc array shape descriptor.
- [x] Gate alloc-only integration tests and benchmarks with `alloc` feature
      requirements.
- [x] Pass no-default and default Nextest suites for both crates.
- [x] Pass warning-denied Clippy for both crates in both feature modes.
- [x] Record the remaining workspace no-default blockers in `gap_audit.md`.

## CONSUS-NODEF-GATE-001 — Close the next workspace no-default blocker

- [x] Correct the alloc-gated `consus-arrow::datatype` re-export and the
      no-alloc test import in `fa314cb`.
- [x] Pass package no-default check, strict Clippy, and no-default Nextest
      2/2; pass default strict Clippy and Nextest 79/79.
- [x] Re-run the locked workspace no-default and hosted exact-head gates from
      a standalone provider checkout. The standalone no-default check and
      warning-denied Clippy pass; no-default Nextest is `2031/2031`, default
      Nextest is `2553/2553`, and locked doctests pass. All 17 packages pass
      isolated locked Rustdoc. The aggregate Windows workspace Rustdoc command
      exceeds the bounded local collection window after those package-level
      passes. The provider CI and Documentation jobs now have explicit
      timeouts, with Documentation enforcing `RUSTDOCFLAGS=-Dwarnings`.
- [x] Collect the exact hosted CI and Documentation results for the pushed
      provider head. The Atlas umbrella invocation remains an environment
      boundary: its development overlay requests a `Cargo.lock` rewrite and
      reports unused local patches before compilation. No lockfile was changed.
      Exact head `65a7b28` passed CI `32017963837` (80 jobs), Documentation
      `32017963800`, and Pages `32017962556`.
- [x] Fix the exact hosted rustdoc failures from Documentation run
      `32017157627`: remove the unresolved optional-dependency link in Arrow;
      use the public `consus_core::ParseBudget` path in Zarr; and keep the
      module-private helper reference as code prose. Both affected packages
      pass local warning-denied rustdoc and formatting.
- [x] Fix the two additional exact hosted rustdoc failures from
      Documentation run `32017590806`: qualify `NwbFile` from the NWB
      validation module and qualify `ParseBudget` in MAT. Both affected
      packages pass local warning-denied rustdoc and formatting.
- [x] Fix the three additional exact hosted rustdoc failures from
      Documentation run `32017799064`: qualify both Parquet `ParseBudget`
      references and mark the boolean bit-pack index expression as code prose.
      The Parquet package passes local warning-denied rustdoc and formatting.

## ATLAS-CONSUS-001 — Themis topology partition sizing [minor]

- [x] Make the standard `consus` feature set use Themis CPU topology for the
      default parallel-I/O partition count, retaining the standard-library
      fallback.
- [x] Add the value-semantic Themis partition-count regression.
- [x] Restore no-default compilation by gating alloc-only facade modules and
      preserving compact no-alloc error construction in compression parsing.
- [x] Pass formatting, default and no-default package checks, warning-denied
      Clippy for the affected packages, Nextest 7/7, doctests, and rustdoc.
- [x] Reconcile the standalone Git-source lock and merge current `origin/main`
      in `005d0a7`; root Atlas records the exact provider head separately.

## CRATES-REL-003 — Facade package documentation [patch]

- [x] Declare the workspace README as facade package metadata.
- [x] Remove source-tree-relative paths from crate-level Rustdoc.
- [x] Verify the exact packaged source with a locked facade publish dry-run.

## CRATES-REL-002 — Moirai package identity [patch]

- [x] Bind the `moirai` Rust import to registry package `moirai-runtime`.
- [x] Regenerate the standalone lockfile and pass locked metadata plus the
      strongest independent leaf-package dry run.
- [x] Deliver the focused release-manifest correction for hosted verification.

## DOCS-001 — Documentation without mandatory Pages [patch]

- [x] Preserve unconditional workspace Rustdoc and redirect generation.
- [x] Gate Pages configuration, artifact upload, and deployment on repository
      variable `CONSUS_ENABLE_PAGES == 'true'`.
- [x] Leave Pages disabled and require hosted CI to prove a green build with a
      skipped deployment path.

## REL-001 — Python release wheels [patch]

- [x] Make Cargo the Python distribution version source of truth.
- [x] Add the pinned cross-platform wheel, GitHub Release, and PyPI workflow.
- [x] Build, install, import, and inspect a production wheel locally.
- [x] Pass workflow lint and focused warning-denied Rust checks.
- [x] Remove the CI-blocking unused compression-test import and pass its
      all-feature Nextest target under committed 30/60-second budgets.
- [x] Pass warning-denied all-target Clippy for the touched compression package.
- [x] Delete the empty HDF5 property-test artifact, remove the unused
      large-file-test local, and pass focused warning-denied Clippy plus the
      all-feature HDF5 Nextest target.
- [x] Resolve the cross-platform Arrow, IO, and Zarr test build defects and
      pass each affected all-feature package through Nextest.
- [x] Pass warning-denied all-target Clippy for IO and Zarr, replacing the
      tautological chunk-count property with an independent oracle.
- [x] Remove the process-global AWS credential race and pass both S3
      differential tests against live MinIO.
- [x] Commit the workspace dependency lock required by release `--locked`.
- [x] Route every hosted native test through pinned cargo-nextest with the
      committed timeout budget; pin the CI action and MinIO supply chain.
- [x] Hoist workspace formatting to one prerequisite and lint every package
      target without repeating the format pass across the package matrix.
- [x] Verify the live MinIO ranged read against deterministic source bytes.
- [x] Create the protected `pypi` GitHub environment restricted to release tags.
- [x] Resolve the Rust 1.97.1 test-only Clippy findings from hosted run
      `29795739435`, the remaining exact-fixture findings, and the macOS mmap
      test-path collision; pass the affected checks on the replacement PR head.
- [x] Pass exact code head `a558e79` across all 58 jobs in hosted run
      `29797846759`.
- [x] Rename the unreleased distribution and release-tag contract from
      `atlas-consus` to `consus-python` and update the protected environment.
      A locked CPython 3.13 wheel rebuilt as `consus-python` 0.1.0, installed
      into an isolated target, and imported as `consus`.
- [x] Exclude the `consus-python` `cdylib` from workspace Rustdoc output so it
      cannot collide with the documented Rust facade's `consus` artifact.
- [ ] Register the PyPI pending trusted publisher after account verification.

## REC-001 — Recover orphaned provider branches [minor]

- [x] Prove the three bounded-read commits are parallel descendants and retain
      the RITK-pinned `ec386e3` contract as the canonical successor.
- [x] Recover the unique ONNX and NPY/NPZ providers onto current `main`.
- [x] Bound hostile NPY payload reservation through `consus-io`, declare the
      complete `std + alloc` feature edge, and use MSRV-compatible `zip` 6.0.
- [x] Add both provider crates to stable, MSRV, cross-platform Clippy, and
      Nextest matrices.
- [x] Pass NPY 4/4, ONNX 3/3, and Consus I/O 251/251 Nextest suites;
      warning-denied Clippy and rustdoc; focused doctests; ONNX `alloc`-only
      compilation; locked metadata; and all 196 applicable Consus I/O semver
      checks.
