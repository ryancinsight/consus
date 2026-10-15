# 0005 — Public async provider identity

Status: Accepted

Date: 2026-10-15

Change class: [major], pre-1.0 minor increment

Driver: ATLAS-PR530-PROVIDER-COHERENCE

## Context

Published consus-hdf5 0.2.0, source cf30dc2971f5fc194f90a296f5b77b739983a618,
depends on registry moirai-async 0.7.0. Its public AsyncHdf5File<R> requires
that provider's AsyncReadAt and AsyncLength traits. Current main exposes
moirai-async 0.9.3 instead. These traits belong to different incompatible
provider packages, even where their definitions have the same text.

The executed identical generic consumer builds and runs on the actual
published HDF5 0.2.0/Moirai 0.7 pair, fails with exactly two E0277 errors on
current Consus with its original bounds, and builds and runs after the caller
selects the current Moirai 0.9 provider. No compatibility adapter, patch, old
Git revision, or production source substitution is used.

The intermediate 0.2.1 preparation compared public APIs with current main,
which already contains the provider migration. Its 223-check pass does not
establish compatibility with published 0.2.0. The automatic 0.2.1 publisher
is cancelled before any publishing job steps execute. The patch-class claim
and its first-upload handoff are withdrawn; the numerical fixes remain valid.

## Decision

- Advance the uniform workspace and internal requirements to 0.3.0. This
  pre-1.0 minor release covers the actual public trait-identity break.
- Keep current Apollo core 0.17, Moirai 0.9.3, canonical Git source identities,
  Rust 1.95, all production Rust, features, numerical behavior and budgets.
- Callers implementing positioned-reader bounds migrate to the same Moirai
  0.9 provider identity as their Consus package. Git consumers use the canonical
  Git cohort; registry consumers use the corresponding published packages.
- Keep the public-boundary regression beside the fresh JPEG consumer. Actual
  compile outcomes supplement API scanners that do not detect this identity
  change. Run both consumer checks in the existing three-OS test job without
  changing its budgets. No scanner rule or release gate is disabled.
- Issue a replacement clean-source first-upload handoff only after verification.
  Authentication and publisher registration remain separate human actions.

## Alternatives

A 0.2.1 patch conceals a breaking public contract. Keeping both old and new
provider identities through adapters or aliases adds a compatibility surface
instead of completing the coordinated migration. Reverting the current
provider graph would abandon the authorized numerical corrections.

## Verification

Run all three public-consumer phases, real JPEG reconstruction, the unchanged
workspace/native/feature/MSRV/strict diagnostic suites, the normalized raster
package and its 76 tests, canonical locks and ADR index, and final exact-head
hosted checks. Verify the actual merged source, registry archives and Pages
result; preserve the earlier owner commits and branches. Incoming consumers
must admit Consus 0.3 before their final integration gates.
