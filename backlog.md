# Consus — Backlog

<a id="consus-npy-header-001"></a>
## CONSUS-NPY-HEADER-001 — Parse complete typed header dictionaries [patch] — blocked

- status: blocked
- priority: correctness
- scope: crates/consus-npy/src/format.rs and format/header.rs, regression tests,
  version-cohort manifests, and authoritative documentation.
- next step: close independent review findings, then qualify the final revision
  and preserve the existing provider/registry/documentation delivery holds.

- Target: 0.4.1, retaining the existing uniform workspace release cohort.
- Reproduced: double-quoted dictionary keys are rejected, while malformed
  boolean and shape tokens can be read as valid metadata.
- Local repair: structural dictionary parsing, exact separators and integer
  tokens, value-semantic NPY/NPZ regressions and NumPy oracle comparison.
- Pending: independent exact-revision review and all applicable gates.
  Provider-refresh PR #186, registry publication and documentation deployment
  configuration remain with their existing owners. Do not merge or publish
  until those delivery holds are resolved.

Completed work is recorded by its pull request and the `Item:` trailer on its
commits. Active delivery holds remain on this board. See `gap_audit.md` for
other risks and `checklist.md` for the historical checklist.

<a id="atlas-pr530-provider-coherence"></a>
## ATLAS-PR530-PROVIDER-COHERENCE — Current public provider contract [major] — blocked

- milestone: M-1.0
- Target: 0.3.0; ADR0005 precedes the version correction. Preserve PR172's
  owner history and numerical repair; its 0.2.1 release claim is withdrawn.
- The actual published HDF50.2.0/Moirai0.7 generic consumer passes; the same
  caller fails with two E0277 errors on current Moirai0.9; migrated bounds
  pass. A current-main API comparison cannot establish published compatibility.
- priority: correctness
- Retain all production Rust, features, Rust1.95, provider/registry identities,
  existing workflow gates and test limits. Re-run the unchanged native/package/final-head
  gates for the corrected uniform workspace and internal 0.3.0 requirements.
- The cancelled 0.2.1 publishing job executes no steps. Both earlier upload
  commands are obsolete. Prepare a new exact-source raster0.3.0 handoff only
  after verification; authentication/publisher registration stay separate.

<a id="consus-parquet-raw-deflate"></a>
## CONSUS-PARQUET-RAW-DEFLATE — Compress Parquet GZIP pages through RawDeflateCodec [patch] — todo

- milestone: M-interop-contracts
- outcome: `consus-parquet`'s `compress_page_values` uses
  `consus_compression::codec::raw_deflate::RawDeflateCodec` instead of its own
  inline `flate2::read::DeflateEncoder`, so one raw-DEFLATE compressor remains.
- acceptance: the existing GZIP/ZLIB writer round-trip tests pass unchanged, and
  `git grep DeflateEncoder -- crates/consus-parquet/src` returns no production hit.
- priority: tightening
- needs: CONSUS-ZIP-OWN (consus#180)
- scope: crates/consus-parquet/src/encoding/compression.rs, crates/consus-parquet/Cargo.toml
- next step: route the `gzip` feature through `consus-compression/deflate`.
