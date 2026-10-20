# Consus — Backlog

No open items. Every item this board carried has been delivered; each one's
delivery is recorded by its pull request and the `Item:` trailer on its
commits, not here. See `gap_audit.md` for risks nobody is fixing and
`checklist.md` for the historical checklist.

<a id="atlas-pr530-provider-coherence"></a>
## ATLAS-PR530-PROVIDER-COHERENCE — Current public provider contract [major] — blocked

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

- outcome: `consus-parquet`'s `compress_page_values` uses
  `consus_compression::codec::raw_deflate::RawDeflateCodec` instead of its own
  inline `flate2::read::DeflateEncoder`, so one raw-DEFLATE compressor remains.
- acceptance: the existing GZIP/ZLIB writer round-trip tests pass unchanged, and
  `git grep DeflateEncoder -- crates/consus-parquet/src` returns no production hit.
- priority: tightening
- needs: CONSUS-ZIP-OWN (consus#180)
- scope: crates/consus-parquet/src/encoding/compression.rs, crates/consus-parquet/Cargo.toml
- next step: route the `gzip` feature through `consus-compression/deflate`.

<a id="consus-python-free-threading"></a>
## CONSUS-PYTHON-FREE-THREADING — Audit the binding surface for free-threaded Python [minor] — todo

- outcome: `consus-python` drops `#[pymodule(gil_used = true)]` after an audit of
  every `#[pyclass]` for `Sync`-safe interior mutability.
- acceptance: a thread-pool test drives one object of each class on a `python3.14t`
  interpreter with `PYTHON_GIL=0`, with value-equal results and no borrow errors.
  A run with `PYTHON_GIL` unset shows `sys._is_gil_enabled()` false after import.
- priority: verification
- needs: CONSUS-ZIP-OWN (consus#180)
- scope: crates/consus-python/src
- next step: list the `#[pyclass]` types and their interior-mutability fields.
