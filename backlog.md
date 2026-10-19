# Consus — Backlog

Current work is recorded below. Earlier delivery checkpoints remain historical.

<a id="CONSUS-MOIRAI-010-PUBLIC-IDENTITY"></a>
## CONSUS-MOIRAI-010-PUBLIC-IDENTITY — Version the current async provider [major]

- Status: in progress; priority: P1; owner: current-provider integration; target: 0.4.0; last-update: 2026-10-19.
- Contract: published HDF5 0.3/beae4ce7 exposes Moirai 0.9.3 traits; current
  Git `441738cd` exposes 0.10 traits under the same version. ADR0006 records the
  uniform 0.4 boundary; implementations, features and source URLs stay fixed.
- Verification: update the existing actual published/unchanged/migrated consumer
  baseline; retain E0277 and AsyncReadAt/AsyncLength assertions. Run full current
  dependency, native, advisory, package, API and exact-head hosted/review gates.
- Security: admit patched ZIP 9, preserving compression/format tests and limits.
- Hold: no release until exact-revision gates and independent review pass; use
  existing authorized credentials only, with no configuration/access changes.

<a id="CONSUS-CHUNKED-VL-WIDTH"></a>
## CONSUS-CHUNKED-VL-WIDTH — Retain the raw chunk descriptor width [patch]

- Status: in progress; priority: P1; target: 0.4.0; last-update: 2026-10-19.
- The unchanged HDMF string-column oracle exposes discarded variable width.
  Preserve the single chunk-reading algorithm, fixed-type dispatch, unknown-width
  errors and all parser limits. Two actual native cases fail before repair;
  all three cases pass after repair. Final native/Python/package/review gates remain.

<a id="CONSUS-PYTHON-314"></a>
## CONSUS-PYTHON-314 — Current binding provider and interpreter [minor]

- Status: in progress; priority: P1; target: 0.4.0; last-update: 2026-10-19.
- PyO3 0.22 rejects the current interpreter. Upgrade runtime/build-config together
  to 0.29.3 and migrate direct calls without aliases or weakened lints. Preserve
  Python names and values; extend the existing wheel matrix to CPython 3.14 and
  execute the unchanged HDF5/HDMF/MAT/netCDF/Parquet/Zarr interoperability suite.
- Existing interpreter support, limits and all previous gates remain required.

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
