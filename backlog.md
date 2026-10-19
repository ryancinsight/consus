# Consus — Backlog

Current work is recorded below. Earlier delivery checkpoints remain historical.

<a id="CONSUS-MOIRAI-010-PUBLIC-IDENTITY"></a>
## CONSUS-MOIRAI-010-PUBLIC-IDENTITY — Version the current async provider [major]

- Status: in progress; priority: P1; owner: current-provider integration; target: 0.4.0; last-update: 2026-10-19.
- Contract: published HDF5 0.3/beae4ce7 exposes Moirai 0.9.3 traits; current
  Git `441738cd` exposes 0.10 traits under the same version. ADR0006 records the
  uniform 0.4 boundary; implementations, features and source URLs stay fixed.
- Verification: the actual published/unchanged/migrated consumer passes,
  retaining two E0277 and AsyncReadAt/AsyncLength assertions. Local native,
  Python, feature/MSRV, advisory, API, fuzz-build and normalized package gates
  pass. Exact-head hosted/review and final synchronized archive identity remain.
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
  to 0.30.1 and migrate direct calls without aliases or weakened lints. Preserve
  Python names and values; extend the existing wheel matrix to CPython 3.14 and
  execute the unchanged HDF5/HDMF/MAT/netCDF/Parquet/Zarr interoperability suite.
- Existing interpreter support, limits and all previous gates remain required.

<a id="CONSUS-RELEASE-MACHO"></a>
## CONSUS-RELEASE-MACHO — Import the optimized macOS extension [patch]

- Status: in progress; priority: P1; target: 0.4.0; last-update: 2026-10-19.
- The release wheel builds but all six format modules fail to import because
  symbol stripping leaves a misaligned LINKEDIT string pool. Preserve the
  rejected binary and offsets, retain release symbols, and execute the unchanged
  45 value-semantic cases against the optimized wheel. They pass after repair.
- Keep thin LTO, opt-level 3, one codegen unit, all limits and existing gates.
  CI tests the production artifact; final provenance/hosted/review gates remain.

<a id="atlas-pr530-provider-coherence"></a>
## ATLAS-PR530-PROVIDER-COHERENCE — Public provider contract [major] — historical publication

- Historical target: 0.3.0. All 16 package names now have an unyanked 0.3.0
  publication; the current registry baseline receipt is independently queried
  on 2026-10-19. This section records the earlier migration, not a new upload
  instruction. The current 0.4 repair is tracked above.
- ADR0005 precedes the version correction. Preserve PR172's
  owner history and numerical repair; its 0.2.1 release claim is withdrawn.
- The actual published HDF50.2.0/Moirai0.7 generic consumer passes; the same
  caller fails with two E0277 errors on current Moirai0.9; migrated bounds
  pass. A current-main API comparison cannot establish published compatibility.
- priority: correctness
- Retain all production Rust, features, Rust1.95, provider/registry identities,
  existing workflow gates and test limits. The earlier uniform workspace/internal 0.3 requirements are published;
  current final-head acceptance belongs to the 0.4 work above.
- The cancelled 0.2.1 publishing job executes no steps. Both earlier upload
  commands are obsolete. The exact-source raster 0.3 publication is independently verified; no
  further 0.3 handoff is requested. Authentication/registration stays separate.
