# Consus — Backlog

## ATLAS-CONSUS-SHUFFLE-038 — HDF5 shuffle filter is a silent no-op on read [major] — complete 2026-08-19

- Owner: Atlas integration. The defect was fixed in provider commit `ef439b2`.
  The synchronous and asynchronous chunk readers now carry dataset element
  size into the filter pipeline, and filter ID 2 delegates both directions to
  the canonical `consus_compression::ShuffleFilter`.
- The wrapper preserves HDF5 framing for trailing partial elements. The
  analytical oracle covers element sizes 1, 2, 4, and 8, the write path's
  on-disk byte planes, the inverse read path, and the h5py shuffle-plus-deflate
  end-to-end value case.
- ADR-0003 records the accepted boundary and alternatives. Provider commit
  `ef439b2` reports Nextest 2759/2759, strict Clippy, and formatting passing.

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


## ATLAS-CONSUS-PARSE-LIMITS-035 — Bound remaining untrusted length/depth sites [minor] — done 2026-08-16

- Owner: Atlas safety audit. Scope: `consus-hdf5/src/file/reader.rs`
  (v1 group B-tree descent) and `consus-fits/src/table/{parse,decode}.rs`
  (TFIELDS column count, TFORM repeat count). Not committed.
- Premise correction: the 11 sites named in the driving brief
  (`btree/v2.rs:685,761`, `dataset/chunk.rs:110,126,166`,
  `datatype/compound.rs:336,522,550`, `consus-fits/table/data.rs:153,187,188`)
  were already bounded by commit `03bb65e` and its follow-ups. Every line
  number in the brief resolves against `03bb65e~1`, three commits behind the
  default head; the brief was assembled from a stale tree. The claim that
  `try_reserve` appears once is also false — `consus-core/src/parse/budget.rs`
  is a dedicated budget module and `consus-io` uses it in five more places.
  The acceptance oracle's three cases are already covered by
  `consus-hdf5/tests/adversarial_input.rs`.
- Delivered: three sites the hardening pass missed are now bounded.
  `collect_btree_v1_leaves` recursed with no depth parameter and re-reads
  `header.level` from each node, so a self-referential child pointer was an
  unbounded descent reachable from `Hdf5File::open` on any v1 group — the
  exact twin of the `btree/v2.rs` fix. `TFIELDS` and the `TFORMn` repeat count
  sized reservations directly from header cards.
- Bounds and their derivation: v1 descent uses the shared
  `ParseBudget::descend` ceiling, matching the v2 tree. `TFIELDS` is bounded by
  `FitsHeader::len()` — each column requires its own `TFORMn` card, so a header
  of N cards cannot describe more than N columns; this is exact, not a
  ceiling. The `TFORMn` repeat count is clamped to `cell.len() / elem_size`,
  the already-materialized cell being the exact bound.
- Evidence: each bound falsified by removing it and observing the failure —
  v1 descent `has overflowed its stack (test aborted)`; `TFIELDS`
  `capacity overflow` panic. Restored, then `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`, `cargo nextest run`
  (2539/2539), and `cargo test --doc` all pass.
- Delivery evidence: current provider default `0ed341c` passes hosted CI
  `31919441650`, Documentation `31919441619`, and Pages
  `31919441097`. The implementation and its adversarial coverage are already
  merged; this closure synchronizes the PM state without changing source.


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

## ATLAS-CONSUS-PARSE-LIMITS-037 — Bound consus-mat, consus-parquet, consus-zarr parse paths [minor] — done 2026-08-15

- Scope: three crates outside the -035/-036 HDF5 and FITS surface.
- Delivered (this pass):
  - `consus-mat::v5::matrix::parse_matrix` now descends through a
    `ParseBudget` ceiling for mxCELL/mxSTRUCT nesting (was unbounded, ~40-50
    input bytes per level) with deep-chain + shallow-chain tests.
  - `consus-mat::v5::decompress_zlib` uses `ParseBudget::read_bounded`,
    capping a zlib decompression bomb at the byte ceiling (was
    `read_to_end`, unbounded).
  - `consus-parquet::wire::thrift::ThriftReader::skip` carries the descent
    ceiling for nested struct/list/map skips (was one frame per input byte)
    with an adversarial depth test.
  - `consus-parquet::dataset::schema::parse_fields` carries the descent
    ceiling for single-child group chains (was bounded only by element
    count) with deep-chain + shallow-chain tests.
  - Verified: `consus-mat` all-feature Nextest 90/90 (incl. compressed-read
    integration); `consus-parquet` lib tests pass; strict Clippy clean.
- Delivered (follow-up pass, 2026-08-15):
  - `consus-parquet::wire::metadata` schema / row-group / column-chunk list
    reservations now use `ParseBudget::vec_with_capacity`, bounding
    `count × size_of<T>` (a small footer can no longer reserve 100+ MiB).
  - `consus-zarr::chunk::ops`: `try_expand_fill_value` bounds
    `num_elements × element_size` against the byte ceiling with a fallible
    allocation; `checked_chunk_bytes` bounds the chunk / padded-chunk /
    shard-chunk byte products at every validation and allocation site; a
    hostile-shape test proves the typed rejection.
  - Verified: `consus-parquet` lib Nextest 227/227, `consus-zarr` lib
    Nextest 110/110 (incl. the hostile fill-value test), strict Clippy,
    formatting. -037 is closed.
- Acceptance: as -036, per crate.
- Re-verification 2026-08-18 (third brief against the same stale tree): a
  driving brief again named ~12 "missed" sites across -036/-037. Every one
  resolves as already bounded at the current head, and the brief's own
  "obvious fuzz gaps" (`consus-mat`, `consus-parquet`) are two of the four
  targets that already exist. Site-by-site: `heap/global.rs` already uses
  `checked_sub` with a typed `InvalidFormat` naming both operands (the
  claimed "unchecked `- header_size`" is absent); `file/mod.rs:204,401,679`
  are not allocation sites at all (a call, a `)?`, an error return) and
  `:696` is `checked_add` + `budget.zeroed`; all five `heap/fractal.rs`
  lines route through `read_bounded_bytes` → `budget.zeroed` with every
  product `checked_mul`/`checked_add`; `consus-mat::parse_matrix` is
  `parse_matrix_depth` under `ParseBudget::descend` and `decompress_zlib`
  uses `read_bounded`; `ThriftReader::skip` delegates to `skip_depth` under
  the same ceiling; `try_expand_fill_value` bounds the shape product via
  `checked_bytes`. No source change was warranted.
- Residuals recorded by that pass (both outside -036/-037 scope, neither a
  parse-path defect): `consus-zarr::chunk::ops::expand_fill_value` is a
  `pub` wrapper that `.expect()`s on an input-dependent value — a panic
  policy violation and a prohibited sibling of `try_expand_fill_value`,
  though no production caller reaches it (bench and unit tests only); and
  the `write_array` paths at `ops.rs:853,1146,1186` compute
  `total_elements * element_size` unchecked, which wraps in release. Both
  need their own items.
- Coverage added by that pass: `fuzz_surface_typecheck.rs` for
  `consus-hdf5`, `consus-mat`, and `consus-parquet`, mirroring the
  `consus-fits` pattern so a signature change in a fuzzed entry point
  breaks the ordinary gate instead of the weekly fuzz job.

## ATLAS-CONSUS-PARQUET-058 — Consolidate PLAIN scalar decoders [major][arch] — done 2026-08-15

- Owner: Atlas provider integration. Scope: `consus-parquet::encoding::plain`
  and its in-repository callers/export modules only; peer-owned FITS/HDF5 work,
  generated lock state, and unrelated type-suffix findings are excluded.
- Acceptance: replace the four duplicated public `decode_plain_*` scalar
  decoders for INT32, INT64, FLOAT, and DOUBLE with one sealed, const-width
  `PlainValue`/`decode_plain<T>` seam; migrate every caller and test without
  compatibility aliases; preserve truncation, overflow, allocation-budget,
  bit-pattern, and value semantics.
- Verification: generic conformance coverage for all four scalar types,
  focused Parquet Nextest, strict Clippy, formatting, doctests, package
  semver analysis, and the exact provider hosted matrix.
- Dependency: ADR 0002 records the breaking public replacement and its
  migration path. The provider scan must reduce `type_suffixed_fns` from 85
  without increasing any other debt class.
- Local evidence: `type_suffixed_fns=81`, with `oversized_files=83`,
  `unwrap_production=383`, `allow_sites=16`, and `orphan_modules=0`; Parquet
  Nextest 249/249, strict Clippy, all-feature and no-default checks, doctests,
  and warning-denied Rustdoc pass. `cargo semver-checks` classifies the four
  removed entry points as a required major release.
- Delivery evidence: implementation commit `e99a73a` merged as provider
  `b20d419`; the required hosted matrix passed at `31880062463`, and exact
  post-merge provider CI, Documentation, and Pages passed at `31880314888`,
  `31880314874`, and `31880314709`. Atlas integrated the exact provider head
  in root commit `1b225ea`.

## ATLAS-CONSUS-TYPES-057 — Consolidate endian scalar reads [arch] — done 2026-08-15

- Owner: Atlas provider integration. Scope: the `consus-core` byte-decoding
  seam and its direct `consus-hdmf`/`consus-nwb` consumers only; peer-owned FITS/HDF5,
  generated lock state, and unrelated type-suffix findings are excluded.
- Acceptance: replace the six duplicated type-named endian readers with one
  generic, const-sized, zero-cost scalar-reading seam; migrate all direct
  callers without compatibility aliases; preserve byte-order and value
  semantics; reduce the provider `type_suffixed_fns` count by the six removed
  reader definitions.
- Verification: generic reader conformance tests for every supported signed
  and unsigned width, focused Consus-core, Consus-HDMF, and Consus-NWB Nextest,
  strict
  Clippy, formatting, doctests, and the exact provider hosted matrix.
- Evidence to date: the six type-named readers now have one
  `EndianScalar`/`read_integer<T>` implementation in `consus-core`; direct
  HDMF and NWB callers are migrated with no aliases. The provider scan reports
  `type_suffixed_fns=85` (91 before this slice). Local focused gates pass:
  313/313 core+HDMF Nextest tests plus 278/278 NWB tests, strict Clippy,
  all affected crate checks, and three doctests.

## ATLAS-CONSUS-UNWRAP-056 — Harden decode test diagnostics [patch] — done 2026-08-15

- Owner: Atlas provider integration. Scope: `consus-core::decode` test
  diagnostics only; peer-owned FITS/HDF5 work and type-suffix cleanup are
  excluded.
- Acceptance: remove all bare test unwraps from the decode module using
  invariant-bearing `expect` messages; preserve value-semantic coverage and
  return the provider conformance `unwrap_production` count to its committed
  baseline without weakening assertions.
- Verification: focused core Nextest, no-default check, strict Clippy,
  formatting, diff check, and the exact provider hosted matrix.
- Evidence: local `cargo fmt --check`, all-target/all-feature check, strict
  Clippy, all-feature Consus-core Nextest, and the no-default-feature check
  passed; the provider scan reports `unwrap_production=383` and no bare
  unwraps remain in `decode.rs`.

## ATLAS-CONSUS-HIERARCHY-055 — Isolate Arrow datatype descriptors [patch] — done 2026-08-15

- Owner: Atlas provider integration. Scope: `consus-arrow::datatype` only;
  peer-owned FITS/HDF5 changes, generated lock state, and unrelated conformance
  classes are excluded.
- Acceptance: move the descriptor families into a named vertical child module
  without changing the public `consus-arrow` exports; the parent module falls
  below the 500-line hierarchy target; value-semantic Arrow tests, strict
  Clippy, formatting, and the provider hosted gates pass at the exact head.
- Method: preserve the public re-export closure, compile-time feature
  boundaries, and conversion behavior; no adapter or duplicate API is allowed.
- Delivered: moved temporal, scalar metadata, and alloc-backed nested
  descriptors into `datatype/descriptors.rs`; `datatype/mod.rs` is 330 lines
  and public `consus-arrow` exports remain unchanged. The provider scan drops
  `oversized_files` from 84 to 83; the separate stale unwrap/type-suffix
  residuals remain tracked for a later scope.
- Local evidence: all-feature Arrow Nextest 81/81, no-default Nextest 2/2,
  strict Clippy/checks, doctests, warning-denied Rustdoc, formatting, and diff
  checks pass.

## ATLAS-CONSUS-RESOURCE-BOUNDARY-097 — Bounded external-input expansion [major] — done 2026-08-14

- Owner: current Atlas safety audit; scope is the Consus core parse budget,
  native compression codecs, Parquet page decompression, and the Moirai S3
  client. The generated `Cargo.lock` remains peer-owned and is excluded from
  this increment.
- Finding: S3 range construction underflowed for zero-length reads and could
  overflow at the end offset; listing responses and accumulated object keys
  had no aggregate resource ceiling; deflate, LZ4, Snappy, and Parquet page
  expansion could allocate from attacker-controlled output sizes.
- Delivered in the working increment: `ParseBudget::read_bounded` provides one
  fallible, byte-capped stream reader; codec paths validate encoded output
  sizes before allocation; S3 ranges are checked and listings cap response,
  key-count, and key-byte growth; value-semantic regressions cover the range
  arithmetic and shared output-budget contract.
- Evidence: exact default head `c3afb406993f6b92e11963100621438064928383`; hosted
  CI run `31851240758` passed every repository-owned check, including the
  Ubuntu, macOS, and Windows package matrices, MSRV jobs, MinIO S3 tests,
  fuzz-target build, and feature checks. Documentation run `31851240739`
  passed its build job.
- Acceptance: focused `cargo nextest` passes for core, compression, Parquet,
  and IO with deflate/LZ4/Snappy/Zstd/GZIP coverage; locked metadata, strict
  Clippy, doctests, and provider hosted gates pass at the final head. No
  decompressor can reserve or grow beyond the shared parse budget.
## CONSUS-TEST-API-001 — Migrate cross-format tests to provider-owned APIs [patch] — done 2026-08-13

- Owner: Atlas integration. Scope: `tests/cross_format_interop.rs` only;
  no provider facade or compatibility layer was added.
- Delivered: HDF5 tests now use `Hdf5FileBuilder`, `list_root_group`,
  `dataset_at`, and the contiguous/chunked read APIs; Zarr tests use the
  canonical `ArrayMetadata` plus `write_chunk`/`read_chunk`; NetCDF tests use
  `NetcdfWriter` and `read_model` over deterministic in-memory HDF5 images.
  The absent-file skip paths and stale Arrow/Parquet call signatures are gone.
- Acceptance: all nine cross-format scenarios execute with value-semantic
  assertions; no test depends on an aspirational `ZarrArray`, `ArrayMetadataV3`,
  `NcFile`, `build_writer`, or `MemCursor::new(buffer)` surface.
- Evidence: focused all-format Nextest 8/8; all-format plus compression 9/9;
  integration-test package all-features Nextest 42/42; warning-denied Clippy
  for the touched target and workspace all-targets/all-features; workspace
  no-default locked check; warning-denied workspace Rustdoc; formatting and
  diff checks. PR #24 source head `a5b9cfdde4c789c237652e0d62c42ce8372005f5`
  merged at `33c2df06b0209f21755462fe44bec85e6a979253`; hosted run
  `31683877253` passed all 68 repository-owned jobs at the exact source head.

## CONSUS-NODEF-FITS-003 — Close FITS no-default cfg boundary [patch] — done 2026-08-13

- Owner: Atlas integration. Scope: `consus-fits` module, re-export, test, and
  format-mapping feature boundaries plus the shared `consus-core` error
  constructor required by workspace feature unification.
- Acceptance: `consus-fits --no-default-features` compiles and tests its
  retained value-semantic descriptors without warnings; default and all-feature
  FITS behavior remains green; alloc-only parsing, HDU, image, table, file, and
  validation APIs are not exposed or compiled in the no-alloc build.
- Current evidence: the no-default package check, strict Clippy, and Nextest
  16/16 pass after gating. Default package check, strict Clippy, Nextest
  170/170, all-features check/Clippy, doctests, and warning-denied Rustdoc pass;
  the workspace no-default check now passes after the shared Error constructor
  and NWB re-export closure, and the workspace Clippy run exposed and fixed a
  Copy-value clone in `tests/property_integration.rs`, stale NWB Option
  assertions, and an approximate-PI fixture literal.
- Completion: the provider workspace no-default check and strict Clippy pass;
  `consus-nwb` and its HDF5 dependency are closed in the companion items
  below. Remaining Consus work is limited to separately tracked test/API audit.

## CONSUS-NODEF-NWB-004 — Close NWB no-default cfg boundary [patch] — done 2026-08-13

- Owner: Atlas integration. Scope: `consus-nwb` module, re-export, version,
  namespace, and alloc-only integration-test boundaries.
- Acceptance: no-default builds retain the conventions/version surface without
  compiling alloc-backed NWB APIs; default behavior remains value-semantically
  tested; the workspace no-default feature graph remains warning-clean.
- Delivered: alloc-only NWB modules, re-exports, and integration tests are
  gated; no-default version matching is exhaustive; the shared workspace
  `Error::invalid_format` constructor prevents dependency-feature shape from
  leaking into FITS consumers.
- Evidence: workspace no-default check and strict Clippy pass; default
  `consus-nwb` Nextest 278/278 passes; no-default `consus-nwb` has no compiled
  tests by contract and passes the explicit no-test gate.

## CONSUS-NODEF-HDF5-005 — Close HDF5 no-default cfg boundary [patch] — done 2026-08-13

- Owner: Atlas integration. Scope: `consus-hdf5` alloc-backed module,
  re-export, test, benchmark, B-tree, and superblock error boundaries.
- Acceptance: no-default HDF5 retains only its allocation-free structural
  modules and compiles warning-cleanly; default HDF5 behavior remains covered
  by its full value-semantic suite.
- Delivered: alloc-backed HDF5 modules and test/bench targets are gated by the
  `alloc` feature; no-default superblock errors use `consus-core`'s shared
  constructor; default B-tree exports remain complete.
- Evidence: no-default check, strict Clippy, and explicit no-test Nextest pass;
  default strict Clippy passes and default HDF5 Nextest passes 405/405.

## CONSUS-NODEF-ARROW-PARQUET-002 — Close Arrow/Parquet no-default cfg boundary [patch] — done 2026-08-13

- Owner: Atlas foundation audit; scope: `consus-arrow` and `consus-parquet`
  no-default module, test, and benchmark surfaces.
- Delivered: alloc-only Parquet schema, bridge, conversion, wire, hybrid, and
  Arrow facade modules are feature-gated at their module boundaries; the
  no-alloc Arrow array descriptor retains a value-semantic shape API; alloc-only
  integration tests and benchmarks declare their `alloc` requirement.
- Acceptance: both crates compile with `--no-default-features`; their
  no-default tests execute real retained surfaces; default behavior remains
  covered by the complete package suites; warning-denied Clippy is clean in
  both feature modes.
- Evidence: `consus-parquet` no-default Nextest 10/10 and default 215/215;
  `consus-arrow` no-default Nextest 2/2 and default 79/79; strict Clippy in
  both modes; focused rustfmt check; workspace no-default sweep reaches the
  next pre-existing `consus-fits` cfg boundary.
- Residual: `CONSUS-NODEF-GATE-001` remains open for `consus-fits`,
  `consus-nwb`, and downstream workspace feature edges; `CONSUS-TEST-API-001`
  remains separate.

## ATLAS-CONSUS-GATE-FIX-001 — Atlas gate fixes and audit record [patch] — done 2026-08-12

- Owner: foundation audit (ATLAS-FOUNDATION-PLANNING-002); scope: canonical
  Atlas engineering gates against this checkout.
- Delivered: `consus-arrow` `--no-default-features` cfg-gating fixes (re-export
  groups, bridge/schema/conversion alloc gating), Clippy lint fixes in
  `consus-nwb` (`report.rs`) and `consus-hdmf` (`tests/integration.rs`), and
  `consus-hdf5` root re-exports for the `Hdf5File`/`Hdf5FileBuilder` facades.
- Verified: default + all-features checks and doctests pass. The initial
  `consus-arrow` no-default re-export closure was completed by the follow-on
  `CONSUS-NODEF-ARROW-PARQUET-002` item; the wider workspace cfg debt remains
  under `CONSUS-NODEF-GATE-001`.
- Remaining (tracked, not part of this patch): `--no-default-features`
  workspace cfg debt (CONSUS-NODEF-GATE-001) and the integration-test
  aspirational I/O API (CONSUS-TEST-API-001); both are inventoried in
  `gap_audit.md`.

## CONSUS-NODEF-GATE-001 — Close the next workspace no-default blocker [patch] — done 2026-08-17

- Owner: Codex. Scope: the next reproducible workspace
  `--no-default-features` blocker and its provider-local gate evidence; no
  changes to the peer-owned HDF5 worktree scope.
- Claim: 2026-08-17. Re-verify the standalone locked workspace gate outside
  the Atlas development overlay and synchronize the provider records.
- Entry evidence: `consus-arrow` unconditionally re-exports the alloc-gated
  `datatype` module from `src/lib.rs:74`, so the workspace no-default check
  fails before it can reach the next format provider.
- Delivered in `fa314cb`: the `datatype` re-export and no-alloc test imports
  now follow the `alloc` boundary. Package no-default check, strict Clippy,
  no-default Nextest 2/2, default strict Clippy, and default Nextest 79/79
  pass; the current worktree workspace no-default check also passes.
- Delivered locally: the standalone provider checkout passes locked workspace
  check and warning-denied Clippy with and without default features. Nextest
  passes `2031/2031` without default features and `2553/2553` with defaults;
  locked workspace doctests pass. Every one of the 17 workspace packages also
  passes isolated locked Rustdoc. The aggregate Windows workspace Rustdoc
  command timed out twice after those package-level passes (at 244 seconds on
  the bounded rerun and beyond the earlier 300-second collection window), so
  it is retained as an orchestration residual rather than misreported as a
  package documentation failure.
- Cleanup delivered in this slice: the provider CI and documentation jobs now
  carry explicit job timeouts, and the documentation job enforces
  warning-denied Rustdoc.
- Hosted Documentation run `32017157627` at `57a4e66` correctly failed on
  warning-denied rustdoc: an optional `zerocopy` link in Arrow and three Zarr
  links (a module-private helper and an unqualified `ParseBudget`) were
  unresolved. Those links are corrected and the affected packages pass local
  `RUSTDOCFLAGS=-Dwarnings` rustdoc; the replacement hosted result is pending.
- Replacement Documentation run `32017590806` at `22294b5` found two further
  unresolved links in NWB and MAT (`NwbFile::validate_conformance` was not
  qualified from the validation module, and `ParseBudget` was unqualified).
  Both are corrected and the affected packages pass local warning-denied
  rustdoc; the next hosted result is pending.
- The next hosted Documentation run `32017799064` at `0b5505a` found three
  more Parquet documentation links: two unqualified `ParseBudget` references
  and a prose index variable parsed as an intra-doc link. They are corrected;
  local warning-denied Parquet rustdoc and formatting pass.
- Final exact-head evidence: CI `32017963837` passed all 80 jobs at
  `65a7b28`; Documentation `32017963800` and Pages deployment `32017962556`
  also passed at that head. The Atlas umbrella overlay still requests a
  `Cargo.lock` rewrite and reports unused local patches before compilation,
  but that is a separate development-environment boundary and not a provider
  gate residual.
- Acceptance: the no-default workspace check reaches the next blocker or
  passes; the affected package's no-default and default value-semantic gates
  are green; the residual record names the exact remaining blocker.

## ATLAS-CONSUS-001 — Themis topology partition sizing [minor] — done

- Owner: Codex; scope: the `consus` facade's default parallel-I/O policy and
  the no-alloc feature boundary in the compression/facade crates.
- Acceptance: the standard feature set derives its default partition count
  from Themis CPU topology; `no_default_features` remains compilable without
  Themis or alloc; malformed low-level inputs preserve the compact no-alloc
  error variant; focused tests, Clippy, doctests, and rustdoc pass.
- Evidence: `cargo fmt --all -- --check`; default `consus` check, Clippy,
  Nextest 7/7, doctests, and rustdoc pass; no-default `consus` check and
  no-default `consus-compression` Clippy pass.
- Delivery: merged as `005d0a7` and present on the current default `3610b45`.
  Hosted CI `31645404672`, Documentation `31645404702`, and Pages deployment
  `31645405182` all passed at the exact default head. The Atlas gitlink is
  advanced separately by the root integration item.

## CRATES-REL-003 — Facade package documentation [patch] — done

- Owner: Codex `/root`; scope: the `consus` facade's packaged README and
  crate-level Rustdoc.
- Root cause: the source-tree-relative Rustdoc include escaped the packaged
  crate archive, so `cargo publish --dry-run --package consus` could not compile
  the verified tarball.
- Acceptance: Cargo packages the workspace README inside the facade archive,
  while self-contained crate Rustdoc compiles from both the workspace and the
  flattened package archive.
- Evidence: source-tree doctests and a locked facade publish dry-run against
  the standalone release tree.

## CRATES-REL-002 — Moirai package identity [patch] — done

- Owner: Codex `/root`; scope: the root and `consus-io` Moirai dependency
  identities, standalone lockfile, and crates.io package verification.
- Acceptance: clean metadata resolves the `moirai` Rust import through package
  `moirai-runtime` 0.4.0, and `consus-core` remains packageable without a local
  overlay.
- Evidence: standalone locked metadata, workspace formatting, and the
  `consus-core` package dry run pass.

## DOCS-001 — Decouple documentation checks from Pages [patch] — done

- Owner: Codex `/root/architecture_audit`; scope: documentation workflow and
  matching PM records only. Pages settings and deployment enablement are
  explicit non-goals.
- Root cause: documentation run `29941230671` built Rustdoc and its redirect,
  then failed in `actions/configure-pages@v5` because this repository does not
  have Pages enabled.
- Acceptance: every documentation run builds Rustdoc and the redirect. Pages
  configuration, artifact upload, and deployment run only when repository
  variable `CONSUS_ENABLE_PAGES` equals `true`; with the variable absent or
  false, the build passes and the deployment path is skipped.

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

## Phase 1: HDF5 MVP (Read + Write)

### P1.1 — HDF5 Read Path
- [x] Object header v1 parser
- [x] Object header v2 parser
- [x] Datatype message parser (all 11 classes)
- [x] Dataspace message parser
- [x] Data layout message parser (v3 and v4 variants currently implemented)
- [x] Filter pipeline message parser
- [x] Symbol table message parser (v1 groups)
- [x] Link message parser (v2 groups)
- [x] B-tree v1 traversal (group navigation)
- [x] B-tree v2 traversal (fractal heap integration)
- [x] Local heap reader
- [x] Global heap reader
- [x] Contiguous dataset read
- [x] Chunked dataset read (single chunk)
- [x] Chunked dataset read (multi-chunk with filter pipeline)
- [x] Hyperslab selection read
- [x] Point selection read
- [x] Compound datatype read
- [x] Variable-length datatype read
- [x] Attribute read
- [x] Dense group link enumeration
- [x] Dense attribute enumeration
- [x] Soft link resolution
- [x] Superblock v0/v1/v2/v3 parsing
- [x] File open with validation
- [x] Chunk index v4 B-tree v2 lookup
- [x] External link traversal beyond typed error reporting
- [x] Reference-file coverage against canonical HDF Group fixtures

### P1.2 — HDF5 Write Path
- [x] Superblock v2 writer
- [x] Object header v2 writer
- [x] Datatype message writer
- [x] Dataspace message writer
- [x] Data layout message writer (contiguous)
- [x] Data layout message writer (chunked with materialized v1 chunk index in current scope)
- [x] Contiguous dataset write
- [x] Group creation at root via link messages
- [x] Attribute write
- [x] File creation (new file from scratch)
- [x] File close with flush and checksum
- [x] Filter pipeline message writer
- [x] Chunk index writer for chunked datasets (v1 raw-data chunk B-tree leaf in current scope)
- [x] Chunked dataset write with persisted chunk index and end-to-end value roundtrip in current scope
- [x] V4 layout message emission with B-tree v2 chunk index (layout_version=4)
- [x] Chunked dataset compression roundtrip coverage (deflate, fletcher32)
- [x] Local heap writer for v1 group emission — `write_local_heap` now emits a valid HEAP header + null-terminated name pool; `write_v1_group_header` emits SNOD + B-tree v1 group index; `add_v1_group_with_children` exposes root-linked v1 group emission; covered by writer roundtrip tests and `list_group_v1` verification
- [x] BUG-HDF5-001 through BUG-HDF5-005 resolved (Milestone 53): `write_v1_group_index` B-tree v1 layout; local heap header size constant; v1 object header `V1_HEADER_PADDING = 4` + async reader correction; compound member `dim_overhead = 28`; variable-length string embedded base type consumption

### P1.3 — HDF5 Verification
- [x] In-memory round-trip tests for contiguous datasets
- [x] In-memory round-trip tests for chunked datasets (v3 layout, single-leaf v1 chunk B-tree scope)
- [x] In-memory round-trip tests for attributes
- [x] Reference-style tests against repository sample files
- [x] Download and validate canonical HDF Group reference files (covered by test_latest.h5 and custom generated fixtures)
- [x] Read tests against `t_vlen.h5`
- [x] Read tests against `t_filter.h5`
- [x] Read tests against `t_compound.h5`
- [x] Read tests against `t_vlen.h5`
- [x] Read tests against `t_string.h5` equivalent — `data/hdf5_string_ref_sample.h5` h5py fixture; 6 value-semantic integration tests in `tests/integration_hdf5_string_ref.rs` (Milestone 53)
- [x] Read tests against `t_group.h5` equivalent — `data/hdf5_group_ref_sample.h5` h5py fixture; 7 value-semantic integration tests in `tests/integration_hdf5_group_ref.rs` (Milestone 53)
- [x] Read tests against `t_chunk.h5`
- [x] Read tests against `t_filter.h5`
- [x] V4 B-tree v2 chunk index roundtrip tests (2D, 3D, single-chunk)
- [x] Compressed chunked dataset roundtrip tests (deflate, fletcher32, deflate+v4)
- [x] Comparison with `h5dump` output for verified fixtures (implemented in `tests/hdump_verify.rs`)

### P1.4 — Performance & Memory
- [x] Fill-value-aware undefined chunk reads
- [x] Parallel chunk I/O via Rayon (serial + parallel paths both verified)
- [x] Criterion benchmarks: contiguous read throughput (1 MB dataset)
- [x] Criterion benchmarks: chunked read throughput (v3 + v4 B-tree v2)
- [x] Criterion benchmarks: compressed read (deflate); zstd/lz4 deferred to codec feature expansion
- [x] Criterion benchmarks: zstd and lz4 compressed read (blocked on HDF5 test-time feature enablement)
- [x] Allocation reduction in object-header and writer message assembly
- [x] Comparison with HDF5 C library via `hdf5-rs`
- [x] Comparison with Python `h5py` (covered by `gen_hdf5_string_ref.py`, `gen_hdf5_group_ref.py` and integration tests)

## Phase 2: Zarr + netCDF-4

### P2.1 — Zarr v2
- [x] `.zarray` JSON metadata parser
- [x] `.zattrs` JSON metadata parser
- [x] `.zgroup` JSON metadata parser
- [x] Directory store implementation
- [x] Chunk read (single chunk)
- [x] Chunk read (multi-chunk)
- [x] Compression pipeline (Zarr codec chain)
- [x] Full array read with selection
- [x] Partial selection read semantics across chunk boundaries
- [x] Partial selection write semantics across chunk boundaries
- [x] Zarr v2 write path
- [x] Chunk-grid bounds validation for `read_chunk` and `write_chunk`
- [x] Repository fixtures generated from Python zarr for v2 and v3 arrays
- [x] Integration tests against Python zarr-produced fixtures
- [x] Python v2 chunk-key interoperability against Python-generated filesystem stores
- [x] Python v2 gzip full-array interoperability against Python-generated filesystem stores
- [x] Python v3 default codec-chain interoperability against Python-generated filesystem stores
- [x] Full read/write interoperability against Python zarr library output

### P2.2 — Zarr v3
- [x] `zarr.json` metadata parser
- [x] Sharding codec
- [x] v3 chunk key encoding
- [x] v3 codec pipeline
- [x] v3 write path
- [x] Interop tests with zarr-python v3

### P2.3 — netCDF-4
- [x] Dimension scale detection via HDF5 attributes
- [x] netCDF-4 classic model read — root-group HDF5 read entry point now extracts a validated canonical `NetcdfModel` from `/`; covered by empty-file, flat root, and nested-group integration tests; `read_nested_group_into_model` test corrected (CLASS=DIMENSION_SCALE on "x" required; 13/13 integration tests verified)
- [x] Variable → HDF5 dataset mapping
- [x] CF conventions attribute parsing
- [x] Unlimited dimension handling
- [x] Full variable byte extraction for contiguous and chunked HDF5-backed netCDF variables
- [x] DIMENSION_LIST-based variable-to-dimension binding for HDF5-backed netCDF extraction
- [x] Nested-group dimension inheritance and nearest-scope shadowing validation
- [x] netCDF-4 classic model read
- [x] netCDF-4 enhanced model read (user-defined types) — `NetcdfUserType { name, datatype }` added to `NetcdfGroup.user_types`; `extract_group` populates from `NodeType::NamedDatatype` children via `Hdf5File::named_datatype_at`; `Hdf5FileBuilder::add_named_datatype` added; 2 integration tests + 1 HDF5 unit test (Milestone 45 — this sprint)
- [x] netCDF-4 write path — `NetcdfWriter::write_model` emits flat classic netCDF-4 HDF5 files: `_nc_properties` root attribute, dimension scales with `CLASS`/`NAME`/`_Netcdf4Dimid`, variables with `DIMENSION_LIST` object-reference bindings, string-valued CF attribute propagation; `Reference(Object/Region)` encoding added to `consus-hdf5::file::writer::encode_datatype`; 7 round-trip integration tests + 4 unit tests + 1 doctest + 2 HDF5 datatype encoding tests (Milestone 42 — this sprint)
- [x] netCDF-4 enhanced model write path — sub-group hierarchy write with DIMENSION_LIST bindings per group scope, numeric CF attribute propagation (Int/Uint/Float/IntArray/UintArray/FloatArray/StringArray), recursive child group nesting; `SubGroupBuilder<'a>` HDF5 builder API; `DatasetTarget` generic zero-cost trait; 7 integration tests; `write_enhanced_model_sub_group_roundtrip` and `write_nested_two_level_sub_group_roundtrip` verify full hierarchy roundtrip (Milestone 43 — this sprint)
- [x] Comparison with Unidata netCDF-C reference files

## Phase 1.5 — Workspace Test Integrity
- [x] Restore compile-valid property integration suite against current stable APIs
- [x] Re-enable value-semantic property coverage for shape, selection, byte-order, datatype sizing, in-memory I/O, compression, Arrow schema conversion, and Parquet schema conversion
- [x] Align integration-test manifest with `consus-io` alloc-gated `MemCursor` support
- [x] Verified `cargo nextest run -p consus-hdf5 --test roundtrip_hdf5 --no-fail-fast`
- [x] Verified `cargo test -p consus-hdf5 --test reference_hdf_group`
- [x] Verified `cargo nextest run -p consus-hdf5 --test roundtrip_hdf5 --no-fail-fast`
- [x] Verified `cargo test -p consus-hdf5 --test reference_hdf_group`

## Phase 2.5: Datatype Mapping Completion

### P2.5a — Arrow ↔ Core Nested-Type Conversion
- [x] `core_datatype_to_arrow_hint`: Compound → Struct with recursive fields
- [x] `core_datatype_to_arrow_hint`: Array → List with correct element type
- [x] `core_datatype_to_arrow_hint`: Complex → Struct with real/imaginary children
- [x] `arrow_datatype_to_core`: Struct → Compound with recursive field conversion
- [x] `arrow_datatype_to_core`: Map → Compound with key/value fields
- [x] `arrow_datatype_to_core`: Union → Compound with variant fields
- [x] Roundtrip tests: Compound → Struct → Compound preserves field names

### P2.5b — FITS Binary Table TFORM → Core Datatype
- [x] `BinaryFormatCode` enum for all 13 FITS Standard 4.0 format codes
- [x] `parse_binary_format` TFORM string parser (repeat + code)
- [x] `binary_format_to_datatype` per-code canonical mapping
- [x] `tform_to_datatype` high-level TFORM → Datatype conversion
- [x] `binary_format_element_size` byte-width lookup
- [x] Array wrapping for repeat > 1 scalar types
- [x] Crate-root re-exports for `BinaryFormatCode` and `tform_to_datatype`
- [x] Comprehensive value-semantic tests for all 13 format codes

### P2.5c — HDF5 Datatype Class Mapping
- [x] `map_string` (fixed/variable, ASCII/UTF-8)
- [x] `map_bitfield` (→ Opaque with HDF5_bitfield tag)
- [x] `map_opaque` (with optional tag)
- [x] `map_compound` (ordered fields + size)
- [x] `map_reference` (Object/Region, size-based default)
- [x] `map_enum` (structural envelope with empty members)
- [x] `map_variable_length` (→ VarLen with base type)
- [x] `map_array` (→ Array with base + dims)
- [x] `charset_from_flags` helper (ASCII/UTF-8/unknown→ASCII)
- [x] Coverage table in module documentation
- [x] Value-semantic tests for all mapping functions

### P2.5d — FITS Column Descriptor Datatype Integration
- [x] `FitsTableColumn` extended with `datatype: Datatype` and `byte_width: usize` fields
- [x] `FitsTableColumn::new()` updated to accept `datatype` and `byte_width` parameters
- [x] `FitsTableColumn::from_binary_tform()` constructor derives `datatype` and `byte_width` from TFORM
- [x] `datatype()` and `byte_width()` accessors on `FitsTableColumn`
- [x] `parse_column` dispatches binary→`from_binary_tform`, ASCII→`FixedString` with `parse_ascii_column_width`
- [x] `FitsBinaryTableDescriptor::from_header()` validates column byte widths sum to `NAXIS1`
- [x] 5 new value-semantic tests (Boolean/Int32/Float64, Array, NAXIS1 mismatch, ASCII FixedString, Complex/Compound)
- [x] Verified `cargo test -p consus-fits --lib` (128/128)

## Phase 3: Parquet + Polish

### P3.1 — Parquet Interop
- [x] Consus ↔ Parquet schema mapping
- [x] Read Parquet files as Consus datasets
  - [x] Canonical in-memory Parquet dataset descriptor model
  - [x] Top-level column descriptors with canonical `Datatype`, storage classification, and `[total_rows]` shape
  - [x] Row-group and column-chunk descriptor validation (`row_count > 0`, schema-order chunk alignment, exact chunk cardinality)
  - [x] Ordered projection API preserving source schema order for selected top-level columns
  - [x] Value-semantic tests for fixed-width, variable-width, nested-group, projection, and invalid row-group layouts
  - [x] Nested group canonicalization to `Datatype::Compound` with ordered child fields and analytically derived fixed-size offsets
  - [x] Repeated field canonicalization to `Datatype::VarLen` for scalar and group columns
  - [x] Value-semantic tests for repeated scalar columns and repeated group columns
  - [x] Byte-level footer trailer validation (`PAR1` magic, little-endian footer length, footer offset bounds)
  - [x] Canonical footer prelude and byte-range metadata descriptors (`FooterPrelude`, `RowGroupLocation`, `ColumnChunkLocation`, `ParquetFooterDescriptor`)
  - [x] Value-semantic tests for valid trailer parsing, short input rejection, invalid magic rejection, footer-length overflow rejection, overlapping chunk rejection, and footer-bound row-group rejection
  - [x] Minimal Thrift compact binary protocol decoder (`wire::thrift::ThriftReader`): zigzag varint, i16/i32/i64, string/binary, field header, list/set/map header, recursive skip
  - [x] Canonical Parquet wire metadata types: `FileMetadata`, `SchemaElement`, `RowGroupMetadata`, `ColumnChunkMetadata`, `ColumnMetadata`, `KeyValue`
  - [x] Footer payload extraction and Thrift decoding: `decode_file_metadata(bytes, prelude)` -> `FileMetadata`
  - [x] Value-semantic tests for FileMetadata decoding from hand-computed Thrift compact byte vectors (valid decode, missing-required-field rejection)
  - [x] Canonical Parquet page header types: `PageHeader`, `DataPageHeader`, `DictionaryPageHeader`, `DataPageHeaderV2`, `PageType`
  - [x] Page header Thrift decoder: `decode_page_header(bytes)` -> `(PageHeader, consumed)`
  - [x] Value-semantic tests for page header decoding (DATA_PAGE, DICTIONARY_PAGE with is_sorted bool, DataPageHeaderV2, empty-input rejection)
  - [x] Schema reconstruction bridge: `schema_elements_to_schema` rebuilds `SchemaDescriptor` from flat pre-order DFS `SchemaElement` list (recursive group support)
  - [x] Dataset materialization bridge: `dataset_from_file_metadata(meta)` -> `ParquetDatasetDescriptor`
  - [x] Value-semantic tests for schema reconstruction and dataset bridge
  - [x] Module decomposition: `wire/thrift.rs`, `wire/metadata.rs`, `wire/page.rs`, `dataset/mod.rs` (all files under 400-line constraint)
  - [x] Physical page payload decoding and level decoding (RLE/bit-packing hybrid, deprecated BIT_PACKED, definition/repetition levels)
  - [x] `encoding/levels.rs`: `decode_levels` (RLE/bit-packing hybrid), `decode_bit_packed_raw` (deprecated BIT_PACKED), `level_bit_width` — 14 value-semantic tests
  - [x] `encoding/plain.rs`: PLAIN encoding decoders for all Parquet physical types (BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY) — 14 value-semantic tests
  - [x] `encoding/rle_dict.rs`: `decode_rle_dict_indices` (RLE_DICTIONARY, encoding ID 8) — 5 value-semantic tests
  - [x] `wire/payload.rs`: `split_data_page_v1` and `split_data_page_v2` payload splitters with `PagePayload` struct — 6 value-semantic tests
  - [x] :  enum (8 variants covering all Parquet physical types), , ,  (PLAIN/PLAIN_DICTIONARY/RLE_DICTIONARY dispatch), ,  — 16 value-semantic tests
  - [x] :  and  mapping parquet.thrift Type enum discriminants 0–7 — 2 tests
  - [x] Typed column value extraction: compression pipeline (decompress values_bytes before decoding)
  - [x] Real file-backed dataset read API (open file -> validate -> decode footer -> materialize dataset)
- [x] Write Consus datasets to Parquet
  - [x] Canonical writer-side planning over `SchemaDescriptor` trees with nested/group lowering to leaf paths
  - [x] Thrift compact footer encoder for `FileMetadata`, `SchemaElement`, `RowGroupMetadata`, `ColumnChunkMetadata`, and `ColumnMetadata`
  - [x] Page header encoder for `PageHeader`, `DataPageHeader`, `DataPageHeaderV2`, and `DictionaryPageHeader`
  - [x] Row-source to leaf-column value lowering for flat and nested/group schemas
  - [x] Complete file emission with trailer validation and `PAR1` footer assembly
  - [x] Footer metadata and trailer roundtrip verification against the existing reader
  - [x] PLAIN value encoder (`encode_cell_plain`) for all non-Boolean physical types (INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY)
  - [x] Boolean column PLAIN bit-packing encoder (`encode_bool_column_plain`): LSB-first, ⌈count/8⌉ bytes
  - [x] `physical_type_discriminant`: maps `ParquetPhysicalType` to parquet.thrift Type enum discriminant
  - [x] `build_file_bytes` emits real DataPage v1 pages with correct byte offsets recorded in `ColumnMetadata`
  - [x] End-to-end writer→reader roundtrip tests: INT32 (3 values), DOUBLE (2 values), BYTE_ARRAY (2 strings), BOOLEAN (4 values), two-column INT32+DOUBLE (2 rows)
  - [x] Negative test: Null in required column returns `InvalidFormat`
  - [x] Verified `cargo test -p consus-parquet --lib` 175/175 pass (default features)
  - [x] Verified `cargo check --workspace`: zero warnings, zero errors
- [x] Hybrid mode: Parquet tables inside Consus containers
- [x] Arrow array bridge (zero-copy) — Milestone 25: `zerocopy` optional feature added to `consus-arrow`; `fixed_to_le_bytes_fast<T: IntoBytes + Immutable>` helper reinterprets native-LE slice as `&[u8]` via `IntoBytes::as_bytes` (one allocation + one bulk memcpy); Int32/Int64/Float/Double arms cfg-selected between fast path (`#[cfg(all(feature = "zerocopy", target_endian = "little"))]`) and element-by-element fallback; Boolean/Int96/ByteArray/FixedLenByteArray unchanged; 2 value-semantic agreement tests verify fast path == `to_le_bytes()` reference for i32 boundary values and f64 non-finite values; 50/50 pass without feature, 52/52 pass with `--features zerocopy`

### P3.2 — FITS Table Wiring
- [x] Wire `tform_to_datatype` into `FitsTableColumn` so parsed TFORMn produces canonical `Datatype`
- [x] Per-column byte-width computation and NAXIS1 validation for binary tables
- [x] ASCII table column value decoding: `decode_ascii_column` extracts A/I/F/E/D fields from raw row bytes; trailing-space stripping; Fortran D-notation normalization; 11 value-semantic unit tests
- [x] Binary table column value decoding: `decode_binary_column` + `decode_scalar_binary` cover all 13 FITS Standard 4.0 TFORM codes (L/X/B/I/J/K/A/E/D/C/M/P/Q); big-endian byte extraction; array repeat handling; 24 value-semantic unit tests
- [x] `FitsTableData::decode_row` and `FitsTableData::decode_column` dispatch to binary/ASCII decoders per table kind

### P3.1 — Parquet Interop (continued)
- [x] Arrow array bridge — `column_values_to_arrow` in `consus-arrow/src/array/materialize.rs`: materializes `ColumnValues` (Boolean/Int32/Int64/Int96/Float/Double/ByteArray/FixedLenByteArray) into canonical `ArrowArray`; fixed-width numerics stored little-endian; variable-width ByteArray produces monotone offsets; 10 value-semantic tests covering all 8 variants plus empty-array boundary cases
- [x] `column_values_to_arrow` exported from `consus-arrow` crate root under `#[cfg(feature = "alloc")]`
- [x] E2E integration test pipeline — `consus-arrow/tests/parquet_arrow_e2e.rs`: 6 integration tests exercising full ParquetWriter → ParquetReader → `column_values_to_arrow` pipeline for INT32, INT64, DOUBLE, BYTE_ARRAY, BOOLEAN, and two-column (INT32+DOUBLE) schemas; byte-level assertions on all output `ArrowArray` buffers
- [x] Compressed page emission — `compress_page_values(data, codec) -> Result<Vec<u8>>` added to `consus-parquet/src/encoding/compression.rs`; `build_file_bytes` now honors codec parameter: compresses each page's PLAIN bytes, records correct `ColumnMetadata.codec` discriminant, and sets correct `uncompressed_page_size` / `compressed_page_size` in each page header; 4 new tests (uncompressed passthrough, brotli unsupported, gzip INT32 writer→reader roundtrip, gzip BYTE_ARRAY writer→reader roundtrip)
- [x] Zero-copy materialization fast path — `zerocopy` optional feature added to `consus-arrow`; `fixed_to_le_bytes_fast<T: IntoBytes + Immutable>` helper uses `zerocopy::IntoBytes::as_bytes().to_vec()` (one allocation + one bulk memcpy) instead of element-by-element `to_le_bytes()` loop; active for Int32/Int64/Float/Double on `#[cfg(all(feature = "zerocopy", target_endian = "little"))]`; 2 agreement tests verify fast path bytes == element-loop reference for i32 and f64 boundary values
- [x] Arrow array bridge (zero-copy) — completed via zerocopy optional feature (see above)
- [x] Multi-row-group writer splitting — `ParquetWriter::with_row_group_size(n: usize) -> Self` builder method; `row_group_size: Option<usize>` field; `encode_leaf_columns(plan, rows, row_start, row_end) -> Result<Vec<Vec<u8>>>` private helper; `build_file_bytes` partitions rows into ⌈N/n⌉ groups (last group ≤ n rows); ≥1 row group invariant preserved for N=0; `FileMetadata.num_rows` = N; each `RowGroupMetadata.num_rows` = group size; each `ColumnMetadata.data_page_offset` = absolute offset; 6 value-semantic tests (even split, uneven split [3,3,1], size>count, exact multiple, default single group, zero size) and 1 proptest (∀ values ∈ Vec<i32>, m ≥ 1: roundtrip identity) in `writer/tests_extra.rs`
- [x] Compressed writer roundtrip tests — SNAPPY, ZSTD, LZ4_RAW, LZ4: 4 feature-gated writer→reader roundtrip tests (`writer_snappy_roundtrip_i32_three_values`, `writer_zstd_roundtrip_i32_three_values`, `writer_lz4_raw_roundtrip_i32_three_values`, `writer_lz4_roundtrip_i32_three_values`) in `writer/tests_extra.rs`; each writes INT32 values under the respective codec and reads back via `ParquetReader`, asserting exact value equality
- [x] proptest roundtrip suite — `encoding/compression_proptest.rs`: 5 compression roundtrip properties (gzip+zlib under `#[cfg(feature="gzip")]`, snappy, zstd, lz4_raw+lz4) asserting `decompress(compress(data, c), c, |data|) == data` for arbitrary byte vectors up to 1 KiB; `encoding/plain_proptest.rs`: 6 PLAIN decode properties (i32, i64, f32-bits, f64-bits, i96-12-bytes, byte_array, fixed_len_byte_array) asserting `decode(encode(v), 1) == [v]`; `writer/tests_extra.rs`: boolean bit-packing property `∀ bools: decode(encode_bool_column_plain(bools), |bools|) == bools` and multi-row-group roundtrip property
- [x] Optional flat column write (def_level encoding, CellValue::Null)
- [x] Repeated flat column write (rep/def level encoding, CellValue::Repeated)
- [x] `ColumnValuesWithLevels` type with Dremel level accessors
- [x] `read_column_chunk_with_levels` reader API
- [x] `dataset_from_file_metadata` row_count correctness fix (use rg.num_rows)
- [x] Nested group column write/read support (Milestone 34 — Dremel full traversal) — `traverse_dremel_into` recursive encoding for Required/Optional/Repeated at any depth; `encode_leaf_columns` unified to single Dremel path; 4 value-semantic roundtrip tests
- [x] Multi-page splitting within a column chunk (Milestone 36/P3.7 — CLOSED) — `ParquetWriter::with_page_row_limit`; 6 deterministic tests + 1 proptest

### P3.3 — Production Readiness
- [x] CI/CD pipeline (GitHub Actions)
- [x] Async HDF5 I/O path via Moirai's native executor and positioned-read contracts
- [x] Memory-mapped I/O backend — `MmapReader` in `consus-io/src/io/sync/mmap.rs`; feature-gated under `mmap` feature; implements `ReadAt + Length` over `memmap2::Mmap`; `open(path)` and `from_file(&File)` constructors; `as_slice() -> &[u8]` accessor; `Send + Sync`; 8 unit tests + 3 integration tests in `tests/unit_mmap.rs`; `memmap2 = { version = "0.9" }` added to workspace deps; verified `cargo test -p consus-io --features mmap` 28+3=31 pass
- [x] Parquet reader proptest suite — `consus-parquet/src/reader/reader_proptest.rs`: 5 proptest roundtrip properties (`prop_reader_i32_roundtrip`, `prop_reader_f64_roundtrip`, `prop_reader_bool_roundtrip`, `prop_reader_byte_array_roundtrip`, `prop_reader_two_column_i32_f64_roundtrip`); all assert computed column values with `prop_assert_eq!`; verified `cargo test -p consus-parquet --lib` 197/197 pass
- [x] Criterion benchmark harness — `consus-parquet/benches/parquet_rw.rs`: `bench_write_i32` + `bench_read_i32` at 1K/10K/100K rows; `consus-arrow/benches/arrow_bridge.rs`: `bench_bridge_i32`, `bench_bridge_double`, `bench_bridge_byte_array`; `[[bench]]` targets added to both Cargo.toml files; verified `cargo check --bench parquet_rw` and `cargo check --bench arrow_bridge` clean
- [x] Large file (>4 GiB) regression tests
- [x] proptest harnesses delivered: `is_valid_iso8601` (4 property tests, consus-nwb) + `decode_attribute_value` (4 property tests, consus-hdf5) — Milestone 52 (this sprint — CLOSED)
- [x] `cargo-fuzz` harness targets (heap-buffer and logic fuzz) — `fuzz/Cargo.toml` + three `fuzz/fuzz_targets/` harnesses: `fuzz_hdf5_parser` (superblock → list_root_group → dataset_at / attributes_at / read_chunked_dataset_all_bytes), `fuzz_parquet_decoder` (footer → Thrift FileMetadata → all rg×col read_column_chunk), `fuzz_mat_reader` (loadmat_bytes v4/v5/v7.3 dispatch); `cargo fuzz list` reports all 3 targets; compilation blocked on Windows (libfuzzer-sys C++ build uses MSVC __pragma incompatible with g++.exe — expected platform limitation); targets compile clean on Linux CI
- [x] WASM target validation
- [x] `no_std` smoke tests (`thumbv7em-none-eabihf`) — workspace `no_std + alloc` compilation now fully clean: NO-STD-001 closed (M-050); embedded target smoke test still pending
- [x] Documentation site — automated via GitHub Actions workflow (`.github/workflows/docs.yml`)
- [x] crates.io publication — automated via `scripts/publish.ps1` for topological release sequence

## Phase 2.6: MATLAB .mat Format Reader (consus-mat)

### P2.6a - MAT v4 (Binary)
- [x] V4Header::parse: type field decoding (M*1000+P*10+T), LE/BE byte-order, name extraction
- [x] read_v4_variable: numeric matrix (all 6 precisions), text matrix (f64->char), complex, LE/BE normalization
- [x] read_mat_v4: sequential record parsing until EOF
- [x] Positive: v4_double_array_shape_and_values (f64, shape [2,3], 6 exact column-major values)
- [x] Negative: v4_truncated_header_returns_error
- [x] Negative: v4_empty_slice_returns_error
- [x] v4 sparse matrix explicit permanent rejection policy with fixture coverage (v4_sparse_matrix_returns_unsupported_feature_error)

### P2.6b - MAT v5 (Structured Binary)
- [x] V5FileHeader::parse: LE/BE detection via endian indicator
- [x] read_tag: standard vs small element detection, all 15 miXXXX type codes
- [x] parse_matrix: mxDOUBLE..mxUINT64 numeric, mxCHAR, mxSPARSE, mxCELL, mxSTRUCT
- [x] Complex flag detection and imaginary sub-element extraction
- [x] Logical flag detection producing MatLogicalArray
- [x] Sparse invariants enforced: `ir.len() == nzmax`, `jc.len() == ncols + 1`
- [x] Expanded synthetic coverage for char, logical, complex, sparse, and cell decoding
- [x] Unknown top-level v5 element skipping with structural validation
- [x] `mxOBJECT_CLASS` explicit permanent rejection policy with integration coverage
- [x] Compressed `miCOMPRESSED` fixture coverage across enabled/disabled feature configurations

### P2.6c - MAT v7.3 (HDF5-backed)
- [x] Root-group traversal with MATLAB_class dispatch through `consus-hdf5`
- [x] Numeric, logical, char, cell, and scalar struct decoding
- [x] Deterministic numeric ordering for cell-group children named `"0"`, `"1"`, ...
- [x] Expanded synthetic HDF5-backed coverage for logical, char, cell, and struct decoding
- [x] Non-scalar struct array decoding with authoritative shape preservation
- [x] Character decoding from dataset datatype byte order instead of hardcoded LE
- [x] Sparse v7.3 decoding or explicit permanent rejection policy
- [x] Compact-layout rejection coverage
- [x] Virtual-layout rejection coverage (DatasetLayout::Virtual in HDF5 builder; rejection test passing)
- [x] Chunked-dataset fixture coverage: v73_chunked_double_array_roundtrip passing
- [x] v7.3 cell array group roundtrip: MATLAB_class="cell" group with decimal-named child datasets; value-semantic integration coverage
- [x] v7.3 struct array group roundtrip: MATLAB_class="struct" group with field-named child datasets; value-semantic integration coverage

### P2.6d - Model, Documentation, and Release Readiness
- [x] Canonical public model types for numeric, char, logical, sparse, cell, and struct arrays
- [x] Invariant-enforcing constructors added for cell, char, logical, sparse, and struct models
- [x] Crate-level documentation updated for feature flags, entry points, contracts, and unsupported cases
- [x] Remove redundant struct field-name storage: MatStructArray.fields removed; data keys are sole SSOT; new() signature changed to (shape, data); field_names() returns impl Iterator
- [x] Add crate README with usage examples, feature matrix, and version-specific behavior notes
- [x] Add CI coverage for `cargo test -p consus-mat` and feature-matrix verification, including `miCOMPRESSED` enabled/disabled configurations
- [x] miCOMPRESSED zlib decompression (compress feature)
- [x] read_mat_v5: sequential element parsing with miMATRIX and miCOMPRESSED dispatch
- [x] Positive: v5_double_array_roundtrip (f64, shape [1,3], 3 exact values)
- [x] Negative: v5_invalid_endian_indicator_returns_error, v5_truncated_header_returns_error
- [x] consus-hdf5 Hdf5FileBuilder extended: ChildDatasetSpec + add_group_with_attributes enables nested group authoring with attached attributes for v73 fixture coverage
- [x] Model unit tests added: 42 tests across all 6 model modules covering constructors, invariant enforcement, and accessor methods
- [x] MatError Display unit tests: 5 tests covering all Display impl variants
- [x] Multi-variable v5 integration test: v5_multiple_variables_roundtrip (2 named scalar doubles, value-semantic)
- [x] loadmat file I/O test: loadmat_from_reader_parses_test_fixture (std::fs::File + test_v5.mat roundtrip)
- [x] Doc test for loadmat_bytes: MAT v4 scalar double byte sequence, verifies variable count and name
- [x] Verified cargo test -p consus-mat: 71/71 pass (42 lib + 4 v4 + 1 v5-compressed + 14 v5 + 9 v73 + 1 doc)
- [x] Verified cargo test -p consus-mat --no-default-features --features std,alloc: 62/62 pass
- [x] Verified cargo test -p consus-hdf5: 321/321 pass
- [x] Verified cargo check --workspace: zero errors

### P2.6c - MAT v7.3 (HDF5-backed)
- [x] Version detection via HDF5 file signature at byte offset 0
- [x] read_mat_v73: root group traversal, MATLAB_class attribute dispatch via consus-hdf5
- [x] Numeric arrays: contiguous+chunked payload, shape reversal (C-order to Fortran-order)
- [x] Complex arrays: compound {real, imag} field deinterleaving
- [x] Logical arrays: uint8 payload decoded to Vec<bool>
- [x] Char arrays: uint16 payload decoded to UTF-8 String
- [x] Struct arrays: group children mapped to MatStructArray
- [x] Cell arrays: group children mapped to MatCellArray
- [x] Positive: v73_double_array_roundtrip (HDF5 + MATLAB_class attr, 3 exact f64 values)

### P2.6d - Version Detection and Entry Points
- [x] detect_version: HDF5 magic -> V73, MAT v5 endian indicator -> V5, fallback -> V4
- [x] loadmat_bytes: auto-detect and dispatch to version-specific parser
- [x] loadmat<R: Read + Seek>: std-feature convenience wrapper
- [x] 5 unit tests in detect::tests module

### P2.6e - Correctness Hardening and Coverage Expansion (this sprint)
- [x] Removed dead byteorder + consus-compression deps from Cargo.toml
- [x] Removed dead UnsupportedVersion variant; fixed lib.rs doc strings
- [x] v5 sparse: ir.len()==nzmax + jc.len()==ncols+1 invariants enforced
- [x] v73 cell group: children sorted by numeric name for deterministic element order
- [x] v5 vacuous truncated test replaced with value-asserting negative test
- [x] v5 synthetic test suite: char, logical, complex, sparse, cell, struct (7 new tests)
- [x] Verified cargo test -p consus-mat: 17/17 pass (3 v4, 10 v5, 4 v73)

## Phase 3: Parquet Nested Column Write + NWB Support

### P3.3b — Parquet Nested Column Write (this sprint — CLOSED)
- [x] `top_field_idx` in `LeafColumnPlan` — correct row value indexing for group schemas
- [x] `traverse_dremel_into` — recursive Dremel encoding for Required/Optional/Repeated at any depth
- [x] `encode_leaf_columns` unified to single Dremel path (replaces 3 flat branches)
- [x] Four value-semantic nested-column roundtrip tests (required group, optional group, repeated group, deeply-nested optional)
- [x] `cargo test -p consus-parquet --lib` → 209/209

### P3.4 — NWB Read Path
- [x] NWBFile open and HDF5 validation (`NwbFile::open`, `validate_root_attributes`)
- [x] Session metadata extraction (`NwbSessionMetadata`, `session_metadata()`)
- [x] TimeSeries read — data array + timestamps (`time_series(path)`)
- [x] Namespace version detection (`NwbVersion::parse`, `detect_version`)
- [x] Conformance validation skeleton (`validate_root_attributes` — neurodata_type_def + nwb_version)
- [x] Integer dataset promotion to f64 in `read_f64_dataset` — all signed/unsigned 8/16/32/64-bit widths, both byte orders
- [x] `starting_time` + `rate` read from `{path}/starting_time` scalar dataset and its `rate` float32 attribute
- [x] `NwbFile::list_time_series(group_path)` — enumerate TimeSeries children at any group path (`""` = root)
- [x] `group/mod.rs` — `NwbGroupChild` + `list_typed_group_children` (NodeType::Group filter, neurodata_type_def/inc extraction)
- [x] `conventions/mod.rs` — `NeuroDataType` enum, `classify_neurodata_type`, `is_timeseries_type` (def + inc + known subtypes)
- [x] `namespace/mod.rs` — `NwbNamespace` with `core()`, `hdmf_common()`, `CORE_NAME` constant
- [x] `consus-hdf5 list_group_at` fix: SYMBOL_TABLE guard prevents v1 fallback error on v2 empty groups
- [x] Multi-level open_path verified in integration test: list_acquisition returns both nested paths; time_series reads through /acquisition/{name}/ hierarchy
- [x] Units table read (spike times) — `NwbFile::units_table()` + `UnitsTable::from_vectordata` (Milestone 40 — closed)
- [x] Subject metadata extraction — `NwbSubjectMetadata` model + `NwbFile::subject()` read path
- [x] `NwbFile::list_acquisition()` — convenience wrapper over `list_time_series("acquisition")`
- [x] `NwbFile::list_processing(module)` — convenience wrapper over `list_time_series("processing/{module}")`
- [x] ElectrodeTable read (electrode metadata) — `NwbFile::electrode_table()` + `read_string_dataset`/`read_u64_dataset` (Milestone 40 — closed)
- [x] Namespace version detection and spec YAML parsing from `/specifications/` (NwbVersion V2_8, NwbNamespaceSpec, parse_nwb_spec_yaml, format_nwb_spec_yaml, list_specifications, read_specification, write_namespace_specs — this sprint)
- [x] Per-type `neurodata_type_inc` inheritance chains in `NwbNamespaceSpec` — `NwbTypeSpec` struct; `neurodata_types: Vec<NwbTypeSpec>`; iterative BFS `is_timeseries_type_with_specs` with depth-64 guard; backward-compatible YAML parse/format (Milestone 44 — this sprint)

### P3.5 — NWB Write Path (Milestone 37 — this sprint — CLOSED)
- [x] `NwbFileBuilder` — construct root HDF5 group with required NWB metadata attributes
- [x] Required root attributes: `neurodata_type_def = "NWBFile"`, `nwb_version`, `identifier`, `session_description`, `session_start_time`
- [x] `write_time_series(ts: &TimeSeries)` — emit group with `data` + `timestamps` or `starting_time` + `rate` datasets
- [x] `neurodata_type_def = "TimeSeries"` attribute on each written TimeSeries group
- [x] Units table write: `Units` group with `spike_times` VectorData dataset
- [x] Roundtrip tests: write then re-open with `NwbFile::open` and verify all fields
- [x] Namespace conformance validation before write (`validate_time_series_for_write`)
- [x] `NwbFile::units_spike_times()` — read path for Units roundtrip verification
- [x] `cargo test -p consus-nwb --lib` → 149/149

### P3.7 — Parquet Multi-Page Column Chunk Splitting (this sprint — CLOSED)
- [x] `ParquetWriter::with_page_row_limit(limit)` builder method
- [x] `build_file_bytes` page range computation: `ceil(group_rows / limit)` pages per column chunk
- [x] Transpose-then-emit pattern: `pages_by_column[leaf_idx][page_idx]` guarantees contiguous column chunk emission
- [x] `data_page_offset` = first page byte offset; `total_uncompressed/compressed_size` and `num_values` summed over all pages
- [x] Six deterministic tests + one proptest (`prop_multi_page_i32_roundtrip`)
- [x] `cargo test -p consus-parquet --lib` → 215/215

### P3.8 — HDF5 Nested Group Write + NWB Extended APIs (Milestone 39 — CLOSED)
- [x] `ChildGroupSpec<'a>` — new public struct in `consus-hdf5::file::writer`
- [x] `write_group_node` — private recursive free function replacing duplicated group-write logic
- [x] `Hdf5FileBuilder::add_group_with_attributes` refactored to delegate to `write_group_node` (backward compat)
- [x] `Hdf5FileBuilder::add_group_with_children` — new method supporting arbitrary-depth nested groups
- [x] `NwbSubjectMetadata` — `consus-nwb::metadata`, 5 optional fields, `from_parts` + accessors
- [x] `NwbFile::subject()` — reads `general/subject` group attributes
- [x] `NwbFileBuilder::write_subject(&NwbSubjectMetadata)` — writes `general/subject` via nested group API
- [x] `NwbFile::list_acquisition()` and `list_processing(module)` convenience methods
- [x] Proptest roundtrips: timestamps, rate (f32 precision invariant), units spike times
- [x] `cargo test -p consus-nwb --lib` → 166/166; `cargo test --workspace` → 2239/2239

### P3.9 — NWB ElectrodeTable + UnitsTable + Storage String/U64 + README (Milestone 40 — CLOSED)
- [x] `read_string_dataset` in `consus-nwb::storage` — decode FixedString dataset → `Vec<String>` (null-stripped)
- [x] `read_u64_dataset` in `consus-nwb::storage` — decode integer dataset → `Vec<u64>` (all 8/16/32/64-bit widths, signed+unsigned)
- [x] `decode_raw_as_u64` private helper — matches `decode_raw_as_f64` pattern; rejects non-integer datatypes
- [x] 8 new value-semantic storage tests (u32→u64 widening, u64 identity, i32 signed bit-pattern, float rejection, FixedString exact-fill, null-padded strip, all-null element, wrong-type rejection)
- [x] `UnitsTable` model in `consus-nwb::model::units` — `spike_times_per_unit: Vec<Vec<f64>>`, `ids: Option<Vec<u64>>`
- [x] `UnitsTable::new`, `from_parts`, `from_vectordata` (VectorIndex decode with monotone + length invariant checks)
- [x] `UnitsTable::flat_spike_times()` + `cumulative_index()` — encode back to VectorData/VectorIndex wire format
- [x] 18 value-semantic `UnitsTable` unit tests (construction, VectorIndex decode, error paths, roundtrip)
- [x] `ElectrodeRow` + `ElectrodeTable` model in `consus-nwb::model::electrode`
- [x] `ElectrodeTable::from_rows`, `from_columns`, `empty`; column iterators `id_column`, `location_column`, `group_name_column`
- [x] 13 value-semantic `ElectrodeTable` unit tests (construction, column mismatch rejection, accessors, Clone/PartialEq)
- [x] `NwbFile::units_table()` — reads `Units/spike_times` + `Units/spike_times_index` + optional `Units/id` via VectorIndex decode
- [x] `NwbFile::electrode_table()` — reads `electrodes/id` + `electrodes/location` + `electrodes/group_name`
- [x] `NwbFileBuilder::write_units_table(&UnitsTable)` — emits VectorData + VectorIndex datasets with `VectorData`/`VectorIndex` `neurodata_type_def` attributes; optional `id` dataset
- [x] `NwbFileBuilder::write_electrode_table(&ElectrodeTable)` — emits `DynamicTable` group with `id`, `location`, `group_name` datasets (null-padded fixed-string columns)
- [x] 7 new file integration tests: 3 UnitsTable roundtrips (with IDs, without IDs, empty), 2 ElectrodeTable roundtrips (3-row, empty), 2 NotFound negative tests
- [x] `crates/consus-nwb/README.md` created — format overview, feature flags, quick-start read/write examples, module architecture, NWB compliance table, license
- [x] `cargo test -p consus-nwb --lib` → 211/211; `cargo test --workspace` → 2285/2285; `cargo check --workspace` → 0 errors, 0 warnings

### P3.6 — NWB Verification
- [x] h5py-generated NWB 2.7 fixture verification (Milestone 46 — this sprint)
- [x] All read paths verified against h5py fixture: session_metadata, list_acquisition, time_series (f64+i16 promotion+rate), units_table, electrode_table (VL strings), subject — `tests/integration_real_file.rs` (Milestone 46 — this sprint)
- [x] Full conformance validation against NWB 2.x schema — `NwbFile::validate_conformance()` + `NwbConformanceReport` + 4 validation layers + 29 new tests (Milestone 47 — this sprint — CLOSED)
- [x] Extended conformance: `timestamps_reference_time` (ISO 8601) + `file_create_date` (≥1 ISO 8601 entry) layer-2 validation; DynamicTable `colnames` layer-5 validation; `NwbFileBuilder::new` writes both new required attrs automatically — 12 new tests (Milestone 48 — this sprint — CLOSED)
- [x] DynamicTable column-content consistency validation — `check_dynamic_table_column_content`, `DynamicTableColumnMissing` variant, layer 6 in `validate_conformance` (Milestone 51 — this sprint — CLOSED)
- [x] no_std + alloc compilation verified for consus-core, consus-io, consus-hdf5, consus-nwb; consus-hdf5 and consus-nwb no_std gaps fixed (Milestone 49 — this sprint — CLOSED)
- [x] no_std + alloc compilation for consus-zarr now passes; NO-STD-001 resolved — gzip/zstd/lz4 feature-gated under `std`, codec paths guarded, FsStore gated (Milestone 50 — this sprint — CLOSED)
- [x] ElectrodeTable read (electrode metadata) — `read_string_dataset` added; `NwbFile::electrode_table()` + `NwbFileBuilder::write_electrode_table()` implemented (Milestone 40)
- [x] Namespace version detection and spec YAML parsing from `/specifications/` (NwbVersion V2_8, NwbNamespaceSpec, parse_nwb_spec_yaml, format_nwb_spec_yaml, list_specifications, read_specification, write_namespace_specs — this sprint)

## Phase 4: Cloud Native Backends

### P4.1 — Async S3 Backend (`consus-io`)
- [x] Implement `AsyncReadAt` and `AsyncLength` using `rusoto_s3` `GetObjectRequest` with HTTP `Range` headers
- [x] Implement `S3Reader` struct holding bucket name, object key, and pre-configured `S3Client`
- [x] Handle AWS credential extraction, error mapping, and region mapping
- [x] Implement in-memory integration testing via mocked responses or MinIO (if available)

### P4.2 — Zarr Cloud Integration
- [x] Add `S3Store` implementation for Zarr v2/v3 using `consus-io::S3Reader`
- [x] Enable parallel HTTP GETs in Zarr chunk reads (already partially supported by async chunk pipeline)

### P4.3 — HDF5 Cloud Integration
- [x] Adapt `Hdf5File::open` to accept an `AsyncReadAt` backend
- [x] Asynchronous B-tree navigation and metadata traversal over HTTP ranges
