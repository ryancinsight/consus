# Consus - Gap Audit

## ATLAS-CONSUS-TYPES-057 — Generic endian scalar reads (2026-08-15)

The core decoder had six duplicated public readers whose names encoded the
scalar type and width. They now share one `EndianScalar` trait and generic
`read_integer<T>` entry point in `decode/endian.rs`. The associated byte width
is compile-time data, the conversion is monomorphized per supported scalar,
and the borrowed input remains allocation-free. Direct HDMF call sites select
the concrete scalar at the decode boundary; no forwarding aliases remain.

The new conformance test instantiates all supported signed and unsigned
16/32/64-bit scalars in both byte orders and checks short-input rejection. The
provider scan drops `type_suffixed_fns` from 91 to 85. Local evidence is
313/313 Consus-core+HDMF Nextest tests, 278/278 Consus-NWB tests, strict
Clippy for all affected crates, all-target checks, and three passing doctests.
Hosted provider checks remain the merge gate; no conformance baseline change
is authorized.

## ATLAS-CONSUS-UNWRAP-056 — Decode test diagnostics (2026-08-15)

The conformance scan classified 14 bare unwraps in the test-only decode
module as production debt because the module uses a feature-qualified test
configuration. The sites were replaced with invariant-bearing `expect`
messages: non-zero test bit widths and valid decode results. Runtime decode
logic and value-semantic assertions are unchanged.

At the provider worktree, the exact focused gates passed: formatting, all-
target/all-feature check, strict Clippy, all-feature Consus-core Nextest, and
the no-default-feature check. The provider conformance scan now reports
`unwrap_production=383` (397 before the slice) and `type_suffixed_fns=91`;
the latter remains a separate cleanup item. Hosted provider checks are the
merge gate before advancing the Atlas gitlink.

## ATLAS-CONSUS-HIERARCHY-055 — Arrow datatype hierarchy (2026-08-15)

The `consus-arrow::datatype` manifest mixed the canonical `ArrowDataType` enum
with temporal/scalar metadata and alloc-backed nested descriptor definitions.
At the exact provider head `419b114b`, that file was 535 lines and contributed
one stale 500-line hierarchy count. The descriptor families now have one named
home in `datatype/descriptors.rs`; `datatype/mod.rs` retains the enum,
conversion, and tests and is 330 lines. Public exports and `alloc` feature
boundaries are unchanged, so consumers do not need a migration or adapter.

Evidence: provider conformance scan `oversized_files=83`, all-feature Arrow
Nextest 81/81, no-default Nextest 2/2, strict Clippy/checks, doctests,
warning-denied Rustdoc, formatting, and diff checks. The provider's separate
`unwrap_production=397` and `type_suffixed_fns=91` residuals are not hidden by
this structural slice and remain open for a later owner-local cleanup.

## ATLAS-CONSUS-BTREE-RESOURCE-039 — v1 chunk B-tree allocation boundary (2026-08-15)

The synchronous and asynchronous v1 raw-data chunk B-tree readers derived the
record region with unchecked `usize` arithmetic, allocated it with `vec!`, and
reserved decoded entry and rank vectors without the shared parser budget. The
record-size calculation is now shared, checked for overflow and the byte
ceiling before I/O, and all v1/v4 entry and rank-vector allocations use
fallible `ParseBudget` reservations.

The value-semantic regression rejects a record region beyond a constrained
budget before any read. Local verification is strict Clippy, formatting,
package check, no-default check, the focused budget test, async Nextest 11/11,
and full HDF5 Nextest 440/440. Exact provider head `6bb060b` passes CI
`31869739418`, Documentation `31869739408`, and manually dispatched Pages
`31869947774`.

## ATLAS-CONSUS-ASYNC-RESOURCE-038 — Async HDF5 allocation boundary (2026-08-15)

The async HDF5 I/O layer allocated `read_region` buffers directly from caller
or file-derived lengths, and async chunked reads recomputed dataset and chunk
payload products outside the shared budget helper. The region helper now
receives the `ParseContext`, uses its fallible byte ceiling, and validates
object-header continuation lengths before converting them for slicing. Async
chunked output and uncompressed chunk sizes reuse the synchronous bounded
payload calculation.

The new async regression requests one byte beyond the default allocation
ceiling and asserts `Error::ResourceLimit` before I/O. Verification at the
implementation tree is async Nextest 11/11, adversarial HDF5 Nextest 16/16,
full HDF5 Nextest 439/439, strict Clippy, formatting, package check, and the
no-default check. Exact provider head `bae27ac` passes hosted CI
`31868746891`, Documentation `31868746908`, and Pages `31868746410`.

## ATLAS-CONSUS-PARSE-LIMITS-036 — HDF5 allocation boundary (2026-08-15)

The HDF5 parser still had file-controlled allocations outside the shared
`ParseBudget`: global-heap collection/object payloads, contiguous and chunked
dataset output, fixed-array data blocks, and direct, indirect, managed, and
huge fractal-heap reads. The slice now checks element and byte products before
allocation, uses fallible zeroed buffers, checks address and row-coverage
arithmetic, and rejects impossible fixed-array metadata.

Five adversarial tests cover the collection, dataset, managed-object,
indirect-block, and huge-object limits; the fixed-array limit has an additional
unit regression. The empty `AsyncFacadeUnavailable` marker module was removed
because it was not an implementation or consumer contract; async behavior
continues to live in the provider-owned backend crates.

Verification at the implementation tree: strict Clippy for `consus-hdf5` and
`consus`, HDF5 Nextest 438/438, Consus Nextest 8/8, both package doctest
commands, `consus --no-default-features` check, formatting, and diff checks.
Provider head `9e11ba7` passed exact-head hosted CI run `31867893038`,
Documentation run `31867893068`, and Pages run `31867892530`.

## ATLAS-CONSUS-HDF5-LINK-098 — Reject overflowing link ranges (2026-08-14)

Hosted fuzz run `31848757905` found a panic in the HDF5 link-message parser at
`message.rs:321`: a fuzzed cursor and field length overflowed during
`cursor + need`. The reproducer was
`crash-3d3e53cb8b8815a14a313be45517235206021935`.

The remaining-range guard now uses checked addition, link-name lengths use a
fallible `u64`→`usize` conversion, and a regression test asserts that an
oversized name length returns `InvalidFormat` without panicking. Local HDF5
nextest (431/431) and strict Clippy pass. Exact default head
`c3afb406993f6b92e11963100621438064928383` is covered by hosted CI run
`31851240758`, which passed the repository-owned matrix.

## Dataset read + decode consolidation (2026-08-14)

The HDF5 dataset read path was triplicated: `consus-hdmf/src/storage.rs`,
`consus-nwb/src/storage/dataset.rs`, and kwavers-diagnostics each hand-rolled
the storage-layout dispatch (contiguous/chunked/compact/virtual) and the
byte-order decode. `Hdf5File::read_dataset_raw` now owns the single
layout-dispatch + raw-payload read (including the variable-length string
element-size case); consus-hdmf's private loader and consus-nwb's
`read_f64_dataset`/`read_u64_dataset`/`read_string_dataset` delegate to it.
The NWB `decode_raw_as_f64` (an exact duplicate of
`consus_core::decode::decode_to_f64`) and the duplicated `primitives.rs`
byte readers were deleted; the 13 value-semantic NWB decode tests now
exercise the consus-core implementation directly.

Also fixed: the NWB fixture-guard test built expected paths from hardcoded
Windows backslash literals (`D:\consus\data\nwb\...`), which failed on the
macOS runner where `Path::join` uses `/`; it now derives expected paths with
`join`, so it is host-portable.

Verification: `cargo test -p consus-hdf5 --lib` 280/280, `consus-nwb --lib`
272/272; workspace formatting clean; hosted CI green for consus-core/hdf5/
hdmf/nwb on all platforms plus MSRV. Net -208 lines.
## HDF5 no-default closure (2026-08-13)

The NWB no-default library gate originally failed in `consus-hdf5` because
alloc-backed file APIs and B-tree re-exports were compiled without `alloc`, and
superblock errors constructed heap-backed payloads unconditionally. The HDF5
root now retains only its allocation-free structural modules in no-default
builds; alloc-backed modules, tests, and the benchmark are feature-gated. The
shared `Error::invalid_format` constructor preserves the error category in
no-alloc builds and retains detailed messages in alloc builds.

Evidence: HDF5 no-default check, strict Clippy, and explicit no-test Nextest
pass; default strict Clippy and Nextest 405/405 pass. The workspace no-default
check and strict Clippy pass after this closure.

## Arrow/Parquet no-default closure (2026-08-13)

`consus-arrow` and `consus-parquet` now honor their declared `alloc` boundary.
Alloc-bearing schema, bridge, conversion, wire, hybrid, reader, writer, and
materialization surfaces are not compiled in a no-default build. The retained
no-alloc surface is the physical/logical Parquet model plus an Arrow fixed-width
shape descriptor; its tests assert value semantics rather than existence only.
Alloc-only integration tests and Criterion benches declare `alloc` as a
required feature, so no-default `--all-targets` checks do not compile APIs that
the feature contract intentionally removes.

Evidence: `consus-parquet` no-default Nextest 10/10 and default 215/215;
`consus-arrow` no-default Nextest 2/2 and default 79/79; strict Clippy passes
for both packages in both modes. A workspace no-default check proceeds through
these two providers and stops at the pre-existing `consus-fits` cfg boundary.

## Pages-disabled documentation CI (2026-07-22)

Documentation run `29941230671` proved that Rustdoc and redirect generation
were healthy, but the same build job unconditionally queried the GitHub Pages
API and failed when Pages was disabled. Documentation verification now remains
unconditional while Pages configuration, artifact upload, and deployment are
one explicit opt-in path controlled by repository variable
`CONSUS_ENABLE_PAGES == 'true'`. An absent or false variable skips publication
without masking Rustdoc failures or changing repository Pages settings.

## Stale provider branch recovery (2026-07-22)

Four stale branch refs retained required provider work absent from `main`.
`codex/consus-bounded-read-main`, `codex/consus-npy-provider`, and
`codex/consus-bounded-read-ritk` each carried a parallel bounded-read commit;
`codex/consus-bounded-read-ritk` then added ONNX and the final consumer
contract. RITK pins `ec386e3` and uses its 16 MiB bounded-capacity policy, so it
supersedes `codex/consus-bounded-capacity-ritk`'s earlier 64 KiB reservation
variant. The independent NPY/NPZ provider from `14bb619` remains required.

Recovery rebases the canonical bounded-read and ONNX history plus the NPY/NPZ
provider onto current `main`. Current-tree verification found and fixed an
unbounded NPY payload reservation, a hidden `consus-io/alloc` feature-unification
dependency, test-only type inference drift, and the stale `zip` 0.6 dependency.
The recovered surface uses `zip` 6.0, the newest maintained line compatible
with the workspace Rust 1.85 MSRV. Focused evidence: NPY 4/4, ONNX 3/3, and
Consus I/O 251/251 Nextest tests; warning-denied Clippy and rustdoc; three
doctests; ONNX `alloc`-only compilation; locked metadata; and 196/196
applicable Consus I/O semver checks.

## M-053 ONNX document ownership (2026-07-10)

RITK required ONNX graph inspection without inheriting a deep-learning tensor
runtime from its parser. `consus-onnx` now owns the protobuf wire subset needed
for ModelProto graph topology, value shapes/types, nodes, operator sets, and
TensorProto initializers. Names and `raw_data` borrow the caller's source bytes;
all allocating collections and length-delimited fields carry explicit bounds.

Evidence tier: exact synthetic ModelProto values, pointer-range proof that the
initializer payload aliases the source buffer, and typed negative tests for
truncation, absent GraphProto, document limits, and node limits. Package
nextest passes 3/3; warning-denied Clippy and Rustdoc are clean.

## Bounded exact streaming reads (2026-07-10)

RITK's MGH decoder needed a generic exact read that did not reserve a hostile
header length upfront. That I/O policy belongs in `consus-io`, not a medical
format or legacy image core. `read_exact_bounded` grows in fixed 64 KiB chunks
only as bytes are confirmed, returns `UnexpectedEof` with the received count,
and performs no source access for a zero-length request.

The ONNX provider integration exposed two downstream contract regressions:
the bounded reservation helper was absent and the exact reader excluded
`dyn Read`. Both contracts are restored at the provider boundary with exact
cap-law and trait-object regressions.

## Typed NPY/NPZ storage closure (2026-07-10)

`consus-npy` now owns bounded NPY header validation and typed NPZ archive I/O.
The public boundary is an owned shape plus boxed scalar payload, so consumers
can construct their native array provider without an ndarray compatibility
layer. Evidence tier: compile-time dtype binding and value-semantic round-trip
tests. Residual: structured/object dtypes and Fortran-order writing are rejected
explicitly; Fortran-order reads preserve the storage-order flag.
## Data Folder Record

- NWB sample acquisition manifest stored at `D:\consus\data\nwb\manifest.txt`
- Acquisition status: `acquisition_failed`
- Fixture data must remain under `D:\consus\data\nwb`
- Test guard recorded in `crates/consus-nwb/tests/roundtrip_proptest.rs` to fail until `D:\consus\data\nwb\allen_brain_observatory_sample.nwb` exists

## Audit Date: 2026-05-08 (updated this sprint)
## Scope: Phase 3 — NWB Verification Against Real Files (Milestone 38)

---

## P-004 — File-backed Parquet reader
- **Status**: RESOLVED (commit 4c9e018)
- **Gap**: No API to read values from a complete Parquet file in memory.
- **Fix**: Added `reader` module with `ColumnPageDecoder` (stateful page iterator, dict retention, v1/v2 decompression dispatch), `merge_column_values`, and `ParquetReader<'a>` (footer validation → metadata decode → dataset materialize → column chunk read).
- **Tests**: 21 value-semantic tests (max levels derivation, merge variants, page decoder v1/v2/dict, reader roundtrip, bounds checks).
- **Residual risk**: Nested (group) column decoding remains incomplete in the file-backed reader; writer-side nested/group lowering, footer/page synthesis, and row-source payload emission remain open.

---

## A-002: Arrow Array Materialization Bridge (resolved this sprint)

- **Status**: RESOLVED (this sprint)
- **Gap**: `ColumnValues` from `consus-parquet` could be decoded but not materialized into the canonical `ArrowArray` model in `consus-arrow`. The bridge layer (`ArrowBridge`, `ArrowIntegrationPlan`) was descriptive-only; no `ColumnValues → ArrowArray` conversion existed.
- **Fix**: Added `consus-arrow/src/array/materialize.rs` with `column_values_to_arrow(values: &ColumnValues) -> ArrowArray`. Physical-type mapping:
  - `Boolean` → `FixedWidth { element_width: 1 }`, 0x00/0x01 per element
  - `Int32/Int64/Float/Double` → `FixedWidth` with little-endian byte encoding (Arrow memory-format convention)
  - `Int96` → `FixedWidth { element_width: 12 }`, raw bytes preserved
  - `ByteArray` → `VariableWidth` with monotone offsets (`offsets.len() == len + 1`)
  - `FixedLenByteArray` → `FixedWidth { element_width: fixed_len }`, concatenated raw bytes
  - Re-exported from `array/mod.rs` and `consus-arrow` crate root under `#[cfg(feature = "alloc")]`.
  - Also resolved two pre-existing warnings: unused `ArrowField` import in `bridge/mod.rs`; duplicated `#[cfg(feature="alloc")] #[test]` attributes in `memory/mod.rs`.
- **Tests**: 10 value-semantic tests covering all 8 `ColumnValues` variants plus empty-array boundary cases for Boolean and ByteArray.
- **Residual risk**: Zero-copy materialization (reinterpret fixed-width slices without allocation) requires alignment guarantees not yet enforced; current implementation performs explicit byte-level conversion for portability. A future zero-copy path requires `bytemuck` or equivalent.

## P-007: Compressed page emission — writer silently ignores codec parameter (resolved this sprint)

- **Status**: RESOLVED (this sprint)
- **Gap**: `ParquetWriter::with_compression(codec)` accepted a codec but `build_file_bytes` ignored it (`_codec` unused). All emitted pages were UNCOMPRESSED regardless of requested codec. `ColumnMetadata.codec` was hardcoded to `0` (UNCOMPRESSED), causing any non-uncompressed write to produce a structurally incorrect file that readers would fail to decompress.
- **Fix**:
  - Added `compress_page_values(data: &[u8], codec: CompressionCodec) -> Result<Vec<u8>>` to `consus-parquet/src/encoding/compression.rs` (declared `pub(crate)`). Mirrors `decompress_page_values` with symmetric codec dispatch: UNCOMPRESSED pass-through, GZIP/ZLIB via `flate2::read::DeflateEncoder`, SNAPPY via `snap::raw::Encoder`, ZSTD via `zstd::bulk::compress`, LZ4_RAW via `lz4_flex::compress`, LZ4 via `lz4_flex::compress_prepend_size`, BROTLI always `UnsupportedFeature`. Disabled features return `UnsupportedFeature` with actionable enable-feature message.
  - Updated `build_file_bytes`: renamed `_codec` → `codec`; applies `compress_page_values` after PLAIN encoding; sets `page_header.uncompressed_page_size = plain_size`, `page_header.compressed_page_size = compressed_size`; sets `ColumnMetadata.codec = codec as i32`; records correct `total_uncompressed_size` and `total_compressed_size` (header + respective payload).
- **Tests**: `compress_page_values_uncompressed_passthrough` (always), `compress_page_values_brotli_returns_unsupported` (always), `writer_gzip_roundtrip_i32_three_values` (`#[cfg(feature="gzip")]`), `writer_gzip_roundtrip_byte_array` (`#[cfg(feature="gzip")]`).
- **Verification**: 177/177 pass default features; 183/183 pass with `--features gzip`.
- **Residual risk**: Multi-codec writer tests (SNAPPY, ZSTD, LZ4) require the respective feature flags. UNCOMPRESSED, GZIP are verified. Writer still emits a single row group; multi-row-group splitting remains open.

---

## M-037: NWB Write Path — NwbFileBuilder + validate_time_series_for_write (resolved this sprint)

- **Status**: Closed.
- **Gap**: `consus-nwb` had no write path. There was no way to construct an NWB 2.x file from Rust, write TimeSeries groups, or write a Units table. The read path (`NwbFile::open`, `time_series`, `list_time_series`) was complete but roundtrip authoring was blocked on missing write surface.
- **Resolution**:
  - `NwbFileBuilder::new(nwb_version, identifier, session_description, session_start_time)` — writes all five required NWB root attributes (`neurodata_type_def = "NWBFile"`, `nwb_version`, `identifier`, `session_description`, `session_start_time`) at construction time; rejects empty `identifier` or `session_description` with `InvalidFormat` before any HDF5 bytes are written.
  - `NwbFileBuilder::write_time_series(ts: &TimeSeries)` — calls `ts.validate()` and `validate_time_series_for_write` before any write; emits `{name}/data` (f64 LE) + `{name}/timestamps` (f64 LE) for timestamp-based TimeSeries, or `{name}/data` (f64 LE) + `{name}/starting_time` scalar dataset with `rate` f32 LE attribute for rate-based TimeSeries; attaches `neurodata_type_def = "TimeSeries"` to the group.
  - `NwbFileBuilder::write_units(spike_times: &[f64])` — emits `Units` group with `neurodata_type_def = "Units"` attribute; `Units/spike_times` dataset (f64 LE) with `neurodata_type_def = "VectorData"` and `description = "spike times"` attributes.
  - `NwbFileBuilder::finish() -> Result<Vec<u8>>` — delegates to `Hdf5FileBuilder::finish()`.
  - `validate_time_series_for_write(ts: &TimeSeries)` added to `consus_nwb::validation` — checks timing representation is present (timestamps or rate) and that rate > 0 when present.
  - `NwbFile::units_spike_times()` — new read method added to `NwbFile` for roundtrip verification; reads `Units/spike_times` via `read_f64_dataset`.
  - Module-level private helpers `fixed_string_bytes`, `f64_le_datatype`, `f32_le_datatype` added to `file/mod.rs`; shared between `NwbFileBuilder` impl and test helpers (SSOT, no duplication).
- **Tests added**:
  - `file::tests`: 12 value-semantic tests — `nwb_file_builder_minimal_file_opens_successfully`, `nwb_file_builder_empty_identifier_returns_error`, `nwb_file_builder_empty_session_description_returns_error`, `write_time_series_with_timestamps_roundtrip`, `write_time_series_with_rate_roundtrip`, `write_multiple_time_series_roundtrip`, `write_empty_time_series_with_timestamps_roundtrip`, `write_time_series_without_timing_returns_conformance_error`, `write_time_series_with_zero_rate_returns_error`, `write_time_series_with_negative_rate_returns_error`, `write_units_spike_times_roundtrip`, `write_units_empty_spike_times_roundtrip`.
  - `validation::tests`: 7 value-semantic tests — `validate_for_write_ok_with_timestamps`, `validate_for_write_ok_with_rate`, `validate_for_write_rejects_no_timing`, `validate_for_write_rejects_zero_rate`, `validate_for_write_rejects_negative_rate`, `validate_for_write_rejects_negative_inf_rate`, `validate_for_write_accepts_very_small_positive_rate`.
- **Verification**: `cargo test -p consus-nwb --lib` → 149/149; `cargo test --workspace` → 2219/2219; `cargo check --workspace` → 0 errors, 0 warnings.

---

## M-046: read_string_dataset VariableString support + NWB h5py fixture integration tests (resolved this sprint)

- **Crates affected**: `consus-nwb`
- **Gap**: `read_string_dataset` only supported `FixedString`; real NWB files use `VariableString` (HDF5 VL type) for dataset columns. Milestone 38/41 NWB fixture verification was blocked on external file acquisition.
- **Resolution**: Extended `read_string_dataset` with a `Datatype::VariableString` arm using `consus_hdf5::heap::resolve_vl_references`. Generated deterministic h5py fixture. Added 10-invariant integration test.
- **Verification**: `cargo test -p consus-nwb --test integration_real_file` → 1/1; `cargo test --workspace` → 0 failures.

---

## Open Gaps

_No open gaps._

---

## Risk Assessment

| Risk | Probability | Impact | Status |
|------|-------------|--------|--------|
| Partial selection write semantics across chunk boundaries | Low | Medium | Reduced by multidimensional value-semantic write tests covering contiguous, strided, and uninitialized-chunk update paths |
| Invalid chunk coordinates accepted as store keys | Low | Low | Reduced by chunk-grid validation in `read_chunk`/`write_chunk` and negative tests for out-of-grid coordinates |
| Python interoperability mismatch | Medium | High | Closed by Z-102: all Python-generated v3 fixtures pass including boundary chunks across padded and partial chunk layouts |
| Sharded v3 high-level API drift | Medium | Medium | Closed by Z-103: high-level read/write dispatches to spec-compliant shard implementation; verified by single-shard and multi-shard round-trip tests |
| Fill-value width mismatch for non-8-byte numeric types | Low | Medium | Closed by Z-107: float32 and float64 fill-value byte expansion now matches the target element width and is covered by dedicated tests plus array roundtrips |
| Store/backend divergence | Low | Medium | Reduced by in-memory, filesystem, and Python-generated fixture coverage; S3 interop remains indirect |
| netCDF HDF5 read coverage (dimension coordinate values and compact/virtual variable payloads not yet read) | Low | Medium | Reduced by N-001, N-002, N-005, N-006, and N-007: attributes, unlimited extents, contiguous/chunked variable payload reads, DIMENSION_LIST-based semantic dimension binding, and ancestor-scope dimension inheritance are now preserved; coordinate-value extraction plus compact/virtual payload extraction remain roadmap work under P2.3 |
| MATLAB .mat v7.3 completeness (non-scalar struct arrays, virtual-layout coverage) | Low | Low | Reduced further by M-001 Sprint 5: cell and struct group roundtrip tests added via extended HDF5 builder; model unit test coverage complete; virtual-layout fixture coverage remains blocked on virtual dataset HDF5 authoring surface; non-scalar struct shape preservation requires MATLAB_dims attribute authoring (roadmap) |
| Parquet datatype, dataset-model, and trailer-validation coverage | Low | Medium | Reduced by P-001: canonical Parquet mappings now cover all core datatype variants present in the crate, including compound, array, enum, varlen, and reference cases; validated dataset descriptor and ordered projection coverage enforce row-group chunk cardinality, schema-order field identity, total-row aggregation, nested-column classification, nested group → canonical `Compound` preservation, and repeated field → canonical `VarLen` preservation; trailer validation now enforces `PAR1` magic, little-endian footer-length decoding, footer-offset bounds, non-overlapping row-group/column-chunk byte ranges, and rejection of row groups extending into the footer payload |
| Arrow nested-type field loss on conversion | Low | Medium | Closed by A-001: Compound/Array/Complex → Arrow Struct/List now preserves recursive field structure; Struct/Map/Union → Compound now preserves child fields |
| FITS binary table column type mapping | Medium | Medium | Closed by F-001: TFORM format codes now map to canonical Datatype for all 13 FITS Standard 4.0 binary table column types |
| FITS column descriptor datatype integration | Medium | Medium | Closed by F-002: FitsTableColumn now carries canonical Datatype and byte_width derived from TFORM; binary table NAXIS1 validation enforced |
| HDF5 datatype class mapping coverage | Medium | Medium | Closed by H-001: all 10 HDF5 datatype classes now have public mapping functions to canonical Datatype |

---

## Atlas gate audit 2026-08-12 (ATLAS-FOUNDATION-PLANNING-002)

Canonical Atlas engineering gates run against this checkout by the foundation
audit. This section is the provider-local record; the Atlas root `backlog.md`
owns the cross-repository matrix.

### Verified green

- `cargo check` (default): pass.
- `cargo check --all-features`: pass.
- Doctests (`--all-features`): pass.
- `consus-arrow` and `consus-parquet` no-default cfg closure is complete under
  `CONSUS-NODEF-ARROW-PARQUET-002`; default and no-default package gates are
  green. The remaining workspace debt is outside these two providers.
- Clippy lint fixes landed: `consus-nwb/src/validation/report.rs` redundant
  borrow; `consus-hdmf/tests/integration.rs` `approx_constant` (2 sites).
- `consus-hdf5` root re-exports added for the `Hdf5File` and `Hdf5FileBuilder`
  facades (mirroring the `consus-fits` facade pattern).

### Verified green — CONSUS-NODEF-GATE-001 (--no-default-features cfg debt)

The previous residual was stale: the current `consus-fits` package no-default
check passes, and the next reproducible failure was the unconditional
`consus-arrow` `datatype` re-export at `src/lib.rs:74`. Commit `fa314cb` gates
that export and fixes the no-alloc test import. A standalone provider checkout
now passes locked workspace check and warning-denied Clippy with and without
default features; no-default Nextest passes `2031/2031`, default Nextest passes
`2553/2553`, and locked workspace doctests pass. All 17 workspace packages pass
isolated locked Rustdoc. The aggregate Windows workspace Rustdoc command timed
out twice after those package-level passes, so the remaining local residual is
workspace documentation orchestration rather than a package documentation
failure. Provider CI and Documentation jobs now have explicit timeouts, and
Documentation enforces `RUSTDOCFLAGS=-Dwarnings`.

The first exact hosted Documentation run at `57a4e66` (`32017157627`) then
failed on four genuine links: the optional `zerocopy` dependency in Arrow, a
module-private Zarr helper, and two unqualified Zarr `ParseBudget` references.
The Arrow reference is now code prose, the private helper is code prose, and
the two public references use `consus_core::ParseBudget`. Local warning-denied
Rustdoc passes for `consus-arrow` and `consus-zarr`; the replacement hosted
Documentation result remains open.

Replacement Documentation run `32017590806` at `22294b5` found two more
unresolved links that the first run did not reach: `NwbFile` from the NWB
validation module and `ParseBudget` in MAT. They now use
`crate::file::NwbFile::validate_conformance` and
`consus_core::ParseBudget`, respectively. Local warning-denied Rustdoc passes
for both affected packages; the next hosted Documentation result remains open.

Documentation run `32017799064` at `0b5505a` found three further Parquet
links: two unqualified `ParseBudget` references and the `value[i]` prose index
expression. They now use `consus_core::ParseBudget` and inline code for the
expression. They now use `consus_core::ParseBudget` and inline code for the
expression. Local warning-denied Parquet Rustdoc passes.

Final exact-head evidence: CI `32017963837` passed all 80 jobs at `65a7b28`;
Documentation `32017963800` and Pages deployment `32017962556` also passed at
that head. The Atlas development overlay still reports unused local patches
and requests a `Cargo.lock` rewrite before compilation; it remains an
environment-boundary note, not a provider source or hosted-gate residual.

The Atlas umbrella still stops the locked invocation before compilation because
the development overlay reports unused local patches and requests a
`Cargo.lock` rewrite. No lockfile change was kept. Collect the exact hosted CI
and Documentation results from the pushed provider head before closing this
item; the umbrella-overlay failure is an environment boundary, not source
evidence.

### Open — CONSUS-TEST-API-001 (integration-test aspirational I/O API)

`--all-targets` clippy/nextest fail on four test targets whose tests target an
aspirational I/O facade that does not exist in the provider:

- `consus-integration-tests` `tests/cross_format_interop.rs` (45 errors): uses
  `MemCursor::new(buffer)` (real API takes 0 args), `Hdf5FileBuilder::new()`
  (real API requires `FileCreationProps`), `build_writer()` (absent), and the
  absent `consus_zarr::{ArrayMetadataV3, ZarrArray}` and `consus_netcdf::NcFile`
  facades.
- `consus-integration-tests` `tests/property_integration.rs` (2 errors): same
  `build_writer`/cursor assumptions.
- `consus-nwb` lib test (2 errors) and `tests/integration_real_file.rs`
  (5 errors): `MemCursor::new(buffer)`, `Hdf5FileBuilder::new()` arg, and
  `as_deref()`-on-`Option<&str>` redundant deref sites.

Closure is a test-API reconciliation slice that maps the tests onto the real
`Hdf5FileBuilder` (in-memory byte emission), `MemCursor`, and either adds the
`ZarrArray`/`ArrayMetadataV3`/`NcFile` facades (the chunk/`read_model` building
blocks exist) or rewrites the tests against the existing chunk API.
