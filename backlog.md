# Consus — Backlog

<a id="consus-stable-qualification"></a>
## CONSUS-STABLE-QUALIFICATION — consus stable-release qualification [patch] — todo
- priority: verification; milestone: M-interop-contracts; basis: `832899c5dbef`.
- Outcome: Close the HDF5 handoff to Helios using the published reader. Follow the [release scope](docs/adr/0008-release-scope-1-0.md) and [milestone exits](backlog/milestones.md). Planning targets for `consus-hdf5` are 0.6.1, then 0.7.0 if interop additions are needed, then 1.0.0. Verification alone can advance the current version.
- Acceptance and sequence: [stable-release checklist](checklist.md#stable-release-checklist).

Risks that no item fixes are recorded in `gap_audit.md`.

<a id="consus-hdf5-exact-ieee"></a>
## CONSUS-HDF5-EXACT-IEEE — Publish exact dataset admission [minor] — todo

- milestone: M-interop-contracts
- priority: correctness
- scope: `consus-hdf5` 0.6.0 only. Verify the published representation API and
  archive identity before the Helios cutover.

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

<a id="consus-ladder-failing-exits"></a>
## CONSUS-LADDER-FAILING-EXITS — Give each milestone a failing exit check [patch] — todo

- milestone: M-core-formats
- outcome: every open block of `backlog/milestones.md` has one failing exit check at the fetched
  default commit (roadmap rule); no exit names an item ID.
- evidence: no block has one at `6f850682`; basis adds only #206 and #208. M-1.0's passes (CI run 43776070564 at `6f850682`, job
  133002546532: 14 passed with and without `PYTHON_GIL=0`); the others are whole-package `cargo test -p`
  runs that the same run's nextest jobs cover green. Only the standing "no open correctness item"
  holds the ladder, via blocked `ATLAS-PR530-PROVIDER-COHERENCE` (consus#208 round-2 judge, F1).
  #208 removed the item-ID exits; at basis only `docs/adr/0008-release-scope-1-0.md:40` names
  `CONSUS-PARQUET-RAW-DEFLATE` and `CONSUS-PYTHON-FREE-THREADING`.
- acceptance: each block's exits run at a fetched default commit leave one failing (0 of 3 at `6f850682`),
  or ADR 0008 records the ladder delivered pending the correctness item.
- priority: verification; needs: none; basis: 21769557a13a64530ba5f07cae0227c617c36f93
- scope: backlog/milestones.md, docs/adr/0008-release-scope-1-0.md
- next step: name what `ATLAS-PR530-PROVIDER-COHERENCE` delivers as an exactly named check, or close
  the passing blocks; either needs an ADR 0008 revision note and a judge verdict.

<a id="consus-release-job-red"></a>
## CONSUS-RELEASE-JOB-RED — Pending publish gets a 403 for consus-raster [patch] — blocked

- milestone: M-core-formats
- outcome: `publish-pending` in `rust-release.yml` publishes `consus-raster` and `consus-zarr` 0.4.1
  and the default-branch run is green.
- evidence: six runs read (42973982746, 43497877937, 43523622004, 43685555434, 43770175761,
  43776070912; 2026-11-11 to 2026-11-15) package both crates, then fail at the `consus-raster`
  upload: 403 "access token is not valid for crate". Plan and SemVer jobs pass; registry max of both
  crates is 0.3.0. Derived cause (OIDC-only publishing): no crates.io trusted-publisher entry for
  `consus-raster` bound to this workflow; unverifiable without owner access. Risk: `gap_audit.md`.
- acceptance: a `rust-release.yml` run on the default branch succeeds and the registry max of
  `consus-raster` and `consus-zarr` is 0.4.1; both fail at basis.
- priority: verification; needs: none; basis: 21769557a13a64530ba5f07cae0227c617c36f93
- scope: crates.io trusted-publisher settings (registry owner), `.github/workflows/rust-release.yml`
- blocked: registry owner. Question: add crates.io trusted-publisher entries for `consus-raster` and
  `consus-zarr` (repository `ryancinsight/consus`, workflow `rust-release.yml`)? Re-open on the
  answer or when an upload no longer returns 403. No repository change cures it.

<a id="consus-npy-error-variants"></a>
## CONSUS-NPY-ERROR-VARIANTS — Type the NPY header rejections [major] — todo

- milestone: M-interop-contracts
- outcome: `consus_npy::Error` distinguishes header failures by variant (unsupported
  version, header over the cap, header encoding, dictionary key/value/shape syntax,
  shape overflow) instead of `InvalidFormat(String)`; the enum is `#[non_exhaustive]`.
- acceptance: `crates/consus-npy/tests` match each rejection by variant and field, and
  `cargo semver-checks` against 0.4.1 records the break under the declared [major] class.
- priority: tightening
- scope: crates/consus-npy/src/error.rs, crates/consus-npy/src/format.rs,
  crates/consus-npy/src/format/header.rs, crates/consus-npy/src/npz.rs, crates/consus-npy/tests
- next step: replace the 16 header `invalid(..)` sites in `header.rs` and the `format.rs`
  version/length/encoding sites; migration note in CHANGELOG under `### Breaking`.

<a id="consus-npy-literal-conformance"></a>
## CONSUS-NPY-LITERAL-CONFORMANCE — Match NumPy on Python-literal headers [patch] — todo

- milestone: M-interop-contracts
- outcome: the header lexer agrees with `numpy.load` 2.5.2, serving the exit check "verify
  document and interchange crates". Consus rejects, and NumPy accepts, `((2,))`, `(-(0), 2)`, `(False)`, `('<i4')`,
  parenthesized keys, octal and `\N{..}` escapes in `descr`, and the `2L` suffix in 1.0/2.0.
  Consus accepts, and NumPy rejects: NUL in a comment, a leading newline then indentation, a
  trailing tab, and a header over 10,000 characters.
- acceptance: one differential case per listed header against the NumPy result.
- priority: feature
- scope: crates/consus-npy/src/format.rs, crates/consus-npy/src/format/header.rs, crates/consus-npy/tests
- next step: pin the cases, change the lexer, derive `MAX_HEADER_BYTES`, delete the CR members
  unreachable after `physical_lines`, and read the header through `consus_io::read_exact_bounded`.

<a id="consus-npy-test-harness"></a>
## CONSUS-NPY-TEST-HARNESS — Pin NPY rejection reasons in one harness [patch] — todo

- milestone: M-interop-contracts
- outcome: the rejection loops in `header_contract/main.rs` and `literal_compatibility.rs` assert
  the reason, not only `InvalidFormat` with a non-empty message, and the two integration
  binaries become modules of one harness (`tests/npy/main.rs`).
- acceptance: `cargo nextest list -p consus-npy` names the same tests before and after,
  and each former non-empty-message loop fails under a mutated rejection reason.
- priority: verification
- needs: CONSUS-NPY-ERROR-VARIANTS
- scope: crates/consus-npy/tests
- next step: after the variants land, replace message matching with variant matching.

<a id="consus-zarr-edge-chunk-padding"></a>
## CONSUS-ZARR-EDGE-CHUNK-PADDING — Write edge chunks at the full chunk shape [patch] — todo

- milestone: M-core-formats
- outcome: `write_array` stores every chunk, edge chunks included, at the full `chunks` shape padded
  with the fill value, as `write_array_selection` does and zarr-python does (`chunk/ops/coords.rs:93-95`).
- evidence: at `31ee569f`, `write_array` sizes each buffer by the clipped extent
  (`crates/consus-zarr/src/chunk/ops/arrays.rs:374-389`) while `write_array_selection` pads
  (`arrays.rs:290-294`). The consus#219 judge reported zarr-python failing on such a store ("cannot
  reshape array of size 8 into shape (4,4)"); recovered from the verdict, not re-run here.
- acceptance: `write_array` of shape `[5, 5]`, chunks `[4, 4]`, dtype `|u1`, `fill_value` 7: chunks (0,1), (1,0)
  and (1,1) are each 16 bytes (4, 4 and 1 today), each in-bounds cell equals its input cell, every padded cell is 7.
- priority: correctness; needs: none; basis: 31ee569f6019c932e4fe074b165a0d6f1447682d
- scope: crates/consus-zarr/src/chunk/ops/arrays.rs, crates/consus-zarr/tests, docs/book/formats.md
- next step: pad in `write_array` to the full chunk shape, as `write_array_selection` does.

<a id="consus-zlib-decode-ceiling"></a>
## CONSUS-ZLIB-DECODE-CEILING — One rule for a declared size above 64 MiB [patch] — todo

- milestone: M-core-formats
- outcome: zlib, gzip and raw-DEFLATE decoding give one result for a declared `expected_size` above 64 MiB, and
  `codec/traits.rs` qualifies its claim that `decompress` inverts `compress` exactly.
- evidence: at `31ee569f`, `ZlibFraming::ceiling` ignores `expected_size` (`crates/consus-compression/src/codec/flate.rs:53-55`),
  gzip and raw DEFLATE take `expected_size.max(64 MiB)` (`flate.rs:74-76`, `91-93`): a 64 MiB + 1 declaration is
  rejected by zlib only (derived from `budget.rs:98-105`, `262-269`; not run). `deflate.rs:75-85` and `gzip.rs:251-261`
  pin only the `usize::MAX` rejection; `traits.rs:5,34` claim exact inversion.
- acceptance: one test gives a small valid stream and `expected_size` 64 MiB + 1 to `ZlibCodec`, `GzipCodec` and `RawDeflateCodec`: all return the input or all `ResourceLimit` (they differ today);
  `git grep -n "can invert exactly" crates/consus-compression/src/codec/traits.rs` prints nothing (two lines today).
- priority: correctness; needs: none; basis: 31ee569f6019c932e4fe074b165a0d6f1447682d
- scope: crates/consus-compression/src/codec, crates/consus-compression/tests
- next step: judgment call: widen the zlib ceiling to the declaration, or hold 64 MiB for all three and document it.

<a id="consus-book-compression-features"></a>
## CONSUS-BOOK-COMPRESSION-FEATURES — Match the codec table to the crate features [patch] — todo

- milestone: M-core-formats
- outcome: the `formats.md` codec table names, per registered codec, the one feature that gates it; no document claims BZip2.
- evidence: at `31ee569f`, `docs/book/formats.md:51` reads `DEFLATE (zlib/gzip) | deflate (default)`;
  `registry/default.rs:56-101` gates DEFLATE (name `deflate`, zlib framing) on `deflate`, gzip and zlib on `gzip`
  (and `std`, `codec/mod.rs:18-22`), Zstandard, LZ4, SZIP, Blosc on `zstd`, `lz4`, `szip`, `blosc`. `formats.md:55`
  lists a `bzip2` feature `Cargo.toml` lacks; `docs/book/stack_position.md:11` lists BZip2.
- acceptance: `grep -cE '^\| (gzip|zlib) \| .gzip.' docs/book/formats.md` prints 2 and
  `grep -cE '^\| DEFLATE [^|]*\| .deflate. \|$' docs/book/formats.md` prints 1 (0 and 0 today); and
  `git grep -n -i bzip2 -- docs README.md 'crates/*/README.md'` prints nothing (two lines today).
- priority: verification; needs: none; basis: 31ee569f6019c932e4fe074b165a0d6f1447682d
- scope: docs/book/formats.md, docs/book/stack_position.md
- next step: rewrite the table rows from `registry/default.rs` and the `[features]` table of `crates/consus-compression/Cargo.toml`.

<a id="consus-zlib-doc-drift"></a>
## CONSUS-ZLIB-DOC-DRIFT — Correct three statements about the zlib codec [patch] — todo

- milestone: M-core-formats
- outcome: the book, a test comment and the error labels say what the zlib, gzip and raw-DEFLATE codecs do.
- evidence: at `31ee569f`, `docs/book/formats.md:100` says a gzip-framed store "is rewritten to be readable by
  numcodecs", yet no code in `crates/consus-zarr/src` rewrites a store; `codec/zlib.rs:163-164` says level 6 output "is the
  CPython bytes" while its test (`zlib.rs:165-171`) compares two header bytes; `ZlibFraming::LABEL` is `deflate (zlib)`
  and `RawDeflateFraming::LABEL` is `deflate` (`flate.rs:45,83`), though the codec named `deflate` is the zlib-framed one.
- acceptance: `git grep -n -E "is rewritten to be readable|Level 6 output is the CPython bytes" -- docs crates` prints
  nothing (two lines today); a test asserts the malformed-stream error messages of `ZlibCodec` and `RawDeflateCodec`
  start with `zlib` and `raw-deflate` (`deflate (zlib)` and `deflate` today).
- priority: verification; needs: none; basis: 31ee569f6019c932e4fe074b165a0d6f1447682d
- scope: docs/book/formats.md, crates/consus-compression/src/codec
- next step: rewrite the sentence and the comment; set the three labels to `zlib`, `gzip` and `raw-deflate`.

<a id="consus-zarr-level-review"></a>
## CONSUS-ZARR-LEVEL-REVIEW — Review the Zarr level fix and commit its independent decode [patch] — todo

- milestone: M-core-formats
- outcome: commits 48cdb05 and c4d053f (Zarr `zlib` and `gzip` levels) get an independent review, and the manual
  CPython decode of their output becomes a committed check.
- evidence: at `31ee569f`, `consus-zarr/src/codec/mod.rs:340-353` reads `level` for both codecs and
  `tests/unit_zlib_codec.rs:105-161` compares stored chunks with the direct encoder at levels 0 and 9; both decode only
  through Consus. The CPython decode of the four explicit-level chunks to 29,696 bytes was a ticked manual line.
- acceptance: `python -m pytest -q scripts/zarr_level_interop.py` passes a test that decodes the `zlib` and `gzip` chunks at
  levels 0 and 9 with CPython to the 29,696-byte input (the file does not exist today).
- priority: verification; needs: none; basis: 31ee569f6019c932e4fe074b165a0d6f1447682d
- scope: crates/consus-zarr/tests, scripts
- next step: review the two commits against the level oracle; have the Rust test write the chunks and `scripts/` decode them.

<a id="consus-flate-surface"></a>
## CONSUS-FLATE-SURFACE — Narrow the unused flate items [major] — todo

- milestone: M-interop-contracts
- outcome: `zlib_compress`, `ZlibFraming` and `GzipFraming` are `pub(crate)`; the public surface of
  `codec::flate` is what `consus-zip` consumes.
- evidence: at `31ee569f`, `crates/consus-compression/src/codec/flate.rs:42,63,139` are `pub`; only
  sibling codec modules use them, and `consus-zip` uses `RawDeflateFraming` and `flate2_decompress`
  (`crates/consus-zip/src/read.rs:21`). No other stack member's tracked `*.rs` names them.
- acceptance: `cargo semver-checks -p consus-compression` against the 0.5.0 baseline reports exactly the
  three removals under the declared [major] class, and `cargo check --workspace --locked --all-features`
  passes.
- priority: tightening; needs: CONSUS-ZLIB-DECODE-CEILING; basis: 31ee569f6019c932e4fe074b165a0d6f1447682d
- scope: crates/consus-compression/src/codec/flate.rs, CHANGELOG.md
- next step: narrow the three items after the ceiling change lands; note the removal under `### Breaking`.

<a id="consus-zarr-test-harness"></a>
## CONSUS-ZARR-TEST-HARNESS — Fold the Zarr integration tests into one harness [patch] — todo

- milestone: M-core-formats
- outcome: the eight `consus-zarr` integration binaries become modules of one harness (`tests/zarr/main.rs`); the
  CPython zlib and gzip vectors stay only in `consus-compression/src/codec/zlib.rs`, with their tests.
- evidence: at `31ee569f`, `crates/consus-zarr/tests` has eight `*.rs` binaries; vectors sit in `unit_zlib_codec.rs:20-31`
  and `zlib.rs:99-110`. A zarr test cannot reach a `#[cfg(test)]` constant, so it asserts against the public encoder
  (`default_registry().get_by_name("zlib")`) instead of a second literal.
- acceptance: `ls crates/consus-zarr/tests/*.rs | wc -l` prints 0 or 1 (8 today); `git grep -n -E "CPYTHON_|0x78, ?0x9c|[\]x9c" -- crates/consus-zarr`
  prints nothing (8 lines today); `cargo nextest list -p consus-compression` still lists `decodes_cpython_zlib_stream`;
  `cargo nextest list -p consus-zarr` is unchanged after rewriting `consus-zarr::zarr <binary>::<test>` to `consus-zarr::<binary> <test>`.
- priority: tightening; needs: CONSUS-ZARR-EDGE-CHUNK-PADDING; basis: 31ee569f6019c932e4fe074b165a0d6f1447682d
- scope: crates/consus-zarr/tests
- next step: move the files under `tests/zarr/`; the two tests that decoded the literals may rename, listed in the body.

<a id="consus-parquet-payload-count"></a>
## CONSUS-PARQUET-PAYLOAD-COUNT — Carry the checked value count in the page payload [major] — todo

- milestone: M-interop-contracts
- outcome: `PagePayload::num_values` is the `usize` the splitters already validated, so the reader
  converts no count.
- evidence: at `31ee569f`, `split_data_page_v1` and `split_data_page_v2` reject negative counts
  (`crates/consus-parquet/src/wire/payload.rs:160,209`), yet `reader/page.rs:281,374` cast the `i32` with
  `as usize` and `page.rs:439-441` holds an error branch no input reaches (both V2 call sites split first,
  `page.rs:301,393`).
- acceptance: a `consus-parquet` test with `let _: usize = payload.num_values;` on a `split_data_page_v2` result compiles (E0308 today);
  `git grep -n -E "payload.num_values as usize|DataPageV2 num_values must be" -- crates/consus-parquet/src` prints nothing (three lines today).
- priority: tightening; needs: none; basis: 31ee569f6019c932e4fe074b165a0d6f1447682d
- scope: crates/consus-parquet/src/wire/payload.rs, crates/consus-parquet/src/reader/page.rs
- next step: store `count` in the payload; the type change is public, so record it under `### Breaking`.
