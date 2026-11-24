# Consus — Backlog

<a id="consus-stable-qualification"></a>
## CONSUS-STABLE-QUALIFICATION — consus stable-release qualification [patch] — todo
- priority: verification; milestone: M-interop-contracts; basis: `832899c5dbef`.
- Outcome: Close the HDF5 handoff to Helios using the published reader. Follow the [release scope](docs/adr/0008-release-scope-1-0.md) and [milestone exits](backlog/milestones.md). Planning targets for `consus-hdf5` are 0.6.1, then 0.7.0 if interop additions are needed, then 1.0.0. Verification alone can advance the current version.
- Acceptance and sequence: [stable-release checklist](checklist.md#stable-release-checklist).

Risks that no item fixes are recorded in `gap_audit.md`.

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

<a id="consus-zarr-edge-chunk-padding"></a>
## CONSUS-ZARR-EDGE-CHUNK-PADDING — Write edge chunks at the full chunk shape [patch] — todo
- outcome: `write_array` stores every chunk, edge chunks included, at the full `chunks` shape padded with the
  fill value, as `write_array_selection` and zarr-python do (`consus-zarr/src/chunk/ops/coords.rs:93-95`).
- evidence: `write_array` sizes each buffer by the clipped extent (`chunk/ops/arrays.rs:374-389`);
  `write_array_selection` pads (`arrays.rs:290-294`). The consus#219 judge reported zarr-python failing on
  such a store ("cannot reshape array of size 8 into shape (4,4)"); recovered from the verdict, not re-run.
- acceptance: `write_array` of shape `[5, 5]`, chunks `[4, 4]`, `|u1`, `fill_value` 7 stores chunks (0,1),
  (1,0) and (1,1) of 16 bytes each (4, 4 and 1 today); in-bounds cells equal the input and padded cells are 7.
- priority: correctness; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: crates/consus-zarr/src/chunk/ops/arrays.rs, crates/consus-zarr/tests, docs/book/formats.md
- next step: pad in `write_array` to the full chunk shape, as `write_array_selection` does.

<a id="consus-zlib-decode-ceiling"></a>
## CONSUS-ZLIB-DECODE-CEILING — One rule for a declared size above 64 MiB [patch] — todo
- outcome: `DeflateCodec`, `ZlibCodec`, `GzipCodec` and `RawDeflateCodec` give one result for a declared
  `expected_size` above 64 MiB: all decode the stream, or all return `ResourceLimit`.
- evidence: `ZlibFraming::ceiling` ignores `expected_size` (`consus-compression/src/codec/flate.rs:53-55`);
  gzip and raw DEFLATE take `expected_size.max(64 MiB)` (`flate.rs:74-76`, `91-93`); `DeflateCodec`
  (`deflate.rs:42`) and `ZlibCodec` use zlib framing, so 64 MiB + 1 fails there only (derived from
  `consus-core` `parse/budget.rs:98-105`; not run).
- acceptance: `cargo nextest run -p consus-compression --all-features above_default_budget` passes two tests
  (none today): the three framings' `ceiling(64 MiB + 1, ..)` are equal (clamping the capacity hint,
  `flate.rs:98`, moves no ceiling), and each codec decodes its own level-1 `compress` output of 64 MiB + 1
  zero bytes with four equal results (its run time against the 30 s test budget is unverified).
- priority: correctness; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: crates/consus-compression/src/codec, crates/consus-compression/tests
- next step: widen the zlib ceiling to the declaration, or hold 64 MiB for all four (a judgment call).

<a id="consus-py-zarr-compressor-name"></a>
## CONSUS-PY-ZARR-COMPRESSOR-NAME — Reject compressor names the binding cannot honor [patch] — todo
- outcome: `consus.ZarrArray(.., compressor=name)` raises `ValueError` for any name but `none`, `None` and
  `gzip` (its docstring, `zarr.rs:75-76`, lists `none` and `gzip`), instead of storing `"compressor": null`
  for `zlib` or a misspelling.
- evidence: `crates/consus-python/src/zarr.rs:32-36` maps `none`/`None` to null and `gzip` to level 6; the `_`
  arm maps every other string to null, so a requested compressor is dropped without a signal (derived; not
  run).
- acceptance: `python -m pytest -q scripts/compare_zarr.py -k compressor_name` passes a test that `zlib`,
  `zstd` and `bogus` raise `ValueError` (each constructs today), `none` stores null and `gzip` stores level 6.
- priority: correctness; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: crates/consus-python/src/zarr.rs, scripts/compare_zarr.py
- next step: return `PyValueError` from `PyZarrArray::new` for other names; supporting `zlib` is a separate
  [minor].

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

<a id="consus-book-compression-features"></a>
## CONSUS-BOOK-COMPRESSION-FEATURES — Match the codec table to the crate features [patch] — todo
- outcome: the `formats.md` codec table names, per registered codec, its one gating feature; no document
  claims BZip2.
- evidence: `formats.md:51` reads `DEFLATE (zlib/gzip) | deflate (default)`. In `consus-compression`
  `registry/default.rs`, `deflate` gates the codec `deflate` (zlib framing, `56-62`), `gzip` gates `gzip` and
  `zlib` (`64-72`, plus `std`, `codec/mod.rs:18-22`); `formats.md:55` lists a `bzip2` feature `Cargo.toml`
  lacks and `stack_position.md:11` lists BZip2.
- acceptance: `grep -cE '^[|] (gzip|zlib) [|] .gzip.|^[|] deflate [|] .deflate.' docs/book/formats.md` prints
  3 and `sed -n '/^| Codec |/,/^| Zstandard /p' docs/book/formats.md | wc -l` prints 6, so no other row
  precedes Zstandard (0 and 4 today); `git grep -n -i bzip2 -- docs README.md 'crates/*/README.md'` prints
  nothing (two lines today).
- priority: verification; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: docs/book/formats.md, docs/book/stack_position.md
- next step: rewrite the rows from `registry/default.rs` and the `[features]` table of `consus-compression`.

<a id="consus-codec-contract-doc"></a>
## CONSUS-CODEC-CONTRACT-DOC — Qualify the exact-inversion claim of the codecs [patch] — todo
- outcome: the codec docs promise that `decompress` returns the input of `compress` only for a declared size
  within the decode ceiling; none promises exact inversion without that bound.
- evidence: four statements say `decompress` inverts `compress` exactly
  (`consus-compression/src/codec/traits.rs:5` and `:34`, `gzip.rs:47`, `zlib.rs:52`), while a declared size
  above 64 MiB is rejected by some codecs.
- acceptance: `git grep -n -E 'inverts? (it )?exactly' -- crates/consus-compression/src/codec` prints nothing
  (four lines today); `git grep -c 'within the decode ceiling' -- crates/consus-compression/src/codec` prints
  a line for each of `traits.rs`, `gzip.rs` and `zlib.rs` (none today).
- priority: verification; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- needs: CONSUS-ZLIB-DECODE-CEILING
- scope: crates/consus-compression/src/codec
- next step: after the ceiling rule lands, reword the four statements and state the ceiling once in the module
  docs.

<a id="consus-zlib-doc-drift"></a>
## CONSUS-ZLIB-DOC-DRIFT — Correct two false statements about the zlib codec [patch] — todo
- outcome: the book and a test comment say what Consus does for gzip-framed `zlib` chunks and for level 6
  output.
- evidence: `docs/book/formats.md:100` says a gzip-framed store "is rewritten to be readable by numcodecs"; no
  code in `consus-zarr/src` rewrites a store. `consus-compression/src/codec/zlib.rs:163-164` says level 6
  output "is the CPython bytes", while its test (`zlib.rs:165-171`) compares two header bytes.
- acceptance: `grep -ci rewritten docs/book/formats.md` prints 0,
  `grep -c 'does not rewrite' docs/book/formats.md` prints 1,
  `grep -c 'CPython bytes' crates/*/src/codec/zlib.rs` prints 0 and
  `grep -c 'CPython header' crates/*/src/codec/zlib.rs` prints 1 (1, 0, 1 and 0 today).
- priority: verification; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: docs/book/formats.md, crates/consus-compression/src/codec/zlib.rs
- next step: say Consus decodes such chunks but does not rewrite the store, without the word `rewritten`; say
  level 6 output starts with the CPython header.

<a id="consus-compression-error-labels"></a>
## CONSUS-COMPRESSION-ERROR-LABELS — Name the failing framing in flate errors [patch] — todo
- outcome: a malformed zlib, gzip or raw-DEFLATE stream returns `CompressionError { message }` whose `message`
  starts `zlib`, `gzip` or `raw-deflate`, the framing that failed.
- evidence: `FlateFraming::LABEL` is `deflate (zlib)` for zlib and `deflate` for raw DEFLATE
  (`flate.rs:45,83`), though the codec named `deflate` is the zlib-framed one; it prefixes decode and compress
  errors (`flate.rs:123,146`) and `Display` renders `compression error: {message}` (`core/error.rs:189`).
- acceptance: `cargo nextest run -p consus-compression --all-features framing_label_prefixes_error` passes (no
  such test today): the `message` of a malformed stream starts `zlib` for `ZlibCodec` and `DeflateCodec`,
  `gzip` for `GzipCodec` and `raw-deflate` for `RawDeflateCodec`; `git grep -c -E
  'LABEL.*"(zlib|gzip|raw-deflate)";' -- '*/codec/flate.rs'` reports 3 matches (1 today) and
  `git grep -n 'LABEL.*"deflate' -- '*/codec/flate.rs'` prints nothing (two lines today).
- priority: verification; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: crates/consus-compression/src/codec/flate.rs, crates/consus-compression/tests
- next step: set the three labels to `zlib`, `gzip` and `raw-deflate`; test the message prefix of each codec.

<a id="consus-zarr-level-interop"></a>
## CONSUS-ZARR-LEVEL-INTEROP — Decode Consus-written level chunks with CPython [patch] — todo
- outcome: the `zlib` and `gzip` chunks `write_array` writes at levels 0 and 9 are committed files that a Rust
  test asserts byte for byte and CI decodes with CPython to the 29,696-byte input.
- evidence: `codec/mod.rs:340-353` reads `level`; `unit_zlib_codec.rs:105-161` decodes the chunks through
  Consus only; CPython decoded them once by hand (input `b"configured compression level " * 1024`).
- acceptance: `python -m pytest -q scripts/zarr_level_interop.py` passes 4 decodes;
  `git ls-files crates/consus-zarr/tests/data/level_chunks` lists the 4 files; `git grep -n -E
  '[.]compress|compressobj|GzipFile' -- scripts/zarr_level_interop.py` prints nothing;
  `git grep -c zarr_level_interop -- .github/workflows/ci.yml` reports a match; `cargo nextest run -p
  consus-zarr --all-features written_level_chunks_equal_committed_files` passes and `git grep -c 'fn
  written_level_chunks_equal_committed_files' -- 'crates/consus-zarr/tests/*.rs'` reports 1 (none today).
- priority: verification; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: crates/consus-zarr/tests, scripts/zarr_level_interop.py, .github/workflows/ci.yml
- next step: assert each `write_array` chunk equals its file in `unit_zlib_codec.rs`.

<a id="consus-adr-0011-accepted"></a>
## CONSUS-ADR-0011-ACCEPTED — Mark the published exact-IEEE ADR Accepted [patch] — todo
- outcome: ADR 0011 reads Accepted as built at the published `consus-hdf5` 0.6.0 revision, b14cbd3.
- evidence: `docs/adr/0011-exact-ieee-dataset-admission.md` is `Status: Proposed` at tag
  `crate-consus-hdf5-v0.6.0` and on the default branch, and calls itself an "additive public API proposal";
  Helios (b7bda4cd62296a3b11869f03405cab7e6d62dffa) calls `dataset_at_with_ieee_float_encoding` at
  `crates/helios-domain/src/storage.rs:154`.
- acceptance: `grep -c '^Status: Accepted$' docs/adr/0011-*.md` prints 1, `grep -c b14cbd3 docs/adr/0011-*.md`
  prints 1 or more and `grep -c 'API proposal' docs/adr/0011-*.md` prints 0 (0, 0 and 1 today);
  `grep -c '^| .0011.*| Accepted |$' docs/adr/README.md` prints 1 (0 today); `python3
  _atlas/scripts/adr-index.py check --directory docs/adr --strict` exits 0 from the repository root with Atlas
  checked out at `_atlas`.
- priority: verification; milestone: M-interop-contracts; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: docs/adr/0011-exact-ieee-dataset-admission.md, docs/adr/README.md
- next step: set the status, add a dated note citing b14cbd3, reword the proposal line, regenerate the index.

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

<a id="consus-zlib-vector-dedup"></a>
## CONSUS-ZLIB-VECTOR-DEDUP — Keep the CPython zlib and gzip vectors in one crate [patch] — todo
- outcome: the CPython `zlib` and `gzip` vectors live in `zlib.rs` only; the Zarr tests in (3) use `compress`.
- evidence: the vectors sit in `crates/consus-zarr/tests/unit_zlib_codec.rs:20-31` and `zlib.rs:99-110`.
- acceptance: (1) `git ls-files 'crates/consus-zarr/tests/*.rs' | grep -v roundtrip_zarr | xargs cat | perl
  -0pe 's{/\*.*?\*/|//[^\n]*}{}gs' | tr -d '[:space:]' | grep -cE '([0-9][0-9a-fA-Fxob_u]*,){8}'` prints 0 (1
  today), and 1 on `zlib.rs`; (2) `git grep -n -E
  'CPYTHON|[\\][xu]|include_bytes|fs::read|const[[:space:]][A-Z_0-9]+:[[:space:]]*&?[[]u8|"[A-Za-z0-9+/=]{24,}"|"[0-9a-fA-F,[:space:]]{24,}"'
  -- crates/consus-zarr/tests ':!*.proptest-regressions'` prints nothing (7 today); (3) in
  `crates/consus-zarr/tests`, `for t in decodes_numcodecs_zlib_stream
  legacy_gzip_chunk_still_decodes_under_zlib malformed_zlib_stream_is_a_compression_error; do sed -n
  "/^fn.$t/,/^}/p" unit_zlib_codec.rs | grep -c '[.]compress[(]'; done` prints 1+ for each (0, 1, 0 today).
- priority: tightening; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: crates/consus-zarr/tests, crates/consus-compression/src/codec/zlib.rs
- next step: build both streams with `default_registry().get_by_name(..).compress`.

<a id="consus-zarr-test-harness"></a>
## CONSUS-ZARR-TEST-HARNESS — Fold the Zarr integration tests into one harness [patch] — todo
- outcome: the eight `consus-zarr` integration binaries become modules of one harness, `tests/zarr/main.rs`;
  no test changes its name.
- evidence: `crates/consus-zarr/tests` holds eight `*.rs` binaries, each a separate link of the crate graph;
  `property_zarr.proptest-regressions` sits beside its binary.
- acceptance: `ls crates/consus-zarr/tests/*.rs | wc -l` prints 0 (8 today);
  `cargo nextest list -p consus-zarr --all-features` before and after hold the same tests once
  `consus-zarr::<binary> <test>` is read as `consus-zarr::zarr <binary>::<test>`; `git ls-files
  crates/consus-zarr/tests/zarr/property_zarr.proptest-regressions` lists the replay file (none today).
- priority: tightening; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- needs: CONSUS-ZARR-EDGE-CHUNK-PADDING, CONSUS-ZLIB-VECTOR-DEDUP, CONSUS-ZARR-LEVEL-INTEROP
- scope: crates/consus-zarr/tests
- next step: `git mv` the files under `tests/zarr/` with the regression file, and add `main.rs` with one `mod`
  per file.

<a id="consus-flate-surface"></a>
## CONSUS-FLATE-SURFACE — Narrow the unused flate items [major] — todo
- outcome: `zlib_compress`, `ZlibFraming` and `GzipFraming` are `pub(crate)`; the public surface of
  `codec::flate` is what `consus-zip` consumes.
- evidence: `crates/consus-compression/src/codec/flate.rs:42,63,139` are `pub`; only sibling codec modules use
  them, and `consus-zip` uses `RawDeflateFraming` and `flate2_decompress`
  (`crates/consus-zip/src/read.rs:21`). No other stack member's tracked `*.rs` names them.
- acceptance: `cargo semver-checks -p consus-compression` against the 0.5.0 baseline reports exactly the three
  removals under the declared [major] class, and `cargo check --workspace --locked --all-features` passes.
- priority: tightening; milestone: M-interop-contracts; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- needs: CONSUS-ZLIB-DECODE-CEILING
- scope: crates/consus-compression/src/codec/flate.rs, CHANGELOG.md
- next step: narrow the three items after the ceiling change lands; note the removal under `### Breaking`.

<a id="consus-parquet-payload-count"></a>
## CONSUS-PARQUET-PAYLOAD-COUNT — Carry the checked value count in the page payload [major] — todo
- outcome: `PagePayload::num_values` is the checked `usize`; no reader cast remains; both splitters test it.
- evidence: `split_data_page_v1` and `_v2` reject negative counts
  (`consus-parquet/src/wire/payload.rs:160,209`), yet `reader/page.rs:281,374` cast with `as usize` and
  `page.rs:439-441` holds a branch no input reaches (both V2 call sites split first, `page.rs:301,393`).
  `payload.rs:317` tests the V1 count, `:326` the V2 level lengths, nothing the V2 count.
- acceptance: `cargo nextest run -p consus-parquet --all-features split_v2_rejects_negative_value_count
  split_v2_carries_checked_value_count` passes both (E0308 today): the first asserts `Error::InvalidFormat`
  with `num_values` in its message for count -1, the second binds `let count: usize = payload.num_values;`;
  `git grep -n -E 'payload.num_values as usize|DataPageV2 num_values must be' -- crates/consus-parquet/src`
  prints nothing (three lines today).
- priority: tightening; milestone: M-interop-contracts; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- scope: crates/consus-parquet/src/wire/payload.rs, crates/consus-parquet/src/reader/page.rs
- next step: store `count` in the payload; the type change is public, so record it under `### Breaking`.

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

<a id="consus-zarr-zlib-level"></a>
## CONSUS-ZARR-ZLIB-LEVEL — Honor configured write levels [patch] — blocked

- milestone: M-core-formats
- priority: correctness; needs: none
- basis: 0ad1040530ba854520d58e721fff0e801beb55fb
- scope: Zarr codec level selection and real v2 array-write regressions.
- blocked: installed-hook delivery hold; independent review and normal gates
  must pass before delivery. Registry/provider release holds are unchanged.
- evidence: at `0ad1040`, a zlib level-0 write uses level 6; the gzip control
  passes. The shared compression arm called a gzip-only accessor.
- acceptance: level 0/9 and default/6 writes match explicitly configured
  encoders, preserve gzip behavior, and decode independently without data loss.
- next step: final independent review and applicable delivery gates; existing
  provider, registry, and installed-hook holds remain. No publication here.
