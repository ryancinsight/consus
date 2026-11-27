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
  (`consus-compression/src/codec/traits.rs:5` and `:34`, `gzip.rs:47`, `zlib.rs:52`), while a codec that sizes
  its buffer from `expected_size` (`lz4.rs:40`, under the shared 64 MiB budget) rejects a larger declaration;
  the four DEFLATE codecs no longer do (the declaration is a capacity hint).
- acceptance: `git grep -n -E 'inverts? (it )?exactly' -- crates/consus-compression/src/codec` prints nothing
  (four lines today); `git grep -c 'within the decode ceiling' -- crates/consus-compression/src/codec` prints
  a line for each of `traits.rs`, `gzip.rs` and `zlib.rs` (none today).
- priority: verification; milestone: M-core-formats; basis: 673e774cd89064060c515df8bfe7d732d232128e
- scope: crates/consus-compression/src/codec
- next step: reword the four statements for the codecs that still bound the declaration, and state that bound
  once in the module docs.

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
  (`flate.rs:51,81`), though the codec named `deflate` is the zlib-framed one; it prefixes decode and compress
  errors (`flate.rs:159,182`) and `Display` renders `compression error: {message}` (`core/error.rs:189`).
- acceptance: `cargo nextest run -p consus-compression --all-features framing_label_prefixes_error` passes (no
  such test today): the `message` of a malformed stream starts `zlib` for `ZlibCodec` and `DeflateCodec`,
  `gzip` for `GzipCodec` and `raw-deflate` for `RawDeflateCodec`; `git grep -c -E
  'LABEL.*"(zlib|gzip|raw-deflate)";' -- '*/codec/flate.rs'` reports 3 matches (1 today) and
  `git grep -n 'LABEL.*"deflate' -- '*/codec/flate.rs'` prints nothing (two lines today).
- priority: verification; milestone: M-core-formats; basis: 673e774cd89064060c515df8bfe7d732d232128e
- scope: crates/consus-compression/src/codec/flate.rs, crates/consus-compression/tests
- next step: set the three labels to `zlib`, `gzip` and `raw-deflate`; test the message prefix of each codec.

<a id="consus-zarr-level-interop"></a>
## CONSUS-ZARR-LEVEL-INTEROP — Decode Consus-written level chunks with CPython [patch] — todo
- outcome: the `zlib` and `gzip` chunks `write_array` writes at levels 0 and 9 are committed files that a Rust
  test asserts byte for byte and CI decodes with CPython to the 29,696-byte input.
- evidence: the gzip/zlib arm of `codec/mod.rs:351-358` reads `level` after `declared_level`; `unit_zlib_codec.rs:272`
  decodes through Consus only, on `level_input()` (:117), not the phrase x 1024 that CPython decoded once by hand.
- acceptance: `python -m pytest -q scripts/zarr_level_interop.py` passes 4 decodes;
  `git ls-files crates/consus-zarr/tests/data/level_chunks` lists the 4 files; `git grep -n -E
  '[.]compress|compressobj|GzipFile' -- scripts/zarr_level_interop.py` prints nothing;
  `git grep -c zarr_level_interop -- .github/workflows/ci.yml` reports a match; `cargo nextest run -p
  consus-zarr --all-features written_level_chunks_equal_committed_files` passes and `git grep -c 'fn
  written_level_chunks_equal_committed_files' -- 'crates/consus-zarr/tests/*.rs'` reports 1 (none today).
- priority: verification; needs: CONSUS-ZLIB-VECTOR-DEDUP; milestone: M-core-formats; basis: 56a417b33afa
- scope: crates/consus-zarr/tests, scripts/zarr_level_interop.py (new), .github/workflows/ci.yml
- next step: assert each `write_array` chunk equals its file in `unit_zlib_codec.rs`.

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
  crates/consus-zarr/tests/proptest-regressions/property_zarr.txt` lists the replay file (none today).
- priority: tightening; milestone: M-core-formats; basis: 216bde10c66f0f85b7d084aef8b1f19296fb1ae2
- needs: CONSUS-ZARR-EDGE-CHUNK-PADDING, CONSUS-ZLIB-VECTOR-DEDUP, CONSUS-ZARR-LEVEL-INTEROP
- scope: crates/consus-zarr/tests
- next step: `git mv` the files under `tests/zarr/`, the regression file to
  `tests/proptest-regressions/property_zarr.txt`, and add `main.rs` with one `mod` per file.

<a id="consus-flate-surface"></a>
## CONSUS-FLATE-SURFACE — Narrow the unused flate items [major] — todo
- outcome: `zlib_compress`, `ZlibFraming` and `GzipFraming` are `pub(crate)`; the public surface of
  `codec::flate` is what `consus-zip` consumes.
- evidence: `crates/consus-compression/src/codec/flate.rs:48,65,175` are `pub`; only sibling codec modules use
  them, and `consus-zip` uses `RawDeflateFraming` and `flate2_decompress`
  (`crates/consus-zip/src/read.rs:21`). No other stack member's tracked `*.rs` names them.
- acceptance: `cargo semver-checks -p consus-compression` against the 0.5.0 baseline reports exactly the three
  removals under the declared [major] class, and `cargo check --workspace --locked --all-features` passes.
- priority: tightening; milestone: M-interop-contracts; basis: 673e774cd89064060c515df8bfe7d732d232128e
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

<a id="consus-zarr-level-clamps"></a>
## CONSUS-ZARR-LEVEL-CLAMPS — Reject unusable levels for zstd, deflate, lz4 and fallthrough codecs [major] — todo
- outcome: a `level` a codec cannot honor is a typed `InvalidCodecConfiguration`, not a silent clamp or default.
- evidence: zstd clamps to -8..=4 and maps 0 to 1 (consus-compression zstd.rs:28-29); deflate clamps to 0..=9
  (flate.rs:140, so -1 stores); `codec/mod.rs:360,366,378` fall back to 3, 0 and 6 when a level does not parse
  (`"abc"` becomes 6); `deflate` is reachable from numeric filter id 1 when its feature is on (v2.rs:436). The `crc32`
  arm comments (codec/mod.rs:344-346 and :398-400) say the codec is rejected; both arms apply CRC-32C.
- acceptance: zstd and `deflate` document their accepted range as a constant with a source and a level outside it is
  the typed error; an unparsable level is the typed error for every codec (lz4, blosc and szip ignore a parsed one:
  consus-compression lz4.rs:29, blosc.rs:170, szip.rs:162); the zstd test pins singleton levels (-6, -5, -2); the
  `crc32` comments state what the arms do; `git grep -n 'unwrap_or(3)' -- crates/consus-zarr/src` prints nothing.
- priority: correctness; milestone: M-core-formats; basis: 56a417b33afa
- scope: crates/consus-zarr/src/codec/mod.rs, crates/consus-zarr/src/metadata/codec.rs, crates/consus-zarr/tests,
  crates/consus-compression/src/codec
- next step: look up libzstd's level range, then decide reject versus document for 0 and for values above 4.

<a id="consus-zarr-index-codecs"></a>
## CONSUS-ZARR-INDEX-CODECS — Apply a sharded array's index codecs [patch] — todo
- outcome: a v3 sharded array whose `index_codecs` list a codec (crc32c, bytes endianness) is written and read through
  them instead of as raw 16-byte entries.
- evidence: `shard/io.rs:54`, `:63-77` and `:134` treat the index as raw 16-byte entries; `shard/config.rs:74` is the
  only consumer of `index_codecs`. Hypothesis, to verify against a zarr-python sharded array: its default index codecs
  include crc32c, so Consus would read a misaligned index.
- acceptance: a committed zarr-python sharded fixture with default index codecs reads back to its input and a Consus
  write of the same array reads back in zarr-python; a declared index codec that is unsupported is a typed error.
- priority: correctness; milestone: M-core-formats; basis: 56a417b33afa
- scope: crates/consus-zarr/src/shard, crates/consus-zarr/tests, data/zarr_python_fixtures
- next step: generate the sharded fixture with `generate_fixtures.py` and assert the failing read.

<a id="consus-zarr-v2-entry-parity"></a>
## CONSUS-ZARR-V2-ENTRY-PARITY — Make v2 codec entries parse the same in every placement [major] — todo
- outcome: one codec entry JSON gives one parse and write result as a compressor, a numeric-id compressor and a
  filter entry; inline members of a string-id compressor other than `level` are kept.
- evidence: `CompressorNamed` has a typed `level: Option<i32>` (v2.rs:199-213) and drops other inline members
  (`{"id":"blosc","cname":"lz4"}`); `FilterIdConfig` and `FilterConfig` keep them in `members` (:229, :281). A duplicate
  inline `level` key fails `parse` on a string-id compressor, last-wins elsewhere; inline `"level": null` is absent
  there (`to_json` writes it) and rejected on a filter. Found in review of consus#241.
- acceptance: one member map serves all placements; `to_json` writes an absent level in one decided form (null today,
  v2.rs:820); a table test asserts one outcome per case and placement and pins a numeric-id `clevel` in its canonical
  codecs (a mutant carrying only `level` passes 334 tests); CHANGELOG says `to_json` writes numeric-id members back.
- priority: correctness; milestone: M-core-formats; basis: 56a417b33afa
- scope: crates/consus-zarr/src/metadata/v2.rs, crates/consus-zarr/src/metadata/codec.rs, crates/consus-zarr/tests,
  CHANGELOG.md
- next step: decide the written form of an absent level (omit versus null); review it as a persisted-format change.

<a id="consus-zarr-level-text-edges"></a>
## CONSUS-ZARR-LEVEL-TEXT-EDGES — Compare declared levels as numbers, not text [patch] — todo
- outcome: a level declared as `-0` is read as 0, and two declarations that select the same level do not conflict.
- evidence: `config_text` (metadata/codec.rs:44-50) renders a nested `-0`, which serde_json reads as float -0.0, as
  `-0.0`, and the level parse rejects it, where CPython reads int 0; `declared_level` (:151-169) compares text, so `6`
  and `-1` conflict; a `configuration` given as a JSON array no longer parses. Found in review of consus#236 and #241
  by probe, not re-run at 56a417b.
- acceptance: `cargo nextest run -p consus-zarr --all-features declared_levels_compare_as_numbers` passes (no such test
  today): `-0` is accepted as 0, `6` beside `-1` agree for gzip and zlib, and an array `configuration` is a typed
  parse error with its text asserted.
- priority: correctness; milestone: M-core-formats; basis: 56a417b33afa
- scope: crates/consus-zarr/src/metadata/codec.rs, crates/consus-zarr/tests
- next step: parse each declaration to a typed level before comparing.

<a id="consus-zarr-error-types"></a>
## CONSUS-ZARR-ERROR-TYPES — Type configuration faults for Python callers [major] — todo
- outcome: `ChunkError::InvalidCodecConfiguration` carries typed fields and its source, and consus-python raises a
  distinct exception for it instead of `RuntimeError`.
- evidence: consus-python maps a `write_chunk` error (zarr.rs:131) and a `read_chunk` error (:140) to `PyRuntimeError`;
  `InvalidCodecConfiguration` holds a `String` (chunk/error.rs:27) and `ChunkError::from_compression` (:60-67) drops
  the source error and maps any pipeline `InvalidFormat` (including a missing `bytes` `endian`) to the configuration
  variant.
- acceptance: `python -m pytest -q scripts/compare_zarr.py -k codec_configuration` passes a test (none today; ci.yml:164
  and :205 run that file) that opens a zlib level 10 `.zarray` with `from_store` and asserts `write_chunk` raises the
  binding's configuration exception class (`RuntimeError` today); a Rust unit test in chunk/error.rs matches the
  variant fields; a missing `endian` is not the level variant.
- priority: tightening; milestone: M-interop-contracts; basis: 56a417b33afa
- scope: crates/consus-zarr/src/chunk/error.rs, crates/consus-python/src/zarr.rs, scripts/compare_zarr.py
- next step: replace the string payload with the codec name, declared text and source.

<a id="consus-zarr-gzip-level-accessor"></a>
## CONSUS-ZARR-GZIP-LEVEL-ACCESSOR — Remove the level accessors that swallow parse errors [major] — todo
- outcome: no `Codec` accessor returns `None` for a level that does not parse: `gzip_level` is removed, and `zstd_level`
  and `lz4_level` are removed or validate; tests and the write path read levels through validating readers only.
- evidence: the three are public (metadata/codec.rs:189, :198, :221) and read through `parsed_config` (:97), which drops
  the parse error. `gzip_level` has only test callers (codec/mod.rs:544, metadata/codec.rs:278, metadata/v3.rs:772) and
  disagrees with the validating `deflate_level` (:181): -1 gives None and 10 gives Some(10). The write path defaults
  the other two (codec/mod.rs:360, :366) and one test reads `zstd_level` (:553).
- acceptance: `git grep -n 'pub fn gzip_level' -- crates` prints nothing (one line today); `git grep -n
  'parsed_config("level")' -- crates/consus-zarr/src` prints nothing (three today); `cargo semver-checks -p consus-zarr`
  against the published baseline records the changed accessors under the declared [major] class.
- priority: tightening; needs: CONSUS-ZARR-LEVEL-CLAMPS; milestone: M-1.0; basis: 56a417b33afa
- scope: crates/consus-zarr/src/metadata/codec.rs, crates/consus-zarr/src/codec/mod.rs,
  crates/consus-zarr/src/metadata/v3.rs, CHANGELOG.md
- next step: after the clamps item, switch the four test call sites, remove `gzip_level`, note it under `### Breaking`.

<a id="consus-changelog-zarr-headings"></a>
## CONSUS-CHANGELOG-ZARR-HEADINGS — Date the published CHANGELOG headings [patch] — todo
- outcome: `CHANGELOG.md` heads the published 0.5.0 and 0.4.0 entries with their release dates, not `Unreleased`, and
  the 0.5.0 entry no longer says publication is gated, so the Zarr level entry under `## [Unreleased]` reads as a
  change after 0.5.0.
- evidence: `CHANGELOG.md:3` is `## [Unreleased]`, `:45` is `## [0.5.0] - Unreleased` and `:63` says approvals "still
  gate publication"; crates.io (API read at this filing) lists consus-zarr 0.5.0 as its newest version, uploaded
  2026-11-19, and tag `crate-consus-zarr-v0.5.0` holds the 0.5.0 clamp. `:89` is `## [0.4.0] - Unreleased`;
  consus-compression 0.4.0 was uploaded 2026-10-20, and the other crates' 0.4.0 dates were not read.
- acceptance: `git grep -n -E '^## \[0\.[45]\.0\] - Unreleased' -- CHANGELOG.md` prints nothing (two lines today), as
  does `git grep -n 'still gate publication' -- CHANGELOG.md` (one line today); the 0.5.0 heading reads
  `## [0.5.0] - 2026-11-19` and the 0.4.0 heading carries the first crates.io upload date of a 0.4.0 crate.
- priority: verification; milestone: M-core-formats; basis: 56a417b33afa
- scope: CHANGELOG.md
- next step: read the 0.4.0 upload dates from crates.io, then set both headings and drop the gating sentence.

<a id="consus-zarr-v3-level-text"></a>
## CONSUS-ZARR-V3-LEVEL-TEXT — Reject a v3 level that is not a JSON integer [patch] — todo
- outcome: a Zarr v3 codec `configuration.level` that is a JSON string, `null` or a non-integer number is a typed
  `InvalidCodecConfiguration` for every codec that reads a level, not text that is accepted or replaced by a default.
- evidence: `CodecSpec::to_codec` (metadata/v3.rs:415-421) renders each value with `config_text`, so the JSON string
  `"5"` is accepted as level 5 for gzip and zlib; under zstd and lz4 a v3 `null`, `"six"` or `1.5` writes the
  default-level output (zstd 3, lz4 0), run for zstd at 920a56b and derived for lz4 from codec/mod.rs:366. gzip and zlib
  already reject `null`, `"six"` and `1.5`.
- acceptance: a v3 table test over gzip, zlib, zstd and lz4 asserts `"5"`, `null`, `"six"` and `1.5` return the typed
  error with the whole message and an empty store; a JSON integer still selects its level.
- priority: correctness; needs: CONSUS-ZARR-LEVEL-CLAMPS; milestone: M-core-formats; basis: 56a417b33afa
- scope: crates/consus-zarr/src/metadata/v3.rs, crates/consus-zarr/src/metadata/codec.rs, crates/consus-zarr/tests
- next step: decide whether the check lives in the `config_text` callers or in a typed level reader shared with v2.

<a id="consus-zarr-v2-level-string"></a>
## CONSUS-ZARR-V2-LEVEL-STRING — Reject a v2 level given as a JSON string [patch] — todo
- outcome: a Zarr v2 `level` given as a JSON string in a `configuration` object, on a filter entry or on a numeric-id
  compressor is a typed `InvalidCodecConfiguration`, not text read as an integer.
- evidence: `member_pairs` (metadata/v2.rs:425-431) renders each member with `config_text`, which unquotes a string
  (metadata/codec.rs:46), and `DeflateLevel::parse` (:27-38) reads the text with `str::parse::<i32>`, so the filter
  `{"id":"zlib","level":"05"}` writes level 5, as `"5"` and `"+5"` do (derived, not run). numcodecs and zarr 3.2.1
  raise `TypeError` for a string level (relayed from the first review of consus#236, not re-run).
- acceptance: `cargo nextest run -p consus-zarr --all-features v2_string_level_is_rejected` passes (no such test today):
  for gzip, zlib and zstd in a `configuration` object and on a filter entry, and for zstd (id 32015) on a numeric-id
  compressor, `"5"`, `"+5"` and `"05"` return the typed error with the whole message and an empty store, and the
  JSON integer 5 still writes level 5. `hdf5_filter_id_to_name` (v2.rs:434-447) maps no id to gzip or zlib.
- priority: correctness; needs: CONSUS-ZARR-V3-LEVEL-TEXT; milestone: M-core-formats; basis: 56a417b33afa
- scope: crates/consus-zarr/src/metadata/v2.rs, crates/consus-zarr/src/metadata/codec.rs, crates/consus-zarr/tests
- next step: reuse the typed level reader the v3 item settles on for the v2 member maps.

<a id="consus-zarr-alloc-build"></a>
## CONSUS-ZARR-ALLOC-BUILD — Build consus-zarr with only the alloc feature [patch] — todo
- outcome: `cargo check --locked -p consus-zarr --no-default-features --features alloc` compiles, and CI builds that
  configuration.
- evidence: at an export of 56a417b that command exits 101 with 9 errors: E0432 for `crate::codec::default_registry`
  (chunk/ops/chunk_io.rs:8) and E0599 for `to_string` and `to_owned` without their traits in scope (metadata/codec.rs:48
  and :121; chunk/ops arrays.rs:124 and :389, chunk_io.rs:35 and :82, sharded.rs:76 and :312). ci.yml:111-112 builds only
  consus-hdf5 with `alloc`.
- acceptance: the command exits 0 and a ci.yml step runs it; `git grep -n 'consus-zarr --no-default-features' --
  .github/workflows/ci.yml` lists that step.
- priority: verification; milestone: M-core-formats; basis: 56a417b33afa
- scope: crates/consus-zarr/src, crates/consus-zarr/Cargo.toml, .github/workflows/ci.yml
- next step: import `ToString` and `ToOwned` from `alloc` where missing, and gate `default_registry` on `std`.

<a id="consus-ci-aggregate-check"></a>
## CONSUS-CI-AGGREGATE-CHECK — Gate merges on one aggregate CI check [patch] — todo
- outcome: `ci.yml` ends in one `always()` job that needs every other job and fails when one of them fails or is
  cancelled and on a draft pull request, and the `main` branch protection requires that one context.
- evidence: `ci.yml` defines ten jobs (semver, lockfile, fmt, check, test, test-free-threaded, msrv, test-mat-features,
  fuzz-build, fuzz-run), none with `always()`. `gh api repos/ryancinsight/consus/branches/main/protection`, read at this
  filing, requires seven contexts by job name; `Test Python bindings (free-threaded 3.14t)` is not one, so a red run of
  that job holds no merge (derived). Found in review of consus#236, #240 and #242.
- acceptance: `git grep -c 'always()' -- .github/workflows/ci.yml` reports 1 (no match today), on a job whose `needs:`
  names the other ten job ids and which fails a draft; `on.pull_request.types` includes `ready_for_review` (no `types`
  today, ci.yml:6-7); `required_status_checks.contexts` of the same API read lists that job (absent today).
- priority: verification; milestone: M-core-formats; basis: 56a417b33afa
- scope: .github/workflows/ci.yml, branch protection of `main` (repository setting)
- next step: add the job, let one pull-request run report it, then require its context in place of the seven names.

<a id="consus-ladder-release-versions"></a>
## CONSUS-LADDER-RELEASE-VERSIONS — Align the ladder's planned versions with released crates [patch] — todo
- outcome: the ladder's planned versions agree with the per-crate versions already released (consus-hdf5 0.6.0 and
  0.6.1, consus-zarr and consus 0.6.0), and the [major] items CONSUS-ZARR-ERROR-TYPES, CONSUS-ZARR-LEVEL-CLAMPS and
  CONSUS-ZARR-V2-ENTRY-PARITY sit in milestones that admit them.
- evidence: backlog/milestones.md plans M-core-formats at 0.5.0, M-interop-contracts at 0.6.0 and M-1.0 at 1.0.0; ADR
  0008 (:9, :18-22) repeats them beside "workspace 0.4.0". crates.io (generic client, read at this filing) lists
  consus-zarr and consus 0.6.0, consus-hdf5 0.6.0 and 0.6.1, with tags `crate-consus-zarr-v0.6.0` and
  `crate-consus-hdf5-v0.6.1`. ERROR-TYPES sits on M-interop-contracts, CLAMPS and ENTRY-PARITY on M-core-formats.
- acceptance: `git grep -n -E '[*][*]M-[a-z0-9.-]+[*][*] [(]0[.][56][.]0[)]' -- backlog/milestones.md` prints nothing
  (two lines today) and `git grep -n 'workspace 0.4.0' -- docs/adr/0008-release-scope-1-0.md` prints nothing (one).
  With no block at 0.5.0 or 0.6.0, each of the three [major] items sits in a milestone that admits its break.
- priority: verification; needs: none; milestone: M-core-formats; basis: 673e774cd89064060c515df8bfe7d732d232128e
- scope: backlog/milestones.md, docs/adr/0008-release-scope-1-0.md, backlog.md
- next step: move the open blocks past 0.6.0 under an ADR 0008 revision note, then retarget the three items.
