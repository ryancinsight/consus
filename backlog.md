# Consus — Backlog

<a id="consus-raster-ljif"></a>
## CONSUS-RASTER-LJIF — Read grayscale lossless JPEG with LJIF metadata [patch]

- Target: `consus-raster` 0.5.1.
- Accept the observed grayscale LJIF header before the frame. Reject malformed
  lengths, duplicate headers and unsupported modes.
- [Acceptance and release gates](checklist.md#consus-raster-051-release).

<a id="consus-stable-qualification"></a>
## CONSUS-STABLE-QUALIFICATION — consus stable-release qualification [patch] — todo
- priority: verification; milestone: M-interop-contracts; basis: `d1319313201c`.
- Outcome: Close the HDF5 handoff to Helios using the published reader. Follow the [release scope](docs/adr/0008-release-scope-1-0.md) and [milestone exits](backlog/milestones.md).
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
