# 0006 — Version the Moirai 0.10 public async identity

Status: Proposed

Exact-head validation and independent review remain required.

Date: 2026-10-19

Change class: [major], pre-1.0 minor increment

## Contract

Published consus-hdf5 0.3.0 comes from beae4ce7 and admits Moirai 0.9.3.
Its public AsyncHdf5File and positioned async read functions use
Moirai-owned AsyncReadAt and AsyncLength traits. Current Git `441738cd` admits
Moirai 0.10 under the same Consus 0.3 version. Identical trait definitions or
unchanged Consus implementation files do not preserve nominal trait identity
across incompatible provider versions.

## Decision

Prepare the existing uniformly versioned workspace/internal requirements at
0.4.0. Keep the Moirai 0.10 Git source, features and defaults; do not restore an
older source, add aliases, bridge traits or hide the migration behind a patch.
Consumers crossing the async boundary migrate to the same Moirai 0.10 identity.
The facade and internal path graph remain one versioned release cohort.

Refresh the consumer regression's published baseline to actual HDF5 0.3 and
Moirai 0.9.3. The identical caller must compile there, fail nominal trait bounds
against current Consus with its unchanged old provider, and compile after its
provider migrates. A comparison with HDF5 0.2 alone misses this new boundary.

Also admit patched ZIP 9 for the existing format implementations, retaining the
declared compression feature and real archive tests. Current ZIP 6 is affected
by the same RustSec 2026-0367/0368/0369 family as the Gaia renewal. Do not ignore
the advisories, remove format support or weaken extraction tests.

## Binding and storage checks

The full graph previously failed because PyO3 0.22 rejects Python 3.14.
Upgrade PyO3 and its build-config package together to 0.30.1, migrate the
binding calls without changing Python names or numeric values, and explicitly
retain the cloned MatVariable extraction contract. Remove the obsolete macro
lint exemption. Exercise Python 3.14 in CI and add it to the existing wheel
matrix, retaining every previous interpreter.
PyO3 0.30.1's minimum CPython version is 3.9, matching the existing floor.
Build and exercise the optimized wheel locally and in the existing CI job.

The unchanged HDMF reference suite then exposes a discarded parameter:
`read_dataset_raw` accepts a variable element width but loses it at chunked
dispatch. Forward that width into the existing single chunk-reading algorithm.
Fixed-width datatypes retain their own width, and unknown variable widths
retain the existing error. A h5py-generated native fixture covers plain and
compressed two-chunk UTF-8/empty/unequal strings plus fixed-width dispatch;
the existing HDMF oracle remains unchanged.

## Release artifact contract

With Rust 1.97 and Apple ld27037.1, `strip = "symbols"` produces a release
extension whose string pool begins at 0x001AB024 (4 modulo the 8-byte alignment).
The optimized wheel builds but dyld rejects it before all six format modules
can import. Retain symbols with `strip = "none"`; the resulting string pool
is aligned, and the unchanged 45 interoperability cases pass. Thin LTO,
opt-level 3, one codegen unit, tests and timeouts remain unchanged. CI builds
and imports the release wheel so a development-only pass cannot conceal this
artifact failure. No global linker or credential configuration changes.

## Acceptance

Run bounded published/current/migrated consumer checks, unchanged workspace
native/Clippy/docs/feature gates, advisory checks and actual normalized package
verification. Preserve owner source and existing workflow/time budgets. Publish
only reviewed packages after all applicable exact-revision gates pass, using
existing authorized credentials; no authentication/configuration change.
