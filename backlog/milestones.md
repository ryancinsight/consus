# Milestone ladder ([ADR 0008](../docs/adr/0008-release-scope-1-0.md))
<a id="M-core-formats"></a> **M-core-formats** (0.5.0) needs: (none) — core format read and write.
- exit: read and write HDF5 data (`cargo test -p consus-hdf5 --locked --all-features` passes).
- exit: read and write Zarr data (`cargo test -p consus-zarr --locked --all-features` passes).
- exit: interpret netCDF-4 models over HDF5 (`cargo test -p consus-netcdf --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-interop-contracts"></a> **M-interop-contracts** (0.6.0) needs: M-core-formats — interop and compression contracts.
- exit: verify compression and container contracts (`cargo test -p consus-compression --locked --all-features` and `cargo test -p consus-zip --locked --all-features` pass).
- exit: verify document and interchange crates (`cargo test -p consus-parquet --locked --all-features`, `cargo test -p consus-npy --locked --all-features`, and `cargo test -p consus-onnx --locked --all-features` pass).
- exit: resolve `CONSUS-PARQUET-RAW-DEFLATE` with landed evidence on the default branch.
- standing: no open correctness item.
<a id="M-1.0"></a> **M-1.0** (1.0.0) needs: M-interop-contracts — stable storage-format release (authority-gated).
- exit: resolve `ATLAS-PR530-PROVIDER-COHERENCE` and `CONSUS-PYTHON-FREE-THREADING` with landed evidence on the default branch.
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.
