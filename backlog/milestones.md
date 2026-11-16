# Milestone ladder ([ADR 0008](../docs/adr/0008-release-scope-1-0.md))
<a id="M-core-formats"></a> **M-core-formats** (0.5.0) needs: (none) — core format read and write.
- exit: read and write HDF5 data (`cargo test -p consus-hdf5 --locked --all-features` passes).
- exit: read and write Zarr data (`cargo test -p consus-zarr --locked --all-features` passes).
- exit: interpret netCDF-4 models over HDF5 (`cargo test -p consus-netcdf --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-interop-contracts"></a> **M-interop-contracts** (0.6.0) needs: M-core-formats — interop and compression contracts.
- exit: verify compression and container contracts (`cargo test -p consus-compression --locked --all-features` and `cargo test -p consus-zip --locked --all-features` pass).
- exit: verify document and interchange crates (`cargo test -p consus-parquet --locked --all-features`, `cargo test -p consus-npy --locked --all-features`, and `cargo test -p consus-onnx --locked --all-features` pass).
- standing: no open correctness item.
<a id="M-1.0"></a> **M-1.0** (1.0.0) needs: M-interop-contracts — stable storage-format release (authority-gated).
- exit: import the Python extension without re-enabling the GIL and drive one shared object of every binding class from a thread pool (on a free-threaded `python3.14t` interpreter with the release wheel installed, `python -m pytest -q scripts/concurrent_bindings.py` passes with `PYTHON_GIL=0` and with `PYTHON_GIL` unset; `test_import_keeps_gil_disabled` removes the variable itself, so the module's declaration decides).
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.
