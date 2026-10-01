//! Pins the `extension-module` link contract for the `consus` Python extension.
//!
//! A Python extension module must not link a Python interpreter library: the
//! interpreter that loads it supplies the Python symbols. On Apple targets that
//! contract is spelled `-undefined dynamic_lookup`, and PyO3 leaves emitting it
//! to the extension crate -- `pyo3-ffi`'s build script skips its link
//! configuration whenever `is_linking_libpython_for_target` is false, which is
//! exactly the `extension-module`-on-Darwin case, and `pyo3`'s own build script
//! never emits the flag. Without it, the cdylib link fails on undefined
//! `_PyBaseObject_Type` and its siblings.
//!
//! The defect was latent rather than absent: while `crate-type` was `cdylib`
//! alone, `cargo test` built this crate for the test harness and never
//! performed the cdylib's final link, so no CI lane ever reached it. Adding
//! `rlib` (needed for the doctest gate) put the cdylib link into
//! `cargo nextest run`, which is where `Test (macos-latest)` failed.

fn main() {
    // Apple targets: `-undefined` + `dynamic_lookup`. wasm32-unknown-emscripten:
    // `-sSIDE_MODULE=2` + `-sWASM_BIGINT`. Every other target is a no-op.
    pyo3_build_config::add_extension_module_link_args();
}
