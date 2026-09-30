//! Link behavior for the `consus` Python extension cdylib.
//!
//! The cdylib is a Python extension module: it must not link a Python
//! interpreter library, because the interpreter that loads it supplies the
//! Python symbols. PyO3 does not emit that contract's linker arguments for
//! us — `pyo3-ffi`'s build script runs `emit_link_config` only when
//! `is_linking_libpython_for_target` is true, and that predicate ends in
//! `|| !is_extension_module()`, so it is false for exactly this case
//! (`extension-module` on Darwin). `pyo3`'s own build script emits only
//! cfgs. PyO3 documents the remaining step as the extension crate's job:
//! `pyo3_build_config::add_extension_module_link_args`, "should be called
//! from a build script".
//!
//! Calling it covers Apple (`-undefined dynamic_lookup`) and
//! `wasm32-unknown-emscripten` (`-sSIDE_MODULE=2 -sWASM_BIGINT`), and is a
//! no-op everywhere else, so the target condition stays upstream's rather
//! than ours. The arguments reach the final cdylib link of this crate only.
//!
//! Why this is load-bearing rather than belt-and-braces: the failure it
//! fixes is deterministic, not a flake. `cargo test` performs the cdylib's
//! final link only when the lib target is built with `cdylib` alongside
//! another crate type; with `cdylib` alone it builds the test harness and
//! never links the dylib, which is why the defect sat latent in this crate
//! until `rlib` was added to `crate-type` for the doctest gate
//! (`b392c17`). From that commit on, every macOS push run failed
//! `Test (macos-latest)` on undefined `_PyBaseObject_Type`.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
