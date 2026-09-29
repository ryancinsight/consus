//! macOS link behavior for the `consus` Python extension cdylib.
//!
//! The cdylib is a Python extension module: it must never link a Python
//! interpreter library, because the interpreter that loads it supplies the
//! Python symbols at load time. On Apple's linker that contract is spelled
//! `-undefined dynamic_lookup`; without it, the link intermittently fails
//! with undefined `_PyBaseObject_Type` (observed as a flake on the macOS CI
//! runner: identical commits link on one run and fail on the next, because
//! the exact link behavior ended up dependent on pyo3's feature resolution
//! and runner environment rather than being pinned).
//!
//! Emitting the flag from here pins it for every build path that links this
//! cdylib: `cargo test --workspace`, nextest, maturin wheel builds, and
//! release packaging. It is emitted only for Apple targets and only for the
//! final cdylib link of this crate, so dependency builds and other platforms
//! are untouched.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }
    // `-undefined dynamic_lookup` defers Python symbol resolution to load
    // time, which is the defining property of an extension module.
    println!("cargo:rustc-link-arg-bins=-Wl,-undefined,dynamic_lookup");
    println!("cargo:rustc-link-arg-cdylib=-Wl,-undefined,dynamic_lookup");
}
