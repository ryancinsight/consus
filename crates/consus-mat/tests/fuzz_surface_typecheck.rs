//! The MAT fuzz target's API surface, type-checked by the normal test build.
//!
//! `fuzz/fuzz_targets/fuzz_mat_reader.rs` cannot be compiled on every
//! developer host: `libfuzzer-sys` builds a C++ shim that requires the MSVC
//! toolchain on Windows, so `cargo fuzz build` runs only in CI. This test
//! exercises the exact call that target makes, so a rename or signature
//! change in the fuzzed surface breaks the ordinary gate rather than the
//! weekly fuzz job.
#![cfg(feature = "alloc")]

use consus_mat::loadmat_bytes;

#[test]
fn mat_reader_fuzz_surface_is_callable() {
    // Too short for any version-detection prefix: rejected deterministically,
    // exactly the early return the fuzz target takes for most inputs.
    assert!(loadmat_bytes(&[0u8; 3]).is_err());

    // A 128-byte zero block is a well-formed v5 header length with a zeroed
    // version/endian field, so version detection runs and then rejects. The
    // contract under test is that every outcome is a `Result`, never a panic.
    let _ = loadmat_bytes(&[0u8; 128]);
}
