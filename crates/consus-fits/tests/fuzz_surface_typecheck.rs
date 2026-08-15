//! The FITS fuzz target's API surface, type-checked by the normal test build.
//!
//! `fuzz/fuzz_targets/fuzz_fits_table.rs` cannot be compiled on every
//! developer host: `libfuzzer-sys` builds a C++ shim that requires the MSVC
//! toolchain on Windows, so `cargo fuzz build` runs only in CI. This test
//! exercises the exact calls that target makes, so a rename or signature
//! change in the fuzzed surface breaks the ordinary gate rather than the
//! weekly fuzz job.
#![cfg(feature = "alloc")]

use consus_fits::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor, parse_header_bytes};

#[test]
fn fits_table_fuzz_surface_is_callable() {
    // A non-multiple of 80 bytes: rejected deterministically, exactly the
    // early return the fuzz target takes for most inputs.
    assert!(parse_header_bytes(&[0u8; 3]).is_err());

    // 80 zero bytes tokenise to one card; neither descriptor accepts it,
    // and both must return rather than panic.
    let Ok(header) = parse_header_bytes(&[b' '; 80]) else {
        return;
    };
    assert!(FitsAsciiTableDescriptor::from_header(&header).is_err());
    assert!(FitsBinaryTableDescriptor::from_header(&header).is_err());
}
