//! The HDF5 fuzz target's API surface, type-checked by the normal test build.
//!
//! `fuzz/fuzz_targets/fuzz_hdf5_parser.rs` cannot be compiled on every
//! developer host: `libfuzzer-sys` builds a C++ shim that requires the MSVC
//! toolchain on Windows, so `cargo fuzz build` runs only in CI. This test
//! exercises the exact calls that target makes — including the
//! `list_root_group` / `dataset_at` / `attributes_at` /
//! `read_chunked_dataset_all_bytes` chain reached only after a successful
//! `open` — so a rename or signature change in the fuzzed surface breaks the
//! ordinary gate rather than the weekly fuzz job.
//!
//! Adversarial *behaviour* is covered by `adversarial_input.rs`; this file
//! covers only that the fuzzed entry points still exist with the shapes the
//! target calls them at.
#![cfg(feature = "alloc")]

use consus_hdf5::file::Hdf5File;
use consus_io::MemCursor;

#[test]
fn hdf5_parser_fuzz_surface_is_callable() {
    // Not an HDF5 superblock signature: rejected deterministically, exactly
    // the early return the fuzz target takes for most inputs.
    let cursor = MemCursor::from_bytes(vec![0u8; 64]);
    let Ok(file) = Hdf5File::open(cursor) else {
        return;
    };

    // Stage 2 and 3: the exact traversal the fuzz target drives. Reached only
    // when a superblock parses, which adversarial bytes rarely achieve.
    let Ok(entries) = file.list_root_group() else {
        return;
    };
    for (_name, addr, _link_type) in &entries {
        let _ = file.dataset_at(*addr);
        let _ = file.attributes_at(*addr);
        let _ = file.read_chunked_dataset_all_bytes(*addr);
    }
}
