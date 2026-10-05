//! Replay of committed fuzz crashes through the same surface the fuzzer drives.
//!
//! Each test runs the `fuzz_hdf5_parser` sequence on a crash artifact a
//! scheduled fuzz run produced. The failure mode they guard against is
//! uncatchable from inside the parser — a panic — so the contract asserted
//! here is the fuzz target's own: every `Result` outcome is acceptable, a
//! panic is not. A regression re-introduces the panic and fails the test
//! deterministically on every host.

#![cfg(feature = "alloc")]

use consus_hdf5::file::Hdf5File;
use consus_io::MemCursor;

/// The exact sequence `fuzz/fuzz_targets/fuzz_hdf5_parser.rs` drives.
fn drive(data: &[u8]) {
    let cursor = MemCursor::from_bytes(data.to_vec());
    let Ok(file) = Hdf5File::open(cursor) else {
        return;
    };
    let Ok(entries) = file.list_root_group() else {
        return;
    };
    for (_name, addr, _link_type) in &entries {
        let _ = file.dataset_at(*addr);
        let _ = file.attributes_at(*addr);
        let _ = file.read_chunked_dataset_all_bytes(*addr);
    }
}

/// crash-b789d8a4 (2026-10-05 scheduled fuzz run): a chunked layout carrying a
/// zero chunk dimension reached `dataset_dim.div_ceil(chunk_dim)` in
/// `read_chunked_dataset_all_bytes` and panicked "attempt to divide by zero".
/// Chunk dimensions come from the layout message and must be rejected as
/// `Error::InvalidFormat` at parse, before any division consumes them.
#[test]
fn crash_b789d8a4_zero_chunk_dimension_is_a_typed_error() {
    drive(include_bytes!(
        "data/crash-b789d8a4784a410385e58d531799ac60713f9312"
    ));
}
