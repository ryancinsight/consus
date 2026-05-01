//! Fuzz target: HDF5 parser (heap-buffer and logic).
//!
//! ## Strategy
//!
//! Drive `Hdf5File::open` with adversarial byte sequences to exercise:
//!
//! 1. Superblock detection and version dispatch (v0/v1/v2/v3).
//! 2. Object header parsing for the root group.
//! 3. B-tree v1/v2 traversal through `list_root_group`.
//! 4. Dataset metadata decoding through `dataset_at`.
//! 5. Attribute message decoding through `attributes_at`.
//! 6. Chunked B-tree data reads through `read_chunked_dataset_all_bytes`.
//!
//! All `Result` errors are discarded; only panics cause fuzzer failures.
#![no_main]

use consus_hdf5::file::Hdf5File;
use consus_io::MemCursor;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // MemCursor::from_bytes takes Vec<u8>; the to_vec() copy is the only
    // allocation here.  The fuzzer corpus owns `data`; MemCursor owns the copy.
    let cursor = MemCursor::from_bytes(data.to_vec());

    // Stage 1: superblock detection and superblock-version dispatch.
    // Fails deterministically for non-HDF5 input; no panic must occur.
    let Ok(file) = Hdf5File::open(cursor) else {
        return;
    };

    // Stage 2: root object header + group B-tree traversal.
    let Ok(entries) = file.list_root_group() else {
        return;
    };

    // Stage 3: per-entry dataset metadata, attribute, and chunked-data paths.
    for (_name, addr, _link_type) in &entries {
        // Dataset layout message and dataspace/datatype decoding.
        let _ = file.dataset_at(*addr);

        // Attribute message set attached to this object header address.
        let _ = file.attributes_at(*addr);

        // Chunked B-tree v1/v4 traversal and raw chunk assembly.
        // Returns InvalidFormat for contiguous/compact layouts; both code
        // paths are exercised without panicking.
        let _ = file.read_chunked_dataset_all_bytes(*addr);
    }
});
