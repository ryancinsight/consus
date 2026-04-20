//! Property-based tests for HDF5 write → read roundtrip invariants.
//!
//! ## Specification
//!
//! For any valid combination of:
//! - Datatype ∈ {Integer(8..64, signed/unsigned, LE/BE), Float(32/64, LE/BE), FixedString}
//! - Shape ∈ {scalar, 1D, 2D, 3D} with bounded dimension sizes
//! - Layout ∈ {Contiguous, Chunked(v3), Chunked(v4)}
//! - Compression ∈ {None, Deflate}
//!
//! The invariant  must hold at the byte level.

use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Compression, Datatype, Shape, StringEncoding};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use consus_hdf5::property_list::DatasetLayout;
use consus_io::MemCursor;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_and_open(builder: Hdf5FileBuilder) -> Hdf5File<MemCursor> {
    let bytes = builder.finish().expect("finalize file");
    let cursor = MemCursor::from_bytes(bytes);
    Hdf5File::open(cursor).expect("open file")
}

fn find_dataset_addr(file: &Hdf5File<MemCursor>, name: &str) -> u64 {
    let children = file.list_root_group().expect("list root");
    children
        .iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("dataset '{}' not found", name))
        .1
}
