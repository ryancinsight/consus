#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../../../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Core types, traits, and error definitions.
///
/// Available in all configurations including `no_std`.
pub use consus_core as core;

/// Sync and async I/O abstractions.
pub use consus_io as io;

/// Compression codec registry.
pub use consus_compression as compression;

/// HDF5 format backend.
#[cfg(feature = "hdf5")]
pub use consus_hdf5 as hdf5;

/// Zarr v2/v3 format backend.
#[cfg(feature = "zarr")]
pub use consus_zarr as zarr;

/// netCDF-4 format backend.
#[cfg(feature = "netcdf")]
pub use consus_netcdf as netcdf;

/// Apache Parquet interop layer.
#[cfg(feature = "parquet")]
pub use consus_parquet as parquet;

/// Apache Arrow interop layer.
#[cfg(feature = "arrow")]
pub use consus_arrow as arrow;

pub mod builders;
pub mod highlevel;
pub mod sync;

#[cfg(feature = "async-io")]
pub mod r#async;

pub use consus_core::{
    ByteOrder, ChunkShape, Compression, Datatype, Error, Extent, Hyperslab, HyperslabDim, Layout,
    LinkType, NodeType, PointSelection, ReferenceType, Result, Selection, Shape, StringEncoding,
};

#[cfg(feature = "alloc")]
pub use builders::{
    DatasetBuilder, DatasetBuilderSpec, FileBuilder, FileOpenOptions, GroupBuilder,
};

#[cfg(feature = "alloc")]
pub use highlevel::{
    BackendFactory, BackendRegistry, Dataset, DatasetCreateSpec, File, FileOptions, Group,
    UnifiedBackend, ZeroCopyBytes,
};

pub use sync::{
    ByteView as ZeroCopySlice, IoRange, TypedByteView, ZeroCopyRead, par_read_ranges,
    partition_range, read_ranges, read_typed, selection_byte_len, source_len, write_ranges,
};

#[cfg(feature = "std")]
use alloc::sync::Arc;

/// Returns the process-wide default backend registry.
///
/// This crate contains no format-specific dispatch logic. The default registry
/// is intentionally empty until backend adapter crates or application code
/// register factories explicitly.
#[cfg(all(feature = "alloc", feature = "std"))]
pub fn default_backend_registry() -> BackendRegistry {
    BackendRegistry::new()
}

/// Returns an empty shared backend registry handle.
///
/// This is the canonical entry point for applications that want to assemble
/// their own registry once and share it across multiple `File` operations.
#[cfg(all(feature = "alloc", feature = "std"))]
pub fn shared_backend_registry() -> Arc<BackendRegistry> {
    Arc::new(default_backend_registry())
}

/// Prelude for the unified facade API.
pub mod prelude {
    pub use crate::{Compression, Error, Result, Selection, Shape};

    #[cfg(feature = "alloc")]
    pub use crate::{
        Dataset, DatasetBuilder, File, FileBuilder, FileOpenOptions, Group, GroupBuilder,
    };

    #[cfg(feature = "async-io")]
    pub use crate::r#async::{
        AsyncDataset, AsyncDatasetHandle, AsyncFile, AsyncFileHandle, AsyncGroup, AsyncGroupHandle,
        ParallelReadPlan as AsyncParallelReadPlan, ZeroCopyBytes as AsyncZeroCopyBytes,
    };

    pub use crate::sync::{ByteView, IoRange, ZeroCopyRead};
}
