//! # consus-zarr
//!
//! Pure-Rust implementation of the Zarr storage format (v2 and v3).
//!
//! ## Zarr Specification
//!
//! - **Zarr v2**: <https://zarr.readthedocs.io/en/stable/spec/v2.html>
//! - **Zarr v3**: <https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html>
//!
//! ## Module Hierarchy
//!
//! ```text
//! consus-zarr
//! ├── lib.rs               # Facade + public API re-exports
//! ├── metadata/            # Array and group metadata parsing/serialization
//! │   ├── mod.rs           # Shared types: ArrayMetadata, GroupMetadata,
//! │   │                    #   Codec, FillValue, AttributeValue, ZarrVersion,
//! │   │                    #   dtype_to_element_size()
//! │   ├── v2.rs            # .zarray, .zgroup, .zattrs (Zarr v2 JSON)
//! │   ├── v3.rs            # zarr.json (Zarr v3 JSON)
//! │   └── consolidated.rs  # .zmetadata consolidated format (v2 and v3)
//! ├── chunk/               # Chunk indexing and key generation
//! │   ├── mod.rs           # ChunkIndex, chunk decomposition, selection
//! │   └── key_encoding.rs  # ChunkKeySeparator, chunk_key()
//! ├── codec/               # Compression codec pipeline execution
//! │   └── mod.rs           # CodecPipeline, CompressionLevel, default_registry()
//! ├── store/               # Storage backend abstraction and implementations
//! │   ├── mod.rs           # Store trait, ReadWriteStore, PrefixedStore,
//! │   │                    #   SplitStore, re-exports
//! │   ├── memory.rs        # InMemoryStore (BTreeMap-backed)
//! │   ├── filesystem.rs    # FsStore (local directory tree)
//! │   └── s3.rs            # S3Store (rusoto_s3, feature = "s3")
//! ├── shard/               # Zarr v3 sharding codec support
//! │   └── mod.rs           # ShardConfig, ShardIndexReader,
//! │                        #   read_chunk_from_shard, compute_linear_index
//! └── tests/
//!     └── integration.rs   # Round-trip, store, metadata, codec tests
//! ```
//!
//! ## Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `std` (default) | Full functionality: JSON, compression, all stores |
//! | `alloc` | Core without std; enables JSON and in-memory stores |
//! | `s3` | S3-compatible object store via rusoto_s3 |
//!
//! ## Architecture (DIP/SSOT)
//!
//! Format logic depends on abstract traits (`Store`, `CompressionRegistry`)
//! defined in lower-layer crates. No format-specific details leak upward.
//!
//! ## Python Interoperability
//!
//! - Metadata JSON is spec-compliant: `.zarray`, `.zgroup`, `.zattrs`,
//!   `zarr.json`, and `.zmetadata` are byte-for-byte compatible.
//! - Chunk keys follow the canonical Zarr convention.
//! - Codec parameters (gzip level, etc.) match zarr-python's defaults.
//!
//! ## Quick Start
//!
//! ```ignore
//! use consus_zarr::store::{InMemoryStore, Store};
//! use consus_zarr::metadata::{ArrayMetadataV2, ZarrVersion, FillValue, Codec,
//!     ChunkKeyEncoding};
//! use consus_zarr::chunk::{chunk_key, ChunkKeySeparator};
//!
//! // Build a v2 array metadata and store it
//! let mut store = InMemoryStore::new();
//!
//! let zarray = r#"{
//!   "zarr_format": 2,
//!   "shape": [100, 100],
//!   "chunks": [10, 10],
//!   "dtype": "<f8",
//!   "fill_value": 0.0,
//!   "order": "C",
//!   "compressor": {"id": "gzip", "configuration": {"level": 1}},
//!   "filters": null
//! }"#;
//!
//! store.set("my_array/.zarray", zarray.as_bytes()).unwrap();
//!
//! // Read it back
//! let data = store.get("my_array/.zarray").unwrap();
//! let meta = ArrayMetadataV2::parse(std::str::from_utf8(&data).unwrap()).unwrap();
//! assert_eq!(meta.shape, &[100, 100]);
//!
//! // Generate chunk keys for the array
//! for i in 0..10 {
//!     for j in 0..10 {
//!         let key = chunk_key(&[i, j], ChunkKeySeparator::Dot);
//!         println!("chunk key: {key}");
//!     }
//! }
//! ```
//!
//! ## S3 Example
//!
//! ```ignore
//! use consus_zarr::store::S3Store;
//! use consus_zarr::store::Store;
//!
//! let store = S3Store::new("my-zarr-bucket")
//!     .with_prefix("experiment_001.zarr")
//!     .with_region(Region::UsEast1)
//!     .with_md5()  // Required for Python zarr interop
//!     .build()
//!     .unwrap();
//!
//! // Now use `store` with the Zarr API...
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

// Re-export the Store trait at crate root for convenience.
#[cfg(feature = "alloc")]
pub use crate::store::Store;

// ---------------------------------------------------------------------------
// Public sub-module re-exports
// ---------------------------------------------------------------------------

/// Metadata types and JSON parsing/serialization for Zarr v2 and v3.
#[cfg(feature = "alloc")]
pub mod metadata;

#[cfg(feature = "alloc")]
pub use metadata::{
    ArrayMetadata, AttributeValue, ChunkKeyEncoding, Codec, ConsolidatedMetadata, FillValue,
    GroupMetadata, ZarrVersion, dtype_to_element_size,
};

#[cfg(feature = "alloc")]
pub use metadata::{ArrayMetadataV2, GroupMetadataV2};

#[cfg(feature = "alloc")]
pub use metadata::ZarrJson;
#[cfg(feature = "std")]
pub use metadata::{WriteZarrJsonError, write_group_json, write_zarr_json};

/// Chunk indexing, decomposition, and key encoding.
pub mod chunk;

pub use chunk::{
    ChunkError, ChunkKeySeparator, Selection, SelectionStep, chunk_key, expand_fill_value,
    read_array, read_chunk, write_array, write_array_selection, write_chunk,
};

/// Compression codec pipeline execution.
#[cfg(feature = "alloc")]
pub mod codec;

/// Storage backend abstraction and implementations.
#[cfg(feature = "alloc")]
pub mod store;

#[cfg(feature = "alloc")]
pub use store::{FsStore, InMemoryStore, PrefixedStore, ReadWriteStore, SplitStore};

/// Zarr v3 sharding codec support.
#[cfg(feature = "alloc")]
pub mod shard;
