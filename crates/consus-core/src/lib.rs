//! # consus-core
//!
//! Core types, traits, and error definitions for the Consus scientific storage library.
//!
//! This crate provides the abstract storage model shared across all format
//! backends (HDF5, Zarr, netCDF-4, Parquet). It is `no_std`-compatible by
//! default; enable the `std` feature for `std::io` integration and
//! `std::error::Error` implementations.
//!
//! ## Architecture (Dependency Inversion Principle)
//!
//! Format-specific crates depend on the abstract traits defined in
//! [`core::traits`]. Type definitions are consolidated in [`types`] as
//! the Single Source of Truth (SSOT).
//!
//! ## Module Hierarchy
//!
//! ```text
//! consus-core
//! ├── core/            # Abstract traits and error hierarchy
//! │   ├── traits       # File, Group, Dataset, Attribute, Link, Selection traits
//! │   └── error        # Comprehensive error types and Result alias
//! └── types/           # Canonical type definitions (SSOT)
//!     ├── datatype     # Datatype, ByteOrder, StringEncoding, CompoundField, EnumMember
//!     ├── dimension    # Extent, Shape, ChunkShape, Layout
//!     ├── selection    # Selection, Hyperslab, HyperslabDim, PointSelection
//!     ├── reference    # ReferenceType
//!     ├── attribute    # AttributeValue, Attribute, UserMetadata
//!     └── node         # NodeType, LinkType, Compression, DatasetConfig
//! ```
//!
//! ## Invariants
//!
//! - All types are `Send + Sync` when their contents are.
//! - Datatype representations are canonical: two equivalent logical types
//!   produce identical `Datatype` values regardless of source format.
//! - Dimension ordering follows row-major (C) convention by default;
//!   column-major (Fortran) is explicitly represented via `Layout`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod core;
pub mod types;

// ---------------------------------------------------------------------------
// Re-export error types at the crate root for convenience.
// ---------------------------------------------------------------------------

pub use self::core::error::{Error, Result};

// ---------------------------------------------------------------------------
// Re-export abstract traits at the crate root.
// ---------------------------------------------------------------------------

pub use self::core::traits::{
    AttributeWrite, DatasetRead, DatasetWrite, FileRead, FileWrite, GroupRead, GroupWrite,
    HasAttributes, LinkRead, Node, SelectionOps,
};

// ---------------------------------------------------------------------------
// Re-export primary value types at the crate root.
// ---------------------------------------------------------------------------

pub use types::{
    ByteOrder, ChunkShape, Compression, Datatype, Extent, Hyperslab, HyperslabDim, Layout,
    LinkType, NodeType, PointSelection, ReferenceType, Selection, Shape, StringEncoding,
};

pub use types::AttributeValue;

#[cfg(feature = "alloc")]
pub use types::{Attribute, CompoundField, DatasetConfig, EnumMember, UserMetadata};
