//! Canonical type definitions for the Consus storage model.
//!
//! This module consolidates all value types used across the storage
//! abstraction layer. Types defined here are the Single Source of Truth
//! (SSOT) for the entire workspace. Format-specific crates depend on
//! these definitions via `consus-core` and never redefine them.
//!
//! ## Module Hierarchy
//!
//! ```text
//! types/
//! ├── datatype     # Datatype, ByteOrder, StringEncoding, CompoundField, EnumMember
//! ├── dimension    # Extent, Shape, ChunkShape, Layout
//! ├── selection    # Selection, Hyperslab, HyperslabDim, PointSelection
//! ├── reference    # ReferenceType
//! ├── attribute    # AttributeValue, Attribute, UserMetadata
//! └── node         # NodeType, LinkType, Compression, DatasetConfig
//! ```

pub mod attribute;
pub mod datatype;
pub mod dimension;
pub mod node;
pub mod reference;
pub mod selection;

// Re-export primary types at the module level for ergonomic access.
pub use attribute::AttributeValue;
#[cfg(feature = "alloc")]
pub use attribute::{Attribute, UserMetadata};

pub use datatype::{ByteOrder, Datatype, StringEncoding};
#[cfg(feature = "alloc")]
pub use datatype::{CompoundField, EnumMember};

pub use dimension::{ChunkShape, Extent, Layout, Shape};

#[cfg(feature = "alloc")]
pub use node::DatasetConfig;
pub use node::{Compression, LinkType, NodeType};

pub use reference::ReferenceType;

pub use selection::{Hyperslab, HyperslabDim, PointSelection, Selection};
