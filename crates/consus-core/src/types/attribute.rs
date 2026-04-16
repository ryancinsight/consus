//! Attribute value representation.
//!
//! Attributes are small named metadata values attached to nodes (groups,
//! datasets) in the storage hierarchy. Unlike datasets, attributes are
//! not chunked, compressed, or partially selectable.
//!
//! ## Specification
//!
//! An attribute has:
//! - A name (unique within the parent node).
//! - A datatype (from the canonical `Datatype` enum).
//! - A value (represented as `AttributeValue`).

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use super::datatype::Datatype;

/// Attribute value representation.
///
/// Values are stored in a widened representation to avoid generic
/// proliferation. The original precision is recoverable from the
/// associated `Datatype`.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// Scalar signed integer (widened to i64).
    Int(i64),
    /// Scalar unsigned integer (widened to u64).
    Uint(u64),
    /// Scalar IEEE 754 float (widened to f64).
    Float(f64),
    /// UTF-8 string value.
    #[cfg(feature = "alloc")]
    String(String),
    /// Raw byte array (opaque or binary).
    #[cfg(feature = "alloc")]
    Bytes(Vec<u8>),
    /// Array of signed integers.
    #[cfg(feature = "alloc")]
    IntArray(Vec<i64>),
    /// Array of unsigned integers.
    #[cfg(feature = "alloc")]
    UintArray(Vec<u64>),
    /// Array of floats.
    #[cfg(feature = "alloc")]
    FloatArray(Vec<f64>),
    /// Array of strings.
    #[cfg(feature = "alloc")]
    StringArray(Vec<String>),
}

/// A named, typed attribute attached to a node.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// Attribute name (unique within the parent node).
    pub name: String,
    /// Datatype of the attribute value.
    pub datatype: Datatype,
    /// The attribute value.
    pub value: AttributeValue,
}

/// User-defined metadata block (ordered key-value pairs).
///
/// Used for format-specific metadata that does not map directly to the
/// canonical attribute model (e.g., HDF5 property lists, Zarr `.zattrs`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserMetadata {
    /// Ordered key-value entries.
    pub entries: Vec<(String, AttributeValue)>,
}
