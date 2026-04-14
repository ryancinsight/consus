//! Attribute and metadata types for the Consus storage model.
//!
//! ## Specification
//!
//! Attributes are small named values attached to groups or datasets.
//! They carry metadata (units, descriptions, provenance) rather than
//! bulk array data. Attributes have a datatype and a value.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use crate::datatype::Datatype;

/// An attribute value.
///
/// Attributes are small metadata values attached to objects in the hierarchy.
/// Unlike datasets, attributes are not chunked or compressed.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// Scalar integer (widened to i64).
    Int(i64),
    /// Scalar unsigned integer (widened to u64).
    Uint(u64),
    /// Scalar float (widened to f64).
    Float(f64),
    /// String value.
    #[cfg(feature = "alloc")]
    String(String),
    /// Array of bytes (opaque or raw).
    #[cfg(feature = "alloc")]
    Bytes(Vec<u8>),
    /// Array of integer values.
    #[cfg(feature = "alloc")]
    IntArray(Vec<i64>),
    /// Array of float values.
    #[cfg(feature = "alloc")]
    FloatArray(Vec<f64>),
}

/// A named attribute with type information.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// Attribute name.
    pub name: String,
    /// Datatype of the attribute value.
    pub datatype: Datatype,
    /// The attribute value.
    pub value: AttributeValue,
}

/// User-defined metadata block (key-value pairs).
///
/// Used for format-specific metadata that does not map to the canonical
/// attribute model (e.g., HDF5 property lists, Zarr `.zattrs`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserMetadata {
    pub entries: Vec<(String, AttributeValue)>,
}
