//! Arrow datatype descriptors used by the canonical datatype enum.

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec::Vec};

use super::ArrowDataType;

/// Temporal unit for Arrow temporal types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Millisecond resolution.
    Millisecond,
    /// Microsecond resolution.
    Microsecond,
    /// Nanosecond resolution.
    Nanosecond,
}

/// Signedness for integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntSign {
    /// Signed integer.
    Signed,
    /// Unsigned integer.
    Unsigned,
}

/// Decimal type metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecimalType {
    /// Total number of significant digits.
    pub precision: usize,
    /// Digits to the right of the decimal point.
    pub scale: isize,
}

impl DecimalType {
    /// Create a decimal descriptor.
    #[must_use]
    pub const fn new(precision: usize, scale: isize) -> Self {
        Self { precision, scale }
    }
}

/// Fixed-size binary metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedSizeBinaryType {
    /// Size in bytes.
    pub size: usize,
}

impl FixedSizeBinaryType {
    /// Create a fixed-size binary descriptor.
    #[must_use]
    pub const fn new(size: usize) -> Self {
        Self { size }
    }
}

/// Timestamp metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimestampType {
    /// Temporal unit.
    pub unit: TimeUnit,
    /// Whether the value is adjusted to UTC.
    pub is_adjusted_to_utc: bool,
}

impl TimestampType {
    /// Create a timestamp descriptor.
    #[must_use]
    pub const fn new(unit: TimeUnit, is_adjusted_to_utc: bool) -> Self {
        Self {
            unit,
            is_adjusted_to_utc,
        }
    }
}

/// Duration metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DurationType {
    /// Temporal unit.
    pub unit: TimeUnit,
}

impl DurationType {
    /// Create a duration descriptor.
    #[must_use]
    pub const fn new(unit: TimeUnit) -> Self {
        Self { unit }
    }
}

/// A dictionary-encoded type.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct DictionaryType {
    /// Key datatype used for dictionary indices.
    pub index_type: Box<ArrowDataType>,
    /// Value datatype stored in the dictionary.
    pub value_type: Box<ArrowDataType>,
    /// Whether the dictionary is ordered.
    pub ordered: bool,
}

#[cfg(feature = "alloc")]
impl DictionaryType {
    /// Create a dictionary type descriptor.
    #[must_use]
    pub fn new(index_type: ArrowDataType, value_type: ArrowDataType, ordered: bool) -> Self {
        Self {
            index_type: Box::new(index_type),
            value_type: Box::new(value_type),
            ordered,
        }
    }
}

/// A list type with a single child element type.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ListType {
    /// Child element type.
    pub element_type: Box<ArrowDataType>,
    /// Whether the list is nullable.
    pub nullable: bool,
}

#[cfg(feature = "alloc")]
impl ListType {
    /// Create a list type descriptor.
    #[must_use]
    pub fn new(element_type: ArrowDataType, nullable: bool) -> Self {
        Self {
            element_type: Box::new(element_type),
            nullable,
        }
    }
}

/// A map type represented as key/value entries.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct MapType {
    /// Key type.
    pub key_type: Box<ArrowDataType>,
    /// Value type.
    pub value_type: Box<ArrowDataType>,
    /// Whether the value field is nullable.
    pub value_nullable: bool,
}

#[cfg(feature = "alloc")]
impl MapType {
    /// Create a map type descriptor.
    #[must_use]
    pub fn new(key_type: ArrowDataType, value_type: ArrowDataType, value_nullable: bool) -> Self {
        Self {
            key_type: Box::new(key_type),
            value_type: Box::new(value_type),
            value_nullable,
        }
    }
}

/// A struct type composed of named fields.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct StructType {
    /// Ordered child fields.
    pub fields: Vec<crate::field::ArrowField>,
}

#[cfg(feature = "alloc")]
impl StructType {
    /// Create a struct type descriptor.
    #[must_use]
    pub fn new(fields: Vec<crate::field::ArrowField>) -> Self {
        Self { fields }
    }
}

/// A union type.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct UnionType {
    /// Ordered variant fields.
    pub fields: Vec<crate::field::ArrowField>,
    /// Whether the union is sparse.
    pub sparse: bool,
}

#[cfg(feature = "alloc")]
impl UnionType {
    /// Create a union type descriptor.
    #[must_use]
    pub fn new(fields: Vec<crate::field::ArrowField>, sparse: bool) -> Self {
        Self { fields, sparse }
    }
}
