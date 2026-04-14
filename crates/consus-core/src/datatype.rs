//! Canonical datatype representation for the Consus storage model.
//!
//! ## Specification
//!
//! The datatype system is a union of types expressible across HDF5, Zarr,
//! netCDF-4, and Parquet. Each format backend maps its native type system
//! to/from these canonical types.
//!
//! ### Invariant
//!
//! For any format-specific type `T_f` that has a canonical mapping:
//!   `canonicalize(T_f) == canonicalize(T_f')` iff `T_f` and `T_f'` represent
//!   the same logical type, regardless of format origin.

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::String, vec::Vec};

use core::num::NonZeroUsize;

/// Byte order for multi-byte scalar types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    /// Least-significant byte first (x86, ARM default).
    LittleEndian,
    /// Most-significant byte first (network order, POWER).
    BigEndian,
}

/// Canonical datatype for the Consus storage model.
///
/// This enum represents every type expressible across supported formats.
/// Format-specific backends convert to/from this representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Datatype {
    /// Boolean (1-bit logical).
    Boolean,

    /// Signed integer with specified bit width and byte order.
    Integer {
        bits: NonZeroUsize,
        byte_order: ByteOrder,
        signed: bool,
    },

    /// IEEE 754 floating-point with specified bit width.
    Float {
        bits: NonZeroUsize,
        byte_order: ByteOrder,
    },

    /// Complex number (real + imaginary), each component a float.
    Complex {
        component_bits: NonZeroUsize,
        byte_order: ByteOrder,
    },

    /// Fixed-length string.
    FixedString {
        length: usize,
        encoding: StringEncoding,
    },

    /// Variable-length string (requires alloc).
    VariableString { encoding: StringEncoding },

    /// Opaque blob of fixed size.
    Opaque {
        size: usize,
        #[cfg(feature = "alloc")]
        tag: Option<String>,
    },

    /// Compound type (struct-like): ordered named fields.
    #[cfg(feature = "alloc")]
    Compound {
        fields: Vec<CompoundField>,
        size: usize,
    },

    /// Array type: fixed-size array of another datatype.
    #[cfg(feature = "alloc")]
    Array {
        base: Box<Datatype>,
        dims: Vec<usize>,
    },

    /// Enumeration: named integer constants.
    #[cfg(feature = "alloc")]
    Enum {
        base: Box<Datatype>,
        members: Vec<EnumMember>,
    },

    /// Variable-length sequence of another datatype.
    #[cfg(feature = "alloc")]
    VarLen { base: Box<Datatype> },

    /// Reference to another object in the hierarchy.
    Reference(ReferenceType),
}

/// String encoding variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringEncoding {
    Ascii,
    Utf8,
}

/// A field within a compound datatype.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundField {
    /// Field name.
    pub name: String,
    /// Field datatype.
    pub datatype: Datatype,
    /// Byte offset within the compound type.
    pub offset: usize,
}

/// A member of an enumeration datatype.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct EnumMember {
    /// Member name.
    pub name: String,
    /// Integer value (stored as i64; width determined by base type).
    pub value: i64,
}

/// Object reference types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceType {
    /// Reference to an object (group, dataset).
    Object,
    /// Reference to a region within a dataset.
    Region,
}

impl Datatype {
    /// Returns the size in bytes of a single element of this type.
    ///
    /// Returns `None` for variable-length types.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Datatype::Boolean => Some(1),
            Datatype::Integer { bits, .. } | Datatype::Float { bits, .. } => Some(bits.get() / 8),
            Datatype::Complex { component_bits, .. } => Some(component_bits.get() / 4),
            Datatype::FixedString { length, .. } => Some(*length),
            Datatype::VariableString { .. } => None,
            Datatype::Opaque { size, .. } => Some(*size),
            #[cfg(feature = "alloc")]
            Datatype::Compound { size, .. } => Some(*size),
            #[cfg(feature = "alloc")]
            Datatype::Array { base, dims } => base
                .element_size()
                .map(|s| s * dims.iter().product::<usize>()),
            #[cfg(feature = "alloc")]
            Datatype::Enum { base, .. } => base.element_size(),
            #[cfg(feature = "alloc")]
            Datatype::VarLen { .. } => None,
            Datatype::Reference(_) => Some(8), // 8-byte object reference
        }
    }
}
