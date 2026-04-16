//! Canonical datatype representation for the Consus storage model.
//!
//! ## Specification
//!
//! The datatype system is a union of types expressible across HDF5, Zarr,
//! netCDF-4, and Parquet. Each format backend maps its native type system
//! to/from these canonical types.
//!
//! ### Canonicalization Invariant
//!
//! For any format-specific type `T_f` with a canonical mapping:
//!   `canonicalize(T_f) == canonicalize(T_f')` iff `T_f` and `T_f'` represent
//!   the same logical type, regardless of format origin.
//!
//! ### Element Size Contract
//!
//! For fixed-size types, `element_size()` returns `Some(n)` where `n > 0`.
//! For variable-length types (`VariableString`, `VarLen`), returns `None`.
//! Array and compound sizes are computed recursively from their components.

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::String, vec::Vec};

use ::core::num::NonZeroUsize;

use super::reference::ReferenceType;

/// Byte order for multi-byte scalar types.
///
/// Determines the byte-level representation of integers, floats, and
/// complex components in storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    /// Least-significant byte first (x86, ARM default).
    LittleEndian,
    /// Most-significant byte first (network byte order, POWER).
    BigEndian,
}

/// String character encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringEncoding {
    /// ASCII (7-bit, single-byte characters).
    Ascii,
    /// UTF-8 (variable-width Unicode encoding).
    Utf8,
}

/// A field within a compound datatype.
///
/// Compound fields are ordered and named. The `offset` is the byte
/// offset within the compound type's in-memory representation.
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
///
/// Enum members map symbolic names to integer values of the base type.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct EnumMember {
    /// Member name.
    pub name: String,
    /// Integer value (widened to i64; actual width is the enum's base type).
    pub value: i64,
}

/// Canonical datatype for the Consus storage model.
///
/// This enum represents every type expressible across supported formats.
/// Format-specific backends convert to/from this representation.
///
/// ## Element Size
///
/// - Fixed-size types: `element_size()` returns `Some(bytes)`.
/// - Variable-length types: `element_size()` returns `None`.
///
/// ## Representation Invariants
///
/// - `Integer { bits, .. }`: `bits` is a multiple of 8, in {8, 16, 32, 64, 128}.
/// - `Float { bits, .. }`: `bits` ∈ {16, 32, 64, 128}.
/// - `Complex { component_bits, .. }`: `component_bits` ∈ {32, 64}
///   (total size = 2 × component_bits / 8).
/// - `FixedString { length, .. }`: `length > 0`.
/// - `Compound { fields, size }`: `size >= sum(field sizes)`, fields non-overlapping.
/// - `Array { dims, .. }`: all `dims[i] > 0`.
#[derive(Debug, Clone, PartialEq)]
pub enum Datatype {
    /// Boolean (1-byte logical, 0 = false, nonzero = true).
    Boolean,

    /// Signed or unsigned integer.
    Integer {
        /// Bit width (multiple of 8).
        bits: NonZeroUsize,
        /// Byte order.
        byte_order: ByteOrder,
        /// Whether the integer is signed.
        signed: bool,
    },

    /// IEEE 754 floating-point.
    Float {
        /// Bit width (16, 32, 64, or 128).
        bits: NonZeroUsize,
        /// Byte order.
        byte_order: ByteOrder,
    },

    /// Complex number (real + imaginary), each component an IEEE 754 float.
    Complex {
        /// Bit width of each component (32 or 64).
        component_bits: NonZeroUsize,
        /// Byte order.
        byte_order: ByteOrder,
    },

    /// Fixed-length string.
    FixedString {
        /// Byte length of the string storage.
        length: usize,
        /// Character encoding.
        encoding: StringEncoding,
    },

    /// Variable-length string (requires alloc for reading).
    VariableString {
        /// Character encoding.
        encoding: StringEncoding,
    },

    /// Opaque blob of fixed size.
    Opaque {
        /// Size in bytes.
        size: usize,
        /// Optional application-defined tag.
        #[cfg(feature = "alloc")]
        tag: Option<String>,
    },

    /// Compound type (struct-like): ordered named fields.
    #[cfg(feature = "alloc")]
    Compound {
        /// Ordered fields.
        fields: Vec<CompoundField>,
        /// Total size in bytes (may include padding).
        size: usize,
    },

    /// Fixed-size array of another datatype.
    #[cfg(feature = "alloc")]
    Array {
        /// Element datatype.
        base: Box<Datatype>,
        /// Array dimensions (each > 0).
        dims: Vec<usize>,
    },

    /// Enumeration: named integer constants over a base integer type.
    #[cfg(feature = "alloc")]
    Enum {
        /// Base integer datatype.
        base: Box<Datatype>,
        /// Named members.
        members: Vec<EnumMember>,
    },

    /// Variable-length sequence of another datatype.
    #[cfg(feature = "alloc")]
    VarLen {
        /// Element datatype.
        base: Box<Datatype>,
    },

    /// Reference to another object or region in the hierarchy.
    Reference(ReferenceType),
}

impl Datatype {
    /// Returns the size in bytes of a single element of this type.
    ///
    /// Returns `None` for variable-length types (`VariableString`, `VarLen`).
    ///
    /// ## Derivation
    ///
    /// - `Boolean` → 1 byte.
    /// - `Integer { bits }` → `bits / 8`.
    /// - `Float { bits }` → `bits / 8`.
    /// - `Complex { component_bits }` → `2 × component_bits / 8`.
    /// - `FixedString { length }` → `length`.
    /// - `Opaque { size }` → `size`.
    /// - `Compound { size }` → `size`.
    /// - `Array { base, dims }` → `base.element_size() × ∏ dims`.
    /// - `Enum { base }` → `base.element_size()`.
    /// - `Reference(_)` → 8 (HDF5 object reference size).
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Self::Boolean => Some(1),
            Self::Integer { bits, .. } | Self::Float { bits, .. } => Some(bits.get() / 8),
            Self::Complex { component_bits, .. } => Some(2 * component_bits.get() / 8),
            Self::FixedString { length, .. } => Some(*length),
            Self::VariableString { .. } => None,
            Self::Opaque { size, .. } => Some(*size),
            #[cfg(feature = "alloc")]
            Self::Compound { size, .. } => Some(*size),
            #[cfg(feature = "alloc")]
            Self::Array { base, dims } => base
                .element_size()
                .and_then(|s| s.checked_mul(dims.iter().product::<usize>())),
            #[cfg(feature = "alloc")]
            Self::Enum { base, .. } => base.element_size(),
            #[cfg(feature = "alloc")]
            Self::VarLen { .. } => None,
            Self::Reference(_) => Some(8),
        }
    }

    /// Whether this type is variable-length.
    pub fn is_variable_length(&self) -> bool {
        self.element_size().is_none()
    }

    /// Whether this type is a scalar numeric type (integer, float, or complex).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Integer { .. } | Self::Float { .. } | Self::Complex { .. }
        )
    }
}
