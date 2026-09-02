//! Arrow datatype model for the Consus workspace.
//!
//! This module defines a Rust-native Arrow type vocabulary without
//! depending on the upstream Arrow crate. It is intended to be used
//! by the `consus-arrow` crate as the semantic layer for arrays,
//! schemas, and bridge descriptors.
//!
//! ## Invariants
//!
//! - Types are explicit and value-semantic.
//! - Fixed-width and variable-width types are distinguished.
//! - Nested types carry their element or field structure.
//! - Temporal units are explicit for timestamp and duration values.
//!
//! ## Scope
//!
//! This module defines the canonical type model used by the crate:
//! - primitive scalar types
//! - variable-width binary and UTF-8 strings
//! - nested list, struct, and map types
//! - temporal and decimal types
//! - dictionary and union descriptors
//! - conversion into `consus_core::Datatype`

mod descriptors;

pub use descriptors::{
    DecimalType, DurationType, FixedSizeBinaryType, IntSign, TimeUnit, TimestampType,
};
#[cfg(feature = "alloc")]
pub use descriptors::{DictionaryType, ListType, MapType, StructType, UnionType};

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::String, vec::Vec};

use consus_core::{ByteOrder, Datatype};

/// Canonical Arrow datatype model for Consus.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrowDataType {
    /// Boolean value.
    Boolean,
    /// Signed integer with bit width.
    Int {
        /// Bit width.
        bit_width: u8,
        /// Signedness.
        sign: IntSign,
    },
    /// Floating-point value.
    Float {
        /// Bit width.
        bit_width: u8,
    },
    /// UTF-8 string.
    Utf8,
    /// Binary value.
    Binary,
    /// Fixed-size binary.
    FixedSizeBinary(FixedSizeBinaryType),
    /// Decimal value.
    Decimal(DecimalType),
    /// Date type with day resolution.
    Date32,
    /// Date type with millisecond resolution.
    Date64,
    /// Timestamp value.
    Timestamp(TimestampType),
    /// Duration value.
    Duration(DurationType),
    /// Interval type.
    Interval,
    /// List type.
    #[cfg(feature = "alloc")]
    List(ListType),
    /// Map type.
    #[cfg(feature = "alloc")]
    Map(MapType),
    /// Struct type.
    #[cfg(feature = "alloc")]
    Struct(StructType),
    /// Union type.
    #[cfg(feature = "alloc")]
    Union(UnionType),
    /// Dictionary encoded type.
    #[cfg(feature = "alloc")]
    Dictionary(DictionaryType),
    /// Null type.
    Null,
    /// Extension type name.
    #[cfg(feature = "alloc")]
    Extension {
        /// Extension identifier.
        name: String,
    },
}

impl ArrowDataType {
    /// Returns `true` if the datatype has fixed-width physical storage.
    #[must_use]
    pub fn is_fixed_width(&self) -> bool {
        match self {
            Self::Boolean => true,
            Self::Int { .. } => true,
            Self::Float { .. } => true,
            Self::Utf8 => false,
            Self::Binary => false,
            Self::FixedSizeBinary(_) => true,
            Self::Decimal(_) => true,
            Self::Date32 => true,
            Self::Date64 => true,
            Self::Timestamp(_) => true,
            Self::Duration(_) => true,
            Self::Interval => true,
            #[cfg(feature = "alloc")]
            Self::List(_) => false,
            #[cfg(feature = "alloc")]
            Self::Map(_) => false,
            #[cfg(feature = "alloc")]
            Self::Struct(_) => false,
            #[cfg(feature = "alloc")]
            Self::Union(_) => false,
            #[cfg(feature = "alloc")]
            Self::Dictionary(_) => false,
            Self::Null => true,
            #[cfg(feature = "alloc")]
            Self::Extension { .. } => false,
        }
    }

    /// Returns `true` if the datatype is nested.
    #[must_use]
    pub fn is_nested(&self) -> bool {
        #[cfg(feature = "alloc")]
        {
            matches!(
                self,
                Self::List(_) | Self::Map(_) | Self::Struct(_) | Self::Union(_)
            )
        }

        #[cfg(not(feature = "alloc"))]
        {
            false
        }
    }

    /// Returns `true` if the datatype is variable-width.
    #[must_use]
    pub fn is_variable_width(&self) -> bool {
        !self.is_fixed_width()
    }

    /// Returns `true` if the datatype is temporal.
    #[must_use]
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            Self::Date32 | Self::Date64 | Self::Timestamp(_) | Self::Duration(_)
        )
    }

    /// Convert to the closest `consus_core::Datatype`.
    ///
    /// This conversion is conservative and preserves structure by choosing
    /// the nearest canonical `consus_core` representation.
    #[must_use]
    pub fn to_consus_datatype(&self) -> Datatype {
        match self {
            Self::Boolean => Datatype::Boolean,
            Self::Int { bit_width, sign } => Datatype::Integer {
                bits: core::num::NonZeroUsize::new((*bit_width).into())
                    .expect("Arrow integer bit width must be non-zero"),
                byte_order: ByteOrder::LittleEndian,
                signed: matches!(sign, IntSign::Signed),
            },
            Self::Float { bit_width } => Datatype::Float {
                bits: core::num::NonZeroUsize::new((*bit_width).into())
                    .expect("Arrow float bit width must be non-zero"),
                byte_order: ByteOrder::LittleEndian,
            },
            Self::Utf8 => Datatype::VariableString {
                encoding: consus_core::StringEncoding::Utf8,
            },
            Self::Binary => Datatype::Opaque {
                size: 0,
                #[cfg(feature = "alloc")]
                tag: None,
            },
            Self::FixedSizeBinary(size) => Datatype::Opaque {
                size: size.size,
                #[cfg(feature = "alloc")]
                tag: None,
            },
            Self::Decimal(decimal) => Datatype::FixedString {
                length: decimal
                    .precision
                    .saturating_add(decimal.scale.unsigned_abs()),
                encoding: consus_core::StringEncoding::Ascii,
            },
            Self::Date32 => Datatype::Integer {
                bits: core::num::NonZeroUsize::new(32).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            },
            Self::Date64 => Datatype::Integer {
                bits: core::num::NonZeroUsize::new(64).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            },
            Self::Timestamp(_) => Datatype::Integer {
                bits: core::num::NonZeroUsize::new(64).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            },
            Self::Duration(_) => Datatype::Integer {
                bits: core::num::NonZeroUsize::new(64).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            },
            Self::Interval => Datatype::Compound {
                #[cfg(feature = "alloc")]
                fields: Vec::new(),
                #[cfg(feature = "alloc")]
                size: 16,
            },
            #[cfg(feature = "alloc")]
            Self::List(list) => Datatype::VarLen {
                base: Box::new(list.element_type.to_consus_datatype()),
            },
            #[cfg(feature = "alloc")]
            Self::Map(map) => Datatype::Compound {
                fields: Vec::new(),
                size: map
                    .value_type
                    .to_consus_datatype()
                    .element_size()
                    .unwrap_or(0),
            },
            #[cfg(feature = "alloc")]
            Self::Struct(_) => Datatype::Compound {
                fields: Vec::new(),
                size: 0,
            },
            #[cfg(feature = "alloc")]
            Self::Union(_) => Datatype::Compound {
                fields: Vec::new(),
                size: 0,
            },
            #[cfg(feature = "alloc")]
            Self::Dictionary(dictionary) => dictionary.value_type.to_consus_datatype(),
            Self::Null => Datatype::Opaque {
                size: 0,
                #[cfg(feature = "alloc")]
                tag: None,
            },
            #[cfg(feature = "alloc")]
            Self::Extension { .. } => Datatype::Opaque {
                size: 0,
                #[cfg(feature = "alloc")]
                tag: None,
            },
            #[cfg(not(feature = "alloc"))]
            _ => Datatype::Opaque { size: 0 },
        }
    }
}

/// Convert a fixed bit width into a byte width when it is a multiple of 8.
#[must_use]
pub const fn bit_width_to_byte_width(bit_width: u8) -> Option<usize> {
    if bit_width.is_multiple_of(8) {
        Some((bit_width / 8) as usize)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_detection_matches_primitives() {
        assert!(ArrowDataType::Boolean.is_fixed_width());
        assert!(
            ArrowDataType::Int {
                bit_width: 32,
                sign: IntSign::Signed
            }
            .is_fixed_width()
        );
        assert!(ArrowDataType::Float { bit_width: 64 }.is_fixed_width());
        assert!(!ArrowDataType::Utf8.is_fixed_width());
        assert!(!ArrowDataType::Binary.is_fixed_width());
        assert!(ArrowDataType::FixedSizeBinary(FixedSizeBinaryType::new(16)).is_fixed_width());
        assert!(ArrowDataType::Date32.is_fixed_width());
    }

    #[test]
    fn nested_detection_matches_container_types() {
        #[cfg(feature = "alloc")]
        {
            assert!(ArrowDataType::List(ListType::new(ArrowDataType::Boolean, true)).is_nested());
            assert!(
                ArrowDataType::Map(MapType::new(
                    ArrowDataType::Utf8,
                    ArrowDataType::Binary,
                    true
                ))
                .is_nested()
            );
            assert!(ArrowDataType::Struct(StructType { fields: Vec::new() }).is_nested());
        }
        assert!(!ArrowDataType::Boolean.is_nested());
    }

    #[test]
    fn bit_width_to_byte_width_requires_multiples_of_eight() {
        assert_eq!(bit_width_to_byte_width(8), Some(1));
        assert_eq!(bit_width_to_byte_width(16), Some(2));
        assert_eq!(bit_width_to_byte_width(32), Some(4));
        assert_eq!(bit_width_to_byte_width(7), None);
        assert_eq!(bit_width_to_byte_width(15), None);
    }

    #[test]
    fn conversion_to_consus_datatype_is_defined() {
        let ty = ArrowDataType::Int {
            bit_width: 32,
            sign: IntSign::Signed,
        };
        match ty.to_consus_datatype() {
            Datatype::Integer { bits, signed, .. } => {
                assert_eq!(bits.get(), 32);
                assert!(signed);
            }
            other => panic!("expected Integer, got {other:?}"),
        }

        let utf8 = ArrowDataType::Utf8.to_consus_datatype();
        match utf8 {
            Datatype::VariableString { .. } => {}
            other => panic!("expected VariableString, got {other:?}"),
        }
    }
}
