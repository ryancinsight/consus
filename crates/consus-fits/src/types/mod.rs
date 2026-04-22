//! FITS scalar, structural, and HDU classification types.
//!
//! ## Scope
//!
//! This module is the single source of truth for FITS-facing type semantics in
//! `consus-fits`. It defines:
//! - `BITPIX` storage encodings
//! - canonical mapping from FITS image element types to `consus_core::Datatype`
//! - HDU classification for primary and extension units
//!
//! ## Specification
//!
//! FITS Standard 4.0 defines the following `BITPIX` values for image arrays:
//! - `8`    => unsigned 8-bit integer
//! - `16`   => signed 16-bit integer
//! - `32`   => signed 32-bit integer
//! - `64`   => signed 64-bit integer
//! - `-32`  => IEEE 754 binary32
//! - `-64`  => IEEE 754 binary64
//!
//! FITS stores multi-byte numeric values in big-endian byte order.
//!
//! ## Invariants
//!
//! - Every supported `BITPIX` value maps to exactly one canonical
//!   `consus_core::Datatype`.
//! - Unsupported `BITPIX` values are rejected explicitly.
//! - HDU classification is value-semantic and does not depend on I/O state.
//! - This module contains no parsing or wire-format logic.
//!
//! ## Architecture
//!
//! This module is intentionally self-contained and depends only on
//! `consus-core`. Header parsing and keyword interpretation must depend on this
//! module rather than duplicating `BITPIX` or HDU classification logic.

pub mod format;

use core::fmt;
use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Datatype, Error, Result};

pub use format::{
    BinaryFormatCode, binary_format_element_size, binary_format_to_datatype, parse_binary_format,
    tform_to_datatype,
};

/// FITS image storage encoding derived from the `BITPIX` header keyword.
///
/// ## Mapping
///
/// | `BITPIX` | Meaning | Canonical datatype |
/// |----------|---------|--------------------|
/// | `8` | unsigned 8-bit integer | `Datatype::Integer { bits: 8, signed: false }` |
/// | `16` | signed 16-bit integer | `Datatype::Integer { bits: 16, signed: true }` |
/// | `32` | signed 32-bit integer | `Datatype::Integer { bits: 32, signed: true }` |
/// | `64` | signed 64-bit integer | `Datatype::Integer { bits: 64, signed: true }` |
/// | `-32` | IEEE 754 binary32 | `Datatype::Float { bits: 32 }` |
/// | `-64` | IEEE 754 binary64 | `Datatype::Float { bits: 64 }` |
///
/// FITS uses big-endian storage for all multi-byte numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bitpix {
    /// Unsigned 8-bit integer image element.
    U8,
    /// Signed 16-bit integer image element.
    I16,
    /// Signed 32-bit integer image element.
    I32,
    /// Signed 64-bit integer image element.
    I64,
    /// IEEE 754 binary32 image element.
    F32,
    /// IEEE 754 binary64 image element.
    F64,
}

impl Bitpix {
    /// Parse a FITS `BITPIX` integer value.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if `value` is not one of the FITS
    /// Standard 4.0 image encodings.
    pub fn from_i64(value: i64) -> Result<Self> {
        match value {
            8 => Ok(Self::U8),
            16 => Ok(Self::I16),
            32 => Ok(Self::I32),
            64 => Ok(Self::I64),
            -32 => Ok(Self::F32),
            -64 => Ok(Self::F64),
            _ => Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!("unsupported FITS BITPIX value: {value}"),
            }),
        }
    }

    /// Return the canonical FITS integer value for this encoding.
    pub const fn to_i64(self) -> i64 {
        match self {
            Self::U8 => 8,
            Self::I16 => 16,
            Self::I32 => 32,
            Self::I64 => 64,
            Self::F32 => -32,
            Self::F64 => -64,
        }
    }

    /// Return the canonical Consus datatype for this FITS image encoding.
    ///
    /// ## Invariant
    ///
    /// The returned datatype is the unique canonical representation for the
    /// corresponding FITS image element type.
    pub fn to_datatype(self) -> Datatype {
        match self {
            Self::U8 => Datatype::Integer {
                bits: nonzero(8),
                byte_order: ByteOrder::BigEndian,
                signed: false,
            },
            Self::I16 => Datatype::Integer {
                bits: nonzero(16),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            },
            Self::I32 => Datatype::Integer {
                bits: nonzero(32),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            },
            Self::I64 => Datatype::Integer {
                bits: nonzero(64),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            },
            Self::F32 => Datatype::Float {
                bits: nonzero(32),
                byte_order: ByteOrder::BigEndian,
            },
            Self::F64 => Datatype::Float {
                bits: nonzero(64),
                byte_order: ByteOrder::BigEndian,
            },
        }
    }

    /// Return the element size in bytes.
    pub const fn element_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Return whether this encoding is floating-point.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Return whether this encoding is integral.
    pub const fn is_integer(self) -> bool {
        !self.is_float()
    }
}

impl fmt::Display for Bitpix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_i64())
    }
}

impl TryFrom<i64> for Bitpix {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        Self::from_i64(value)
    }
}

impl From<Bitpix> for Datatype {
    fn from(value: Bitpix) -> Self {
        value.to_datatype()
    }
}

/// FITS Header/Data Unit classification.
///
/// ## Semantics
///
/// - `Primary` denotes the first HDU in a FITS file.
/// - `Image` denotes an `IMAGE` extension HDU.
/// - `Table` denotes an ASCII table extension (`TABLE`).
/// - `BinTable` denotes a binary table extension (`BINTABLE`).
///
/// This enum is intentionally limited to the foundational HDU classes requested
/// for the current implementation stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HduType {
    /// Primary HDU.
    Primary,
    /// Image extension HDU.
    Image,
    /// ASCII table extension HDU.
    Table,
    /// Binary table extension HDU.
    BinTable,
}

impl HduType {
    /// Classify an HDU from the FITS `XTENSION` keyword value.
    ///
    /// `None` denotes the primary HDU, which is identified by `SIMPLE = T`
    /// rather than `XTENSION`.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if the extension type is not one of the
    /// foundational FITS HDU classes supported by this crate stage.
    pub fn from_xtension(xtension: Option<&str>) -> Result<Self> {
        match xtension {
            None => Ok(Self::Primary),
            Some(value) => match value.trim() {
                "IMAGE" => Ok(Self::Image),
                "TABLE" => Ok(Self::Table),
                "BINTABLE" => Ok(Self::BinTable),
                other => Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: alloc::format!("unsupported FITS XTENSION value: {other}"),
                }),
            },
        }
    }

    /// Return the canonical `XTENSION` keyword value for extension HDUs.
    ///
    /// Returns `None` for the primary HDU.
    pub const fn xtension_keyword_value(self) -> Option<&'static str> {
        match self {
            Self::Primary => None,
            Self::Image => Some("IMAGE"),
            Self::Table => Some("TABLE"),
            Self::BinTable => Some("BINTABLE"),
        }
    }

    /// Return whether this HDU is the primary HDU.
    pub const fn is_primary(self) -> bool {
        matches!(self, Self::Primary)
    }

    /// Return whether this HDU is an extension HDU.
    pub const fn is_extension(self) -> bool {
        !self.is_primary()
    }
}

impl fmt::Display for HduType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primary => f.write_str("PRIMARY"),
            Self::Image => f.write_str("IMAGE"),
            Self::Table => f.write_str("TABLE"),
            Self::BinTable => f.write_str("BINTABLE"),
        }
    }
}

fn nonzero(bits: usize) -> NonZeroUsize {
    match NonZeroUsize::new(bits) {
        Some(value) => value,
        None => unreachable!("BITPIX mapping requires non-zero bit widths"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitpix_parses_all_standard_values() {
        assert_eq!(Bitpix::from_i64(8).unwrap(), Bitpix::U8);
        assert_eq!(Bitpix::from_i64(16).unwrap(), Bitpix::I16);
        assert_eq!(Bitpix::from_i64(32).unwrap(), Bitpix::I32);
        assert_eq!(Bitpix::from_i64(64).unwrap(), Bitpix::I64);
        assert_eq!(Bitpix::from_i64(-32).unwrap(), Bitpix::F32);
        assert_eq!(Bitpix::from_i64(-64).unwrap(), Bitpix::F64);
    }

    #[test]
    fn bitpix_rejects_non_standard_values() {
        assert!(Bitpix::from_i64(0).is_err());
        assert!(Bitpix::from_i64(24).is_err());
        assert!(Bitpix::from_i64(-16).is_err());
    }

    #[test]
    fn bitpix_round_trips_to_integer_value() {
        let values = [
            Bitpix::U8,
            Bitpix::I16,
            Bitpix::I32,
            Bitpix::I64,
            Bitpix::F32,
            Bitpix::F64,
        ];

        for bitpix in values {
            let parsed = Bitpix::from_i64(bitpix.to_i64()).unwrap();
            assert_eq!(parsed, bitpix);
        }
    }

    #[test]
    fn bitpix_maps_to_canonical_consus_datatype() {
        assert_eq!(
            Bitpix::U8.to_datatype(),
            Datatype::Integer {
                bits: NonZeroUsize::new(8).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: false,
            }
        );

        assert_eq!(
            Bitpix::I16.to_datatype(),
            Datatype::Integer {
                bits: NonZeroUsize::new(16).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            }
        );

        assert_eq!(
            Bitpix::I32.to_datatype(),
            Datatype::Integer {
                bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            }
        );

        assert_eq!(
            Bitpix::I64.to_datatype(),
            Datatype::Integer {
                bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            }
        );

        assert_eq!(
            Bitpix::F32.to_datatype(),
            Datatype::Float {
                bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );

        assert_eq!(
            Bitpix::F64.to_datatype(),
            Datatype::Float {
                bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );
    }

    #[test]
    fn bitpix_reports_element_size_and_kind() {
        assert_eq!(Bitpix::U8.element_size(), 1);
        assert_eq!(Bitpix::I16.element_size(), 2);
        assert_eq!(Bitpix::I32.element_size(), 4);
        assert_eq!(Bitpix::I64.element_size(), 8);
        assert_eq!(Bitpix::F32.element_size(), 4);
        assert_eq!(Bitpix::F64.element_size(), 8);

        assert!(Bitpix::U8.is_integer());
        assert!(Bitpix::I16.is_integer());
        assert!(Bitpix::I32.is_integer());
        assert!(Bitpix::I64.is_integer());
        assert!(Bitpix::F32.is_float());
        assert!(Bitpix::F64.is_float());
    }

    #[test]
    fn hdu_type_classifies_primary_and_extensions() {
        assert_eq!(HduType::from_xtension(None).unwrap(), HduType::Primary);
        assert_eq!(
            HduType::from_xtension(Some("IMAGE")).unwrap(),
            HduType::Image
        );
        assert_eq!(
            HduType::from_xtension(Some("TABLE")).unwrap(),
            HduType::Table
        );
        assert_eq!(
            HduType::from_xtension(Some("BINTABLE")).unwrap(),
            HduType::BinTable
        );
    }

    #[test]
    fn hdu_type_rejects_unsupported_extensions() {
        assert!(HduType::from_xtension(Some("A3DTABLE")).is_err());
        assert!(HduType::from_xtension(Some("RANDOM")).is_err());
    }

    #[test]
    fn hdu_type_reports_keyword_value_and_role() {
        assert_eq!(HduType::Primary.xtension_keyword_value(), None);
        assert_eq!(HduType::Image.xtension_keyword_value(), Some("IMAGE"));
        assert_eq!(HduType::Table.xtension_keyword_value(), Some("TABLE"));
        assert_eq!(HduType::BinTable.xtension_keyword_value(), Some("BINTABLE"));

        assert!(HduType::Primary.is_primary());
        assert!(!HduType::Primary.is_extension());
        assert!(HduType::Image.is_extension());
        assert!(HduType::Table.is_extension());
        assert!(HduType::BinTable.is_extension());
    }
}
