//! Explicit IEEE floating-point storage encodings for opt-in dataset admission.

use consus_core::{ByteOrder, Datatype};
use core::num::NonZeroUsize;

/// An IEEE binary interchange representation with an explicit storage byte order.
///
/// Unlike canonical `Datatype::Float`, this requests the full standard sign,
/// exponent, significand, bias, precision, offset and normalization contract.
/// No native-endian alias is used at a persisted-data boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IeeeFloatEncoding {
    /// IEEE binary16, little-endian.
    Binary16LittleEndian,
    /// IEEE binary16, big-endian.
    Binary16BigEndian,
    /// IEEE binary32, little-endian.
    Binary32LittleEndian,
    /// IEEE binary32, big-endian.
    Binary32BigEndian,
    /// IEEE binary64, little-endian.
    Binary64LittleEndian,
    /// IEEE binary64, big-endian.
    Binary64BigEndian,
}

impl IeeeFloatEncoding {
    pub(crate) fn datatype(self) -> Datatype {
        let (width, byte_order) = match self {
            Self::Binary16LittleEndian => (16, ByteOrder::LittleEndian),
            Self::Binary16BigEndian => (16, ByteOrder::BigEndian),
            Self::Binary32LittleEndian => (32, ByteOrder::LittleEndian),
            Self::Binary32BigEndian => (32, ByteOrder::BigEndian),
            Self::Binary64LittleEndian => (64, ByteOrder::LittleEndian),
            Self::Binary64BigEndian => (64, ByteOrder::BigEndian),
        };
        Datatype::Float {
            bits: NonZeroUsize::new(width).expect("IEEE interchange widths are nonzero"),
            byte_order,
        }
    }
}
