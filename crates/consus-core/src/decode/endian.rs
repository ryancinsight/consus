//! Zero-copy fixed-width scalar reads with explicit byte order.

use core::convert::TryInto;

use super::super::types::datatype::ByteOrder;

mod sealed {
    pub trait Sealed {}

    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
}

/// Describes a scalar that can be read from an ordered fixed-width byte slice.
///
/// Implementations are intentionally limited to the scalar widths supported
/// by the Consus datatype model. The associated width is resolved at compile
/// time, so [`read_integer`] monomorphizes to a direct native-endian
/// conversion for each scalar type.
pub trait EndianScalar: sealed::Sealed + Sized {
    /// The number of bytes required by the scalar representation.
    const BYTE_WIDTH: usize;

    /// Converts an exact-width byte slice using the requested byte order.
    fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self>;
}

/// Reads one fixed-width scalar without allocation or runtime type dispatch.
///
/// Returns `None` when `bytes` is shorter than the scalar's compile-time
/// width. The slice is borrowed for the duration of the conversion and no
/// intermediate buffer is allocated.
pub fn read_integer<T: EndianScalar>(bytes: &[u8], byte_order: ByteOrder) -> Option<T> {
    T::from_bytes(bytes.get(..T::BYTE_WIDTH)?, byte_order)
}

impl EndianScalar for u16 {
    const BYTE_WIDTH: usize = 2;

    fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(match byte_order {
            ByteOrder::LittleEndian => Self::from_le_bytes(bytes),
            ByteOrder::BigEndian => Self::from_be_bytes(bytes),
        })
    }
}

impl EndianScalar for i16 {
    const BYTE_WIDTH: usize = 2;

    fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(match byte_order {
            ByteOrder::LittleEndian => Self::from_le_bytes(bytes),
            ByteOrder::BigEndian => Self::from_be_bytes(bytes),
        })
    }
}

impl EndianScalar for u32 {
    const BYTE_WIDTH: usize = 4;

    fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(match byte_order {
            ByteOrder::LittleEndian => Self::from_le_bytes(bytes),
            ByteOrder::BigEndian => Self::from_be_bytes(bytes),
        })
    }
}

impl EndianScalar for i32 {
    const BYTE_WIDTH: usize = 4;

    fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(match byte_order {
            ByteOrder::LittleEndian => Self::from_le_bytes(bytes),
            ByteOrder::BigEndian => Self::from_be_bytes(bytes),
        })
    }
}

impl EndianScalar for u64 {
    const BYTE_WIDTH: usize = 8;

    fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(match byte_order {
            ByteOrder::LittleEndian => Self::from_le_bytes(bytes),
            ByteOrder::BigEndian => Self::from_be_bytes(bytes),
        })
    }
}

impl EndianScalar for i64 {
    const BYTE_WIDTH: usize = 8;

    fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(match byte_order {
            ByteOrder::LittleEndian => Self::from_le_bytes(bytes),
            ByteOrder::BigEndian => Self::from_be_bytes(bytes),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{EndianScalar, read_integer};
    use crate::types::datatype::ByteOrder;

    fn assert_round_trip<T>(little: &[u8], big: &[u8], expected: T)
    where
        T: EndianScalar + Copy + PartialEq + core::fmt::Debug,
    {
        assert_eq!(
            read_integer(little, ByteOrder::LittleEndian),
            Some(expected)
        );
        assert_eq!(read_integer(big, ByteOrder::BigEndian), Some(expected));
        assert_eq!(read_integer::<T>(&[], ByteOrder::LittleEndian), None);
    }

    #[test]
    fn reads_all_supported_scalar_widths_and_orders() {
        assert_round_trip(&[0x34, 0x12], &[0x12, 0x34], 0x1234_u16);
        assert_round_trip(&[0xCC, 0xED], &[0xED, 0xCC], -4_660_i16);
        assert_round_trip(
            &[0x78, 0x56, 0x34, 0x12],
            &[0x12, 0x34, 0x56, 0x78],
            0x1234_5678_u32,
        );
        assert_round_trip(
            &[0x88, 0xA9, 0xCB, 0xED],
            &[0xED, 0xCB, 0xA9, 0x88],
            -305_419_896_i32,
        );
        assert_round_trip(
            &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            0x0102_0304_0506_0708_u64,
        );
        assert_round_trip(
            &[0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF],
            &[0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8],
            -283_686_952_306_184_i64,
        );
    }
}
