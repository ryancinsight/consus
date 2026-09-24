//! Zero-copy fixed-width scalar reads and multi-byte integer read/write.
//!
//! This module is the single home for fixed-width byte-order decoding in
//! Consus: the [`EndianScalar`] scalars, the arbitrary-width integer readers
//! used by the HDF5 variable `offset_size`/`length_size` fields, and the
//! little/big-endian read/write and byte-swap helpers. Format crates decode
//! through these functions rather than re-implementing the byte grafts.

use core::convert::TryInto;

use super::super::types::datatype::ByteOrder;

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for i8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Describes a scalar that can be read from an ordered fixed-width byte slice.
///
/// Implementations cover the fixed-width numeric scalars (`u8`/`i8` through
/// `u64`/`i64`, plus `f32`/`f64`). The associated width is resolved at compile
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

/// Reads an unsigned integer of `width` bytes (`0..=8`) with `byte_order`.
///
/// `width == 0` yields `Some(0)`. Returns `None` when `width > 8` or `bytes`
/// is shorter than `width`. This is the runtime-width form used by HDF5's
/// variable `offset_size`/`length_size` fields; [`read_uint_arbitrary`] is the
/// const-generic spelling of the same operation.
pub fn read_uint_width(bytes: &[u8], width: usize, byte_order: ByteOrder) -> Option<u64> {
    if width > 8 {
        return None;
    }
    let bytes = bytes.get(..width)?;
    Some(match byte_order {
        ByteOrder::LittleEndian => bytes
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &b)| acc | (u64::from(b) << (8 * i))),
        ByteOrder::BigEndian => bytes.iter().fold(0u64, |acc, &b| (acc << 8) | u64::from(b)),
    })
}

/// Const-generic width form of [`read_uint_width`], covering widths `0..=8`.
///
/// # Panics
///
/// Panics if `W > 8` or `bytes` is shorter than `W`.
pub fn read_uint_arbitrary<const W: usize>(bytes: &[u8], byte_order: ByteOrder) -> u64 {
    read_uint_width(bytes, W, byte_order)
        .expect("read_uint_arbitrary: width must be 0..=8 and buffer long enough")
}

/// Reads a signed integer of `width` bytes (`0..=8`), sign-extended.
///
/// Returns `None` under the same conditions as [`read_uint_width`].
pub fn read_int_width(bytes: &[u8], width: usize, byte_order: ByteOrder) -> Option<i64> {
    let value = read_uint_width(bytes, width, byte_order)?;
    Some(sign_extend(value, width))
}

/// Sign-extends the low `width * 8` bits of `value` to `i64`.
///
/// If the most-significant bit of the `width`-byte value is set, the upper
/// bits of the returned `i64` are filled with ones (arithmetic shift).
pub fn sign_extend(value: u64, width: usize) -> i64 {
    let bits = width * 8;
    if bits == 0 || bits >= 64 {
        return value as i64;
    }
    let shift = 64 - bits;
    ((value as i64) << shift) >> shift
}

/// Read a little-endian unsigned integer of `width` bytes from `buf`.
///
/// ## Supported widths
///
/// 1, 2, 4, 8 bytes.
///
/// ## Panics
///
/// Panics if `buf.len() < width` or `width` is not in {1, 2, 4, 8}.
#[inline]
pub fn read_uint_le(buf: &[u8], width: usize) -> u64 {
    match width {
        1 => u64::from(buf[0]),
        2 => u64::from(u16::from_le_bytes([buf[0], buf[1]])),
        4 => u64::from(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])),
        8 => u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]),
        _ => panic!("unsupported integer width: {width}"),
    }
}

/// Read a big-endian unsigned integer of `width` bytes from `buf`.
///
/// ## Supported widths
///
/// 1, 2, 4, 8 bytes.
///
/// ## Panics
///
/// Panics if `buf.len() < width` or `width` is not in {1, 2, 4, 8}.
#[inline]
pub fn read_uint_be(buf: &[u8], width: usize) -> u64 {
    match width {
        1 => u64::from(buf[0]),
        2 => u64::from(u16::from_be_bytes([buf[0], buf[1]])),
        4 => u64::from(u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])),
        8 => u64::from_be_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]),
        _ => panic!("unsupported integer width: {width}"),
    }
}

/// Write a little-endian unsigned integer of `width` bytes to `buf`.
///
/// ## Supported widths
///
/// 1, 2, 4, 8 bytes.
///
/// ## Panics
///
/// Panics if `buf.len() < width`, `width` is not in {1, 2, 4, 8},
/// or `value` exceeds the range representable in `width` bytes
/// (for width < 8).
#[inline]
pub fn write_uint_le(buf: &mut [u8], width: usize, value: u64) {
    match width {
        1 => buf[0] = value as u8,
        2 => buf[..2].copy_from_slice(&(value as u16).to_le_bytes()),
        4 => buf[..4].copy_from_slice(&(value as u32).to_le_bytes()),
        8 => buf[..8].copy_from_slice(&value.to_le_bytes()),
        _ => panic!("unsupported integer width: {width}"),
    }
}

/// Write a big-endian unsigned integer of `width` bytes to `buf`.
///
/// ## Supported widths
///
/// 1, 2, 4, 8 bytes.
///
/// ## Panics
///
/// Panics if `buf.len() < width`, `width` is not in {1, 2, 4, 8},
/// or `value` exceeds the range representable in `width` bytes
/// (for width < 8).
#[inline]
pub fn write_uint_be(buf: &mut [u8], width: usize, value: u64) {
    match width {
        1 => buf[0] = value as u8,
        2 => buf[..2].copy_from_slice(&(value as u16).to_be_bytes()),
        4 => buf[..4].copy_from_slice(&(value as u32).to_be_bytes()),
        8 => buf[..8].copy_from_slice(&value.to_be_bytes()),
        _ => panic!("unsupported integer width: {width}"),
    }
}

/// Read a file offset of `size` bytes (little-endian) from a buffer.
///
/// Supports 2, 4, and 8 byte offsets as per the HDF5 specification.
///
/// ## Panics
///
/// Panics if `size` is not in {2, 4, 8}, or if `buf.len() < size`.
#[inline]
pub fn read_offset(buf: &[u8], size: usize) -> u64 {
    read_uint_le(buf, size)
}

/// Read a file length of `size` bytes (little-endian) from a buffer.
///
/// Semantically identical to [`read_offset`] but distinct for documentation
/// clarity. HDF5 files store lengths and offsets with the same encoding but
/// distinct semantic roles.
///
/// ## Panics
///
/// Panics if `size` is not in {2, 4, 8}, or if `buf.len() < size`.
#[inline]
pub fn read_length(buf: &[u8], size: usize) -> u64 {
    read_uint_le(buf, size)
}

/// Swap byte order of a value in-place within a mutable buffer.
///
/// Reverses the first `width` bytes of `buf`.
///
/// ## Proof of involution
///
/// Reversing a sequence twice yields the original sequence:
/// `reverse(reverse(s)) = s` for all finite sequences `s`.
///
/// ## Panics
///
/// Panics if `buf.len() < width`.
#[inline]
pub fn swap_bytes(buf: &mut [u8], width: usize) {
    let slice = &mut buf[..width];
    slice.reverse();
}

impl EndianScalar for u8 {
    const BYTE_WIDTH: usize = 1;

    fn from_bytes(bytes: &[u8], _byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(bytes[0])
    }
}

impl EndianScalar for i8 {
    const BYTE_WIDTH: usize = 1;

    fn from_bytes(bytes: &[u8], _byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(bytes[0] as i8)
    }
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

impl EndianScalar for f32 {
    const BYTE_WIDTH: usize = 4;

    fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(match byte_order {
            ByteOrder::LittleEndian => Self::from_le_bytes(bytes),
            ByteOrder::BigEndian => Self::from_be_bytes(bytes),
        })
    }
}

impl EndianScalar for f64 {
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
    use super::{
        EndianScalar, read_int_width, read_integer, read_uint_arbitrary, read_uint_width,
        sign_extend,
    };
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

    #[test]
    fn reads_narrow_scalars_and_floats() {
        assert_eq!(
            read_integer::<u8>(&[0xAB], ByteOrder::LittleEndian),
            Some(0xAB)
        );
        assert_eq!(
            read_integer::<i8>(&[0xFF], ByteOrder::BigEndian),
            Some(-1_i8)
        );
        assert_round_trip(
            &[0x00, 0x00, 0xC0, 0x3F],
            &[0x3F, 0xC0, 0x00, 0x00],
            1.5_f32,
        );
        assert_round_trip(
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x40],
            &[0x40, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            2.5_f64,
        );
    }

    #[test]
    fn read_uint_width_covers_arbitrary_widths() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(read_uint_width(&data, 0, ByteOrder::LittleEndian), Some(0));
        assert_eq!(
            read_uint_width(&data, 3, ByteOrder::LittleEndian),
            Some(0x0003_0201)
        );
        assert_eq!(
            read_uint_width(&data, 3, ByteOrder::BigEndian),
            Some(0x0001_0203)
        );
        assert_eq!(
            read_uint_width(&data, 8, ByteOrder::LittleEndian),
            Some(0x0807_0605_0403_0201)
        );
        assert_eq!(read_uint_width(&data, 9, ByteOrder::LittleEndian), None);
        assert_eq!(
            read_uint_width(&data[..2], 4, ByteOrder::LittleEndian),
            None
        );
        assert_eq!(
            read_uint_arbitrary::<2>(&data, ByteOrder::LittleEndian),
            0x0201
        );
    }

    #[test]
    fn read_int_width_sign_extends() {
        assert_eq!(
            read_int_width(&[0xFF], 1, ByteOrder::LittleEndian),
            Some(-1)
        );
        assert_eq!(
            read_int_width(&[0xFE, 0xFF], 2, ByteOrder::LittleEndian),
            Some(-2)
        );
        assert_eq!(
            read_int_width(&[0x00, 0x80], 2, ByteOrder::LittleEndian),
            Some(-32768)
        );
        assert_eq!(sign_extend(0, 2), 0);
    }
}
