//! Zero-copy fixed-width scalar reads and writes, and multi-byte integer read/write.
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

    /// Writes the scalar into an exact-width byte slice in `byte_order`.
    ///
    /// Returns `None`, writing nothing, when `out` is not exactly
    /// [`Self::BYTE_WIDTH`] bytes long.
    fn to_bytes(self, out: &mut [u8], byte_order: ByteOrder) -> Option<()>;
}

/// Widest [`EndianScalar::BYTE_WIDTH`]: every sealed scalar fits a buffer of
/// this size.
#[cfg(any(feature = "std", feature = "alloc"))]
const MAX_SCALAR_WIDTH: usize = 8;

#[cfg(feature = "std")]
mod stream;

#[cfg(feature = "std")]
pub use stream::{read_extend, read_from, write_to};

/// Reads one fixed-width scalar without allocation or runtime type dispatch.
///
/// Returns `None` when `bytes` is shorter than the scalar's compile-time
/// width. The slice is borrowed for the duration of the conversion and no
/// intermediate buffer is allocated.
pub fn read_integer<T: EndianScalar>(bytes: &[u8], byte_order: ByteOrder) -> Option<T> {
    T::from_bytes(bytes.get(..T::BYTE_WIDTH)?, byte_order)
}

/// Writes one fixed-width scalar into the start of `out`, the inverse of
/// [`read_integer`].
///
/// Returns `None`, writing nothing, when `out` is shorter than the scalar's
/// compile-time width. Bytes past that width are left unchanged.
pub fn write_integer<T: EndianScalar>(
    out: &mut [u8],
    value: T,
    byte_order: ByteOrder,
) -> Option<()> {
    value.to_bytes(out.get_mut(..T::BYTE_WIDTH)?, byte_order)
}

/// Decodes every `T` in `bytes` in `byte_order`, handing each to `sink` in
/// order: the bulk form of [`read_integer`] for sample buffers.
///
/// The byte order is resolved once for the whole buffer, outside the loop.
/// Returns `None`, calling `sink` for nothing, when `bytes.len()` is not a
/// whole number of scalars; a trailing partial scalar is never dropped
/// silently.
pub fn decode_each<T: EndianScalar>(
    bytes: &[u8],
    byte_order: ByteOrder,
    mut sink: impl FnMut(T),
) -> Option<()> {
    if !bytes.len().is_multiple_of(T::BYTE_WIDTH) {
        return None;
    }
    let chunks = bytes.chunks_exact(T::BYTE_WIDTH);
    match byte_order {
        ByteOrder::LittleEndian => {
            chunks.for_each(|chunk| sink(scalar(chunk, ByteOrder::LittleEndian)))
        }
        ByteOrder::BigEndian => chunks.for_each(|chunk| sink(scalar(chunk, ByteOrder::BigEndian))),
    }
    Some(())
}

/// Decodes every `T` in `bytes` in `byte_order` and appends `map` of each to
/// `out`: [`decode_each`] for the common case of collecting into a vector.
///
/// `out` is extended from an exact-length iterator, so the elements are
/// written into reserved capacity with no per-element capacity check. Returns
/// `None`, leaving `out` unchanged, when `bytes.len()` is not a whole number
/// of scalars.
#[cfg(feature = "alloc")]
pub fn decode_extend<T: EndianScalar, U>(
    bytes: &[u8],
    byte_order: ByteOrder,
    out: &mut alloc::vec::Vec<U>,
    mut map: impl FnMut(T) -> U,
) -> Option<()> {
    if !bytes.len().is_multiple_of(T::BYTE_WIDTH) {
        return None;
    }
    let chunks = bytes.chunks_exact(T::BYTE_WIDTH);
    match byte_order {
        ByteOrder::LittleEndian => {
            out.extend(chunks.map(|chunk| map(scalar(chunk, ByteOrder::LittleEndian))));
        }
        ByteOrder::BigEndian => {
            out.extend(chunks.map(|chunk| map(scalar(chunk, ByteOrder::BigEndian))));
        }
    }
    Some(())
}

/// Appends every value to `out` as `T` in `byte_order`: the bulk form of
/// [`write_integer`]. The byte order is resolved once, outside the loop.
#[cfg(feature = "alloc")]
pub fn extend_encoded<T: EndianScalar>(
    out: &mut alloc::vec::Vec<u8>,
    values: impl IntoIterator<Item = T>,
    byte_order: ByteOrder,
) {
    let mut encode = |value: T, order: ByteOrder| {
        let mut buf = [0_u8; MAX_SCALAR_WIDTH];
        let bytes = &mut buf[..T::BYTE_WIDTH];
        value
            .to_bytes(bytes, order)
            .expect("invariant: the buffer holds exactly BYTE_WIDTH bytes");
        out.extend_from_slice(bytes);
    };
    match byte_order {
        ByteOrder::LittleEndian => values
            .into_iter()
            .for_each(|v| encode(v, ByteOrder::LittleEndian)),
        ByteOrder::BigEndian => values
            .into_iter()
            .for_each(|v| encode(v, ByteOrder::BigEndian)),
    }
}

/// One scalar from an exact-width chunk, with the byte order a constant at
/// each call site so the per-element conversion carries no branch on it.
#[inline(always)]
fn scalar<T: EndianScalar>(chunk: &[u8], byte_order: ByteOrder) -> T {
    T::from_bytes(chunk, byte_order).expect("invariant: chunks_exact yields BYTE_WIDTH bytes")
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

    fn to_bytes(self, out: &mut [u8], _byte_order: ByteOrder) -> Option<()> {
        let out: &mut [u8; Self::BYTE_WIDTH] = out.try_into().ok()?;
        out[0] = self;
        Some(())
    }
}

impl EndianScalar for i8 {
    const BYTE_WIDTH: usize = 1;

    fn from_bytes(bytes: &[u8], _byte_order: ByteOrder) -> Option<Self> {
        let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
        Some(bytes[0] as i8)
    }

    fn to_bytes(self, out: &mut [u8], _byte_order: ByteOrder) -> Option<()> {
        let out: &mut [u8; Self::BYTE_WIDTH] = out.try_into().ok()?;
        *out = self.to_ne_bytes();
        Some(())
    }
}

macro_rules! impl_endian_scalar {
    ($(($ty:ty, $width:expr)),+ $(,)?) => {
        $(
            impl EndianScalar for $ty {
                const BYTE_WIDTH: usize = $width;

                fn from_bytes(bytes: &[u8], byte_order: ByteOrder) -> Option<Self> {
                    let bytes: [u8; Self::BYTE_WIDTH] = bytes.try_into().ok()?;
                    Some(match byte_order {
                        ByteOrder::LittleEndian => Self::from_le_bytes(bytes),
                        ByteOrder::BigEndian => Self::from_be_bytes(bytes),
                    })
                }

                fn to_bytes(self, out: &mut [u8], byte_order: ByteOrder) -> Option<()> {
                    let out: &mut [u8; Self::BYTE_WIDTH] = out.try_into().ok()?;
                    *out = match byte_order {
                        ByteOrder::LittleEndian => self.to_le_bytes(),
                        ByteOrder::BigEndian => self.to_be_bytes(),
                    };
                    Some(())
                }
            }
        )+
    };
}

impl_endian_scalar!(
    (u16, 2),
    (i16, 2),
    (u32, 4),
    (i32, 4),
    (u64, 8),
    (i64, 8),
    (f32, 4),
    (f64, 8),
);

#[cfg(test)]
mod tests;
