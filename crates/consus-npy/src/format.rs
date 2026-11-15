//! NPY stream encoding and decoding.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use consus_io::bounded_capacity;
use std::io::{Read, Write};

use crate::{Error, NpyArray, NpyElement, Result};

mod header;

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const MAX_HEADER_BYTES: usize = 1 << 20;

/// Format versions NumPy reads; the header-length width and text encoding
/// follow from the version.
#[derive(Clone, Copy)]
enum Version {
    V1,
    V2,
    V3,
}

impl Version {
    fn new(major: u8, minor: u8) -> Result<Self> {
        match (major, minor) {
            (1, 0) => Ok(Self::V1),
            (2, 0) => Ok(Self::V2),
            (3, 0) => Ok(Self::V3),
            _ => Err(Error::InvalidFormat(format!(
                "unsupported NPY version {major}.{minor}"
            ))),
        }
    }

    /// Decodes header bytes as NumPy does: Latin-1 for 1.0 and 2.0, UTF-8
    /// for 3.0.
    fn decode(self, bytes: Vec<u8>) -> Result<String> {
        match self {
            Self::V1 | Self::V2 => Ok(bytes.into_iter().map(char::from).collect()),
            Self::V3 => String::from_utf8(bytes)
                .map_err(|error| Error::InvalidFormat(format!("header is not UTF-8: {error}"))),
        }
    }
}

/// Reads one typed NPY array.
///
/// Only format versions 1.0, 2.0 and 3.0 are accepted; every other
/// `major.minor` pair is rejected with [`Error::InvalidFormat`]. The set
/// follows the `numpy.lib.format` specification, "Format Version 1.0", "2.0"
/// and "3.0", and NumPy's reader (`_check_version`, `_header_size_info`):
/// <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>.
/// The header of versions 1.0 and 2.0 is decoded as Latin-1, the encoding
/// NumPy's reader applies to what the specification calls ASCII, and the
/// version 3.0 header as UTF-8.
///
/// # Errors
///
/// Returns [`Error::InvalidFormat`] for a missing magic, an unsupported
/// version, a header over the size limit, undecodable text or a malformed
/// dictionary, [`Error::DtypeMismatch`] when the stored dtype is not `T`'s,
/// and [`Error::Io`] when the stream ends before the declared payload.
///
/// # Examples
///
/// ```
/// use consus_npy::{Error, NpyArray, read_npy, write_npy};
///
/// let array = NpyArray::new([2], [7_i32, -3])?;
/// let mut bytes = Vec::new();
/// write_npy(&mut bytes, &array)?;
/// assert_eq!(read_npy::<i32>(bytes.as_slice())?, array);
///
/// // Byte 7 is the minor version; NumPy reads only 1.0, 2.0 and 3.0.
/// bytes[7] = 7;
/// match read_npy::<i32>(bytes.as_slice()) {
///     Err(Error::InvalidFormat(message)) => {
///         assert_eq!(message, "unsupported NPY version 1.7");
///     }
///     other => panic!("version 1.7 must be rejected, received {other:?}"),
/// }
/// # Ok::<(), Error>(())
/// ```
pub fn read_npy<T: NpyElement>(mut reader: impl Read) -> Result<NpyArray<T>> {
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::InvalidFormat("missing NPY magic".into()));
    }
    let major = reader.read_u8()?;
    let minor = reader.read_u8()?;
    let version = Version::new(major, minor)?;
    let header_len = match version {
        Version::V1 => usize::from(reader.read_u16::<LittleEndian>()?),
        Version::V2 | Version::V3 => usize::try_from(reader.read_u32::<LittleEndian>()?)
            .map_err(|_| Error::InvalidFormat("header length does not fit usize".into()))?,
    };
    if header_len > MAX_HEADER_BYTES {
        return Err(Error::InvalidFormat(format!(
            "header length {header_len} exceeds {MAX_HEADER_BYTES} byte limit"
        )));
    }
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let header = version.decode(header)?;
    let header = header::Header::parse(&header)?;
    let dtype = header.dtype;
    if dtype != T::DTYPE && dtype != T::DTYPE.replacen('<', "=", 1) {
        return Err(Error::DtypeMismatch {
            stored: dtype,
            requested: T::DTYPE,
        });
    }
    let fortran_order = header.fortran_order;
    let shape = header.shape;
    let count = shape.iter().try_fold(1usize, |count, &axis| {
        count
            .checked_mul(axis)
            .ok_or_else(|| Error::InvalidFormat(format!("shape overflows usize: {shape:?}")))
    })?;
    let mut values = Vec::with_capacity(bounded_capacity(count, core::mem::size_of::<T>()));
    for _ in 0..count {
        values.push(T::read_from(&mut reader)?);
    }
    Ok(NpyArray::from_parts(
        shape.into_boxed_slice(),
        fortran_order,
        values.into_boxed_slice(),
    ))
}

/// Writes one typed array using NPY version 1.0.
pub fn write_npy<T: NpyElement>(mut writer: impl Write, array: &NpyArray<T>) -> Result<()> {
    if array.is_fortran_order() {
        return Err(Error::InvalidFormat(
            "writing Fortran-order arrays is not supported".into(),
        ));
    }
    let shape = if array.shape().is_empty() {
        "()".to_owned()
    } else if array.shape().len() == 1 {
        format!("({},)", array.shape()[0])
    } else {
        format!(
            "({})",
            array
                .shape()
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let prefix_len = MAGIC.len() + 2 + 2;
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        T::DTYPE,
        shape
    );
    let padding = (64 - ((prefix_len + header.len() + 1) % 64)) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let header_len = u16::try_from(header.len())
        .map_err(|_| Error::InvalidFormat("version 1 header exceeds u16 length".into()))?;

    writer.write_all(MAGIC)?;
    writer.write_all(&[1, 0])?;
    writer.write_u16::<LittleEndian>(header_len)?;
    writer.write_all(header.as_bytes())?;
    for &value in array.values() {
        value.write_to(&mut writer)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn round_trip_preserves_shape_and_values() {
        let array = NpyArray::new([2, 3], [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut bytes = Vec::new();
        write_npy(&mut bytes, &array).unwrap();
        let decoded = read_npy::<f64>(Cursor::new(bytes)).unwrap();
        assert_eq!(decoded, array);
    }

    #[test]
    fn reads_numpy_generated_version_one_fixture() {
        // Generated by NumPy `save` from [[1.5, -2.0], [3.25, 4.5]] with
        // dtype `<f8`; the fixed bytes provide an implementation-independent
        // interoperability oracle for header padding and scalar decoding.
        let header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (2, 2), }                                                          \n";
        assert_eq!(header.len(), 118);
        let mut bytes = b"\x93NUMPY\x01\x00\x76\x00".to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[
            0, 0, 0, 0, 0, 0, 248, 63, 0, 0, 0, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 10, 64, 0, 0, 0,
            0, 0, 0, 18, 64,
        ]);
        let array = read_npy::<f64>(Cursor::new(bytes)).unwrap();
        assert_eq!(array.shape(), [2, 2]);
        assert_eq!(array.values(), [1.5, -2.0, 3.25, 4.5]);
        assert!(!array.is_fortran_order());
    }

    #[test]
    fn hostile_shape_does_not_reserve_declared_payload() {
        let header = format!(
            "{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}",
            usize::MAX
        );
        let header_len = u16::try_from(header.len()).expect("invariant: test header fits NPY v1");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());

        let error = read_npy::<f64>(Cursor::new(bytes))
            .expect_err("a declared payload with no stored values must be truncated");
        match error {
            Error::Io(error) => assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof),
            other => panic!("expected truncated payload, received {other}"),
        }
    }
}
