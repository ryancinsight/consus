//! NPY stream encoding and decoding.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use consus_io::bounded_capacity;
use std::io::{Read, Write};

use crate::{Error, NpyArray, NpyElement, Result};

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const MAX_HEADER_BYTES: usize = 1 << 20;

/// Reads one typed NPY array.
pub fn read_npy<T: NpyElement>(mut reader: impl Read) -> Result<NpyArray<T>> {
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::InvalidFormat("missing NPY magic".into()));
    }
    let major = reader.read_u8()?;
    let _minor = reader.read_u8()?;
    let header_len = match major {
        1 => usize::from(reader.read_u16::<LittleEndian>()?),
        2 | 3 => usize::try_from(reader.read_u32::<LittleEndian>()?)
            .map_err(|_| Error::InvalidFormat("header length does not fit usize".into()))?,
        _ => {
            return Err(Error::InvalidFormat(format!(
                "unsupported NPY version {major}"
            )));
        }
    };
    if header_len > MAX_HEADER_BYTES {
        return Err(Error::InvalidFormat(format!(
            "header length {header_len} exceeds {MAX_HEADER_BYTES} byte limit"
        )));
    }
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header)
        .map_err(|error| Error::InvalidFormat(format!("header is not UTF-8: {error}")))?;
    let dtype = quoted_value(header, "'descr'")?;
    if dtype != T::DTYPE && dtype != T::DTYPE.replacen('<', "=", 1) {
        return Err(Error::DtypeMismatch {
            stored: dtype,
            requested: T::DTYPE,
        });
    }
    let fortran_order = bool_value(header, "'fortran_order'")?;
    let shape = shape_value(header)?;
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

fn value_tail<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let start = header
        .find(key)
        .ok_or_else(|| Error::InvalidFormat(format!("header missing {key}")))?;
    let tail = &header[start + key.len()..];
    tail.split_once(':')
        .map(|(_, value)| value.trim_start())
        .ok_or_else(|| Error::InvalidFormat(format!("header key {key} has no value")))
}

fn quoted_value(header: &str, key: &str) -> Result<String> {
    let tail = value_tail(header, key)?;
    let quote = tail
        .chars()
        .next()
        .filter(|character| *character == '\'' || *character == '"')
        .ok_or_else(|| Error::InvalidFormat(format!("{key} value is not quoted")))?;
    let remainder = &tail[quote.len_utf8()..];
    let end = remainder
        .find(quote)
        .ok_or_else(|| Error::InvalidFormat(format!("{key} quote is unterminated")))?;
    Ok(remainder[..end].to_owned())
}

fn bool_value(header: &str, key: &str) -> Result<bool> {
    let tail = value_tail(header, key)?;
    if tail.starts_with("True") {
        Ok(true)
    } else if tail.starts_with("False") {
        Ok(false)
    } else {
        Err(Error::InvalidFormat(format!("{key} is not a boolean")))
    }
}

fn shape_value(header: &str) -> Result<Vec<usize>> {
    let tail = value_tail(header, "'shape'")?;
    let end = tail
        .find(')')
        .ok_or_else(|| Error::InvalidFormat("shape tuple is unterminated".into()))?;
    let body = tail
        .strip_prefix('(')
        .ok_or_else(|| Error::InvalidFormat("shape value is not a tuple".into()))?
        .get(..end.saturating_sub(1))
        .ok_or_else(|| Error::InvalidFormat("invalid shape tuple".into()))?;
    body.split(',')
        .map(str::trim)
        .filter(|axis| !axis.is_empty())
        .map(|axis| {
            axis.parse::<usize>().map_err(|error| {
                Error::InvalidFormat(format!("invalid shape axis {axis}: {error}"))
            })
        })
        .collect()
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
