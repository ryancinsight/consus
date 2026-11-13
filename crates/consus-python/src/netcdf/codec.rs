//! Conversion between raw variable bytes and Python lists.

extern crate alloc;

use alloc::vec::Vec;

use consus_core::{ByteOrder, Datatype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::dtype::dtype_tag;

/// Decode a raw byte buffer into a Python list of numeric values according to
/// `dt`.  Handles both little-endian and big-endian variants.
pub(super) fn decode_bytes(py: Python<'_>, data: &[u8], dt: &Datatype) -> PyResult<Py<PyAny>> {
    let nbytes = match dt.element_size() {
        Some(n) if n > 0 => n,
        _ => {
            return Err(PyValueError::new_err(format!(
                "cannot decode datatype '{}'",
                dtype_tag(dt)
            )));
        }
    };
    if !data.len().is_multiple_of(nbytes) {
        return Err(PyValueError::new_err(format!(
            "data length {} is not a multiple of element size {}",
            data.len(),
            nbytes
        )));
    }
    let n = data.len() / nbytes;
    let list = PyList::empty(py);
    for i in 0..n {
        let s = &data[i * nbytes..(i + 1) * nbytes];
        match dt {
            Datatype::Float { bits, byte_order } if bits.get() == 32 => {
                let arr: [u8; 4] = s.try_into().expect("validated slice length");
                let v = match byte_order {
                    ByteOrder::LittleEndian => f32::from_le_bytes(arr),
                    ByteOrder::BigEndian => f32::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Float { bits, byte_order } if bits.get() == 64 => {
                let arr: [u8; 8] = s.try_into().expect("validated slice length");
                let v = match byte_order {
                    ByteOrder::LittleEndian => f64::from_le_bytes(arr),
                    ByteOrder::BigEndian => f64::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer {
                bits,
                signed: true,
                byte_order,
            } if bits.get() == 8 => {
                let v = s[0].cast_signed();
                list.append(v)?;
            }
            Datatype::Integer {
                bits,
                signed: false,
                ..
            } if bits.get() == 8 => {
                list.append(s[0])?;
            }
            Datatype::Integer {
                bits,
                signed: true,
                byte_order,
            } if bits.get() == 16 => {
                let arr: [u8; 2] = s.try_into().expect("validated slice length");
                let v = match byte_order {
                    ByteOrder::LittleEndian => i16::from_le_bytes(arr),
                    ByteOrder::BigEndian => i16::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer {
                bits,
                signed: false,
                byte_order,
            } if bits.get() == 16 => {
                let arr: [u8; 2] = s.try_into().expect("validated slice length");
                let v = match byte_order {
                    ByteOrder::LittleEndian => u16::from_le_bytes(arr),
                    ByteOrder::BigEndian => u16::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer {
                bits,
                signed: true,
                byte_order,
            } if bits.get() == 32 => {
                let arr: [u8; 4] = s.try_into().expect("validated slice length");
                let v = match byte_order {
                    ByteOrder::LittleEndian => i32::from_le_bytes(arr),
                    ByteOrder::BigEndian => i32::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer {
                bits,
                signed: false,
                byte_order,
            } if bits.get() == 32 => {
                let arr: [u8; 4] = s.try_into().expect("validated slice length");
                let v = match byte_order {
                    ByteOrder::LittleEndian => u32::from_le_bytes(arr),
                    ByteOrder::BigEndian => u32::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer {
                bits,
                signed: true,
                byte_order,
            } if bits.get() == 64 => {
                let arr: [u8; 8] = s.try_into().expect("validated slice length");
                let v = match byte_order {
                    ByteOrder::LittleEndian => i64::from_le_bytes(arr),
                    ByteOrder::BigEndian => i64::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer {
                bits,
                signed: false,
                byte_order,
            } if bits.get() == 64 => {
                let arr: [u8; 8] = s.try_into().expect("validated slice length");
                let v = match byte_order {
                    ByteOrder::LittleEndian => u64::from_le_bytes(arr),
                    ByteOrder::BigEndian => u64::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported datatype variant '{}'",
                    dtype_tag(dt)
                )));
            }
        }
    }
    Ok(list.into_any().unbind())
}

/// Encode a Python list of numbers into little-endian raw bytes for `dt`.
pub(super) fn encode_bytes(data: &Bound<'_, PyList>, dt: &Datatype) -> PyResult<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::with_capacity(data.len() * dt.element_size().unwrap_or(0));
    for item in data {
        match dt {
            Datatype::Float { bits, .. } if bits.get() == 32 => {
                let v: f32 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Float { bits, .. } if bits.get() == 64 => {
                let v: f64 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer {
                bits, signed: true, ..
            } if bits.get() == 8 => {
                let v: i8 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer {
                bits,
                signed: false,
                ..
            } if bits.get() == 8 => {
                let v: u8 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer {
                bits, signed: true, ..
            } if bits.get() == 16 => {
                let v: i16 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer {
                bits,
                signed: false,
                ..
            } if bits.get() == 16 => {
                let v: u16 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer {
                bits, signed: true, ..
            } if bits.get() == 32 => {
                let v: i32 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer {
                bits,
                signed: false,
                ..
            } if bits.get() == 32 => {
                let v: u32 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer {
                bits, signed: true, ..
            } if bits.get() == 64 => {
                let v: i64 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer {
                bits,
                signed: false,
                ..
            } if bits.get() == 64 => {
                let v: u64 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported datatype '{}' for encoding",
                    dtype_tag(dt)
                )));
            }
        }
    }
    Ok(bytes)
}
