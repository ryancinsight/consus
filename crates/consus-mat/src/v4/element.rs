//! MAT v4 variable element reader.
//!
//! Reads one variable record from a MAT v4 file byte slice and converts
//! it to the canonical [`MatArray`] representation.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use crate::error::MatError;
use crate::model::{MatArray, MatCharArray, MatNumericArray, MatNumericClass};

use super::header::V4Header;

/// Reverse byte order in `data` for elements of `elem_size` bytes.
#[cfg(feature = "alloc")]
fn byteswap(mut data: Vec<u8>, elem_size: usize) -> Vec<u8> {
    if elem_size <= 1 {
        return data;
    }
    for chunk in data.chunks_exact_mut(elem_size) {
        chunk.reverse();
    }
    data
}

/// Map MAT v4 precision code to [`MatNumericClass`].
pub fn precision_to_class(precision: u8) -> MatNumericClass {
    match precision {
        0 => MatNumericClass::Double,
        1 => MatNumericClass::Single,
        2 => MatNumericClass::Int32,
        3 => MatNumericClass::Int16,
        4 => MatNumericClass::Uint16,
        5 => MatNumericClass::Uint8,
        _ => MatNumericClass::Double, // unreachable after header validation
    }
}

/// Read one MAT v4 variable from `data` starting at `*pos`.
///
/// Returns `Ok(None)` when the slice is exhausted (EOF).
/// Advances `*pos` past the entire variable record on success.
#[cfg(feature = "alloc")]
pub fn read_v4_variable(
    data: &[u8],
    pos: &mut usize,
) -> Result<Option<(String, MatArray)>, MatError> {
    if *pos >= data.len() {
        return Ok(None);
    }
    if *pos + 20 > data.len() {
        return Err(MatError::InvalidFormat(
            String::from("MAT v4 record truncated at header"),
        ));
    }

    // Peek at the type code to determine byte order before full header parse.
    let type_le = u32::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
    ]);
    let big_endian = (type_le / 1000) == 1; // M=1 → Sun/IEEE BE

    let hdr = V4Header::parse(data, pos, big_endian)?;
    let numel = hdr.mrows.saturating_mul(hdr.ncols);
    let data_len = numel.saturating_mul(hdr.elem_size);

    if *pos + data_len > data.len() {
        return Err(MatError::InvalidFormat(alloc::format!(
            "MAT v4 real data truncated for variable '{}'",
            hdr.name
        )));
    }

    let mut real_data = data[*pos..*pos + data_len].to_vec();
    *pos += data_len;

    let mut imag_data = if hdr.imagf {
        if *pos + data_len > data.len() {
            return Err(MatError::InvalidFormat(alloc::format!(
                "MAT v4 imaginary data truncated for variable '{}'",
                hdr.name
            )));
        }
        let im = data[*pos..*pos + data_len].to_vec();
        *pos += data_len;
        Some(im)
    } else {
        None
    };

    // Normalize to little-endian (only swap for BE machine type).
    if big_endian {
        real_data = byteswap(real_data, hdr.elem_size);
        imag_data = imag_data.map(|d| byteswap(d, hdr.elem_size));
    }

    let array = match hdr.matrix_type {
        0 => {
            // Full numeric matrix (real or complex).
            let class = precision_to_class(hdr.precision);
            MatArray::Numeric(MatNumericArray {
                class,
                shape: alloc::vec![hdr.mrows, hdr.ncols],
                real_data,
                imag_data,
            })
        }
        1 => {
            // Text matrix: each f64 element encodes one character code (column-major).
            let chars: String = real_data
                .chunks_exact(8)
                .map(|b| {
                    let val = f64::from_le_bytes([
                        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                    ]);
                    char::from_u32(val as u32).unwrap_or('\u{FFFD}')
                })
                .collect();
            MatArray::Char(MatCharArray {
                shape: alloc::vec![hdr.mrows, hdr.ncols],
                data: chars,
            })
        }
        2 => {
            return Err(MatError::UnsupportedFeature(
                String::from("MAT v4 sparse matrices are not supported"),
            ));
        }
        _ => {
            return Err(MatError::InvalidFormat(alloc::format!(
                "unknown MAT v4 matrix type {}",
                hdr.matrix_type
            )))
        }
    };

    Ok(Some((hdr.name, array)))
}
