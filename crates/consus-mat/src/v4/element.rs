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
    let header_end = (*pos)
        .checked_add(20)
        .ok_or_else(|| MatError::InvalidFormat(String::from("MAT v4 header offset overflow")))?;
    if header_end > data.len() {
        return Err(MatError::InvalidFormat(String::from(
            "MAT v4 record truncated at header",
        )));
    }

    // Peek at the type code to determine byte order before full header parse.
    let type_le = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    let big_endian = (type_le / 1000) == 1; // M=1 → Sun/IEEE BE

    let hdr = V4Header::parse(data, pos, big_endian)?;
    let numel = hdr.mrows.checked_mul(hdr.ncols).ok_or_else(|| {
        MatError::InvalidFormat(alloc::format!(
            "MAT v4 dimensions overflow for variable '{}'",
            hdr.name
        ))
    })?;
    let data_len = numel.checked_mul(hdr.elem_size).ok_or_else(|| {
        MatError::InvalidFormat(alloc::format!(
            "MAT v4 data size overflow for variable '{}'",
            hdr.name
        ))
    })?;

    let real_end = (*pos).checked_add(data_len).ok_or_else(|| {
        MatError::InvalidFormat(alloc::format!(
            "MAT v4 real data range overflow for variable '{}'",
            hdr.name
        ))
    })?;
    if real_end > data.len() {
        return Err(MatError::InvalidFormat(alloc::format!(
            "MAT v4 real data truncated for variable '{}'",
            hdr.name
        )));
    }

    let mut real_data = data[*pos..real_end].to_vec();
    *pos = real_end;

    let mut imag_data = if hdr.imagf {
        let imag_end = (*pos).checked_add(data_len).ok_or_else(|| {
            MatError::InvalidFormat(alloc::format!(
                "MAT v4 imaginary data range overflow for variable '{}'",
                hdr.name
            ))
        })?;
        if imag_end > data.len() {
            return Err(MatError::InvalidFormat(alloc::format!(
                "MAT v4 imaginary data truncated for variable '{}'",
                hdr.name
            )));
        }
        let im = data[*pos..imag_end].to_vec();
        *pos = imag_end;
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
                    let val = f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                    char::from_u32(val as u32).unwrap_or('\u{FFFD}')
                })
                .collect();
            MatArray::Char(MatCharArray {
                shape: alloc::vec![hdr.mrows, hdr.ncols],
                data: chars,
            })
        }
        2 => {
            return Err(MatError::UnsupportedFeature(String::from(
                "MAT v4 sparse matrices are not supported",
            )));
        }
        _ => {
            return Err(MatError::InvalidFormat(alloc::format!(
                "unknown MAT v4 matrix type {}",
                hdr.matrix_type
            )));
        }
    };

    Ok(Some((hdr.name, array)))
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    /// The hosted fuzz regression must return an error instead of overflowing
    /// while computing the MAT v4 real-data range.
    #[test]
    fn rejects_dimension_data_range_overflow() {
        let data = [
            0x02, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0xF0, 0x3F, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x32, 0x00, 0x02, 0x00, 0x00, 0x08, 0x4A,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, 0x00, 0x00, 0x00, 0x20, 0x00, 0x3F,
            0x14, 0x40, 0x00, 0x00, 0x00, 0x00, 0xAA, 0x00, 0x18, 0x40,
        ];
        let mut pos = 0;
        let err = read_v4_variable(&data, &mut pos).expect_err("hostile dimensions must fail");
        assert!(matches!(err, MatError::InvalidFormat(_)));
    }
}
