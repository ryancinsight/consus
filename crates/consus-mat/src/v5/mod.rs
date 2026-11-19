//! MAT v5 file reader.
pub mod element;
pub mod header;
pub mod matrix;
#[cfg(feature = "alloc")]
mod numeric;
pub mod tag;
use crate::error::MatError;
use crate::model::MatArray;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};
use header::V5FileHeader;
use tag::{MiType, read_element_bytes, read_tag};
/// Decode all top-level variables in file order.
#[cfg(feature = "alloc")]
pub fn read_mat_v5(data: &[u8]) -> Result<Vec<(String, MatArray)>, MatError> {
    let mut variables = Vec::new();
    visit_matrices(data, |payload, big_endian| {
        if let Some(variable) = matrix::parse_matrix(payload, big_endian)? {
            variables.push(variable);
        }
        Ok(None::<()>)
    })?;
    Ok(variables)
}

/// Decode the first top-level variable with the requested name.
///
/// Reads the names of preceding matrices without decoding their values.
/// Returns `Ok(None)` if the variable is absent.
///
/// # Errors
///
/// Returns a format error for a non-Level-5 file or invalid elements reached
/// during the search. Compressed elements require the `compress` feature.
#[cfg(feature = "alloc")]
pub fn read_mat_v5_variable(data: &[u8], name: &str) -> Result<Option<MatArray>, MatError> {
    if crate::detect::detect_version(data) != Some(crate::MatVersion::V5) {
        return Err(MatError::InvalidFormat(
            "expected a MATLAB Level-5 file".into(),
        ));
    }
    visit_matrices(data, |payload, big_endian| {
        matrix::parse_named_matrix(payload, big_endian, Some(name))
            .map(|variable| variable.map(|(_, value)| value))
    })
}

#[cfg(feature = "alloc")]
fn visit_matrices<T>(
    data: &[u8],
    mut visit: impl FnMut(&[u8], bool) -> Result<Option<T>, MatError>,
) -> Result<Option<T>, MatError> {
    let big_endian = V5FileHeader::parse(data)?.big_endian;
    let mut pos = 128usize;
    while data.len() - pos >= 8 {
        let tag = read_tag(data, &mut pos, big_endian)?;
        match tag.mi_type {
            MiType::Matrix if tag.nbytes != 0 => {
                let payload = read_element_bytes(data, &mut pos, &tag)?;
                if let Some(value) = visit(&payload, big_endian)? {
                    return Ok(Some(value));
                }
            }
            MiType::Compressed => {
                let compressed = read_compressed(data, &mut pos, &tag, big_endian)?;
                #[cfg(feature = "compress")]
                {
                    let decompressed = decompress_zlib(compressed)?;
                    let mut inner_pos = 0usize;
                    let inner_tag = read_tag(&decompressed, &mut inner_pos, big_endian)?;
                    if inner_tag.mi_type != MiType::Matrix {
                        return Err(MatError::InvalidFormat(String::from(
                            "miCOMPRESSED: inner element is not miMATRIX",
                        )));
                    }
                    if inner_tag.nbytes != 0 {
                        let payload =
                            read_element_bytes(&decompressed, &mut inner_pos, &inner_tag)?;
                        if let Some(value) = visit(&payload, big_endian)? {
                            return Ok(Some(value));
                        }
                    }
                }
                #[cfg(not(feature = "compress"))]
                {
                    let _ = compressed;
                    return Err(MatError::UnsupportedFeature(String::from(
                        "miCOMPRESSED requires the 'compress' feature",
                    )));
                }
            }
            _ if tag.nbytes != 0 => {
                read_element_bytes(data, &mut pos, &tag)?;
            }
            _ => {}
        }
    }
    Ok(None)
}

#[cfg(feature = "alloc")]
fn read_compressed<'a>(
    data: &'a [u8],
    pos: &mut usize,
    tag: &tag::DataTag,
    big_endian: bool,
) -> Result<&'a [u8], MatError> {
    let end = pos
        .checked_add(tag.nbytes)
        .ok_or_else(|| MatError::InvalidFormat("compressed MAT element offset overflow".into()))?;
    let compressed = data
        .get(*pos..end)
        .ok_or_else(|| MatError::InvalidFormat("compressed MAT element is truncated".into()))?;
    *pos = end;

    // SciPy writes no padding after miCOMPRESSED. Some producers pad to the
    // regular element layout. Prefer a valid tag at the declared payload end.
    let padding = if tag.small {
        4 - tag.nbytes
    } else {
        tag.nbytes.wrapping_neg() % 8
    };
    let mut next = end;
    if padding != 0
        && read_tag(data, &mut next, big_endian).is_err()
        && let Some(padded_end) = end.checked_add(padding)
        && data
            .get(end..padded_end)
            .is_some_and(|bytes| bytes.iter().all(|&byte| byte == 0))
    {
        *pos = padded_end;
    }
    Ok(compressed)
}

#[cfg(feature = "compress")]
fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>, MatError> {
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    // The compressed payload is attacker-chosen; `read_to_end` would grow the
    // output without bound (a decompression bomb). `ParseBudget::read_bounded`
    // caps the expansion and makes allocator exhaustion a typed error.
    consus_core::ParseBudget::DEFAULT
        .read_bounded(&mut decoder, 0, "mat v5 zlib decompression")
        .map_err(|e| MatError::CompressionError(alloc::format!("zlib decompression failed: {e}")))
}
