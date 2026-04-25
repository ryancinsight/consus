//! MAT v5 file reader.
pub mod element;
pub mod header;
pub mod matrix;
pub mod tag;
use crate::error::MatError;
use crate::model::MatArray;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};
use header::V5FileHeader;
use tag::{MiType, read_element_bytes, read_tag};
#[cfg(feature = "alloc")]
pub fn read_mat_v5(data: &[u8]) -> Result<Vec<(String, MatArray)>, MatError> {
    let hdr = V5FileHeader::parse(data)?;
    let big_endian = hdr.big_endian;
    let mut pos = 128usize;
    let mut variables = Vec::new();
    while pos < data.len() {
        if pos + 8 > data.len() {
            break;
        }
        let etag = read_tag(data, &mut pos, big_endian)?;
        match etag.mi_type {
            MiType::Matrix => {
                let payload = if etag.nbytes == 0 {
                    Vec::new()
                } else {
                    read_element_bytes(data, &mut pos, &etag)?
                };
                if let Some(var) = matrix::parse_matrix(&payload, big_endian)? {
                    variables.push(var);
                }
            }
            MiType::Compressed => {
                let compressed = read_element_bytes(data, &mut pos, &etag)?;
                #[cfg(feature = "compress")]
                {
                    let decompressed = decompress_zlib(&compressed)?;
                    let mut inner_pos = 0usize;
                    let inner_tag = read_tag(&decompressed, &mut inner_pos, big_endian)?;
                    if inner_tag.mi_type != MiType::Matrix {
                        return Err(MatError::InvalidFormat(String::from(
                            "miCOMPRESSED: inner element is not miMATRIX",
                        )));
                    }
                    let inner_payload = if inner_tag.nbytes == 0 {
                        Vec::new()
                    } else {
                        read_element_bytes(&decompressed, &mut inner_pos, &inner_tag)?
                    };
                    if let Some(var) = matrix::parse_matrix(&inner_payload, big_endian)? {
                        variables.push(var);
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
            _ => {
                if etag.nbytes != 0 {
                    let _ = read_element_bytes(data, &mut pos, &etag)?;
                }
            }
        }
    }
    Ok(variables)
}
#[cfg(feature = "compress")]
fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>, MatError> {
    use std::io::Read as _;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).map_err(|e| {
        MatError::CompressionError(alloc::format!("zlib decompression failed: {}", e))
    })?;
    Ok(out)
}
