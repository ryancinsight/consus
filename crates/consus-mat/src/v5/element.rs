//! Byte-level helpers for MAT v5 numeric sub-element data.
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use crate::error::MatError;

pub fn read_u32(buf: &[u8], offset: usize, big_endian: bool) -> Result<u32, MatError> {
    if offset + 4 > buf.len() {
        return Err(MatError::InvalidFormat(alloc::string::String::from("read_u32: buffer too short")));
    }
    let b = &buf[offset..offset + 4];
    Ok(if big_endian {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    } else {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    })
}

pub fn read_i32(buf: &[u8], offset: usize, big_endian: bool) -> Result<i32, MatError> {
    read_u32(buf, offset, big_endian).map(|v| v as i32)
}

#[cfg(feature = "alloc")]
pub fn normalize_endian(mut data: Vec<u8>, elem_size: usize, big_endian: bool) -> Vec<u8> {
    if !big_endian || elem_size <= 1 { return data; }
    for chunk in data.chunks_exact_mut(elem_size) { chunk.reverse(); }
    data
}

#[cfg(feature = "alloc")]
pub fn decode_i32_vec(buf: &[u8], big_endian: bool) -> Result<Vec<i32>, MatError> {
    if buf.len() % 4 != 0 {
        return Err(MatError::InvalidFormat(
            alloc::string::String::from("i32 data length not multiple of 4"),
        ));
    }
    buf.chunks_exact(4)
        .map(|b| Ok(if big_endian {
            i32::from_be_bytes([b[0], b[1], b[2], b[3]])
        } else {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }))
        .collect()
}
