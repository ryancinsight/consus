//! MAT v5 data element tag parsing and element type codes.
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use crate::error::MatError;

/// MAT v5 element type codes (miXXXX constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiType {
    Int8 = 1, Uint8 = 2, Int16 = 3, Uint16 = 4,
    Int32 = 5, Uint32 = 6, Single = 7, Double = 9,
    Int64 = 12, Uint64 = 13, Matrix = 14, Compressed = 15,
    Utf8 = 16, Utf16 = 17, Utf32 = 18,
}

impl MiType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Int8), 2 => Some(Self::Uint8),
            3 => Some(Self::Int16), 4 => Some(Self::Uint16),
            5 => Some(Self::Int32), 6 => Some(Self::Uint32),
            7 => Some(Self::Single), 9 => Some(Self::Double),
            12 => Some(Self::Int64), 13 => Some(Self::Uint64),
            14 => Some(Self::Matrix), 15 => Some(Self::Compressed),
            16 => Some(Self::Utf8), 17 => Some(Self::Utf16),
            18 => Some(Self::Utf32), _ => None,
        }
    }
    pub const fn element_size(self) -> usize {
        match self {
            Self::Int8 | Self::Uint8 | Self::Utf8 => 1,
            Self::Int16 | Self::Uint16 | Self::Utf16 => 2,
            Self::Int32 | Self::Uint32 | Self::Single | Self::Utf32 => 4,
            Self::Double | Self::Int64 | Self::Uint64 => 8,
            _ => 0,
        }
    }
}

#[inline]
pub const fn pad8(n: usize) -> usize { (n + 7) & !7 }

#[derive(Debug, Clone)]
pub struct DataTag {
    pub mi_type: MiType,
    pub nbytes: usize,
    pub small: bool,
}

pub fn read_tag(data: &[u8], pos: &mut usize, big_endian: bool) -> Result<DataTag, MatError> {
    if *pos + 8 > data.len() {
        return Err(MatError::InvalidFormat(
            alloc::string::String::from("data element tag truncated"),
        ));
    }
    let b = &data[*pos..*pos + 4];
    let (small_nbytes, small_type) = if big_endian {
        (u16::from_be_bytes([b[0], b[1]]) as usize, u16::from_be_bytes([b[2], b[3]]))
    } else {
        (u16::from_le_bytes([b[2], b[3]]) as usize, u16::from_le_bytes([b[0], b[1]]))
    };
    if small_nbytes != 0 {
        let mi_type = MiType::from_u32(small_type as u32).ok_or_else(|| {
            MatError::InvalidFormat(alloc::format!("unknown small element type {small_type}"))
        })?;
        *pos += 4;
        return Ok(DataTag { mi_type, nbytes: small_nbytes, small: true });
    }
    let type_code = if big_endian {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    } else {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    let nb = &data[*pos + 4..*pos + 8];
    let nbytes = if big_endian {
        u32::from_be_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize
    } else {
        u32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize
    };
    let mi_type = MiType::from_u32(type_code).ok_or_else(|| {
        MatError::InvalidFormat(alloc::format!("unknown element type {type_code}"))
    })?;
    *pos += 8;
    Ok(DataTag { mi_type, nbytes, small: false })
}

#[cfg(feature = "alloc")]
pub fn read_element_bytes(data: &[u8], pos: &mut usize, tag: &DataTag) -> Result<Vec<u8>, MatError> {
    if tag.small {
        if *pos + 4 > data.len() {
            return Err(MatError::InvalidFormat(
                alloc::string::String::from("small element data truncated"),
            ));
        }
        let d = data[*pos..*pos + tag.nbytes].to_vec();
        *pos += 4;
        Ok(d)
    } else {
        if *pos + tag.nbytes > data.len() {
            return Err(MatError::InvalidFormat(alloc::format!(
                "element data truncated: need {}, available {}", tag.nbytes, data.len() - *pos
            )));
        }
        let d = data[*pos..*pos + tag.nbytes].to_vec();
        *pos += pad8(tag.nbytes);
        Ok(d)
    }
}

#[cfg(feature = "alloc")]
pub fn read_subelement_bytes(
    payload: &[u8],
    local_pos: &mut usize,
    big_endian: bool,
) -> Result<(MiType, Vec<u8>), MatError> {
    let tag = read_tag(payload, local_pos, big_endian)?;
    let bytes = read_element_bytes(payload, local_pos, &tag)?;
    Ok((tag.mi_type, bytes))
}
