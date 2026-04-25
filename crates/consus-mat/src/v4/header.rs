//! MAT v4 per-variable header parser.
//!
//! ## Type Field Encoding
//!
//! `type = M * 1000 + P * 10 + T` where:
//! - `M`: machine/byte-order (0 = PC/IEEE LE, 1 = Sun/IEEE BE, 2–4 = VAX/Cray)
//! - `P`: precision (0 = f64, 1 = f32, 2 = i32, 3 = i16, 4 = u16, 5 = u8)
//! - `T`: matrix type (0 = full numeric, 1 = text, 2 = sparse)
//!
//! ## Reference
//!
//! MATLAB Level 4 MAT-File Format, MathWorks Engineering Development Group.

#[cfg(feature = "alloc")]
use alloc::string::String;

use crate::error::MatError;

/// Decoded MAT v4 variable header.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct V4Header {
    /// Machine/byte-order code (0 = LE, 1 = BE).
    pub machine: u32,
    /// Precision code (0-5).
    pub precision: u8,
    /// Matrix type code (0 = full, 1 = text, 2 = sparse).
    pub matrix_type: u8,
    /// Number of rows.
    pub mrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// True if the variable is complex.
    pub imagf: bool,
    /// Variable name.
    pub name: String,
    /// Bytes per element derived from `precision`.
    pub elem_size: usize,
}

#[cfg(feature = "alloc")]
impl V4Header {
    /// Parse a v4 variable header from `data` starting at `*pos`.
    ///
    /// Advances `*pos` past the 20-byte fixed header and the variable-length
    /// name field on success.
    pub fn parse(data: &[u8], pos: &mut usize, big_endian: bool) -> Result<Self, MatError> {
        if *pos + 20 > data.len() {
            return Err(MatError::InvalidFormat(
                String::from("MAT v4 header truncated"),
            ));
        }

        let read_u32 = |b: &[u8]| -> u32 {
            if big_endian {
                u32::from_be_bytes([b[0], b[1], b[2], b[3]])
            } else {
                u32::from_le_bytes([b[0], b[1], b[2], b[3]])
            }
        };

        let type_code = read_u32(&data[*pos..*pos + 4]);
        let mrows    = read_u32(&data[*pos + 4..*pos + 8]) as usize;
        let ncols    = read_u32(&data[*pos + 8..*pos + 12]) as usize;
        let imagf    = read_u32(&data[*pos + 12..*pos + 16]) != 0;
        let namlen   = read_u32(&data[*pos + 16..*pos + 20]) as usize;
        *pos += 20;

        let machine    = type_code / 1000;
        let remainder  = type_code % 1000;
        let precision  = ((remainder / 10) % 10) as u8;
        let matrix_type = (remainder % 10) as u8;

        if matches!(machine, 2..=4) {
            return Err(MatError::UnsupportedFeature(alloc::format!(
                "MAT v4 machine type {machine} (VAX/Cray) is not supported"
            )));
        }

        let elem_size = match precision {
            0 => 8, // f64
            1 => 4, // f32
            2 => 4, // i32
            3 => 2, // i16
            4 => 2, // u16
            5 => 1, // u8
            _ => {
                return Err(MatError::UnsupportedFeature(alloc::format!(
                    "MAT v4 precision code {precision}"
                )))
            }
        };

        if *pos + namlen > data.len() {
            return Err(MatError::InvalidFormat(
                String::from("MAT v4 variable name truncated"),
            ));
        }

        let name_bytes = &data[*pos..*pos + namlen];
        *pos += namlen;
        let nul_end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(namlen);
        let name = String::from_utf8_lossy(&name_bytes[..nul_end]).into_owned();

        Ok(V4Header {
            machine,
            precision,
            matrix_type,
            mrows,
            ncols,
            imagf,
            name,
            elem_size,
        })
    }
}
