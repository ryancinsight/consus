//! miMATRIX element parser for MAT v5.
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

use super::element::{decode_i32_vec, normalize_endian};
use super::tag::{MiType, pad8, read_subelement_bytes, read_tag};
use crate::error::MatError;
use crate::model::{
    MatArray, MatCellArray, MatCharArray, MatLogicalArray, MatNumericArray, MatNumericClass,
    MatSparseArray, MatStructArray,
};
const MX_CELL_CLASS: u8 = 1;
const MX_STRUCT_CLASS: u8 = 2;
const MX_OBJECT_CLASS: u8 = 3;
const MX_CHAR_CLASS: u8 = 4;
const MX_SPARSE_CLASS: u8 = 5;
const MX_DOUBLE_CLASS: u8 = 6;
const MX_SINGLE_CLASS: u8 = 7;
const MX_INT8_CLASS: u8 = 8;
const MX_UINT8_CLASS: u8 = 9;
const MX_INT16_CLASS: u8 = 10;
const MX_UINT16_CLASS: u8 = 11;
const MX_INT32_CLASS: u8 = 12;
const MX_UINT32_CLASS: u8 = 13;
const MX_INT64_CLASS: u8 = 14;
const MX_UINT64_CLASS: u8 = 15;
const FLAG_COMPLEX: u32 = 1 << 11;
const FLAG_LOGICAL: u32 = 1 << 9;
fn mx_class_to_numeric(code: u8) -> Option<MatNumericClass> {
    match code {
        MX_DOUBLE_CLASS => Some(MatNumericClass::Double),
        MX_SINGLE_CLASS => Some(MatNumericClass::Single),
        MX_INT8_CLASS => Some(MatNumericClass::Int8),
        MX_UINT8_CLASS => Some(MatNumericClass::Uint8),
        MX_INT16_CLASS => Some(MatNumericClass::Int16),
        MX_UINT16_CLASS => Some(MatNumericClass::Uint16),
        MX_INT32_CLASS => Some(MatNumericClass::Int32),
        MX_UINT32_CLASS => Some(MatNumericClass::Uint32),
        MX_INT64_CLASS => Some(MatNumericClass::Int64),
        MX_UINT64_CLASS => Some(MatNumericClass::Uint64),
        _ => None,
    }
}
#[cfg(feature = "alloc")]
pub fn parse_matrix(
    payload: &[u8],
    big_endian: bool,
) -> Result<Option<(String, MatArray)>, MatError> {
    if payload.is_empty() {
        return Ok(None);
    }
    let mut pos = 0usize;
    let (flags_type, flags_bytes): (MiType, Vec<u8>) =
        read_subelement_bytes(payload, &mut pos, big_endian)?;
    if flags_type != MiType::Uint32 || flags_bytes.len() < 8 {
        return Err(MatError::InvalidFormat(String::from(
            "miMATRIX: array flags sub-element malformed",
        )));
    }
    let rd32 = |b: &[u8], o: usize| -> u32 {
        let s = &b[o..o + 4];
        if big_endian {
            u32::from_be_bytes([s[0], s[1], s[2], s[3]])
        } else {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        }
    };
    let flags0 = rd32(&flags_bytes, 0);
    let flags1 = rd32(&flags_bytes, 4);
    let mx_class = (flags0 & 0xFF) as u8;
    let is_complex = (flags0 & FLAG_COMPLEX) != 0;
    let is_logical = (flags0 & FLAG_LOGICAL) != 0;
    let nzmax = flags1 as usize;
    let (_dt, dim_bytes): (MiType, Vec<u8>) = read_subelement_bytes(payload, &mut pos, big_endian)?;
    let dims_i32 = decode_i32_vec(&dim_bytes, big_endian)?;
    let shape: Vec<usize> = dims_i32.iter().map(|d| (*d).max(0) as usize).collect();
    let (_nt, name_bytes): (MiType, Vec<u8>) =
        read_subelement_bytes(payload, &mut pos, big_endian)?;
    let nul_end = name_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(name_bytes.len());
    let name = String::from_utf8_lossy(&name_bytes[..nul_end]).into_owned();
    let numel: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let array = match mx_class {
        code if mx_class_to_numeric(code).is_some() => {
            let nc = mx_class_to_numeric(code).unwrap();
            if is_logical {
                let (_rt, raw): (MiType, Vec<u8>) =
                    read_subelement_bytes(payload, &mut pos, big_endian)?;
                let data: Vec<bool> = raw.iter().map(|b| *b != 0).collect();
                MatArray::Logical(MatLogicalArray::new(shape, data)?)
            } else {
                let esz = nc.element_size();
                let (_rt, rr): (MiType, Vec<u8>) =
                    read_subelement_bytes(payload, &mut pos, big_endian)?;
                if rr.len() != numel * esz {
                    return Err(MatError::ShapeError(alloc::format!(
                        "real {} != numel {}*esz {}",
                        rr.len(),
                        numel,
                        esz
                    )));
                }
                let real_data = normalize_endian(rr, esz, big_endian);
                let imag_data = if is_complex {
                    let (_it, ir) = read_subelement_bytes(payload, &mut pos, big_endian)?;
                    Some(normalize_endian(ir, esz, big_endian))
                } else {
                    None
                };
                MatArray::Numeric(MatNumericArray {
                    class: nc,
                    shape,
                    real_data,
                    imag_data,
                })
            }
        }
        MX_CHAR_CLASS => {
            let (cmi, cb): (MiType, Vec<u8>) =
                read_subelement_bytes(payload, &mut pos, big_endian)?;
            let data = match cmi {
                MiType::Uint16 | MiType::Utf16 => {
                    let norm = normalize_endian(cb, 2, big_endian);
                    norm.chunks_exact(2)
                        .map(|b| {
                            char::from_u32(u16::from_le_bytes([b[0], b[1]]) as u32)
                                .unwrap_or(char::REPLACEMENT_CHARACTER)
                        })
                        .collect::<String>()
                }
                MiType::Utf8 | MiType::Int8 | MiType::Uint8 => {
                    String::from_utf8_lossy(&cb).into_owned()
                }
                _ => {
                    return Err(MatError::InvalidFormat(String::from(
                        "mxCHAR_CLASS: unexpected char data element type",
                    )));
                }
            };
            MatArray::Char(MatCharArray::new(shape, data)?)
        }
        MX_SPARSE_CLASS => {
            let nrows = shape.first().copied().unwrap_or(0);
            let ncols = shape.get(1).copied().unwrap_or(0);
            let (_it, ir_b): (MiType, Vec<u8>) =
                read_subelement_bytes(payload, &mut pos, big_endian)?;
            let row_indices = decode_i32_vec(&ir_b, big_endian)?;
            // Invariant: ir must contain exactly nzmax elements (the allocated non-zero buffer).
            if row_indices.len() != nzmax {
                return Err(MatError::ShapeError(alloc::format!(
                    "sparse: ir.len() {} != nzmax {}",
                    row_indices.len(),
                    nzmax
                )));
            }
            let (_jt, jc_b): (MiType, Vec<u8>) =
                read_subelement_bytes(payload, &mut pos, big_endian)?;
            let col_ptrs = decode_i32_vec(&jc_b, big_endian)?;
            // Invariant: jc must contain exactly ncols+1 elements.
            if col_ptrs.len() != ncols + 1 {
                return Err(MatError::ShapeError(alloc::format!(
                    "sparse: jc.len() {} != ncols+1 {}",
                    col_ptrs.len(),
                    ncols + 1
                )));
            }
            let (_pt, pr_b): (MiType, Vec<u8>) =
                read_subelement_bytes(payload, &mut pos, big_endian)?;
            let real_data = normalize_endian(pr_b, 8, big_endian);
            let imag_data = if is_complex {
                let (_pit, pi_b): (MiType, Vec<u8>) =
                    read_subelement_bytes(payload, &mut pos, big_endian)?;
                Some(normalize_endian(pi_b, 8, big_endian))
            } else {
                None
            };
            MatArray::Sparse(MatSparseArray::new(
                nrows,
                ncols,
                row_indices,
                col_ptrs,
                real_data,
                imag_data,
            )?)
        }
        MX_CELL_CLASS => {
            let mut cells: Vec<MatArray> = Vec::with_capacity(numel);
            for _ in 0..numel {
                let ctag = read_tag(payload, &mut pos, big_endian)?;
                if ctag.mi_type != MiType::Matrix {
                    return Err(MatError::InvalidFormat(String::from(
                        "mxCELL_CLASS: expected miMATRIX",
                    )));
                }
                let cp = if ctag.nbytes == 0 {
                    vec![]
                } else {
                    if pos + ctag.nbytes > payload.len() {
                        return Err(MatError::InvalidFormat(String::from(
                            "mxCELL_CLASS: payload truncated",
                        )));
                    }
                    let p = payload[pos..pos + ctag.nbytes].to_vec();
                    pos += pad8(ctag.nbytes);
                    p
                };
                let arr = match parse_matrix(&cp, big_endian)? {
                    Some((_, a)) => a,
                    None => MatArray::Numeric(MatNumericArray {
                        class: MatNumericClass::Double,
                        shape: vec![0, 0],
                        real_data: vec![],
                        imag_data: None,
                    }),
                };
                cells.push(arr);
            }
            MatArray::Cell(MatCellArray::new(shape, cells)?)
        }
        MX_STRUCT_CLASS => {
            let (_fnlt, fnl_b): (MiType, Vec<u8>) =
                read_subelement_bytes(payload, &mut pos, big_endian)?;
            if fnl_b.len() < 4 {
                return Err(MatError::InvalidFormat(String::from(
                    "mxSTRUCT_CLASS: field name length too short",
                )));
            }
            let fnl = (if big_endian {
                i32::from_be_bytes([fnl_b[0], fnl_b[1], fnl_b[2], fnl_b[3]])
            } else {
                i32::from_le_bytes([fnl_b[0], fnl_b[1], fnl_b[2], fnl_b[3]])
            }) as usize;
            let (_fnt, fn_b): (MiType, Vec<u8>) =
                read_subelement_bytes(payload, &mut pos, big_endian)?;
            let nfields = if fnl > 0 { fn_b.len() / fnl } else { 0 };
            let mut field_names: Vec<String> = Vec::with_capacity(nfields);
            for i in 0..nfields {
                let s = i * fnl;
                let slot = &fn_b[s..s + fnl];
                let nul = slot.iter().position(|&b| b == 0).unwrap_or(fnl);
                field_names.push(String::from_utf8_lossy(&slot[..nul]).into_owned());
            }
            let mut field_data: Vec<Vec<MatArray>> = vec![Vec::with_capacity(numel); nfields];
            for f in 0..nfields {
                for _ in 0..numel {
                    let ftag = read_tag(payload, &mut pos, big_endian)?;
                    if ftag.mi_type != MiType::Matrix {
                        return Err(MatError::InvalidFormat(String::from(
                            "mxSTRUCT_CLASS: expected miMATRIX",
                        )));
                    }
                    let fp = if ftag.nbytes == 0 {
                        vec![]
                    } else {
                        if pos + ftag.nbytes > payload.len() {
                            return Err(MatError::InvalidFormat(String::from(
                                "mxSTRUCT_CLASS: payload truncated",
                            )));
                        }
                        let p = payload[pos..pos + ftag.nbytes].to_vec();
                        pos += pad8(ftag.nbytes);
                        p
                    };
                    let arr = match parse_matrix(&fp, big_endian)? {
                        Some((_, a)) => a,
                        None => MatArray::Numeric(MatNumericArray {
                            class: MatNumericClass::Double,
                            shape: vec![0, 0],
                            real_data: vec![],
                            imag_data: None,
                        }),
                    };
                    field_data[f].push(arr);
                }
            }
            let data: Vec<(String, Vec<MatArray>)> = field_names
                .iter()
                .zip(field_data.into_iter())
                .map(|(n, v)| (n.clone(), v))
                .collect();
            MatArray::Struct(MatStructArray::new(shape, data)?)
        }
        MX_OBJECT_CLASS => {
            return Err(MatError::UnsupportedFeature(String::from(
                "MAT v5 mxOBJECT_CLASS is not supported",
            )));
        }
        other => return Err(MatError::InvalidClass(other)),
    };
    Ok(Some((name, array)))
}
