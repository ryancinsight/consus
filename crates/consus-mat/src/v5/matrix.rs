//! miMATRIX element parser for MAT v5.
// Explicit `for f in 0..nfields` loops index parallel field arrays; reviewed-policy allow.
#![allow(clippy::needless_range_loop)]
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

use super::element::{decode_i32_vec, normalize_endian};
use super::tag::{MiType, read_element_bytes, read_subelement_bytes, read_tag};
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
    parse_matrix_depth(payload, big_endian, 0, &consus_core::ParseBudget::DEFAULT)
}

/// Depth-bounded recursive miMATRIX parser.
///
/// mxCELL/mxSTRUCT arrays nest arbitrary sub-matrices, each of which is a
/// self-similar payload; a hostile file can chain ~40-50 input bytes per
/// level. Rust performs no tail-call elimination, so unbounded recursion
/// overflows the stack — an abort no `Result` can express. `depth` counts the
/// levels already entered and is checked against the [`ParseBudget`] ceiling
/// before recursing.
#[cfg(feature = "alloc")]
fn parse_matrix_depth(
    payload: &[u8],
    big_endian: bool,
    depth: u16,
    budget: &consus_core::ParseBudget,
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
        shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or_else(|| {
                MatError::ShapeError(String::from(
                    "miMATRIX dimensions exceed the addressable element count",
                ))
            })?
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
                let expected_bytes = numel.checked_mul(esz).ok_or_else(|| {
                    MatError::ShapeError(String::from(
                        "miMATRIX dimensions exceed the addressable byte count",
                    ))
                })?;
                if rr.len() != expected_bytes {
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
            let available_element_bytes = payload.len().saturating_sub(pos);
            if numel > available_element_bytes / 8 {
                return Err(MatError::ShapeError(String::from(
                    "mxCELL_CLASS dimensions exceed the available element records",
                )));
            }
            let mut cells: Vec<MatArray> = Vec::with_capacity(numel);
            for _ in 0..numel {
                let ctag = read_tag(payload, &mut pos, big_endian)?;
                if ctag.mi_type != MiType::Matrix {
                    return Err(MatError::InvalidFormat(String::from(
                        "mxCELL_CLASS: expected miMATRIX",
                    )));
                }
                let cp = read_element_bytes(payload, &mut pos, &ctag)?;
                let next_depth = budget
                    .descend(depth, "mat v5 cell-array nesting")
                    .map_err(|e| MatError::InvalidFormat(e.to_string()))?;
                let arr = match parse_matrix_depth(&cp, big_endian, next_depth, budget)? {
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
            let nfields = fn_b.len().checked_div(fnl).unwrap_or(0);
            let total_elements = nfields.checked_mul(numel).ok_or_else(|| {
                MatError::ShapeError(String::from(
                    "mxSTRUCT_CLASS dimensions exceed the addressable element count",
                ))
            })?;
            let available_element_bytes = payload.len().saturating_sub(pos);
            if total_elements > available_element_bytes / 8 {
                return Err(MatError::ShapeError(String::from(
                    "mxSTRUCT_CLASS dimensions exceed the available element records",
                )));
            }
            let mut field_names: Vec<String> = Vec::with_capacity(nfields);
            for i in 0..nfields {
                let s = i * fnl;
                let slot = &fn_b[s..s + fnl];
                let nul = slot.iter().position(|&b| b == 0).unwrap_or(fnl);
                field_names.push(String::from_utf8_lossy(&slot[..nul]).into_owned());
            }
            let mut field_data: Vec<Vec<MatArray>> =
                (0..nfields).map(|_| Vec::with_capacity(numel)).collect();
            for f in 0..nfields {
                for _ in 0..numel {
                    let ftag = read_tag(payload, &mut pos, big_endian)?;
                    if ftag.mi_type != MiType::Matrix {
                        return Err(MatError::InvalidFormat(String::from(
                            "mxSTRUCT_CLASS: expected miMATRIX",
                        )));
                    }
                    let fp = read_element_bytes(payload, &mut pos, &ftag)?;
                    let next_depth = budget
                        .descend(depth, "mat v5 struct-array nesting")
                        .map_err(|e| MatError::InvalidFormat(e.to_string()))?;
                    let arr = match parse_matrix_depth(&fp, big_endian, next_depth, budget)? {
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
                .zip(field_data)
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    fn u32le(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn element(mi_type: u32, data: &[u8]) -> Vec<u8> {
        let mut e = Vec::with_capacity(8 + data.len() + 7);
        u32le(&mut e, mi_type);
        u32le(&mut e, data.len() as u32);
        e.extend_from_slice(data);
        let pad = (8 - (data.len() % 8)) % 8;
        e.resize(e.len() + pad, 0u8);
        e
    }

    /// One `mxCELL_CLASS` matrix content wrapping a single child miMATRIX
    /// element. The child is the *content* of an inner matrix (no outer
    /// `miMATRIX` tag), matching what `read_element_bytes` yields at each
    /// recursion level.
    fn cell_wrapping(child: &[u8]) -> Vec<u8> {
        let mut p = Vec::new();
        // Flags: class 1 (mxCELL_CLASS), no flags bits.
        let mut fdata = [0u8; 8];
        fdata[0..4].copy_from_slice(&(1u32).to_le_bytes());
        p.extend(element(6, &fdata));
        // Dims: 1x1 cell array.
        let ddata = [1i32, 1i32];
        let mut dims = Vec::new();
        for &d in &ddata {
            dims.extend_from_slice(&d.to_le_bytes());
        }
        p.extend(element(5, &dims));
        p.extend(element(1, b""));
        // One cell: the child matrix as a miMATRIX element.
        p.extend(element(14, child));
        p
    }

    /// A deeply nested chain of cell arrays must be rejected by the depth
    /// ceiling, not recurse until the stack overflows (an uncatchable abort).
    #[test]
    fn deeply_nested_cell_chain_is_rejected_by_the_depth_ceiling() {
        // Build a chain ~5x the default depth ceiling (64): each level wraps
        // the previous one in a 1x1 cell matrix.
        let mut payload = scalar_double_leaf();
        for _ in 0..(5 * consus_core::ParseBudget::DEFAULT.max_depth) {
            payload = cell_wrapping(&payload);
        }
        let result = parse_matrix(&payload, false);
        match result {
            Err(MatError::InvalidFormat(message)) if message.contains("nesting") => {}
            other => panic!(
                "a deep cell chain must be rejected as a nesting resource limit, got {other:?}"
            ),
        }
    }

    /// A shallow cell chain (well within the ceiling) still parses.
    #[test]
    fn shallow_cell_chain_still_parses() {
        let payload = cell_wrapping(&scalar_double_leaf());
        let result = parse_matrix(&payload, false);
        assert!(
            result.is_ok() && result.as_ref().unwrap().is_some(),
            "a shallow cell chain must parse, got {result:?}"
        );
    }

    /// Minimal `mxDOUBLE_CLASS` matrix content (no outer `miMATRIX` tag).
    fn scalar_double_leaf() -> Vec<u8> {
        let re_el = element(9, &1.0f64.to_le_bytes());
        let mut p = Vec::new();
        let mut fdata = [0u8; 8];
        fdata[0..4].copy_from_slice(&(6u32).to_le_bytes()); // mxDOUBLE_CLASS
        p.extend(element(6, &fdata));
        let ddata = [1i32, 1i32];
        let mut dims = Vec::new();
        for &d in &ddata {
            dims.extend_from_slice(&d.to_le_bytes());
        }
        p.extend(element(5, &dims));
        p.extend(element(1, b""));
        p.extend(re_el);
        p
    }
}
