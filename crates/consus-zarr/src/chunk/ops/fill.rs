//! Fill-value expansion and checked chunk byte-size arithmetic.

#[cfg(feature = "alloc")]
use crate::chunk::error::ChunkError;
#[cfg(feature = "alloc")]
use crate::metadata::FillValue;
#[cfg(feature = "alloc")]
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// Expands a fill value to a byte vector of the specified length.
///
/// The caller must have already bounded `num_elements`; this helper trusts it.
/// Production chunk-read paths use `try_expand_fill_value` instead, which
/// bounds the resulting byte size before allocating.
#[cfg(feature = "alloc")]
pub fn expand_fill_value(fill_value: &FillValue, dtype: &str, num_elements: u64) -> Vec<u8> {
    try_expand_fill_value(fill_value, dtype, num_elements)
        .expect("fill-value expansion must fit the parser byte ceiling")
}

/// Bounded variant of [`expand_fill_value`].
///
/// `num_elements` comes from an attacker-chosen `.zarray`/`zarr.json` chunk
/// shape, so `num_elements × element_size` is checked against the
/// [`consus_core::ParseBudget`] byte ceiling before the allocation, and the allocation
/// itself is fallible — a hostile shape is a typed error, not an
/// allocation abort or a multiply overflow panic.
#[cfg(feature = "alloc")]
pub fn try_expand_fill_value(
    fill_value: &FillValue,
    dtype: &str,
    num_elements: u64,
) -> Result<Vec<u8>, ChunkError> {
    let element_size = crate::metadata::dtype_to_element_size(dtype).unwrap_or(8);
    let total_size = consus_core::ParseBudget::DEFAULT
        .checked_bytes(
            num_elements.saturating_mul(element_size as u64),
            "zarr fill-value expansion",
        )
        .map_err(|e| ChunkError::StoreError(e.to_string()))?;

    let fill_bytes: Vec<u8> = match fill_value {
        FillValue::Default => vec![0u8; element_size],
        FillValue::Null => vec![0u8; element_size],
        FillValue::Bool(b) => {
            let mut bytes = vec![0u8; element_size];
            if element_size >= 1 {
                bytes[0] = if *b { 1 } else { 0 };
            }
            bytes
        }
        FillValue::Int(i) => {
            let mut bytes = vec![0u8; element_size];
            let val = *i;
            for (idx, byte) in bytes.iter_mut().enumerate() {
                *byte = (val >> (idx * 8)) as u8;
            }
            bytes
        }
        FillValue::Uint(u) => {
            let mut bytes = vec![0u8; element_size];
            let val = *u;
            for (idx, byte) in bytes.iter_mut().enumerate() {
                *byte = (val >> (idx * 8)) as u8;
            }
            bytes
        }
        FillValue::Float(s) => {
            let val: f64 = s.parse().unwrap_or(f64::NAN);
            match element_size {
                4 => (val as f32).to_le_bytes().to_vec(),
                8 => val.to_le_bytes().to_vec(),
                _ => {
                    let raw = val.to_le_bytes();
                    let mut bytes = vec![0u8; element_size];
                    let copy_len = core::cmp::min(element_size, raw.len());
                    bytes[..copy_len].copy_from_slice(&raw[..copy_len]);
                    bytes
                }
            }
        }
        FillValue::String(_) => vec![0u8; element_size],
        FillValue::Bytes(b) => {
            let mut bytes = b.clone();
            bytes.resize(element_size, 0);
            bytes
        }
    };

    let mut result = consus_core::ParseBudget::DEFAULT
        .zeroed(total_size as u64, "zarr fill-value buffer")
        .map_err(|e| ChunkError::StoreError(e.to_string()))?;
    for i in (0..total_size).step_by(element_size) {
        result[i..i + element_size].copy_from_slice(&fill_bytes);
    }
    Ok(result)
}

/// Compute `chunk_elements × element_size` under the parser byte ceiling.
///
/// Both operands derive from attacker-chosen `.zarray`/`zarr.json` shapes, so
/// the product is checked for overflow and against the [`consus_core::ParseBudget`] byte
/// ceiling before it is used to validate or allocate chunk buffers — a hostile
/// shape is a typed error, not a multiply-overflow panic.
#[cfg(feature = "alloc")]
pub(crate) fn checked_chunk_bytes(
    chunk_elements: usize,
    element_size: usize,
) -> Result<usize, ChunkError> {
    consus_core::ParseBudget::DEFAULT
        .checked_bytes(
            u64::try_from(chunk_elements)
                .map_err(|_| {
                    ChunkError::StoreError(String::from("zarr chunk element count overflow"))
                })?
                .checked_mul(element_size as u64)
                .ok_or_else(|| {
                    ChunkError::StoreError(String::from("zarr chunk byte size overflow"))
                })?,
            "zarr chunk byte size",
        )
        .map_err(|e| ChunkError::StoreError(e.to_string()))
}
