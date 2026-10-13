//! Internal HDF5 read helpers for the HDMF crate.
//!
//! Adapted from `consus-nwb/src/storage` with HDMF-specific error messages
//! and an additional `detect_column_data` dispatcher.

#![allow(dead_code)]

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec, vec::Vec};

use consus_core::decode::{decode_to_f64, decode_to_u64};
use consus_core::{AttributeValue, Datatype, Error, Result};
use consus_hdf5::attribute::Hdf5Attribute;
use consus_hdf5::file::Hdf5File;
use consus_io::ReadAt;

use crate::table::ColumnData;

// ---------------------------------------------------------------------------
// Raw-bytes loader
// ---------------------------------------------------------------------------

/// Read the complete raw byte payload of a dataset at `addr`.
///
/// Supports contiguous and chunked layouts; delegates layout dispatch to
/// [`Hdf5File::read_dataset_raw`].
#[cfg(feature = "alloc")]
fn read_dataset_raw<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<u8>> {
    let ds = file.dataset_at(addr)?;
    // For variable-length types, `element_size()` returns `None` because the
    // in-memory size is unknown. On disk, each VL element is encoded as a
    // fixed-size global-heap reference: sequence_length (4 bytes, LE) +
    // heap_collection_address (offset_size bytes) + object_index (4 bytes).
    let variable_element_size = match &ds.datatype {
        Datatype::VariableString { .. } => Some(4 + file.context().offset_bytes() + 4),
        _ => None,
    };
    file.read_dataset_raw(&ds, variable_element_size)
}

// ---------------------------------------------------------------------------
// Typed decoders
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
fn decode_as_f64(raw: &[u8], dtype: &Datatype) -> Result<Vec<f64>> {
    decode_to_f64(raw, dtype).map_err(|err| Error::UnsupportedFeature {
        feature: format!("HDMF: decode as f64: {err}"),
    })
}

#[cfg(feature = "alloc")]
fn decode_as_i64(raw: &[u8], dtype: &Datatype) -> Result<Vec<i64>> {
    // Widen via the core `u64` decoder: unsigned inputs zero-extend
    // through `u64` while signed inputs wrap around and back, so the
    // composition reproduces the direct per-width widening bit for bit
    // (`u64::MAX` decodes to `-1`). One core entry
    // point serves both integer targets.
    decode_to_u64(raw, dtype)
        .map(|values| values.into_iter().map(u64::cast_signed).collect())
        .map_err(|err| Error::UnsupportedFeature {
            feature: format!("HDMF: decode as i64: {err}"),
        })
}

#[cfg(feature = "alloc")]
fn decode_as_u64(raw: &[u8], dtype: &Datatype) -> Result<Vec<u64>> {
    decode_to_u64(raw, dtype).map_err(|err| Error::UnsupportedFeature {
        feature: format!("HDMF: decode as u64: {err}"),
    })
}

#[cfg(feature = "alloc")]
fn decode_as_bool(raw: &[u8], dtype: &Datatype) -> Result<Vec<bool>> {
    match dtype {
        Datatype::Boolean => Ok(raw.iter().map(|&v| v != 0).collect()),
        Datatype::Integer { bits, .. } if bits.get() == 8 => {
            Ok(raw.iter().map(|&v| v != 0).collect())
        }
        other => Err(Error::UnsupportedFeature {
            feature: format!("HDMF: decode as bool not supported for {:?}", other),
        }),
    }
}

#[cfg(feature = "alloc")]
fn decode_as_str<R: ReadAt + Sync>(
    file: &Hdf5File<R>,
    addr: u64,
    raw: &[u8],
    dtype: &Datatype,
) -> Result<Vec<String>> {
    match dtype {
        Datatype::FixedString { length, .. } => {
            if *length == 0 {
                return Ok(vec![]);
            }
            let n = raw.len() / length;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks(*length) {
                let trimmed = match chunk.iter().rposition(|&b| b != 0) {
                    Some(pos) => &chunk[..=pos],
                    None => &chunk[..0],
                };
                let s = core::str::from_utf8(trimmed)
                    .map_err(|e| Error::InvalidFormat {
                        message: format!("HDMF: invalid UTF-8 in string dataset: {}", e),
                    })
                    .map(String::from)?;
                out.push(s);
            }
            Ok(out)
        }
        Datatype::VariableString { .. } => {
            let n_elements = {
                let ds = file.dataset_at(addr)?;
                ds.shape.num_elements()
            };
            if n_elements == 0 {
                return Ok(vec![]);
            }
            let ctx = file.context();
            let byte_vecs = consus_hdf5::heap::resolve_vl_references(file.source(), raw, ctx)?;
            let mut out = Vec::with_capacity(byte_vecs.len());
            for bytes in byte_vecs {
                let s = core::str::from_utf8(&bytes)
                    .map_err(|e| Error::InvalidFormat {
                        message: format!("HDMF: variable-length string UTF-8 error: {}", e),
                    })
                    .map(String::from)?;
                out.push(s);
            }
            Ok(out)
        }
        other => Err(Error::UnsupportedFeature {
            feature: format!("HDMF: decode as str not supported for {:?}", other),
        }),
    }
}

// ---------------------------------------------------------------------------
// Public attribute helpers
// ---------------------------------------------------------------------------

/// Find and return the string value of a named attribute.
///
/// Returns `Error::NotFound` when absent, `Error::InvalidFormat` when the
/// attribute exists but does not decode as a string.
#[cfg(feature = "alloc")]
pub fn read_string_attr(attrs: &[Hdf5Attribute], name: &str) -> Result<String> {
    for attr in attrs {
        if attr.name == name {
            return match attr.decode_value() {
                Ok(AttributeValue::String(s)) => Ok(s),
                Ok(_) => Err(Error::InvalidFormat {
                    message: format!("HDMF: attribute '{}' is not a string", name),
                }),
                Err(e) => Err(e),
            };
        }
    }
    Err(Error::NotFound {
        path: format!("attribute '{}'", name),
    })
}

/// Find and return the string value of a named attribute, decoding variable-length
/// strings via the HDF5 global heap when necessary.
///
/// Handles both `FixedString` attributes (decoded by [`read_string_attr`]) and
/// `VariableString` attributes written by tools such as hdmf Python.
#[cfg(feature = "alloc")]
pub fn read_string_attr_any<R: ReadAt + Sync>(
    attrs: &[Hdf5Attribute],
    name: &str,
    file: &Hdf5File<R>,
) -> Result<String> {
    for attr in attrs {
        if attr.name == name {
            match attr.decode_value() {
                Ok(AttributeValue::String(s)) => return Ok(s),
                Ok(AttributeValue::Bytes(_))
                    if matches!(attr.datatype, Datatype::VariableString { .. }) =>
                {
                    let byte_vecs = consus_hdf5::heap::resolve_vl_references(
                        file.source(),
                        &attr.raw_data,
                        file.context(),
                    )?;
                    let raw = byte_vecs.into_iter().next().unwrap_or_default();
                    return core::str::from_utf8(&raw).map(String::from).map_err(|e| {
                        Error::InvalidFormat {
                            message: format!(
                                "HDMF: VL string attribute '{}' contains invalid UTF-8: {}",
                                name, e
                            ),
                        }
                    });
                }
                Ok(_) => {
                    return Err(Error::InvalidFormat {
                        message: format!("HDMF: attribute '{}' is not a string", name),
                    });
                }
                Err(e) => return Err(e),
            }
        }
    }
    Err(Error::NotFound {
        path: format!("attribute '{}'", name),
    })
}

/// Find and return a string-array attribute by name.
///
/// Accepts both `AttributeValue::StringArray` (1-D fixed-string attribute)
/// and `AttributeValue::String` (wraps it in a single-element Vec).
#[cfg(feature = "alloc")]
pub fn read_string_array_attr(attrs: &[Hdf5Attribute], name: &str) -> Result<Vec<String>> {
    for attr in attrs {
        if attr.name == name {
            return match attr.decode_value() {
                Ok(AttributeValue::StringArray(v)) => Ok(v),
                Ok(AttributeValue::String(s)) => Ok(vec![s]),
                Ok(_) => Err(Error::InvalidFormat {
                    message: format!("HDMF: attribute '{}' is not a string array", name),
                }),
                Err(e) => Err(e),
            };
        }
    }
    Err(Error::NotFound {
        path: format!("attribute '{}'", name),
    })
}

/// Find and return a string-array attribute by name, decoding variable-length
/// strings via the HDF5 global heap when necessary.
///
/// Handles both fixed-length and variable-length string array attributes.
/// An empty VL string array (0 elements) returns an empty `Vec`.
#[cfg(feature = "alloc")]
pub fn read_string_array_attr_any<R: ReadAt + Sync>(
    attrs: &[Hdf5Attribute],
    name: &str,
    file: &Hdf5File<R>,
) -> Result<Vec<String>> {
    for attr in attrs {
        if attr.name == name {
            match attr.decode_value() {
                Ok(AttributeValue::StringArray(v)) => return Ok(v),
                Ok(AttributeValue::String(s)) => return Ok(vec![s]),
                Ok(AttributeValue::Bytes(_))
                    if matches!(attr.datatype, Datatype::VariableString { .. }) =>
                {
                    if attr.raw_data.is_empty() {
                        return Ok(vec![]);
                    }
                    let byte_vecs = consus_hdf5::heap::resolve_vl_references(
                        file.source(),
                        &attr.raw_data,
                        file.context(),
                    )?;
                    let mut out = Vec::with_capacity(byte_vecs.len());
                    for bytes in byte_vecs {
                        let s = core::str::from_utf8(&bytes)
                            .map(String::from)
                            .map_err(|e| Error::InvalidFormat {
                                message: format!(
                                    "HDMF: VL string array attribute '{}' contains invalid UTF-8: {}",
                                    name, e
                                ),
                            })?;
                        out.push(s);
                    }
                    return Ok(out);
                }
                // Empty integer array: hdmf Python may write an empty colnames as
                // a zero-element array of some numeric placeholder type.
                Ok(AttributeValue::UintArray(v)) if v.is_empty() => return Ok(vec![]),
                Ok(AttributeValue::IntArray(v)) if v.is_empty() => return Ok(vec![]),
                Ok(_) => {
                    return Err(Error::InvalidFormat {
                        message: format!("HDMF: attribute '{}' is not a string array", name),
                    });
                }
                Err(e) => return Err(e),
            }
        }
    }
    Err(Error::NotFound {
        path: format!("attribute '{}'", name),
    })
}

// ---------------------------------------------------------------------------
// Public dataset helpers
// ---------------------------------------------------------------------------

/// Load dataset bytes once, then hand them to a typed decode routine.
///
/// The five typed readers below share this fetch path; only the
/// `decode` argument varies. Mirrors the `consus-nwb`
/// storage helper over this crates raw loader.
#[cfg(feature = "alloc")]
fn read_dataset_decoded<R, T, F>(file: &Hdf5File<R>, addr: u64, decode: F) -> Result<Vec<T>>
where
    R: ReadAt + Sync,
    F: FnOnce(&[u8], &Datatype) -> Result<Vec<T>>,
{
    let ds = file.dataset_at(addr)?;
    let dtype = ds.datatype.clone();
    let raw = read_dataset_raw(file, addr)?;
    decode(&raw, &dtype)
}

/// Read a signed-integer dataset as `Vec<i64>`.
#[cfg(feature = "alloc")]
pub fn read_i64_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<i64>> {
    read_dataset_decoded(file, addr, decode_as_i64)
}

/// Read an unsigned-integer dataset as `Vec<u64>`.
#[cfg(feature = "alloc")]
pub fn read_u64_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<u64>> {
    read_dataset_decoded(file, addr, decode_as_u64)
}

/// Read a numeric dataset and auto-widen to `Vec<f64>`.
#[cfg(feature = "alloc")]
pub fn read_f64_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<f64>> {
    read_dataset_decoded(file, addr, decode_as_f64)
}

/// Read a boolean dataset as `Vec<bool>`.
#[cfg(feature = "alloc")]
pub fn read_bool_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<bool>> {
    read_dataset_decoded(file, addr, decode_as_bool)
}

/// Read a string dataset as `Vec<String>`.
#[cfg(feature = "alloc")]
pub fn read_string_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<String>> {
    let ds = file.dataset_at(addr)?;
    let dtype = ds.datatype.clone();
    let raw = read_dataset_raw(file, addr)?;
    decode_as_str(file, addr, &raw, &dtype)
}

/// Detect the HDMF column type from the HDF5 dataset dtype and return as [`ColumnData`].
///
/// Dispatch order:
/// 1. `Float` → [`ColumnData::F64`]
/// 2. `Integer { signed: true }` → [`ColumnData::I64`]
/// 3. `Integer { signed: false }` → [`ColumnData::U64`]
/// 4. `Boolean` → [`ColumnData::Bool`]
/// 5. `FixedString` / `VariableString` → [`ColumnData::Str`]
#[cfg(feature = "alloc")]
pub fn detect_column_data<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<ColumnData> {
    let ds = file.dataset_at(addr)?;
    let dtype = ds.datatype.clone();
    let raw = read_dataset_raw(file, addr)?;

    match &dtype {
        Datatype::Float { .. } => Ok(ColumnData::F64(decode_as_f64(&raw, &dtype)?)),
        Datatype::Integer { signed: true, .. } => Ok(ColumnData::I64(decode_as_i64(&raw, &dtype)?)),
        Datatype::Integer { signed: false, .. } => {
            Ok(ColumnData::U64(decode_as_u64(&raw, &dtype)?))
        }
        Datatype::Boolean => Ok(ColumnData::Bool(decode_as_bool(&raw, &dtype)?)),
        Datatype::FixedString { .. } | Datatype::VariableString { .. } => {
            Ok(ColumnData::Str(decode_as_str(file, addr, &raw, &dtype)?))
        }
        other => Err(Error::UnsupportedFeature {
            feature: format!("HDMF: unsupported column dtype {:?}", other),
        }),
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use consus_core::ByteOrder;
    use core::num::NonZeroUsize;

    fn le_int(bits: usize, signed: bool) -> Datatype {
        Datatype::Integer {
            bits: NonZeroUsize::new(bits).expect("test bit width is non-zero"),
            signed,
            byte_order: ByteOrder::LittleEndian,
        }
    }

    #[test]
    fn integer_widths_widen_to_i64() {
        let cases: &[(Vec<u8>, Datatype, Vec<i64>)] = &[
            (
                vec![0x00u8, 0x80, 0xFF],
                le_int(8, false),
                vec![0, 128, 255],
            ),
            (vec![0xFFu8, 0x01], le_int(8, true), vec![-1, 1]),
            (
                0x1234u16.to_le_bytes().to_vec(),
                le_int(16, false),
                vec![0x1234],
            ),
            ((-2i16).to_le_bytes().to_vec(), le_int(16, true), vec![-2]),
            (
                (-123456i32).to_le_bytes().to_vec(),
                le_int(32, true),
                vec![-123456],
            ),
            (
                u32::MAX.to_le_bytes().to_vec(),
                le_int(32, false),
                vec![4294967295],
            ),
            (
                i64::MIN.to_le_bytes().to_vec(),
                le_int(64, true),
                vec![i64::MIN],
            ),
        ];
        for (raw, dtype, expected) in cases {
            assert_eq!(
                &decode_as_i64(raw, dtype).expect("valid integer decodes"),
                expected
            );
        }
    }

    #[test]
    fn u64_max_wraps_to_negative_one() {
        let raw = u64::MAX.to_le_bytes().to_vec();
        let out = decode_as_i64(&raw, &le_int(64, false)).expect("u64 decodes");
        assert_eq!(out, vec![-1i64]);
    }

    #[test]
    fn u64_decoding_matches_u64_core() {
        let raw = 0xCAFEBABEu32.to_le_bytes().to_vec();
        let via_i64 = decode_as_i64(&raw, &le_int(32, false)).expect("decodes")[0];
        let via_u64 = decode_as_u64(&raw, &le_int(32, false)).expect("decodes")[0];
        assert_eq!(via_i64, via_u64.cast_signed());
        assert_eq!(via_i64, 0xCAFEBABE);
    }

    #[test]
    fn ragged_buffer_fails_closed() {
        let err =
            decode_as_i64(&[0u8; 3], &le_int(16, false)).expect_err("ragged buffer must fail");
        assert!(matches!(err, Error::UnsupportedFeature { .. }));
    }

    #[test]
    fn unsupported_width_fails_closed() {
        let err =
            decode_as_i64(&[0u8; 16], &le_int(128, false)).expect_err("wide integer must fail");
        assert!(matches!(err, Error::UnsupportedFeature { .. }));
    }
}
