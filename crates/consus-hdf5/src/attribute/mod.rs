//! HDF5 attribute message parsing (header message type 0x000C).
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.2.m)
//!
//! Attributes are small named datasets attached to any HDF5 object.
//! Each attribute is stored as a header message containing the name,
//! datatype, dataspace, and raw data inline.
//!
//! ### Version 1 Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 1 | Version (1) |
//! | 1 | 1 | Reserved |
//! | 2 | 2 | Name size (including null terminator) |
//! | 4 | 2 | Datatype size |
//! | 6 | 2 | Dataspace size |
//! | 8 | var | Name (null-terminated, padded to 8-byte boundary) |
//! | var | var | Datatype (padded to 8-byte boundary) |
//! | var | var | Dataspace (padded to 8-byte boundary) |
//! | var | var | Data |
//!
//! ### Version 2 Layout
//!
//! Identical field order to v1 but components are NOT padded to
//! 8-byte boundaries. A flags byte replaces the reserved byte.
//!
//! ### Version 3 Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 1 | Version (3) |
//! | 1 | 1 | Flags (bit 0: shared datatype, bit 1: shared dataspace) |
//! | 2 | 2 | Name size (byte count, NOT null-terminated) |
//! | 4 | 2 | Datatype size |
//! | 6 | 2 | Dataspace size |
//! | 8 | 1 | Character encoding (0=ASCII, 1=UTF-8) |
//! | 9 | var | Name (length from name_size, no null terminator) |
//! | var | var | Datatype |
//! | var | var | Dataspace |
//! | var | var | Data |

pub mod info;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};

#[cfg(feature = "alloc")]
use byteorder::BigEndian;

#[cfg(feature = "alloc")]
use consus_core::{Error, Result, Shape};

/// Parsed HDF5 attribute from an attribute header message.
///
/// Contains the attribute name, decoded datatype descriptor, dataspace
/// (shape), and the raw attribute data bytes. The caller is responsible
/// for interpreting `raw_data` according to `datatype` and `shape`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct Hdf5Attribute {
    /// Attribute name.
    pub name: String,
    /// Attribute datatype (canonical representation).
    pub datatype: consus_core::Datatype,
    /// Attribute dataspace (shape).
    pub shape: Shape,
    /// Raw attribute data bytes (uninterpreted).
    pub raw_data: Vec<u8>,
    /// Character encoding of the name (0 = ASCII, 1 = UTF-8).
    /// Versions 1 and 2 default to 0 (ASCII).
    pub name_encoding: u8,
    /// Creation order index, if tracked by the containing object header.
    /// Populated by the caller; not present in the attribute message itself.
    pub creation_order: Option<u16>,
}

/// Alignment boundary for version 1 attribute message components.
const V1_ALIGNMENT: usize = 8;

/// Round `offset` up to the next multiple of `alignment`.
///
/// ## Invariant
///
/// `align_up(n, a) % a == 0` for all `n` and power-of-two `a`.
const fn align_up(offset: usize, alignment: usize) -> usize {
    let mask = alignment - 1;
    (offset + mask) & !mask
}

/// Minimum header size shared by all attribute message versions.
/// version(1) + flags/reserved(1) + name_size(2) + dt_size(2) + ds_size(2) = 8.
const MIN_HEADER_SIZE: usize = 8;

/// Extended header size for version 3 (includes encoding byte).
/// version(1) + flags(1) + name_size(2) + dt_size(2) + ds_size(2) + encoding(1) = 9.
const V3_HEADER_SIZE: usize = 9;

#[cfg(feature = "alloc")]
impl Hdf5Attribute {
    /// Parse an attribute from raw header message bytes.
    ///
    /// Dispatches to the version-specific parser based on the first byte.
    ///
    /// ## Arguments
    ///
    /// - `data`: raw bytes of the attribute header message payload.
    /// - `ctx`: parsing context (offset/length sizes from the superblock).
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the version byte is unsupported (not 1, 2, or 3).
    /// - [`Error::InvalidFormat`] if the message is truncated.
    pub fn parse(data: &[u8], ctx: &crate::address::ParseContext) -> Result<Self> {
        if data.len() < MIN_HEADER_SIZE {
            return Err(Error::InvalidFormat {
                message: String::from("attribute message too short for header"),
            });
        }

        let version = data[0];
        match version {
            1 => Self::parse_v1(data, ctx),
            2 => Self::parse_v2(data, ctx),
            3 => Self::parse_v3(data, ctx),
            _ => Err(Error::InvalidFormat {
                message: alloc::format!("unsupported attribute message version: {version}"),
            }),
        }
    }

    /// Parse a version 1 attribute message.
    ///
    /// Components (name, datatype, dataspace) are each padded to 8-byte boundaries.
    fn parse_v1(data: &[u8], ctx: &crate::address::ParseContext) -> Result<Self> {
        // data[1] is reserved.
        let name_size = LittleEndian::read_u16(&data[2..4]) as usize;
        let dt_size = LittleEndian::read_u16(&data[4..6]) as usize;
        let ds_size = LittleEndian::read_u16(&data[6..8]) as usize;

        let mut cursor = MIN_HEADER_SIZE;

        // --- Name (null-terminated, padded to 8-byte boundary) ---
        let name = Self::read_null_terminated_name(data, cursor, name_size)?;
        cursor += align_up(name_size, V1_ALIGNMENT);

        // --- Datatype (padded to 8-byte boundary) ---
        let datatype = Self::read_datatype(data, cursor, dt_size)?;
        cursor += align_up(dt_size, V1_ALIGNMENT);

        // --- Dataspace (padded to 8-byte boundary) ---
        let shape = Self::read_dataspace(data, cursor, ds_size, ctx)?;
        cursor += align_up(ds_size, V1_ALIGNMENT);

        // --- Data (remainder) ---
        let raw_data = Self::read_raw_data(data, cursor);

        Ok(Self {
            name,
            datatype,
            shape,
            raw_data,
            name_encoding: 0, // v1 assumes ASCII
            creation_order: None,
        })
    }

    /// Parse a version 2 attribute message.
    ///
    /// Identical field layout to v1 but without padding between components.
    /// Byte 1 is a flags byte instead of reserved.
    fn parse_v2(data: &[u8], ctx: &crate::address::ParseContext) -> Result<Self> {
        let _flags = data[1];
        let name_size = LittleEndian::read_u16(&data[2..4]) as usize;
        let dt_size = LittleEndian::read_u16(&data[4..6]) as usize;
        let ds_size = LittleEndian::read_u16(&data[6..8]) as usize;

        let mut cursor = MIN_HEADER_SIZE;

        // --- Name (null-terminated, no padding) ---
        let name = Self::read_null_terminated_name(data, cursor, name_size)?;
        cursor += name_size;

        // --- Datatype (no padding) ---
        let datatype = Self::read_datatype(data, cursor, dt_size)?;
        cursor += dt_size;

        // --- Dataspace (no padding) ---
        let shape = Self::read_dataspace(data, cursor, ds_size, ctx)?;
        cursor += ds_size;

        // --- Data (remainder) ---
        let raw_data = Self::read_raw_data(data, cursor);

        Ok(Self {
            name,
            datatype,
            shape,
            raw_data,
            name_encoding: 0, // v2 assumes ASCII
            creation_order: None,
        })
    }

    /// Parse a version 3 attribute message.
    ///
    /// Name is NOT null-terminated; length is exact byte count.
    /// Includes a character encoding byte (0=ASCII, 1=UTF-8).
    /// No padding between components.
    fn parse_v3(data: &[u8], ctx: &crate::address::ParseContext) -> Result<Self> {
        if data.len() < V3_HEADER_SIZE {
            return Err(Error::InvalidFormat {
                message: String::from("attribute v3 message too short for header"),
            });
        }

        let _flags = data[1];
        let name_size = LittleEndian::read_u16(&data[2..4]) as usize;
        let dt_size = LittleEndian::read_u16(&data[4..6]) as usize;
        let ds_size = LittleEndian::read_u16(&data[6..8]) as usize;
        let name_encoding = data[8];

        let mut cursor = V3_HEADER_SIZE;

        // --- Name (exact length, no null terminator, no padding) ---
        if cursor + name_size > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "attribute v3 name overflows message: need {} bytes at offset {}, have {}",
                    name_size,
                    cursor,
                    data.len(),
                ),
            });
        }
        let name_bytes = &data[cursor..cursor + name_size];
        let name = core::str::from_utf8(name_bytes)
            .map_err(|_| Error::InvalidFormat {
                message: String::from("attribute v3 name is not valid UTF-8"),
            })?
            .into();
        cursor += name_size;

        // --- Datatype (no padding) ---
        let datatype = Self::read_datatype(data, cursor, dt_size)?;
        cursor += dt_size;

        // --- Dataspace (no padding) ---
        let shape = Self::read_dataspace(data, cursor, ds_size, ctx)?;
        cursor += ds_size;

        // --- Data (remainder) ---
        let raw_data = Self::read_raw_data(data, cursor);

        Ok(Self {
            name,
            datatype,
            shape,
            raw_data,
            name_encoding,
            creation_order: None,
        })
    }

    /// Read a null-terminated name from `data[offset..offset+size]`.
    ///
    /// The `size` field includes the null terminator. The returned string
    /// excludes the terminator. Validates UTF-8 encoding.
    fn read_null_terminated_name(data: &[u8], offset: usize, size: usize) -> Result<String> {
        if size == 0 {
            return Ok(String::new());
        }
        if offset + size > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "attribute name overflows message: need {} bytes at offset {}, have {}",
                    size,
                    offset,
                    data.len(),
                ),
            });
        }
        let name_bytes = &data[offset..offset + size];
        // Strip trailing null bytes.
        let end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        core::str::from_utf8(&name_bytes[..end])
            .map(|s| String::from(s))
            .map_err(|_| Error::InvalidFormat {
                message: String::from("attribute name is not valid UTF-8"),
            })
    }

    /// Read and parse a datatype from `data[offset..offset+size]`.
    ///
    /// Delegates to `crate::datatype::compound::parse_datatype` which handles
    /// all HDF5 datatype classes.
    fn read_datatype(data: &[u8], offset: usize, size: usize) -> Result<consus_core::Datatype> {
        if size == 0 {
            return Err(Error::InvalidFormat {
                message: String::from("attribute datatype has zero size"),
            });
        }
        if offset + size > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "attribute datatype overflows message: need {} bytes at offset {}, have {}",
                    size,
                    offset,
                    data.len(),
                ),
            });
        }
        let dt_bytes = &data[offset..offset + size];
        crate::datatype::compound::parse_datatype(dt_bytes)
    }

    /// Read and parse a dataspace from `data[offset..offset+size]`.
    ///
    /// Delegates to `crate::dataspace::parse_dataspace`. A zero-size dataspace
    /// is treated as scalar (rank-0).
    fn read_dataspace(
        data: &[u8],
        offset: usize,
        size: usize,
        ctx: &crate::address::ParseContext,
    ) -> Result<Shape> {
        if size == 0 {
            // Scalar dataspace: rank 0, no dimensions.
            return Ok(Shape::scalar());
        }
        if offset + size > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "attribute dataspace overflows message: need {} bytes at offset {}, have {}",
                    size,
                    offset,
                    data.len(),
                ),
            });
        }
        let ds_bytes = &data[offset..offset + size];
        crate::dataspace::parse_dataspace(ds_bytes, ctx.offset_size)
    }

    /// Read raw data bytes from `data[offset..]`.
    ///
    /// Returns all remaining bytes as the attribute's raw data payload.
    fn read_raw_data(data: &[u8], offset: usize) -> Vec<u8> {
        if offset >= data.len() {
            Vec::new()
        } else {
            Vec::from(&data[offset..])
        }
    }

    /// Decode the raw attribute data bytes into a typed [`consus_core::AttributeValue`].
    ///
    /// Interprets `raw_data` according to `datatype` and `shape`.
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::UnsupportedFeature`] for variable-length or
    ///   compound types that require heap traversal.
    /// - [`consus_core::Error::InvalidFormat`] if `raw_data` is too short.
    #[cfg(feature = "alloc")]
    pub fn decode_value(&self) -> consus_core::Result<consus_core::AttributeValue> {
        decode_attribute_value(&self.raw_data, &self.datatype, &self.shape)
    }
}

// ---------------------------------------------------------------------------
// Attribute value decoding
// ---------------------------------------------------------------------------

/// Decode raw attribute bytes into a typed [`consus_core::AttributeValue`].
///
/// ## Algorithm
///
/// 1. Determine element size from `datatype`. Returns `UnsupportedFeature`
///    for variable-length types.
/// 2. Determine total element count from `shape` (`1` for scalar).
/// 3. Dispatch on `datatype` class to interpret the raw bytes:
///    - `Integer`/`Boolean` → `Int`/`Uint`/`IntArray`/`UintArray`.
///    - `Float` → `Float`/`FloatArray`.
///    - `FixedString` → `String`/`StringArray` (null-stripped, UTF-8 lossy).
///    - All others → `Bytes` (opaque copy).
#[cfg(feature = "alloc")]
pub fn decode_attribute_value(
    raw: &[u8],
    dtype: &consus_core::Datatype,
    shape: &Shape,
) -> consus_core::Result<consus_core::AttributeValue> {
    use consus_core::{AttributeValue, Datatype, Error};

    let total_elements: usize = if shape.rank() == 0 {
        1
    } else {
        shape.num_elements()
    };
    let is_scalar = shape.rank() == 0;

    match dtype {
        Datatype::Boolean => {
            if is_scalar {
                let v: u64 = if raw.first().copied().unwrap_or(0) != 0 {
                    1
                } else {
                    0
                };
                Ok(AttributeValue::Uint(v))
            } else {
                let vals: Vec<u64> = raw
                    .iter()
                    .take(total_elements)
                    .map(|&b| if b != 0 { 1 } else { 0 })
                    .collect();
                Ok(AttributeValue::UintArray(vals))
            }
        }

        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            let sz = bits.get() / 8;
            let need = sz * total_elements;
            if raw.len() < need {
                return Err(Error::InvalidFormat {
                    message: alloc::format!(
                        "attribute integer data too short: need {need}, have {}",
                        raw.len()
                    ),
                });
            }
            if *signed {
                if is_scalar {
                    Ok(AttributeValue::Int(read_int_le(raw, sz, *byte_order)?))
                } else {
                    let vals: Vec<i64> = (0..total_elements)
                        .map(|i| read_int_le(&raw[i * sz..], sz, *byte_order))
                        .collect::<consus_core::Result<_>>()?;
                    Ok(AttributeValue::IntArray(vals))
                }
            } else {
                if is_scalar {
                    Ok(AttributeValue::Uint(read_uint_le(raw, sz, *byte_order)?))
                } else {
                    let vals: Vec<u64> = (0..total_elements)
                        .map(|i| read_uint_le(&raw[i * sz..], sz, *byte_order))
                        .collect::<consus_core::Result<_>>()?;
                    Ok(AttributeValue::UintArray(vals))
                }
            }
        }

        Datatype::Float { bits, byte_order } => {
            let sz = bits.get() / 8;
            let need = sz * total_elements;
            if raw.len() < need {
                return Err(Error::InvalidFormat {
                    message: alloc::format!(
                        "attribute float data too short: need {need}, have {}",
                        raw.len()
                    ),
                });
            }
            if is_scalar {
                Ok(AttributeValue::Float(read_float(raw, sz, *byte_order)?))
            } else {
                let vals: Vec<f64> = (0..total_elements)
                    .map(|i| read_float(&raw[i * sz..], sz, *byte_order))
                    .collect::<consus_core::Result<_>>()?;
                Ok(AttributeValue::FloatArray(vals))
            }
        }

        Datatype::FixedString { length, .. } => {
            if is_scalar {
                let end = (*length).min(raw.len());
                let s = strip_null_and_decode(&raw[..end]);
                Ok(AttributeValue::String(s))
            } else {
                let strs: Vec<alloc::string::String> = (0..total_elements)
                    .map(|i| {
                        let start = i * length;
                        let end = (start + length).min(raw.len());
                        if start >= raw.len() {
                            alloc::string::String::new()
                        } else {
                            strip_null_and_decode(&raw[start..end])
                        }
                    })
                    .collect();
                Ok(AttributeValue::StringArray(strs))
            }
        }

        _ => {
            // For compound, VL, enum, array, opaque, reference types return raw bytes.
            Ok(AttributeValue::Bytes(Vec::from(raw)))
        }
    }
}

/// Read a signed integer of `size` bytes from `raw` in the given byte order.
///
/// Supported sizes: 1, 2, 4, 8.
#[cfg(feature = "alloc")]
fn read_int_le(raw: &[u8], size: usize, order: consus_core::ByteOrder) -> consus_core::Result<i64> {
    if raw.len() < size {
        return Err(consus_core::Error::InvalidFormat {
            message: alloc::format!("integer value truncated: need {size}, have {}", raw.len()),
        });
    }
    let v = match (size, order) {
        (1, _) => raw[0] as i8 as i64,
        (2, consus_core::ByteOrder::LittleEndian) => LittleEndian::read_i16(raw) as i64,
        (2, consus_core::ByteOrder::BigEndian) => BigEndian::read_i16(raw) as i64,
        (4, consus_core::ByteOrder::LittleEndian) => LittleEndian::read_i32(raw) as i64,
        (4, consus_core::ByteOrder::BigEndian) => BigEndian::read_i32(raw) as i64,
        (8, consus_core::ByteOrder::LittleEndian) => LittleEndian::read_i64(raw),
        (8, consus_core::ByteOrder::BigEndian) => BigEndian::read_i64(raw),
        _ => {
            return Err(consus_core::Error::UnsupportedFeature {
                feature: alloc::format!("signed integer decode for size {size}"),
            });
        }
    };
    Ok(v)
}

/// Read an unsigned integer of `size` bytes from `raw`.
#[cfg(feature = "alloc")]
fn read_uint_le(
    raw: &[u8],
    size: usize,
    order: consus_core::ByteOrder,
) -> consus_core::Result<u64> {
    if raw.len() < size {
        return Err(consus_core::Error::InvalidFormat {
            message: alloc::format!(
                "unsigned integer value truncated: need {size}, have {}",
                raw.len()
            ),
        });
    }
    let v = match (size, order) {
        (1, _) => u64::from(raw[0]),
        (2, consus_core::ByteOrder::LittleEndian) => u64::from(LittleEndian::read_u16(raw)),
        (2, consus_core::ByteOrder::BigEndian) => u64::from(BigEndian::read_u16(raw)),
        (4, consus_core::ByteOrder::LittleEndian) => u64::from(LittleEndian::read_u32(raw)),
        (4, consus_core::ByteOrder::BigEndian) => u64::from(BigEndian::read_u32(raw)),
        (8, consus_core::ByteOrder::LittleEndian) => LittleEndian::read_u64(raw),
        (8, consus_core::ByteOrder::BigEndian) => BigEndian::read_u64(raw),
        _ => {
            return Err(consus_core::Error::UnsupportedFeature {
                feature: alloc::format!("unsigned integer decode for size {size}"),
            });
        }
    };
    Ok(v)
}

/// Read a floating-point value of `size` bytes from `raw`.
///
/// Supported sizes: 4 (f32 → f64) and 8 (f64).
#[cfg(feature = "alloc")]
fn read_float(raw: &[u8], size: usize, order: consus_core::ByteOrder) -> consus_core::Result<f64> {
    if raw.len() < size {
        return Err(consus_core::Error::InvalidFormat {
            message: alloc::format!("float value truncated: need {size}, have {}", raw.len()),
        });
    }
    let v = match (size, order) {
        (4, consus_core::ByteOrder::LittleEndian) => {
            f64::from(f32::from_bits(LittleEndian::read_u32(raw)))
        }
        (4, consus_core::ByteOrder::BigEndian) => {
            f64::from(f32::from_bits(BigEndian::read_u32(raw)))
        }
        (8, consus_core::ByteOrder::LittleEndian) => f64::from_bits(LittleEndian::read_u64(raw)),
        (8, consus_core::ByteOrder::BigEndian) => f64::from_bits(BigEndian::read_u64(raw)),
        _ => {
            return Err(consus_core::Error::UnsupportedFeature {
                feature: alloc::format!("float decode for size {size}"),
            });
        }
    };
    Ok(v)
}

/// Strip null bytes and decode as UTF-8 (lossy).
#[cfg(feature = "alloc")]
fn strip_null_and_decode(raw: &[u8]) -> alloc::string::String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    alloc::string::String::from_utf8_lossy(&raw[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_cases() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(16, 8), 16);
        assert_eq!(align_up(13, 8), 16);
    }

    #[test]
    fn reject_too_short() {
        let data = [0u8; 4];
        let ctx = crate::address::ParseContext::new(8, 8);
        let err = Hdf5Attribute::parse(&data, &ctx).unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { message } => {
                assert!(message.contains("too short"));
            }
            _ => panic!("expected InvalidFormat, got: {err:?}"),
        }
    }

    #[test]
    fn reject_unsupported_version() {
        let mut data = [0u8; 8];
        data[0] = 4; // unsupported version
        let ctx = crate::address::ParseContext::new(8, 8);
        let err = Hdf5Attribute::parse(&data, &ctx).unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { message } => {
                assert!(message.contains("unsupported"));
            }
            _ => panic!("expected InvalidFormat, got: {err:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn decode_value_u32_scalar() {
        use consus_core::{AttributeValue, ByteOrder as CoreByteOrder, Datatype};
        use core::num::NonZeroUsize;

        let dtype = Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: CoreByteOrder::LittleEndian,
            signed: false,
        };
        let shape = Shape::scalar();
        let raw = 42u32.to_le_bytes().to_vec();
        let attr = Hdf5Attribute {
            name: alloc::string::String::from("x"),
            datatype: dtype,
            shape,
            raw_data: raw,
            name_encoding: 0,
            creation_order: None,
        };
        match attr.decode_value().unwrap() {
            AttributeValue::Uint(v) => assert_eq!(v, 42),
            other => panic!("expected Uint, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn decode_value_f32_scalar() {
        use consus_core::{AttributeValue, ByteOrder as CoreByteOrder, Datatype};
        use core::num::NonZeroUsize;

        let dtype = Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: CoreByteOrder::LittleEndian,
        };
        let shape = Shape::scalar();
        let raw = 3.14f32.to_bits().to_le_bytes().to_vec();
        let attr = Hdf5Attribute {
            name: alloc::string::String::from("pi"),
            datatype: dtype,
            shape,
            raw_data: raw,
            name_encoding: 0,
            creation_order: None,
        };
        match attr.decode_value().unwrap() {
            AttributeValue::Float(v) => {
                assert!((v - 3.14f64).abs() < 1e-5, "expected ~3.14, got {v}");
            }
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn decode_value_i16_array() {
        use consus_core::{AttributeValue, ByteOrder as CoreByteOrder, Datatype, Extent};
        use core::num::NonZeroUsize;

        let dtype = Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: CoreByteOrder::LittleEndian,
            signed: true,
        };
        let shape = Shape::new(&[Extent::Fixed(3)]);
        let mut raw = Vec::new();
        raw.extend_from_slice(&(-1i16).to_le_bytes());
        raw.extend_from_slice(&0i16.to_le_bytes());
        raw.extend_from_slice(&100i16.to_le_bytes());
        let attr = Hdf5Attribute {
            name: alloc::string::String::from("arr"),
            datatype: dtype,
            shape,
            raw_data: raw,
            name_encoding: 0,
            creation_order: None,
        };
        match attr.decode_value().unwrap() {
            AttributeValue::IntArray(v) => assert_eq!(v, &[-1, 0, 100]),
            other => panic!("expected IntArray, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn decode_value_fixed_string() {
        use consus_core::{AttributeValue, Datatype, StringEncoding};

        let dtype = Datatype::FixedString {
            length: 8,
            encoding: StringEncoding::Ascii,
        };
        let shape = Shape::scalar();
        let raw = b"hello\0\0\0".to_vec();
        let attr = Hdf5Attribute {
            name: alloc::string::String::from("label"),
            datatype: dtype,
            shape,
            raw_data: raw,
            name_encoding: 0,
            creation_order: None,
        };
        match attr.decode_value().unwrap() {
            AttributeValue::String(s) => assert_eq!(s, "hello"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    // ── proptest harnesses (M-052) ─────────────────────────────────────────
    #[cfg(feature = "alloc")]
    mod proptest_harnesses {
        use super::*;
        use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
        use proptest::prelude::*;

        proptest! {
            /// Safety invariant: `decode_attribute_value` never panics on arbitrary
            /// raw bytes for a FixedString scalar datatype.
            ///
            /// The function must return Ok or Err, never panic.
            #[test]
            fn decode_attribute_value_never_panics_on_arbitrary_bytes(
                raw in proptest::collection::vec(any::<u8>(), 0..=256),
            ) {
                let dtype = Datatype::FixedString {
                    length: 8,
                    encoding: StringEncoding::Ascii,
                };
                let shape = Shape::scalar();
                let _ = decode_attribute_value(&raw, &dtype, &shape);
            }

            /// Safety invariant: `decode_attribute_value` never panics for f64
            /// scalar with arbitrary raw bytes.
            #[test]
            fn decode_attribute_value_f64_scalar_never_panics(
                raw in proptest::collection::vec(any::<u8>(), 0..=256),
            ) {
                use core::num::NonZeroUsize;
                let dtype = Datatype::Float {
                    bits: NonZeroUsize::new(64).unwrap(),
                    byte_order: ByteOrder::LittleEndian,
                };
                let shape = Shape::scalar();
                let _ = decode_attribute_value(&raw, &dtype, &shape);
            }

            /// Safety invariant: `decode_attribute_value` never panics for i32
            /// scalar with arbitrary raw bytes.
            #[test]
            fn decode_attribute_value_i32_scalar_never_panics(
                raw in proptest::collection::vec(any::<u8>(), 0..=256),
            ) {
                use core::num::NonZeroUsize;
                let dtype = Datatype::Integer {
                    bits: NonZeroUsize::new(32).unwrap(),
                    signed: true,
                    byte_order: ByteOrder::LittleEndian,
                };
                let shape = Shape::scalar();
                let _ = decode_attribute_value(&raw, &dtype, &shape);
            }

            /// Safety invariant: `decode_attribute_value` never panics for 1-D
            /// FixedString array with arbitrary raw bytes and arbitrary element counts.
            #[test]
            fn decode_attribute_value_string_array_never_panics(
                raw in proptest::collection::vec(any::<u8>(), 0..=256),
                n in 1usize..=8,
            ) {
                let dtype = Datatype::FixedString {
                    length: 4,
                    encoding: StringEncoding::Ascii,
                };
                let shape = Shape::fixed(&[n]);
                let _ = decode_attribute_value(&raw, &dtype, &shape);
            }
        }
    }
}
