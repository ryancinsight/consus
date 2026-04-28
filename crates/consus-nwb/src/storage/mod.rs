//! HDF5-backed storage helpers for NWB readers.
//!
//! Provides typed attribute and dataset access utilities used by
//! [`crate::file`] to extract NWB session metadata and neurodata values
//! from an HDF5 file without duplicating low-level decoding logic.
//!
//! ## Scope
//!
//! | Helper                    | Input                       | Output         |
//! |---------------------------|-----------------------------|----------------|
//! | [`read_string_attr`]      | attribute list + name       | `String`       |
//! | [`read_f64_attr`]         | attribute list + name       | `f64`          |
//! | [`read_f64_dataset`]      | `Hdf5File` + object address | `Vec<f64>`     |
//! | [`read_scalar_f64_dataset`]| `Hdf5File` + object address | `f64`         |
//!
//! All helpers propagate [`consus_core::Error`] variants directly; no
//! intermediate error type is introduced.
//!
//! ## Invariants
//!
//! - [`read_string_attr`] returns `Error::NotFound` when the named attribute
//!   is absent; `Error::InvalidFormat` when the attribute exists but its
//!   decoded value is not a `String`.
//! - [`read_f64_attr`] returns `Error::NotFound` when absent; `Error::InvalidFormat`
//!   when the attribute exists but is non-numeric.
//! - [`read_f64_dataset`] supports `f32`, `f64`, and signed/unsigned integer
//!   (8, 16, 32, 64-bit) contiguous and chunked datasets. All values are
//!   promoted to `f64` (IEEE 754 double).
//! - [`read_scalar_f64_dataset`] is a thin wrapper over [`read_f64_dataset`]
//!   that extracts the single first element.
//! - All functions are pure with respect to the file image: they read but
//!   never mutate the underlying HDF5 source.

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec, vec::Vec};

use consus_core::{AttributeValue, ByteOrder, Datatype, Error, Result};
use consus_hdf5::attribute::Hdf5Attribute;
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use consus_io::ReadAt;

// ---------------------------------------------------------------------------
// Private byte-reading helpers
// ---------------------------------------------------------------------------

/// Interpret the first 2 bytes of `c` as a u16 with the given byte order.
fn read_u16(c: &[u8], bo: ByteOrder) -> u16 {
    let arr = [c[0], c[1]];
    match bo {
        ByteOrder::LittleEndian => u16::from_le_bytes(arr),
        ByteOrder::BigEndian => u16::from_be_bytes(arr),
    }
}

/// Interpret the first 2 bytes of `c` as an i16 with the given byte order.
fn read_i16(c: &[u8], bo: ByteOrder) -> i16 {
    let arr = [c[0], c[1]];
    match bo {
        ByteOrder::LittleEndian => i16::from_le_bytes(arr),
        ByteOrder::BigEndian => i16::from_be_bytes(arr),
    }
}

/// Interpret the first 4 bytes of `c` as a u32 with the given byte order.
fn read_u32(c: &[u8], bo: ByteOrder) -> u32 {
    let arr = [c[0], c[1], c[2], c[3]];
    match bo {
        ByteOrder::LittleEndian => u32::from_le_bytes(arr),
        ByteOrder::BigEndian => u32::from_be_bytes(arr),
    }
}

/// Interpret the first 4 bytes of `c` as an i32 with the given byte order.
fn read_i32(c: &[u8], bo: ByteOrder) -> i32 {
    let arr = [c[0], c[1], c[2], c[3]];
    match bo {
        ByteOrder::LittleEndian => i32::from_le_bytes(arr),
        ByteOrder::BigEndian => i32::from_be_bytes(arr),
    }
}

/// Interpret the first 8 bytes of `c` as a u64 with the given byte order.
fn read_u64(c: &[u8], bo: ByteOrder) -> u64 {
    let arr = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
    match bo {
        ByteOrder::LittleEndian => u64::from_le_bytes(arr),
        ByteOrder::BigEndian => u64::from_be_bytes(arr),
    }
}

/// Interpret the first 8 bytes of `c` as an i64 with the given byte order.
fn read_i64(c: &[u8], bo: ByteOrder) -> i64 {
    let arr = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
    match bo {
        ByteOrder::LittleEndian => i64::from_le_bytes(arr),
        ByteOrder::BigEndian => i64::from_be_bytes(arr),
    }
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

/// Find and decode a scalar string attribute by name from an attribute list.
///
/// Iterates `attrs` in order and returns the decoded string value of the
/// first attribute whose `name` field equals the requested `name`.
///
/// ## Errors
///
/// - [`Error::NotFound`] — no attribute with the given name exists in `attrs`.
/// - [`Error::InvalidFormat`] — an attribute with the given name exists but
///   its decoded value is not an [`AttributeValue::String`].
/// - Propagates any error from [`Hdf5Attribute::decode_value`].
#[cfg(feature = "alloc")]
pub fn read_string_attr(attrs: &[Hdf5Attribute], name: &str) -> Result<String> {
    for attr in attrs {
        if attr.name == name {
            return match attr.decode_value() {
                Ok(AttributeValue::String(s)) => Ok(s),
                Ok(_) => Err(Error::InvalidFormat {
                    message: format!("NWB: attribute '{}' value is not a string", name),
                }),
                Err(e) => Err(e),
            };
        }
    }
    Err(Error::NotFound {
        path: format!("attribute '{}'", name),
    })
}

/// Find a float-valued attribute by name and return it as `f64`.
///
/// Accepts [`AttributeValue::Float`], [`AttributeValue::Int`], and
/// [`AttributeValue::Uint`] variants, widening integer values to `f64`
/// via `as f64` cast.
///
/// ## Errors
///
/// - [`Error::NotFound`] — no attribute with the given name exists in `attrs`.
/// - [`Error::InvalidFormat`] — an attribute with the given name exists but
///   its decoded value is not a numeric type.
/// - Propagates any error from [`Hdf5Attribute::decode_value`].
#[cfg(feature = "alloc")]
pub fn read_f64_attr(attrs: &[Hdf5Attribute], name: &str) -> Result<f64> {
    for attr in attrs {
        if attr.name == name {
            return match attr.decode_value() {
                Ok(AttributeValue::Float(v)) => Ok(v),
                Ok(AttributeValue::Int(v)) => Ok(v as f64),
                Ok(AttributeValue::Uint(v)) => Ok(v as f64),
                Ok(_) => Err(Error::InvalidFormat {
                    message: format!("NWB: attribute '{}' is not a numeric type", name),
                }),
                Err(e) => Err(e),
            };
        }
    }
    Err(Error::NotFound {
        path: format!("attribute '{}'", name),
    })
}

// ---------------------------------------------------------------------------
// Dataset helpers
// ---------------------------------------------------------------------------

/// Read a numeric dataset and return its values as `Vec<f64>`.
///
/// Supports the following source datatypes:
///
/// | Source type        | Conversion                                  |
/// |--------------------|---------------------------------------------|
/// | `f64` (any order)  | identity (byte-reinterpret per byte order)  |
/// | `f32` (any order)  | widening cast via `as f64`                  |
/// | `u8` / `i8`        | zero-/sign-extend then `as f64`             |
/// | `u16` / `i16`      | byte-order decode then `as f64`             |
/// | `u32` / `i32`      | byte-order decode then `as f64`             |
/// | `u64` / `i64`      | byte-order decode then `as f64`             |
///
/// Both contiguous and chunked storage layouts are supported. Compact and
/// virtual layouts return [`Error::UnsupportedFeature`].
///
/// ## Algorithm
///
/// 1. Read dataset metadata from the object header at `addr`.
/// 2. Read the raw byte payload (contiguous or chunked path).
/// 3. Dispatch on the element datatype to interpret the raw bytes as `f64`.
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] — contiguous dataset has no data address.
/// - [`Error::UnsupportedFeature`] — compact or virtual layout, or an
///   integer bit-width other than 8, 16, 32, or 64.
/// - Propagates HDF5 I/O errors from `Hdf5File::dataset_at`,
///   `read_contiguous_dataset_bytes`, and `read_chunked_dataset_all_bytes`.
#[cfg(feature = "alloc")]
pub fn read_f64_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<f64>> {
    let ds = file.dataset_at(addr)?;

    let raw: Vec<u8> = match ds.layout {
        StorageLayout::Contiguous => {
            let element_size = ds.datatype.element_size().unwrap_or(0);
            let n_bytes = ds.shape.num_elements() * element_size;
            let data_addr = ds.data_address.ok_or_else(|| Error::InvalidFormat {
                message: String::from("NWB: contiguous dataset has no data address"),
            })?;
            let mut buf = vec![0u8; n_bytes];
            file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)?;
            buf
        }
        StorageLayout::Chunked => file.read_chunked_dataset_all_bytes(addr)?,
        StorageLayout::Compact => {
            return Err(Error::UnsupportedFeature {
                feature: String::from("NWB: compact dataset layout is not supported"),
            });
        }
        StorageLayout::Virtual => {
            return Err(Error::UnsupportedFeature {
                feature: String::from("NWB: virtual dataset layout is not supported"),
            });
        }
    };

    decode_raw_as_f64(&raw, &ds.datatype)
}

/// Read a scalar numeric dataset and return its single `f64` value.
///
/// Calls [`read_f64_dataset`] and extracts the first element.
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] — when the dataset is empty.
/// - Propagates all errors from [`read_f64_dataset`].
#[cfg(feature = "alloc")]
pub fn read_scalar_f64_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<f64> {
    let vals = read_f64_dataset(file, addr)?;
    vals.into_iter().next().ok_or_else(|| Error::InvalidFormat {
        message: String::from("NWB: scalar dataset is empty"),
    })
}

/// Interpret raw bytes as `Vec<f64>` according to `dtype`.
///
/// ## Supported datatypes
///
/// - `Float { bits: 64 }` — direct reinterpretation (both byte orders).
/// - `Float { bits: 32 }` — widening cast (both byte orders).
/// - `Integer { bits: 8|16|32|64, signed: false|true }` — element-wise
///   `as f64` cast after byte-order decode.
///
/// All other datatypes return [`Error::UnsupportedFeature`].
#[cfg(feature = "alloc")]
fn decode_raw_as_f64(raw: &[u8], dtype: &Datatype) -> Result<Vec<f64>> {
    match dtype {
        Datatype::Float { bits, byte_order } if bits.get() == 64 => {
            let vals: Vec<f64> = raw
                .chunks_exact(8)
                .map(|c| {
                    let arr = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    match byte_order {
                        ByteOrder::LittleEndian => f64::from_le_bytes(arr),
                        ByteOrder::BigEndian => f64::from_be_bytes(arr),
                    }
                })
                .collect();
            Ok(vals)
        }
        Datatype::Float { bits, byte_order } if bits.get() == 32 => {
            let vals: Vec<f64> = raw
                .chunks_exact(4)
                .map(|c| {
                    let arr = [c[0], c[1], c[2], c[3]];
                    let v32 = match byte_order {
                        ByteOrder::LittleEndian => f32::from_le_bytes(arr),
                        ByteOrder::BigEndian => f32::from_be_bytes(arr),
                    };
                    v32 as f64
                })
                .collect();
            Ok(vals)
        }
        Datatype::Integer {
            bits,
            signed,
            byte_order,
        } => {
            let b = bits.get();
            let vals: Vec<f64> = match (b, *signed) {
                (8, false) => raw.iter().map(|&v| v as f64).collect(),
                (8, true) => raw.iter().map(|&v| (v as i8) as f64).collect(),
                (16, false) => raw
                    .chunks_exact(2)
                    .map(|c| read_u16(c, *byte_order) as f64)
                    .collect(),
                (16, true) => raw
                    .chunks_exact(2)
                    .map(|c| read_i16(c, *byte_order) as f64)
                    .collect(),
                (32, false) => raw
                    .chunks_exact(4)
                    .map(|c| read_u32(c, *byte_order) as f64)
                    .collect(),
                (32, true) => raw
                    .chunks_exact(4)
                    .map(|c| read_i32(c, *byte_order) as f64)
                    .collect(),
                (64, false) => raw
                    .chunks_exact(8)
                    .map(|c| read_u64(c, *byte_order) as f64)
                    .collect(),
                (64, true) => raw
                    .chunks_exact(8)
                    .map(|c| read_i64(c, *byte_order) as f64)
                    .collect(),
                _ => {
                    return Err(Error::UnsupportedFeature {
                        feature: format!(
                            "NWB: integer dataset element type {} bits is not supported \
                             (only 8/16/32/64)",
                            b
                        ),
                    });
                }
            };
            Ok(vals)
        }
        other => Err(Error::UnsupportedFeature {
            feature: format!(
                "NWB: dataset element type {:?} is not supported (only f32, f64, and \
                 8/16/32/64-bit integers)",
                other
            ),
        }),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use consus_core::ByteOrder;
    use core::num::NonZeroUsize;

    // ── read_string_attr ──────────────────────────────────────────────────

    fn make_string_attr(name: &str, value: &str) -> Hdf5Attribute {
        // Build a minimal v1 attribute message in memory so we can call
        // Hdf5Attribute::parse on it, which is the authoritative decode path.
        //
        // v1 layout: version(1) | reserved(1) | name_sz(2LE) | dt_sz(2LE) |
        //            ds_sz(2LE) | name(aligned8) | datatype(aligned8) |
        //            dataspace(aligned8) | data
        //
        // We use a FixedString datatype so decode_value() returns AttributeValue::String.

        use consus_hdf5::address::ParseContext;

        let ctx = ParseContext::new(8, 8);

        fn align8(n: usize) -> usize {
            (n + 7) & !7
        }

        let name_bytes_raw: Vec<u8> = {
            let mut v: Vec<u8> = name.as_bytes().to_vec();
            v.push(0u8);
            v
        };
        let name_sz = name_bytes_raw.len();
        let name_padded_sz = align8(name_sz);

        let str_len = value.len().max(1);
        let dt_bytes: Vec<u8> = {
            let class_version: u8 = (1 << 4) | 3;
            let class_flags: u8 = 0;
            let reserved: [u8; 2] = [0, 0];
            let size_le = (str_len as u32).to_le_bytes();
            let class_specific: [u8; 4] = [0, 0, 0, 0];
            let mut v = vec![class_version, class_flags];
            v.extend_from_slice(&reserved);
            v.extend_from_slice(&size_le);
            v.extend_from_slice(&class_specific);
            v
        };
        let dt_sz = dt_bytes.len();
        let dt_padded_sz = align8(dt_sz);

        let ds_bytes: Vec<u8> = {
            let version: u8 = 1;
            let rank: u8 = 0;
            let flags: u8 = 0;
            vec![version, rank, flags, 0, 0, 0, 0, 0]
        };
        let ds_sz = ds_bytes.len();
        let ds_padded_sz = align8(ds_sz);

        let data_bytes: Vec<u8> = {
            let mut v = value.as_bytes().to_vec();
            while v.len() < str_len {
                v.push(0u8);
            }
            v
        };

        let mut msg: Vec<u8> = Vec::new();
        msg.push(1u8);
        msg.push(0u8);
        msg.extend_from_slice(&(name_sz as u16).to_le_bytes());
        msg.extend_from_slice(&(dt_sz as u16).to_le_bytes());
        msg.extend_from_slice(&(ds_sz as u16).to_le_bytes());

        let mut name_section = name_bytes_raw.clone();
        while name_section.len() < name_padded_sz {
            name_section.push(0u8);
        }
        msg.extend_from_slice(&name_section);

        let mut dt_section = dt_bytes.clone();
        while dt_section.len() < dt_padded_sz {
            dt_section.push(0u8);
        }
        msg.extend_from_slice(&dt_section);

        let mut ds_section = ds_bytes.clone();
        while ds_section.len() < ds_padded_sz {
            ds_section.push(0u8);
        }
        msg.extend_from_slice(&ds_section);

        msg.extend_from_slice(&data_bytes);

        Hdf5Attribute::parse(&msg, &ctx).expect("test attribute must parse")
    }

    /// Construct an `Hdf5Attribute` carrying a scalar f64 value directly,
    /// bypassing the v1 message builder. Used only in unit tests that need
    /// to call `decode_value()` on a numerically-typed attribute.
    fn make_f64_attr(name: &str, value: f64) -> Hdf5Attribute {
        use consus_core::{Datatype, Shape};
        Hdf5Attribute {
            name: alloc::string::String::from(name),
            datatype: Datatype::Float {
                bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::LittleEndian,
            },
            shape: Shape::scalar(),
            raw_data: value.to_le_bytes().to_vec(),
            name_encoding: 0,
            creation_order: None,
        }
    }

    /// Construct an `Hdf5Attribute` carrying a scalar i64 value.
    fn make_i64_attr(name: &str, value: i64) -> Hdf5Attribute {
        use consus_core::{Datatype, Shape};
        Hdf5Attribute {
            name: alloc::string::String::from(name),
            datatype: Datatype::Integer {
                bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            },
            shape: Shape::scalar(),
            raw_data: value.to_le_bytes().to_vec(),
            name_encoding: 0,
            creation_order: None,
        }
    }

    #[test]
    fn read_string_attr_finds_named_attribute() {
        let attrs = vec![
            make_string_attr("other", "skip"),
            make_string_attr("identifier", "ses-001"),
        ];
        let result = read_string_attr(&attrs, "identifier").unwrap();
        assert_eq!(result, "ses-001");
    }

    #[test]
    fn read_string_attr_returns_not_found_for_absent_name() {
        let attrs = vec![make_string_attr("nwb_version", "2.7.0")];
        let err = read_string_attr(&attrs, "missing_attr").unwrap_err();
        assert!(
            matches!(err, Error::NotFound { .. }),
            "expected NotFound, got {:?}",
            err
        );
    }

    #[test]
    fn read_string_attr_empty_list_returns_not_found() {
        let err = read_string_attr(&[], "any").unwrap_err();
        assert!(matches!(err, Error::NotFound { .. }));
    }

    #[test]
    fn read_string_attr_first_match_wins() {
        let attrs = vec![
            make_string_attr("key", "first"),
            make_string_attr("key", "second"),
        ];
        let result = read_string_attr(&attrs, "key").unwrap();
        assert_eq!(result, "first");
    }

    // ── decode_raw_as_f64 — float paths ───────────────────────────────────

    #[test]
    fn decode_f64_le_identity() {
        // Theorem: 3 × f64 LE bytes → same values.
        let values: [f64; 3] = [1.0, -2.5, 3.14159265358979];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].to_bits(), 1.0f64.to_bits());
        assert_eq!(result[1].to_bits(), (-2.5f64).to_bits());
        assert_eq!(result[2].to_bits(), 3.14159265358979f64.to_bits());
    }

    #[test]
    fn decode_f32_le_widened_to_f64() {
        // Theorem: f32 LE bytes → f64 via widening cast.
        let values: [f32; 2] = [1.0f32, -0.5f32];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0f32 as f64);
        assert_eq!(result[1], (-0.5f32) as f64);
    }

    #[test]
    fn decode_f64_be_decoded_correctly() {
        // Theorem: f64 big-endian bytes → correct f64 values.
        let value: f64 = 42.0;
        let raw = value.to_be_bytes().to_vec();
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::BigEndian,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), 42.0f64.to_bits());
    }

    #[test]
    fn decode_empty_raw_returns_empty_vec() {
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let result = decode_raw_as_f64(&[], &dt).unwrap();
        assert!(result.is_empty());
    }

    // ── decode_raw_as_f64 — integer promotion paths ───────────────────────

    #[test]
    fn decode_integer_u8_promoted_to_f64() {
        // u8 domain: [0, 255]; all values representable exactly as f64.
        let raw: Vec<u8> = vec![0u8, 1u8, 128u8, 255u8];
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 0.0f64);
        assert_eq!(result[1], 1.0f64);
        assert_eq!(result[2], 128.0f64);
        assert_eq!(result[3], 255.0f64);
    }

    #[test]
    fn decode_integer_i8_promoted_to_f64() {
        // i8 domain: [-128, 127]; all values representable exactly as f64.
        // Raw bytes: 0x00=0, 0x7F=127, 0x80=-128, 0xFF=-1.
        let raw: Vec<u8> = vec![0x00u8, 0x7Fu8, 0x80u8, 0xFFu8];
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 0.0f64);
        assert_eq!(result[1], 127.0f64);
        assert_eq!(result[2], -128.0f64);
        assert_eq!(result[3], -1.0f64);
    }

    #[test]
    fn decode_integer_u16_le_promoted_to_f64() {
        // u16 LE: all values in [0, 65535] are exactly representable as f64.
        let values: [u16; 3] = [0, 1000, 65535];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0.0f64);
        assert_eq!(result[1], 1000.0f64);
        assert_eq!(result[2], 65535.0f64);
    }

    #[test]
    fn decode_integer_i16_le_promoted_to_f64() {
        // i16 LE: all values in [-32768, 32767] are exactly representable as f64.
        let values: [i16; 4] = [-32768, -1000, 0, 32767];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], -32768.0f64);
        assert_eq!(result[1], -1000.0f64);
        assert_eq!(result[2], 0.0f64);
        assert_eq!(result[3], 32767.0f64);
    }

    #[test]
    fn decode_integer_i32_le_promoted_to_f64() {
        // i32 LE: common neural data encoding (e.g. raw ADC samples, spike counts).
        // All values in [-2^31, 2^31-1]; representable exactly as f64 up to 2^53.
        let values: [i32; 3] = [-1_000_000, 0, 1_000_000];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], -1_000_000.0f64);
        assert_eq!(result[1], 0.0f64);
        assert_eq!(result[2], 1_000_000.0f64);
    }

    #[test]
    fn decode_integer_i64_le_promoted_to_f64() {
        // i64 → f64 via `as f64`; values at i64::MIN/MAX exceed f64 mantissa
        // precision (53 bits) but the cast is well-defined: nearest IEEE 754 double.
        let values: [i64; 3] = [i64::MIN, 0, i64::MAX];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], i64::MIN as f64);
        assert_eq!(result[1], 0.0f64);
        assert_eq!(result[2], i64::MAX as f64);
    }

    #[test]
    fn decode_integer_unsupported_bit_width_returns_unsupported_feature() {
        // 128-bit integer is not in the supported set {8, 16, 32, 64}.
        let raw = [0u8; 16];
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(128).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        };
        let err = decode_raw_as_f64(&raw, &dt).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFeature { .. }),
            "expected UnsupportedFeature for 128-bit integer, got {:?}",
            err
        );
    }

    // ── read_scalar_f64_dataset (via decode path) ─────────────────────────

    #[test]
    fn read_scalar_f64_dataset_via_i16_decode_path() {
        // Verify the decode path exercised by read_scalar_f64_dataset for
        // i16 LE data: decode_raw_as_f64 produces a 1-element Vec<f64> and
        // read_scalar_f64_dataset extracts the first element.
        //
        // Analytical expectation: i16 -512 → f64 -512.0 (exact, within 2^53).
        let val: i16 = -512;
        let raw = val.to_le_bytes().to_vec();
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        let vals = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(vals.len(), 1, "single i16 element must decode to 1 f64");
        // Simulate read_scalar_f64_dataset's extraction of the first element.
        let scalar = vals.into_iter().next().unwrap();
        assert_eq!(scalar, -512.0f64);
    }

    // ── read_f64_attr ─────────────────────────────────────────────────────

    #[test]
    fn read_f64_attr_finds_float_attr_returns_correct_value() {
        // Analytical: f64 1.5 is exactly representable; bit-exact comparison valid.
        let attrs = vec![make_f64_attr("gain", 1.5f64)];
        let result = read_f64_attr(&attrs, "gain").unwrap();
        assert_eq!(result.to_bits(), 1.5f64.to_bits());
    }

    #[test]
    fn read_f64_attr_finds_int_attr_widens_to_f64() {
        // Analytical: i64 42 → f64 42.0 (exact, within 2^53 mantissa range).
        let attrs = vec![make_i64_attr("channel", 42i64)];
        let result = read_f64_attr(&attrs, "channel").unwrap();
        assert_eq!(result, 42.0f64);
    }

    #[test]
    fn read_f64_attr_skips_non_matching_names() {
        let attrs = vec![
            make_f64_attr("rate", 30_000.0f64),
            make_f64_attr("gain", 1.0f64),
        ];
        let result = read_f64_attr(&attrs, "gain").unwrap();
        assert_eq!(result.to_bits(), 1.0f64.to_bits());
    }

    #[test]
    fn read_f64_attr_returns_not_found_for_absent_name() {
        let attrs = vec![make_f64_attr("rate", 30_000.0f64)];
        let err = read_f64_attr(&attrs, "missing_attr").unwrap_err();
        assert!(
            matches!(err, Error::NotFound { .. }),
            "expected NotFound for absent attribute, got {:?}",
            err
        );
    }

    #[test]
    fn read_f64_attr_returns_not_found_for_empty_list() {
        let err = read_f64_attr(&[], "any").unwrap_err();
        assert!(matches!(err, Error::NotFound { .. }));
    }
}
