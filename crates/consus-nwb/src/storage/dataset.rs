#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec, vec::Vec};

use consus_core::{ByteOrder, Datatype, Error, Result};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use consus_io::ReadAt;

use super::primitives::{read_i16, read_i32, read_i64, read_u16, read_u32, read_u64};

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

/// Read an integer dataset and return its values as `Vec<u64>`.
///
/// Supports the following source datatypes:
///
/// | Source type        | Conversion                                     |
/// |--------------------|------------------------------------------------|
/// | `u8`               | zero-extend to `u64`                           |
/// | `i8`               | bit-pattern cast (sign-extended then `as u64`) |
/// | `u16` (any order)  | byte-order decode then `as u64`                |
/// | `i16` (any order)  | byte-order decode then `as u64`                |
/// | `u32` (any order)  | byte-order decode then `as u64`                |
/// | `i32` (any order)  | byte-order decode then `as u64`                |
/// | `u64` (any order)  | identity (byte-order decode)                   |
/// | `i64` (any order)  | bit-pattern cast (`as u64`)                    |
///
/// Float datatypes and string datatypes return [`Error::UnsupportedFeature`].
/// Both contiguous and chunked storage layouts are supported.
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] — contiguous dataset has no data address.
/// - [`Error::UnsupportedFeature`] — compact or virtual layout, non-integer
///   datatype, or an integer bit-width other than 8, 16, 32, or 64.
/// - Propagates HDF5 I/O errors.
#[cfg(feature = "alloc")]
pub fn read_u64_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<u64>> {
    let ds = file.dataset_at(addr)?;

    let raw: Vec<u8> = match ds.layout {
        StorageLayout::Contiguous => {
            let element_size = ds.datatype.element_size().unwrap_or(0);
            let n_bytes = ds.shape.num_elements() * element_size;
            let data_addr = ds.data_address.ok_or_else(|| Error::InvalidFormat {
                message: String::from("NWB: contiguous u64 dataset has no data address"),
            })?;
            let mut buf = vec![0u8; n_bytes];
            file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)?;
            buf
        }
        StorageLayout::Chunked => file.read_chunked_dataset_all_bytes(addr)?,
        StorageLayout::Compact => {
            return Err(Error::UnsupportedFeature {
                feature: String::from("NWB: compact u64 dataset layout is not supported"),
            });
        }
        StorageLayout::Virtual => {
            return Err(Error::UnsupportedFeature {
                feature: String::from("NWB: virtual u64 dataset layout is not supported"),
            });
        }
    };

    decode_raw_as_u64(&raw, &ds.datatype)
}

/// Read a fixed-string dataset and return its elements as `Vec<String>`.
///
/// Interprets the dataset as a contiguous array of null-padded fixed-length
/// ASCII or UTF-8 strings as written by the NWB / HDMF format.
///
/// Each element is `length` bytes wide.  Trailing null bytes (`\0`) are
/// stripped; the remaining bytes are decoded as UTF-8.
///
/// For `VariableString` datasets each element is an HDF5 VL reference
/// `{sequence_length(4) | heap_address(offset_size) | object_index(4)}` that
/// is resolved against the file's global heap via
/// [`consus_hdf5::heap::resolve_vl_references`].
///
/// ## Errors
///
/// - [`Error::UnsupportedFeature`] — datatype is not `FixedString` or
///   `VariableString`, or the layout is compact or virtual (for `VariableString`
///   chunked is also unsupported).
/// - [`Error::InvalidFormat`] — contiguous dataset has no data address, a
///   VL reference is malformed, or an element contains invalid UTF-8.
/// - Propagates HDF5 I/O errors.
#[cfg(feature = "alloc")]
pub fn read_string_dataset<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<String>> {
    let ds = file.dataset_at(addr)?;

    match &ds.datatype {
        Datatype::FixedString { length, .. } => {
            let length = *length;
            if length == 0 {
                return Ok(alloc::vec![]);
            }
            let n_elements = ds.shape.num_elements();
            let n_bytes = n_elements * length;
            let raw: Vec<u8> = match ds.layout {
                StorageLayout::Contiguous => {
                    let data_addr = ds.data_address.ok_or_else(|| Error::InvalidFormat {
                        message: String::from("NWB: contiguous string dataset has no data address"),
                    })?;
                    let mut buf = vec![0u8; n_bytes];
                    file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)?;
                    buf
                }
                StorageLayout::Chunked => file.read_chunked_dataset_all_bytes(addr)?,
                StorageLayout::Compact => {
                    return Err(Error::UnsupportedFeature {
                        feature: String::from(
                            "NWB: compact FixedString dataset layout is not supported",
                        ),
                    });
                }
                StorageLayout::Virtual => {
                    return Err(Error::UnsupportedFeature {
                        feature: String::from(
                            "NWB: virtual FixedString dataset layout is not supported",
                        ),
                    });
                }
            };
            let mut strings = Vec::with_capacity(n_elements);
            for chunk in raw.chunks(length) {
                // Strip trailing null bytes (NWB uses null-padded fixed-length strings).
                let trimmed = match chunk.iter().rposition(|&b| b != 0) {
                    Some(pos) => &chunk[..=pos],
                    None => &chunk[..0],
                };
                let s = core::str::from_utf8(trimmed)
                    .map_err(|e| Error::InvalidFormat {
                        message: format!("NWB: string dataset contains invalid UTF-8: {}", e),
                    })
                    .map(String::from)?;
                strings.push(s);
            }
            Ok(strings)
        }

        Datatype::VariableString { .. } => {
            // Each element is an HDF5 VL reference:
            // { sequence_length(4 LE) | heap_address(offset_size) | object_index(4 LE) }
            // resolve_vl_references resolves all references against the global heap
            // and returns one Vec<u8> per element.
            let n_elements = ds.shape.num_elements();
            if n_elements == 0 {
                return Ok(alloc::vec![]);
            }
            let ctx = file.context();
            let ref_size = 4 + ctx.offset_bytes() + 4;
            let n_bytes = n_elements * ref_size;
            let raw: Vec<u8> = match ds.layout {
                StorageLayout::Contiguous => {
                    let data_addr = ds.data_address.ok_or_else(|| Error::InvalidFormat {
                        message: String::from(
                            "NWB: contiguous variable-length string dataset has no data address",
                        ),
                    })?;
                    let mut buf = vec![0u8; n_bytes];
                    file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)?;
                    buf
                }
                StorageLayout::Chunked => {
                    return Err(Error::UnsupportedFeature {
                        feature: String::from(
                            "NWB: chunked variable-length string dataset is not supported",
                        ),
                    });
                }
                StorageLayout::Compact => {
                    return Err(Error::UnsupportedFeature {
                        feature: String::from(
                            "NWB: compact variable-length string dataset is not supported",
                        ),
                    });
                }
                StorageLayout::Virtual => {
                    return Err(Error::UnsupportedFeature {
                        feature: String::from(
                            "NWB: virtual variable-length string dataset is not supported",
                        ),
                    });
                }
            };
            let byte_vecs = consus_hdf5::heap::resolve_vl_references(file.source(), &raw, ctx)?;
            let mut strings = Vec::with_capacity(byte_vecs.len());
            for bytes in byte_vecs {
                let s = core::str::from_utf8(&bytes)
                    .map_err(|e| Error::InvalidFormat {
                        message: format!(
                            "NWB: variable-length string dataset contains invalid UTF-8: {}",
                            e
                        ),
                    })
                    .map(String::from)?;
                strings.push(s);
            }
            Ok(strings)
        }

        other => Err(Error::UnsupportedFeature {
            feature: format!(
                "NWB: expected FixedString or VariableString dataset for string read, got {:?}",
                other
            ),
        }),
    }
}

/// Read a scalar fixed-string dataset as a single `String`.
///
/// A thin wrapper over [`read_string_dataset`] for the common NWB pattern of
/// storing a single YAML or plain-text string as a scalar (rank-0)
/// `FixedString` dataset.  The dataset must contain exactly one element
/// (scalar shape has `num_elements() == 1` per the empty-product convention).
///
/// ## Errors
///
/// - Propagates all errors from [`read_string_dataset`].
/// - [`Error::InvalidFormat`] when the dataset contains no elements
///   (length == 0 `FixedString` type).
#[cfg(feature = "alloc")]
pub fn read_scalar_string_dataset<R: ReadAt + Sync>(
    file: &Hdf5File<R>,
    addr: u64,
) -> Result<String> {
    let strings = read_string_dataset(file, addr)?;
    strings
        .into_iter()
        .next()
        .ok_or_else(|| Error::InvalidFormat {
            message: String::from(
                "NWB: scalar string dataset contains no elements (length-0 FixedString)",
            ),
        })
}

/// Interpret raw bytes as `Vec<u64>` according to `dtype`.
///
/// ## Supported datatypes
///
/// - `Integer { bits: 8|16|32|64, signed: false|true }` (both byte orders)
///   — cast to `u64` after byte-order decode; signed values cast as wrapping bit patterns.
///
/// All other datatypes return [`Error::UnsupportedFeature`].
#[cfg(feature = "alloc")]
fn decode_raw_as_u64(raw: &[u8], dtype: &Datatype) -> Result<Vec<u64>> {
    match dtype {
        Datatype::Integer {
            bits,
            signed,
            byte_order,
        } => {
            let bo = *byte_order;
            let vals: Vec<u64> = match (bits.get(), *signed) {
                (8, false) => raw.iter().map(|&v| v as u64).collect(),
                (8, true) => raw.iter().map(|&v| (v as i8) as u64).collect(),
                (16, false) => raw
                    .chunks_exact(2)
                    .map(|c| read_u16(c, bo) as u64)
                    .collect(),
                (16, true) => raw
                    .chunks_exact(2)
                    .map(|c| read_i16(c, bo) as u64)
                    .collect(),
                (32, false) => raw
                    .chunks_exact(4)
                    .map(|c| read_u32(c, bo) as u64)
                    .collect(),
                (32, true) => raw
                    .chunks_exact(4)
                    .map(|c| read_i32(c, bo) as u64)
                    .collect(),
                (64, false) => raw.chunks_exact(8).map(|c| read_u64(c, bo)).collect(),
                (64, true) => raw
                    .chunks_exact(8)
                    .map(|c| read_i64(c, bo) as u64)
                    .collect(),
                (b, _) => {
                    return Err(Error::UnsupportedFeature {
                        feature: format!(
                            "NWB: integer dataset element type {} bits is not supported \
                             for u64 read (only 8/16/32/64)",
                            b
                        ),
                    });
                }
            };
            Ok(vals)
        }
        other => Err(Error::UnsupportedFeature {
            feature: format!(
                "NWB: expected integer datatype for u64 dataset read, got {:?}",
                other
            ),
        }),
    }
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use consus_core::ByteOrder;
    use core::num::NonZeroUsize;

    // ── decode_raw_as_f64 — float paths ───────────────────────────────────

    #[test]
    fn decode_f64_le_identity() {
        // Theorem: 3 × f64 LE bytes → same values.
        let values: [f64; 3] = [1.0, -2.5, core::f64::consts::PI];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let result = decode_raw_as_f64(&raw, &dt).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].to_bits(), 1.0f64.to_bits());
        assert_eq!(result[1].to_bits(), (-2.5f64).to_bits());
        assert_eq!(result[2].to_bits(), core::f64::consts::PI.to_bits());
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
    // ── read_u64_dataset ──────────────────────────────────────────────────

    #[test]
    fn read_u64_dataset_u32_le_values_widened_to_u64() {
        // Analytical: u32 zero-extends to u64; all values exactly representable.
        // Input: [1u32, 100u32, 0xFFFF_FFFFu32] LE → [1u64, 100u64, 0xFFFF_FFFFu64].
        use consus_core::{Datatype, Shape};
        use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
        use consus_hdf5::file::Hdf5File;
        use consus_io::MemCursor;

        let values: [u32; 3] = [1, 100, 0xFFFF_FFFF];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        };
        let shape = Shape::fixed(&[3]);

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let addr = builder
            .add_dataset("ds", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let result = read_u64_dataset(&file, addr).unwrap();
        assert_eq!(result, vec![1u64, 100u64, 0xFFFF_FFFFu64]);
    }

    #[test]
    fn read_u64_dataset_u64_le_identity() {
        // Analytical: u64 LE bytes → same u64 values (identity decode).
        use consus_core::{Datatype, Shape};
        use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
        use consus_hdf5::file::Hdf5File;
        use consus_io::MemCursor;

        let values: [u64; 2] = [42, 9999];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        };
        let shape = Shape::fixed(&[2]);

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let addr = builder
            .add_dataset("ds", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let result = read_u64_dataset(&file, addr).unwrap();
        assert_eq!(result, vec![42u64, 9999u64]);
    }

    #[test]
    fn read_u64_dataset_i32_signed_bit_pattern() {
        // Analytical: i32 -1 → i64 -1 (sign-extended) → u64 0xFFFF_FFFF_FFFF_FFFF (bit-cast).
        // Raw bytes for -1i32 LE: [0xFF, 0xFF, 0xFF, 0xFF].
        use consus_core::{Datatype, Shape};
        use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
        use consus_hdf5::file::Hdf5File;
        use consus_io::MemCursor;

        let value: i32 = -1;
        let raw = value.to_le_bytes().to_vec();
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        let shape = Shape::fixed(&[1]);

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let addr = builder
            .add_dataset("ds", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let result = read_u64_dataset(&file, addr).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0xFFFF_FFFF_FFFF_FFFFu64);
    }

    #[test]
    fn read_u64_dataset_float_type_returns_unsupported_feature() {
        // Analytical: Float datatype is rejected by decode_raw_as_u64.
        use consus_core::{Datatype, Shape};
        use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
        use consus_hdf5::file::Hdf5File;
        use consus_io::MemCursor;

        let value: f64 = 1.0;
        let raw = value.to_le_bytes().to_vec();
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let shape = Shape::fixed(&[1]);

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let addr = builder
            .add_dataset("ds", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let err = read_u64_dataset(&file, addr).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFeature { .. }),
            "expected UnsupportedFeature for Float dataset, got {:?}",
            err
        );
    }

    // ── read_string_dataset ───────────────────────────────────────────────

    #[test]
    fn read_string_dataset_fixed_ascii_returns_correct_strings() {
        // Analytical: 2 × 5-byte LE fixed-string elements → ["hello", "world"].
        // Raw: b"helloworld" (10 bytes, no null padding needed since each string
        // fills exactly its 5-byte slot).
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
        use consus_hdf5::file::Hdf5File;
        use consus_io::MemCursor;

        let raw = b"helloworld".to_vec();
        let dt = Datatype::FixedString {
            length: 5,
            encoding: StringEncoding::Ascii,
        };
        let shape = Shape::fixed(&[2]);

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let addr = builder
            .add_dataset("ds", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let result = read_string_dataset(&file, addr).unwrap();
        assert_eq!(result, vec!["hello".to_owned(), "world".to_owned()]);
    }

    #[test]
    fn read_string_dataset_null_padded_strips_nulls() {
        // Analytical: NWB uses null-padded fixed-length strings.
        // "CA1\0\0\0\0\0" (8 bytes) → "CA1"; "DG\0\0\0\0\0\0" (8 bytes) → "DG".
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
        use consus_hdf5::file::Hdf5File;
        use consus_io::MemCursor;

        let mut raw = Vec::new();
        raw.extend_from_slice(b"CA1\0\0\0\0\0");
        raw.extend_from_slice(b"DG\0\0\0\0\0\0");
        let dt = Datatype::FixedString {
            length: 8,
            encoding: StringEncoding::Ascii,
        };
        let shape = Shape::fixed(&[2]);

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let addr = builder
            .add_dataset("ds", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let result = read_string_dataset(&file, addr).unwrap();
        assert_eq!(result, vec!["CA1".to_owned(), "DG".to_owned()]);
    }

    #[test]
    fn read_string_dataset_all_null_element_returns_empty_string() {
        // Analytical: an element consisting entirely of null bytes → empty string after strip.
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
        use consus_hdf5::file::Hdf5File;
        use consus_io::MemCursor;

        let raw = vec![0u8, 0u8, 0u8, 0u8];
        let dt = Datatype::FixedString {
            length: 4,
            encoding: StringEncoding::Ascii,
        };
        let shape = Shape::fixed(&[1]);

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let addr = builder
            .add_dataset("ds", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let result = read_string_dataset(&file, addr).unwrap();
        assert_eq!(result, vec![String::new()]);
    }

    #[test]
    fn read_string_dataset_wrong_datatype_returns_unsupported_feature() {
        // Analytical: Float datatype is rejected; only FixedString and VariableString are accepted.
        use consus_core::{Datatype, Shape};
        use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
        use consus_hdf5::file::Hdf5File;
        use consus_io::MemCursor;

        let value: f64 = 1.0;
        let raw = value.to_le_bytes().to_vec();
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let shape = Shape::fixed(&[1]);

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let addr = builder
            .add_dataset("ds", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let err = read_string_dataset(&file, addr).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFeature { .. }),
            "expected UnsupportedFeature for Float dataset, got {:?}",
            err
        );
    }
}
