//! Byte-level scalar decoding for the Consus storage model.
//!
//! ## Specification
//!
//! Decoding converts a raw byte buffer into typed scalars according to a
//! [`Datatype`] (or an equivalent `(element size, signedness, floatness,
//! byte order)` parameter pack). Every format backend maps its native type
//! system onto the canonical [`Datatype`] (see [`super::types::datatype`]), so this
//! module is the single place where byte representation becomes numeric
//! values.
//!
//! ### Decoding Invariant
//!
//! For any fixed-size numeric datatype `D` and any raw buffer `B` whose
//! length is a multiple of `D.element_size()`:
//!   `decode_to_f64(B, D) == decode_to_f64(B, canonicalize(D))`
//!
//! i.e. the same logical type decodes identically regardless of source
//! format (HDF5, Zarr, netCDF-4, Parquet, MetaImage, NRRD).
//!
//! ### Supported Conversions
//!
//! - `Float { bits ∈ {32, 64} }` → `f64` (32-bit widened exactly).
//! - `Integer { bits ∈ {8, 16, 32, 64} }` → `f64` (signed/unsigned per
//!   datatype; `u64` values above `2^53` lose precision by construction).
//! - `Boolean` → `f64` (0.0 / 1.0).
//!
//! Unsupported or variable-length datatypes fail closed with
//! [`Error::UnsupportedFeature`] / [`Error::InvalidFormat`]; a buffer whose
//! length is not a multiple of the element size is a
//! [`Error::InvalidFormat`].

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use super::Error;
#[cfg(feature = "alloc")]
use super::types::datatype::ByteOrder;
#[cfg(feature = "alloc")]
use super::types::datatype::Datatype;

mod endian;

pub use endian::{EndianScalar, read_integer};

/// Decode a raw buffer of fixed-size numeric elements into `Vec<f64>`.
///
/// Supports `Float { 32, 64 }` (32-bit widened exactly), `Integer
/// { 8, 16, 32, 64 }` signed or unsigned, and `Boolean`.
///
/// # Errors
///
/// - The datatype is variable-length or otherwise unsupported →
///   [`Error::UnsupportedFeature`].
/// - The buffer length is not a multiple of the element size →
///   [`Error::InvalidFormat`].
#[cfg(feature = "alloc")]
pub fn decode_to_f64(raw: &[u8], dtype: &Datatype) -> Result<Vec<f64>, Error> {
    if let Some(size) = dtype.element_size() {
        // `is_multiple_of` is stable only since 1.87; the MSRV is 1.85.
        if raw.len() % size != 0 {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "decode_to_f64: buffer length {} is not a multiple of element size {}",
                    raw.len(),
                    size
                ),
            });
        }
    }
    match dtype {
        Datatype::Float { bits, byte_order } => match bits.get() {
            64 => Ok(raw
                .chunks_exact(8)
                .map(|c| {
                    let arr = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    match byte_order {
                        ByteOrder::LittleEndian => f64::from_le_bytes(arr),
                        ByteOrder::BigEndian => f64::from_be_bytes(arr),
                    }
                })
                .collect()),
            32 => Ok(raw
                .chunks_exact(4)
                .map(|c| {
                    let arr = [c[0], c[1], c[2], c[3]];
                    let v32 = match byte_order {
                        ByteOrder::LittleEndian => f32::from_le_bytes(arr),
                        ByteOrder::BigEndian => f32::from_be_bytes(arr),
                    };
                    v32 as f64
                })
                .collect()),
            _ => Err(Error::UnsupportedFeature {
                feature: alloc::format!("decode_to_f64: {}-bit float", bits),
            }),
        },
        Datatype::Integer {
            bits,
            signed,
            byte_order,
        } => {
            let bo = *byte_order;
            match (bits.get(), *signed) {
                (8, false) => Ok(raw.iter().map(|&v| v as f64).collect()),
                (8, true) => Ok(raw.iter().map(|&v| (v as i8) as f64).collect()),
                (16, false) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| {
                        read_integer::<u16>(c, bo).expect("chunks_exact supplies a scalar") as f64
                    })
                    .collect()),
                (16, true) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| {
                        read_integer::<i16>(c, bo).expect("chunks_exact supplies a scalar") as f64
                    })
                    .collect()),
                (32, false) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| {
                        read_integer::<u32>(c, bo).expect("chunks_exact supplies a scalar") as f64
                    })
                    .collect()),
                (32, true) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| {
                        read_integer::<i32>(c, bo).expect("chunks_exact supplies a scalar") as f64
                    })
                    .collect()),
                (64, false) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        read_integer::<u64>(c, bo).expect("chunks_exact supplies a scalar") as f64
                    })
                    .collect()),
                (64, true) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        read_integer::<i64>(c, bo).expect("chunks_exact supplies a scalar") as f64
                    })
                    .collect()),
                (b, _) => Err(Error::UnsupportedFeature {
                    feature: alloc::format!("decode_to_f64: {b}-bit integer"),
                }),
            }
        }
        Datatype::Boolean => Ok(raw
            .iter()
            .map(|&v| if v != 0 { 1.0 } else { 0.0 })
            .collect()),
        other => Err(Error::UnsupportedFeature {
            feature: alloc::format!("decode_to_f64: unsupported datatype {other:?}"),
        }),
    }
}

/// Decode a raw buffer into `Vec<f64>` from the primitive parameter pack used
/// by format readers that translate type-name strings into a single
/// `(element size, signedness, floatness, byte order)` tuple.
///
/// This is the `ritk-codecs`-style entry point: `elem_size` is 1, 2, 4, or
/// 8 bytes; `signed` applies to the integer path; `is_float` selects the
/// float path. Float `elem_size` must be 4 or 8.
///
/// # Errors
///
/// - `is_float` and `elem_size` is not 4 or 8.
/// - `!is_float` and `elem_size` is not 1, 2, 4, or 8.
/// - The buffer length is not a multiple of `elem_size` →
///   [`Error::InvalidFormat`].
#[cfg(feature = "alloc")]
pub fn decode_bytes_to_f64(
    bytes: &[u8],
    elem_size: usize,
    signed: bool,
    is_float: bool,
    byte_order: ByteOrder,
) -> Result<Vec<f64>, Error> {
    // Reject ragged buffers up front instead of letting `chunks_exact` drop
    // the trailing bytes, matching `decode_to_f64` and the module invariant
    // (a non-multiple length is `InvalidFormat`, not silent truncation).
    // `elem_size == 0` cannot divide and falls through to
    // `UnsupportedFeature` below, so guard the modulo.
    if elem_size != 0 && bytes.len() % elem_size != 0 {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "decode_bytes_to_f64: buffer length {} is not a multiple of element size {}",
                bytes.len(),
                elem_size
            ),
        });
    }
    if is_float {
        match elem_size {
            4 => Ok(bytes
                .chunks_exact(4)
                .map(|c| {
                    let arr = [c[0], c[1], c[2], c[3]];
                    match byte_order {
                        ByteOrder::LittleEndian => f32::from_le_bytes(arr) as f64,
                        ByteOrder::BigEndian => f32::from_be_bytes(arr) as f64,
                    }
                })
                .collect()),
            8 => Ok(bytes
                .chunks_exact(8)
                .map(|c| {
                    let arr = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    match byte_order {
                        ByteOrder::LittleEndian => f64::from_le_bytes(arr),
                        ByteOrder::BigEndian => f64::from_be_bytes(arr),
                    }
                })
                .collect()),
            other => Err(Error::UnsupportedFeature {
                feature: alloc::format!("decode_bytes_to_f64: unsupported float size {other}"),
            }),
        }
    } else {
        let bo = byte_order;
        Ok(match (elem_size, signed) {
            (1, false) => bytes.iter().map(|&v| v as f64).collect(),
            (1, true) => bytes.iter().map(|&v| (v as i8) as f64).collect(),
            (2, false) => bytes
                .chunks_exact(2)
                .map(|c| read_integer::<u16>(c, bo).expect("chunks_exact supplies a scalar") as f64)
                .collect(),
            (2, true) => bytes
                .chunks_exact(2)
                .map(|c| read_integer::<i16>(c, bo).expect("chunks_exact supplies a scalar") as f64)
                .collect(),
            (4, false) => bytes
                .chunks_exact(4)
                .map(|c| read_integer::<u32>(c, bo).expect("chunks_exact supplies a scalar") as f64)
                .collect(),
            (4, true) => bytes
                .chunks_exact(4)
                .map(|c| read_integer::<i32>(c, bo).expect("chunks_exact supplies a scalar") as f64)
                .collect(),
            (8, false) => bytes
                .chunks_exact(8)
                .map(|c| read_integer::<u64>(c, bo).expect("chunks_exact supplies a scalar") as f64)
                .collect(),
            (8, true) => bytes
                .chunks_exact(8)
                .map(|c| read_integer::<i64>(c, bo).expect("chunks_exact supplies a scalar") as f64)
                .collect(),
            (other, _) => {
                return Err(Error::UnsupportedFeature {
                    feature: alloc::format!(
                        "decode_bytes_to_f64: unsupported integer size {other}"
                    ),
                });
            }
        })
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::super::Error;
    use super::super::types::datatype::{ByteOrder, Datatype};
    use super::{decode_bytes_to_f64, decode_to_f64};
    use core::num::NonZeroUsize;

    fn le_int(bits: usize, signed: bool) -> Datatype {
        Datatype::Integer {
            bits: NonZeroUsize::new(bits).expect("test bit width is non-zero"),
            signed,
            byte_order: ByteOrder::LittleEndian,
        }
    }

    fn be_float(bits: usize) -> Datatype {
        Datatype::Float {
            bits: NonZeroUsize::new(bits).expect("test bit width is non-zero"),
            byte_order: ByteOrder::BigEndian,
        }
    }

    #[test]
    fn f32_le_widens_exactly() {
        let raw = 1.5f32.to_le_bytes().to_vec();
        let out = decode_to_f64(&raw, &le_float(32)).expect("valid f32 decodes");
        assert_eq!(out, vec![1.5_f64]);
    }

    #[test]
    fn f64_be_roundtrips() {
        let raw = 3.25f64.to_be_bytes().to_vec();
        let out = decode_to_f64(&raw, &be_float(64)).expect("valid f64 decodes");
        assert_eq!(out, vec![3.25_f64]);
    }

    #[test]
    fn i8_signed_preserves_negative() {
        let raw = vec![0xFFu8, 0x01];
        let out = decode_to_f64(&raw, &le_int(8, true)).expect("valid i8 decodes");
        assert_eq!(out, vec![-1.0, 1.0]);
    }

    #[test]
    fn u16_le_value() {
        let raw = 0x1234u16.to_le_bytes().to_vec();
        let out = decode_to_f64(&raw, &le_int(16, false)).expect("valid u16 decodes");
        assert_eq!(out, vec![4660.0_f64]); // 0x1234
    }

    #[test]
    fn u32_be_value() {
        let raw = 0xCAFEBABEu32.to_be_bytes().to_vec();
        let out = decode_to_f64(
            &raw,
            &Datatype::Integer {
                bits: NonZeroUsize::new(32).expect("test bit width is non-zero"),
                signed: false,
                byte_order: ByteOrder::BigEndian,
            },
        )
        .expect("valid u32 decodes");
        assert_eq!(out, vec![3_405_691_582.0_f64]); // 0xCAFEBABE
    }

    #[test]
    fn i64_le_negative() {
        let raw = (-9_223_372_036_854_775_807i64).to_le_bytes().to_vec();
        let out = decode_to_f64(&raw, &le_int(64, true)).expect("valid i64 decodes");
        assert_eq!(out, vec![-9_223_372_036_854_775_807_f64]);
    }

    #[test]
    fn boolean_maps_to_zero_one() {
        let raw = vec![0u8, 5u8, 0u8];
        let out = decode_to_f64(&raw, &Datatype::Boolean).expect("valid boolean decodes");
        assert_eq!(out, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn unsupported_float_width_fails_closed() {
        let raw = vec![0u8; 16];
        let out = decode_to_f64(&raw, &be_float(128));
        assert!(out.is_err());
    }

    #[test]
    fn non_multiple_buffer_length_fails_closed() {
        // 4 bytes of f32 data + 1 trailing byte
        let raw = vec![0u8; 5];
        let out = decode_to_f64(&raw, &le_float(32));
        assert!(out.is_err());
    }

    #[test]
    fn decode_bytes_tuple_matches_datatype_path() {
        // f32 LE via the tuple path vs the datatype path
        let raw = vec![0x00, 0x00, 0xC0, 0x3F]; // 1.5f32 LE
        let tuple = decode_bytes_to_f64(&raw, 4, true, true, ByteOrder::LittleEndian)
            .expect("valid tuple decodes");
        let dtype = decode_to_f64(
            &raw,
            &Datatype::Float {
                bits: NonZeroUsize::new(32).expect("test bit width is non-zero"),
                byte_order: ByteOrder::LittleEndian,
            },
        )
        .expect("valid datatype decodes");
        assert_eq!(tuple, dtype);
    }

    #[test]
    fn empty_buffer_decodes_to_empty_vec() {
        // The exact-size iterator contract (`chunks_exact`/`iter` report
        // `len == 0`) must produce an empty, zero-allocation result rather
        // than erroring or emitting a sentinel element.
        assert_eq!(
            decode_to_f64(&[], &le_float(64)).expect("empty f64 buffer decodes"),
            Vec::<f64>::new()
        );
        assert_eq!(
            decode_to_f64(&[], &le_int(32, false)).expect("empty u32 buffer decodes"),
            Vec::<f64>::new()
        );
        assert_eq!(
            decode_to_f64(&[], &Datatype::Boolean).expect("empty boolean buffer decodes"),
            Vec::<f64>::new()
        );
        assert_eq!(
            decode_bytes_to_f64(&[], 8, false, true, ByteOrder::LittleEndian)
                .expect("empty tuple buffer decodes"),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn single_element_buffer_decodes() {
        // Boundary: one element — the smallest non-trivial exact-size decode.
        let raw = 42u16.to_le_bytes();
        let out = decode_bytes_to_f64(&raw, 2, false, false, ByteOrder::LittleEndian)
            .expect("one u16 decodes");
        assert_eq!(out, vec![42.0_f64]);
    }

    #[test]
    fn tuple_path_rejects_non_multiple_buffer_length() {
        // 4 bytes of f32 data + 1 trailing byte must fail closed instead of
        // silently truncating the ragged tail via `chunks_exact`.
        let raw = vec![0u8; 5];
        let err = decode_bytes_to_f64(&raw, 4, false, true, ByteOrder::LittleEndian)
            .expect_err("ragged buffer must fail");
        assert!(matches!(err, Error::InvalidFormat { .. }));
        // The datatype path is the SSOT: both entries must agree.
        let dtype_err = decode_to_f64(&raw, &le_float(32)).expect_err("ragged buffer must fail");
        assert!(matches!(dtype_err, Error::InvalidFormat { .. }));

        // Even lengths are still accepted on the tuple path (u16: 3 elements).
        let ok = decode_bytes_to_f64(
            &[1, 0, 2, 0, 3, 0],
            2,
            false,
            false,
            ByteOrder::LittleEndian,
        )
        .expect("multiple-of-2 buffer decodes");
        assert_eq!(ok, vec![1.0, 2.0, 3.0]);
    }

    fn le_float(bits: usize) -> Datatype {
        Datatype::Float {
            bits: NonZeroUsize::new(bits).expect("test bit width is non-zero"),
            byte_order: ByteOrder::LittleEndian,
        }
    }
}
