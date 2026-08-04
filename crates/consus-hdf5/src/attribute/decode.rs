#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use byteorder::{BigEndian, ByteOrder, LittleEndian};

#[cfg(feature = "alloc")]
use consus_core::Shape;

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
            } else if is_scalar {
                Ok(AttributeValue::Uint(read_uint_le(raw, sz, *byte_order)?))
            } else {
                let vals: Vec<u64> = (0..total_elements)
                    .map(|i| read_uint_le(&raw[i * sz..], sz, *byte_order))
                    .collect::<consus_core::Result<_>>()?;
                Ok(AttributeValue::UintArray(vals))
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

        _ => Ok(AttributeValue::Bytes(Vec::from(raw))),
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
        match decode_attribute_value(&raw, &dtype, &shape).unwrap() {
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
        let raw = core::f32::consts::PI.to_bits().to_le_bytes().to_vec();
        match decode_attribute_value(&raw, &dtype, &shape).unwrap() {
            AttributeValue::Float(v) => {
                assert_eq!(v, f64::from(core::f32::consts::PI));
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
        match decode_attribute_value(&raw, &dtype, &shape).unwrap() {
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
        match decode_attribute_value(&raw, &dtype, &shape).unwrap() {
            AttributeValue::String(s) => assert_eq!(s, "hello"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    mod proptest_harnesses {
        use super::*;
        use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
        use proptest::prelude::*;

        proptest! {
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
