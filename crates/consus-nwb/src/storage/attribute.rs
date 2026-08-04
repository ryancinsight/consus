#[cfg(feature = "alloc")]
use alloc::{format, string::String};

use consus_core::{AttributeValue, Error, Result};
use consus_hdf5::attribute::Hdf5Attribute;

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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
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
