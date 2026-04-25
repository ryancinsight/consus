//! Dimension scale detection and DIMENSION_LIST parsing from HDF5 attributes.
//!
//! ## Spec: netCDF-4 / HDF5 Dimension Scales (HDF5 Dimension Scales Specification 1.2)
//!
//! A dataset is a netCDF-4 dimension scale iff it has a scalar or fixed-string
//! attribute named `CLASS` whose decoded value equals `"DIMENSION_SCALE"`.
//!
//! The `NAME` attribute, when present, provides the user-facing dimension name;
//! when absent, the dataset name is used as a fallback.
//!
//! The `DIMENSION_LIST` attribute, when present on a variable dataset, stores
//! one object reference per variable axis. Each reference points to the HDF5
//! dataset acting as the dimension scale for that axis.
//!
//! ## Invariants
//!
//! - `is_dimension_scale` returns `true` only when `CLASS == "DIMENSION_SCALE"`.
//! - Attribute decode errors are treated as "not a dimension scale" (safe default).
//! - `dimension_name_from_attrs` returns a non-empty string for any non-empty fallback.
//! - `dimension_list_addresses` returns object-header addresses in axis order.
//! - `resolve_dimension_names_from_list` returns names in the same order as the
//!   decoded object references and only succeeds when every reference resolves.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::AttributeValue;
use consus_hdf5::attribute::{Hdf5Attribute, decode_attribute_value};

use crate::conventions::{DIMENSION_SCALE_CLASS, DIMENSION_SCALE_VALUE};

/// Returns `true` if `attrs` indicate this dataset is a netCDF-4 dimension scale.
///
/// ## Spec
///
/// A dimension scale carries a scalar `CLASS` attribute whose decoded string
/// value, after trimming ASCII whitespace, equals `"DIMENSION_SCALE"`.
///
/// ## Invariants
///
/// - Returns `false` on any attribute decode failure.
/// - Matching is case-sensitive and trim-normalised.
#[cfg(feature = "alloc")]
#[must_use]
pub fn is_dimension_scale(attrs: &[Hdf5Attribute]) -> bool {
    for attr in attrs {
        if attr.name == DIMENSION_SCALE_CLASS {
            if let Ok(val) = decode_attribute_value(&attr.raw_data, &attr.datatype, &attr.shape) {
                if let AttributeValue::String(s) = val {
                    if s.trim() == DIMENSION_SCALE_VALUE {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Extract the dimension name from a `NAME` attribute, falling back to `fallback`.
///
/// ## Spec
///
/// The `NAME` attribute on a dimension scale holds the user-facing dimension
/// identifier.  If the attribute is absent, decoding fails, or its value is
/// empty after trimming, the caller-supplied `fallback` string is returned.
///
/// ## Invariants
///
/// - The returned string is non-empty whenever `fallback` is non-empty.
#[cfg(feature = "alloc")]
#[must_use]
pub fn dimension_name_from_attrs(attrs: &[Hdf5Attribute], fallback: &str) -> String {
    for attr in attrs {
        if attr.name == "NAME" {
            if let Ok(val) = decode_attribute_value(&attr.raw_data, &attr.datatype, &attr.shape) {
                if let AttributeValue::String(s) = val {
                    let trimmed = s.trim().to_string();
                    if !trimmed.is_empty() {
                        return trimmed;
                    }
                }
            }
        }
    }
    fallback.to_string()
}

/// Decode the `DIMENSION_LIST` attribute into object-header addresses.
///
/// ## Spec
///
/// The current bridge accepts the common object-reference encoding used by
/// HDF5-backed netCDF files: one 8-byte little-endian object reference per
/// variable axis. Reference-typed attributes decode to raw bytes in the HDF5
/// attribute layer, so this function interprets those bytes directly.
///
/// ## Invariants
///
/// - Returns `None` when `DIMENSION_LIST` is absent.
/// - Returns `Some(vec![])` only when the attribute is present but empty.
/// - Returns `None` when the raw byte payload length is not a multiple of 8.
#[cfg(feature = "alloc")]
#[must_use]
pub fn dimension_list_addresses(attrs: &[Hdf5Attribute]) -> Option<Vec<u64>> {
    for attr in attrs {
        if attr.name == "DIMENSION_LIST" {
            let value = decode_attribute_value(&attr.raw_data, &attr.datatype, &attr.shape).ok()?;
            let raw = match value {
                AttributeValue::Bytes(bytes) => bytes,
                _ => return None,
            };

            if raw.len() % 8 != 0 {
                return None;
            }

            let refs = raw
                .chunks_exact(8)
                .map(|chunk| {
                    u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();

            return Some(refs);
        }
    }

    None
}

/// Resolve variable dimension names from a decoded `DIMENSION_LIST`.
///
/// ## Spec
///
/// `dimension_scale_names` is the authoritative mapping from HDF5 object-header
/// address to the user-facing dimension-scale name discovered during group
/// traversal. This function preserves the axis order encoded in
/// `DIMENSION_LIST`.
///
/// ## Invariants
///
/// - Returns `Some(names)` only when every referenced address resolves.
/// - Returns `None` when any address is missing from the mapping.
#[cfg(feature = "alloc")]
#[must_use]
pub fn resolve_dimension_names_from_list(
    attrs: &[Hdf5Attribute],
    dimension_scale_names: &[(u64, String)],
) -> Option<Vec<String>> {
    let addresses = dimension_list_addresses(attrs)?;
    let mut names = Vec::with_capacity(addresses.len());

    for address in addresses {
        let name = dimension_scale_names
            .iter()
            .find(|(candidate, _)| *candidate == address)
            .map(|(_, name)| name.clone())?;
        names.push(name);
    }

    Some(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use consus_core::{Datatype, ReferenceType, Shape, StringEncoding};

    fn make_fixed_string_attr(name: &str, value: &str) -> Hdf5Attribute {
        let length = value.len();
        Hdf5Attribute {
            name: name.to_string(),
            datatype: Datatype::FixedString {
                length,
                encoding: StringEncoding::Ascii,
            },
            shape: Shape::scalar(),
            raw_data: value.as_bytes().to_vec(),
            name_encoding: 0,
            creation_order: None,
        }
    }

    fn make_dimension_list_attr(addresses: &[u64]) -> Hdf5Attribute {
        let raw_data: Vec<u8> = addresses
            .iter()
            .flat_map(|address| address.to_le_bytes())
            .collect();

        Hdf5Attribute {
            name: "DIMENSION_LIST".to_string(),
            datatype: Datatype::Reference(ReferenceType::Object),
            shape: Shape::fixed(&[addresses.len()]),
            raw_data,
            name_encoding: 0,
            creation_order: None,
        }
    }

    #[test]
    fn is_dimension_scale_true_for_class_attr() {
        let attrs = vec![make_fixed_string_attr("CLASS", "DIMENSION_SCALE")];
        assert!(is_dimension_scale(&attrs));
    }

    #[test]
    fn is_dimension_scale_false_for_wrong_value() {
        let attrs = vec![make_fixed_string_attr("CLASS", "OTHER_CLASS")];
        assert!(!is_dimension_scale(&attrs));
    }

    #[test]
    fn is_dimension_scale_false_when_no_class_attr() {
        let attrs = vec![make_fixed_string_attr("UNITS", "meters")];
        assert!(!is_dimension_scale(&attrs));
    }

    #[test]
    fn is_dimension_scale_false_for_empty_attrs() {
        assert!(!is_dimension_scale(&[]));
    }

    #[test]
    fn dimension_name_from_attrs_uses_name_attr() {
        let attrs = vec![make_fixed_string_attr("NAME", "latitude")];
        assert_eq!(dimension_name_from_attrs(&attrs, "fallback"), "latitude");
    }

    #[test]
    fn dimension_name_from_attrs_falls_back_when_absent() {
        let attrs = vec![make_fixed_string_attr("CLASS", "DIMENSION_SCALE")];
        assert_eq!(dimension_name_from_attrs(&attrs, "time"), "time");
    }

    #[test]
    fn is_dimension_scale_accepts_whitespace_padded_value() {
        // Some HDF5 writers pad strings with spaces; trim() must handle them.
        let attrs = vec![make_fixed_string_attr("CLASS", "DIMENSION_SCALE   ")];
        assert!(is_dimension_scale(&attrs));
    }

    #[test]
    fn dimension_list_addresses_decode_object_references_in_axis_order() {
        let attrs = vec![make_dimension_list_attr(&[0x10, 0x20, 0x30])];
        assert_eq!(
            dimension_list_addresses(&attrs),
            Some(vec![0x10, 0x20, 0x30])
        );
    }

    #[test]
    fn dimension_list_addresses_reject_non_multiple_of_eight_payload() {
        let attrs = vec![Hdf5Attribute {
            name: "DIMENSION_LIST".to_string(),
            datatype: Datatype::Reference(ReferenceType::Object),
            shape: Shape::fixed(&[1]),
            raw_data: vec![1, 2, 3],
            name_encoding: 0,
            creation_order: None,
        }];
        assert_eq!(dimension_list_addresses(&attrs), None);
    }

    #[test]
    fn resolve_dimension_names_from_list_uses_dimension_scale_mapping() {
        let attrs = vec![make_dimension_list_attr(&[0x200, 0x100])];
        let mapping = vec![(0x100, "x".to_string()), (0x200, "time".to_string())];

        assert_eq!(
            resolve_dimension_names_from_list(&attrs, &mapping),
            Some(vec!["time".to_string(), "x".to_string()])
        );
    }

    #[test]
    fn resolve_dimension_names_from_list_returns_none_for_missing_reference() {
        let attrs = vec![make_dimension_list_attr(&[0x200, 0x999])];
        let mapping = vec![(0x100, "x".to_string()), (0x200, "time".to_string())];

        assert_eq!(resolve_dimension_names_from_list(&attrs, &mapping), None);
    }
}
