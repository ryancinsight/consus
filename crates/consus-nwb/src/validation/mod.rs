//! NWB schema conformance and constraint checking.
//!
//! Validates NWB files against the NWB 2.x specification: required group
//! presence, required attribute values, and type-compatibility checks.
//!
//! ## Specification
//!
//! An NWB 2.x file must satisfy the following root-group constraints:
//!
//! | Attribute             | Required value | Enforcement          |
//! |-----------------------|----------------|----------------------|
//! | `neurodata_type_def`  | `"NWBFile"`    | hard error           |
//! | `nwb_version`         | any string     | hard error (absence) |
//!
//! Additional required attributes (`identifier`, `session_description`,
//! `session_start_time`) are validated by the session-metadata reader in
//! [`crate::file`]; they are not re-checked here to avoid duplicate I/O.
//!
//! ## Invariants
//!
//! - [`validate_root_attributes`] reads the root group's object header
//!   attributes exactly once and checks both constraints in a single pass.
//! - A file that passes validation is guaranteed to have
//!   `neurodata_type_def == "NWBFile"` on its root HDF5 group.
//! - Validation is read-only: no bytes are written to the source.

#[cfg(feature = "alloc")]
use alloc::string::String;

use consus_core::{AttributeValue, Error, Result};
use consus_hdf5::file::Hdf5File;
use consus_io::ReadAt;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Validate the root-group attributes of an NWB file.
///
/// Reads all attributes attached to the root HDF5 group and checks:
///
/// 1. An attribute named `neurodata_type_def` exists and decodes to the
///    string `"NWBFile"`.
/// 2. An attribute named `nwb_version` exists (its value is not inspected
///    here; version parsing is delegated to [`crate::version::detect_version`]).
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] when `neurodata_type_def` is absent, cannot be
///   decoded as a string, or its value is not `"NWBFile"`.
/// - [`Error::NotFound`] when the `nwb_version` attribute is absent.
/// - Propagates any I/O or format error from [`Hdf5File::attributes_at`].
#[cfg(feature = "alloc")]
pub fn validate_root_attributes<R: ReadAt + Sync>(file: &Hdf5File<R>) -> Result<()> {
    let root_addr = file.superblock().root_group_address;
    let attrs = file.attributes_at(root_addr)?;

    let mut found_nwb_file_type = false;
    let mut found_nwb_version = false;

    for attr in &attrs {
        if attr.name == "neurodata_type_def" {
            match attr.decode_value() {
                Ok(AttributeValue::String(ref t)) if t == "NWBFile" => {
                    found_nwb_file_type = true;
                }
                Ok(AttributeValue::String(ref t)) => {
                    return Err(Error::InvalidFormat {
                        message: alloc::format!(
                            "NWB: neurodata_type_def is '{}', expected 'NWBFile'",
                            t
                        ),
                    });
                }
                Ok(_) => {
                    return Err(Error::InvalidFormat {
                        message: String::from(
                            "NWB: neurodata_type_def attribute value is not a string",
                        ),
                    });
                }
                Err(e) => return Err(e),
            }
        } else if attr.name == "nwb_version" {
            found_nwb_version = true;
        }
    }

    if !found_nwb_file_type {
        return Err(Error::InvalidFormat {
            message: String::from(
                "NWB: root group missing required attribute 'neurodata_type_def' = 'NWBFile'",
            ),
        });
    }

    if !found_nwb_version {
        return Err(Error::NotFound {
            path: String::from("nwb_version"),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Write-path conformance
// ---------------------------------------------------------------------------

/// Validate a [`crate::model::TimeSeries`] before writing it to an NWB file.
///
/// ## Checks
///
/// 1. Timing representation is present: either `timestamps` is `Some`, or
///    `rate` is `Some`.  Files omitting both timing fields are technically
///    non-conformant under NWB 2.x strict mode.
/// 2. `rate` is strictly positive when present.  A zero or negative rate is
///    physically invalid (frequency ≤ 0 Hz is undefined).
///
/// This function does **not** re-run [`crate::model::TimeSeries::validate`];
/// callers are responsible for invoking that before `validate_time_series_for_write`.
///
/// ## Errors
///
/// - [`consus_core::Error::InvalidFormat`] when timing is absent.
/// - [`consus_core::Error::InvalidFormat`] when `rate ≤ 0.0`.
#[cfg(feature = "alloc")]
pub fn validate_time_series_for_write(ts: &crate::model::TimeSeries) -> Result<()> {
    if !ts.has_timestamps() && !ts.has_rate() {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "NWB TimeSeries '{}': write requires either timestamps or \
                 (starting_time + rate); neither is present",
                ts.name()
            ),
        });
    }
    if let Some(rate) = ts.rate() {
        if rate <= 0.0 {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "NWB TimeSeries '{}': rate must be > 0.0, got {}",
                    ts.name(),
                    rate
                ),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Build a minimal v1 HDF5 attribute message for a FixedString scalar.
    ///
    /// Layout (v1):
    ///   version(1) | reserved(1) | name_sz(2LE) | dt_sz(2LE) | ds_sz(2LE)
    ///   name (null-terminated, padded to 8 bytes)
    ///   datatype (padded to 8 bytes)
    ///   dataspace (padded to 8 bytes)
    ///   data
    fn make_string_attr_bytes(attr_name: &str, value: &str) -> alloc::vec::Vec<u8> {
        fn align8(n: usize) -> usize {
            (n + 7) & !7
        }

        // Name section: null-terminated, padded to 8-byte boundary.
        let name_raw: alloc::vec::Vec<u8> = {
            let mut v = attr_name.as_bytes().to_vec();
            v.push(0u8);
            v
        };
        let name_sz = name_raw.len();
        let name_padded = align8(name_sz);

        // Datatype: HDF5 FixedString class 3, version 1.
        // Byte layout: class_version(1) | flags(1) | reserved(2) | size(4LE) | class_bits(4)
        let str_len = value.len().max(1);
        let dt_bytes: alloc::vec::Vec<u8> = {
            let class_version: u8 = (1u8 << 4) | 3u8; // version=1, class=3 (string)
            let mut v = alloc::vec![class_version, 0u8, 0u8, 0u8];
            v.extend_from_slice(&(str_len as u32).to_le_bytes());
            v.extend_from_slice(&[0u8; 4]); // class-specific: NullPad + ASCII
            v
        };
        let dt_sz = dt_bytes.len(); // 12
        let dt_padded = align8(dt_sz);

        // Dataspace: scalar (version 1, rank 0).
        let ds_bytes: alloc::vec::Vec<u8> = alloc::vec![1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
        let ds_sz = ds_bytes.len(); // 8
        let ds_padded = align8(ds_sz);

        // Data: value padded with null bytes to str_len.
        let data_bytes: alloc::vec::Vec<u8> = {
            let mut v = value.as_bytes().to_vec();
            while v.len() < str_len {
                v.push(0u8);
            }
            v
        };

        // Assemble the attribute message.
        let mut msg: alloc::vec::Vec<u8> = alloc::vec![
            1u8,                    // version
            0u8,                    // reserved
            (name_sz & 0xFF) as u8, // name_sz low byte
            (name_sz >> 8) as u8,   // name_sz high byte
            (dt_sz & 0xFF) as u8,   // dt_sz low byte
            (dt_sz >> 8) as u8,     // dt_sz high byte
            (ds_sz & 0xFF) as u8,   // ds_sz low byte
            (ds_sz >> 8) as u8,     // ds_sz high byte
        ];

        let mut name_sec = name_raw.clone();
        while name_sec.len() < name_padded {
            name_sec.push(0u8);
        }
        msg.extend_from_slice(&name_sec);

        let mut dt_sec = dt_bytes.clone();
        while dt_sec.len() < dt_padded {
            dt_sec.push(0u8);
        }
        msg.extend_from_slice(&dt_sec);

        let mut ds_sec = ds_bytes.clone();
        while ds_sec.len() < ds_padded {
            ds_sec.push(0u8);
        }
        msg.extend_from_slice(&ds_sec);

        msg.extend_from_slice(&data_bytes);
        msg
    }

    fn parse_attr(name: &str, value: &str) -> consus_hdf5::attribute::Hdf5Attribute {
        use consus_hdf5::address::ParseContext;
        let ctx = ParseContext::new(8, 8);
        let bytes = make_string_attr_bytes(name, value);
        consus_hdf5::attribute::Hdf5Attribute::parse(&bytes, &ctx)
            .expect("test attribute must parse")
    }

    // ── validate_root_attributes via attr list inspection ─────────────────
    //
    // Because validate_root_attributes requires an Hdf5File<R>, we test the
    // underlying logic through the attribute decoding path directly, and
    // verify the full function in the integration tests (src/file/mod.rs).
    // Here we check the per-attribute decoding invariants that the validator
    // depends on.

    #[test]
    fn neurodata_type_def_nwbfile_decodes_as_string() {
        // Theorem: an attribute named "neurodata_type_def" with value "NWBFile"
        // decodes to AttributeValue::String("NWBFile").
        let attr = parse_attr("neurodata_type_def", "NWBFile");
        let val = attr.decode_value().unwrap();
        assert_eq!(val, AttributeValue::String(String::from("NWBFile")));
    }

    #[test]
    fn nwb_version_attr_decodes_as_string() {
        let attr = parse_attr("nwb_version", "2.7.0");
        let val = attr.decode_value().unwrap();
        assert_eq!(val, AttributeValue::String(String::from("2.7.0")));
    }

    #[test]
    fn wrong_neurodata_type_def_value_is_detectable() {
        // Theorem: a value other than "NWBFile" can be distinguished at read time.
        let attr = parse_attr("neurodata_type_def", "TimeSeries");
        match attr.decode_value().unwrap() {
            AttributeValue::String(s) => assert_ne!(s, "NWBFile"),
            other => panic!("expected String, got {:?}", other),
        }
    }

    #[test]
    fn identifier_attr_decodes_correctly() {
        let attr = parse_attr("identifier", "test-session-001");
        let val = attr.decode_value().unwrap();
        assert_eq!(
            val,
            AttributeValue::String(String::from("test-session-001"))
        );
    }

    #[test]
    fn session_description_attr_decodes_correctly() {
        let attr = parse_attr("session_description", "A test NWB session");
        let val = attr.decode_value().unwrap();
        assert_eq!(
            val,
            AttributeValue::String(String::from("A test NWB session"))
        );
    }

    #[test]
    fn session_start_time_attr_decodes_correctly() {
        let attr = parse_attr("session_start_time", "2023-01-01T00:00:00+00:00");
        let val = attr.decode_value().unwrap();
        assert_eq!(
            val,
            AttributeValue::String(String::from("2023-01-01T00:00:00+00:00"))
        );
    }

    // ── Absence detection ─────────────────────────────────────────────────

    #[test]
    fn absent_neurodata_type_def_attr_is_not_found_in_list() {
        // Theorem: searching a list without neurodata_type_def finds nothing.
        let attrs = alloc::vec![
            parse_attr("nwb_version", "2.7.0"),
            parse_attr("identifier", "ses-1"),
        ];
        let found = attrs.iter().any(|a| a.name == "neurodata_type_def");
        assert!(
            !found,
            "neurodata_type_def should not be found in this list"
        );
    }

    #[test]
    fn absent_nwb_version_attr_is_not_found_in_list() {
        let attrs = alloc::vec![parse_attr("neurodata_type_def", "NWBFile")];
        let found = attrs.iter().any(|a| a.name == "nwb_version");
        assert!(!found);
    }

    // ── Multi-attribute scan correctness ──────────────────────────────────

    #[test]
    fn scan_finds_both_required_attrs_in_correct_order() {
        // Theorem: a pass over a list with both attributes finds both.
        let attrs = alloc::vec![
            parse_attr("identifier", "ses-1"),
            parse_attr("neurodata_type_def", "NWBFile"),
            parse_attr("session_description", "desc"),
            parse_attr("nwb_version", "2.7.0"),
            parse_attr("session_start_time", "2023-01-01T00:00:00+00:00"),
        ];

        let mut found_type = false;
        let mut found_version = false;

        for attr in &attrs {
            if attr.name == "neurodata_type_def" {
                if let Ok(AttributeValue::String(ref t)) = attr.decode_value() {
                    if t == "NWBFile" {
                        found_type = true;
                    }
                }
            } else if attr.name == "nwb_version" {
                found_version = true;
            }
        }

        assert!(found_type, "neurodata_type_def='NWBFile' must be found");
        assert!(found_version, "nwb_version must be found");
    }

    // ── validate_time_series_for_write ────────────────────────────────────

    #[test]
    fn validate_for_write_ok_with_timestamps() {
        // Theorem: a TimeSeries with explicit timestamps passes validation.
        let ts = crate::model::TimeSeries::with_timestamps(
            "t",
            alloc::vec![1.0, 2.0],
            alloc::vec![0.0, 0.1],
        );
        validate_time_series_for_write(&ts).unwrap();
    }

    #[test]
    fn validate_for_write_ok_with_rate() {
        // Theorem: a TimeSeries with a positive rate passes validation.
        let ts =
            crate::model::TimeSeries::with_rate("t", alloc::vec![1.0, 2.0], 0.0_f64, 1000.0_f64);
        validate_time_series_for_write(&ts).unwrap();
    }

    #[test]
    fn validate_for_write_rejects_no_timing() {
        // Theorem: a TimeSeries with no timing representation is rejected.
        let ts = crate::model::TimeSeries::without_timing("bare", alloc::vec![1.0]);
        let err = validate_time_series_for_write(&ts).unwrap_err();
        match err {
            Error::InvalidFormat { ref message } => {
                assert!(
                    message.contains("bare"),
                    "error must name the TimeSeries: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn validate_for_write_rejects_zero_rate() {
        // Theorem: rate = 0.0 is physically invalid.
        let ts = crate::model::TimeSeries::with_rate("zero_r", alloc::vec![1.0], 0.0_f64, 0.0_f64);
        let err = validate_time_series_for_write(&ts).unwrap_err();
        match err {
            Error::InvalidFormat { ref message } => {
                assert!(
                    message.contains("rate"),
                    "error must mention 'rate': {message}"
                );
                assert!(
                    message.contains("0"),
                    "error must contain the bad rate value: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn validate_for_write_rejects_negative_rate() {
        // Theorem: negative rate is physically invalid.
        let ts = crate::model::TimeSeries::with_rate("neg_r", alloc::vec![0.5], 0.0_f64, -1.0_f64);
        let err = validate_time_series_for_write(&ts).unwrap_err();
        assert!(
            matches!(err, Error::InvalidFormat { .. }),
            "expected InvalidFormat for negative rate, got {:?}",
            err
        );
    }

    #[test]
    fn validate_for_write_rejects_negative_inf_rate() {
        // Theorem: -inf rate is rejected by rate <= 0.0 check.
        let ts =
            crate::model::TimeSeries::with_rate("inf_r", alloc::vec![], 0.0_f64, f64::NEG_INFINITY);
        let err = validate_time_series_for_write(&ts).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat { .. }));
    }

    #[test]
    fn validate_for_write_accepts_very_small_positive_rate() {
        // Theorem: any strictly positive rate is accepted, including near-zero.
        let ts = crate::model::TimeSeries::with_rate(
            "tiny_r",
            alloc::vec![],
            0.0_f64,
            f64::MIN_POSITIVE,
        );
        validate_time_series_for_write(&ts).unwrap();
    }
}
