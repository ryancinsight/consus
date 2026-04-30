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
// ISO 8601 temporal validation
// ---------------------------------------------------------------------------

/// Returns `true` iff `s` matches the NWB 2.x required ISO 8601 datetime format.
///
/// ## Specification
///
/// NWB 2.x requires session timestamps to be RFC 3339-compatible ISO 8601
/// strings of the form `YYYY-MM-DDTHH:MM:SS[Z|±HH:MM]`.
///
/// ## Proof
///
/// The check is structural: date `YYYY-MM-DD` (10 bytes), `T` separator,
/// time `HH:MM:SS` (8 bytes), then one of:
/// - `Z`                → total 20 bytes (UTC)
/// - `+HH:MM` or `-HH:MM` → total 25 bytes (signed offset)
///
/// No calendar arithmetic is performed; the function validates digit positions
/// and field separators only.
pub fn is_valid_iso8601(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 20 {
        return false;
    }
    // Date: YYYY-MM-DD
    if !all_digits(&b[0..4]) {
        return false;
    }
    if b[4] != b'-' {
        return false;
    }
    if !all_digits(&b[5..7]) {
        return false;
    }
    if b[7] != b'-' {
        return false;
    }
    if !all_digits(&b[8..10]) {
        return false;
    }
    // T separator
    if b[10] != b'T' {
        return false;
    }
    // Time: HH:MM:SS
    if !all_digits(&b[11..13]) {
        return false;
    }
    if b[13] != b':' {
        return false;
    }
    if !all_digits(&b[14..16]) {
        return false;
    }
    if b[16] != b':' {
        return false;
    }
    if !all_digits(&b[17..19]) {
        return false;
    }
    // Timezone: Z (length 20) or ±HH:MM (length 25)
    match b[19] {
        b'Z' => b.len() == 20,
        b'+' | b'-' => {
            b.len() == 25 && all_digits(&b[20..22]) && b[22] == b':' && all_digits(&b[23..25])
        }
        _ => false,
    }
}

#[inline]
fn all_digits(b: &[u8]) -> bool {
    !b.is_empty() && b.iter().all(u8::is_ascii_digit)
}

// ---------------------------------------------------------------------------
// Conformance report types
// ---------------------------------------------------------------------------

/// A single NWB 2.x conformance violation.
///
/// Each variant corresponds to one normative constraint from the NWB 2.x
/// specification.  Multiple violations can be collected before reporting.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConformanceViolation {
    /// A required root-group attribute is absent from the file.
    MissingRootAttribute {
        /// Attribute name as it appears in the NWB 2.x specification.
        name: String,
    },
    /// A required root-group attribute is present but its value is invalid.
    InvalidRootAttributeValue {
        /// Attribute name.
        name: String,
        /// Human-readable description of the constraint that was violated.
        detail: String,
    },
    /// A required top-level NWB group (`acquisition`, `analysis`, etc.) is absent.
    MissingRequiredGroup {
        /// Group name relative to the HDF5 root (e.g. `"acquisition"`).
        path: String,
    },
    /// A `neurodata_type_def` attribute is absent from a group that requires one.
    GroupMissingAttribute {
        /// HDF5 path to the offending group.
        group_path: String,
        /// Name of the expected attribute.
        attr_name: String,
    },
    /// A TimeSeries group is missing the mandatory `data` sub-dataset.
    TimeSeriesMissingData {
        /// HDF5 path to the offending TimeSeries group.
        group_path: String,
    },
}

/// Collected result of a full NWB 2.x conformance check.
///
/// Holds zero or more [`ConformanceViolation`]s gathered during multi-layer
/// validation of a single file.  An empty report indicates full conformance.
///
/// ## Invariant
///
/// `is_conformant()` ⟺ `violations().is_empty()`
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NwbConformanceReport {
    violations: alloc::vec::Vec<ConformanceViolation>,
}

#[cfg(feature = "alloc")]
impl NwbConformanceReport {
    /// Create an empty (conformant) report.
    pub fn new() -> Self {
        Self {
            violations: alloc::vec::Vec::new(),
        }
    }

    /// Returns `true` iff no violations were recorded.
    pub fn is_conformant(&self) -> bool {
        self.violations.is_empty()
    }

    /// Borrow the collected violations in recording order.
    pub fn violations(&self) -> &[ConformanceViolation] {
        &self.violations
    }

    /// Record one violation.  Use [`NwbFile::validate_conformance`] to
    /// trigger full multi-layer validation instead of calling this directly.
    pub(crate) fn push(&mut self, v: ConformanceViolation) {
        self.violations.push(v);
    }

    /// Convert to `Result<()>`, mapping any violation to an `InvalidFormat` error.
    ///
    /// All violations after the first are discarded.  Callers that need the
    /// complete list must inspect [`violations`](Self::violations) before
    /// calling this.
    pub fn into_result(self) -> Result<()> {
        if self.is_conformant() {
            return Ok(());
        }
        Err(Error::InvalidFormat {
            message: alloc::format!(
                "NWB conformance: {} violation(s); first: {:?}",
                self.violations.len(),
                &self.violations[0],
            ),
        })
    }
}

#[cfg(feature = "alloc")]
impl Default for NwbConformanceReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Extended session attribute checking
// ---------------------------------------------------------------------------

/// Scan root-group attributes and record missing or invalid session fields.
///
/// Checks for `identifier` (non-empty string), `session_description`
/// (non-empty string), and `session_start_time` (present + ISO 8601 format).
/// All violations are appended to `report`; the function does not
/// short-circuit on the first failure.
///
/// These attributes are required by NWB 2.x §4.1 but are not checked by
/// [`validate_root_attributes`] to avoid redundant I/O on the normal open path.
///
/// ## Errors
///
/// Returns `Err` only on HDF5 I/O failure during attribute enumeration.
/// Constraint violations are recorded in `report`, not returned as errors.
#[cfg(feature = "alloc")]
pub fn check_root_session_attrs<R: ReadAt + Sync>(
    file: &Hdf5File<R>,
    report: &mut NwbConformanceReport,
) -> Result<()> {
    let root_addr = file.superblock().root_group_address;
    let attrs = file.attributes_at(root_addr)?;

    let mut found_identifier = false;
    let mut found_session_description = false;
    let mut found_session_start_time = false;
    let mut found_timestamps_reference_time = false;
    let mut found_file_create_date = false;

    for attr in &attrs {
        match attr.name.as_str() {
            "identifier" => {
                found_identifier = true;
                match attr.decode_value() {
                    Ok(AttributeValue::String(ref s)) if s.is_empty() => {
                        report.push(ConformanceViolation::InvalidRootAttributeValue {
                            name: String::from("identifier"),
                            detail: String::from("must not be empty"),
                        });
                    }
                    Ok(AttributeValue::String(_)) => {}
                    Ok(_) => {
                        report.push(ConformanceViolation::InvalidRootAttributeValue {
                            name: String::from("identifier"),
                            detail: String::from("must be a string"),
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
            "session_description" => {
                found_session_description = true;
                match attr.decode_value() {
                    Ok(AttributeValue::String(ref s)) if s.is_empty() => {
                        report.push(ConformanceViolation::InvalidRootAttributeValue {
                            name: String::from("session_description"),
                            detail: String::from("must not be empty"),
                        });
                    }
                    Ok(AttributeValue::String(_)) => {}
                    Ok(_) => {
                        report.push(ConformanceViolation::InvalidRootAttributeValue {
                            name: String::from("session_description"),
                            detail: String::from("must be a string"),
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
            "session_start_time" => {
                found_session_start_time = true;
                match attr.decode_value() {
                    Ok(AttributeValue::String(ref s)) => {
                        if !is_valid_iso8601(s) {
                            report.push(ConformanceViolation::InvalidRootAttributeValue {
                                name: String::from("session_start_time"),
                                detail: alloc::format!(
                                    "'{}' does not match ISO 8601 format \
                                     YYYY-MM-DDTHH:MM:SS[Z|±HH:MM]",
                                    s
                                ),
                            });
                        }
                    }
                    Ok(_) => {
                        report.push(ConformanceViolation::InvalidRootAttributeValue {
                            name: String::from("session_start_time"),
                            detail: String::from("must be a string"),
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
            "timestamps_reference_time" => {
                found_timestamps_reference_time = true;
                match attr.decode_value() {
                    Ok(AttributeValue::String(ref s)) => {
                        if !is_valid_iso8601(s) {
                            report.push(ConformanceViolation::InvalidRootAttributeValue {
                                name: String::from("timestamps_reference_time"),
                                detail: alloc::format!(
                                    "'{}' does not match ISO 8601 format \
                                     YYYY-MM-DDTHH:MM:SS[Z|±HH:MM]",
                                    s
                                ),
                            });
                        }
                    }
                    Ok(_) => {
                        report.push(ConformanceViolation::InvalidRootAttributeValue {
                            name: String::from("timestamps_reference_time"),
                            detail: String::from("must be a string"),
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
            "file_create_date" => {
                found_file_create_date = true;
                match attr.decode_value() {
                    Ok(AttributeValue::String(ref s)) => {
                        if !is_valid_iso8601(s) {
                            report.push(ConformanceViolation::InvalidRootAttributeValue {
                                name: String::from("file_create_date"),
                                detail: alloc::format!(
                                    "entry 0 '{}' does not match ISO 8601 format \
                                     YYYY-MM-DDTHH:MM:SS[Z|±HH:MM]",
                                    s
                                ),
                            });
                        }
                    }
                    Ok(AttributeValue::StringArray(ref v)) => {
                        if v.is_empty() {
                            report.push(ConformanceViolation::InvalidRootAttributeValue {
                                name: String::from("file_create_date"),
                                detail: String::from(
                                    "array must contain at least one ISO 8601 entry",
                                ),
                            });
                        } else {
                            for (i, s) in v.iter().enumerate() {
                                if !is_valid_iso8601(s) {
                                    report.push(ConformanceViolation::InvalidRootAttributeValue {
                                        name: String::from("file_create_date"),
                                        detail: alloc::format!(
                                            "entry {} '{}' does not match ISO 8601 format \
                                             YYYY-MM-DDTHH:MM:SS[Z|±HH:MM]",
                                            i,
                                            s
                                        ),
                                    });
                                    break; // report first invalid entry only
                                }
                            }
                        }
                    }
                    Ok(_) => {
                        report.push(ConformanceViolation::InvalidRootAttributeValue {
                            name: String::from("file_create_date"),
                            detail: String::from("must be a string or string array"),
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
            _ => {}
        }
    }

    if !found_identifier {
        report.push(ConformanceViolation::MissingRootAttribute {
            name: String::from("identifier"),
        });
    }
    if !found_session_description {
        report.push(ConformanceViolation::MissingRootAttribute {
            name: String::from("session_description"),
        });
    }
    if !found_session_start_time {
        report.push(ConformanceViolation::MissingRootAttribute {
            name: String::from("session_start_time"),
        });
    }
    if !found_timestamps_reference_time {
        report.push(ConformanceViolation::MissingRootAttribute {
            name: String::from("timestamps_reference_time"),
        });
    }
    if !found_file_create_date {
        report.push(ConformanceViolation::MissingRootAttribute {
            name: String::from("file_create_date"),
        });
    }

    Ok(())
}

/// Validate that every root-level group with `neurodata_type_def == "DynamicTable"`
/// carries a `colnames` attribute.
///
/// ## Specification
///
/// HDMF `DynamicTable` groups must expose a `colnames` attribute that lists
/// the names of all VectorData child columns. Absence of `colnames` renders
/// the table unreadable by HDMF-compliant readers.
///
/// ## Invariants
///
/// - Only direct children of the root group are examined.
/// - A missing `colnames` attribute produces `GroupMissingAttribute`.
/// - No column-content validation is performed (name-to-dataset binding
///   verification is deferred to a future layer).
#[cfg(feature = "alloc")]
pub fn check_dynamic_table_colnames<R: ReadAt + Sync>(
    file: &Hdf5File<R>,
    report: &mut NwbConformanceReport,
) -> Result<()> {
    use consus_core::LinkType;

    let root_addr = file.superblock().root_group_address;
    let children = match file.list_group_at(root_addr) {
        Ok(c) => c,
        Err(consus_core::Error::NotFound { .. }) => return Ok(()),
        Err(e) => return Err(e),
    };

    for (name, addr, link_type) in &children {
        if *link_type != LinkType::Hard {
            continue;
        }
        let attrs = match file.attributes_at(*addr) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let is_dynamic_table = attrs.iter().any(|a| {
            a.name == "neurodata_type_def"
                && matches!(
                    a.decode_value(),
                    Ok(AttributeValue::String(ref s)) if s == "DynamicTable"
                )
        });
        if is_dynamic_table {
            let has_colnames = attrs.iter().any(|a| a.name == "colnames");
            if !has_colnames {
                report.push(ConformanceViolation::GroupMissingAttribute {
                    group_path: name.clone(),
                    attr_name: String::from("colnames"),
                });
            }
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

    // -----------------------------------------------------------------------
    // is_valid_iso8601 tests
    // -----------------------------------------------------------------------

    #[test]
    fn iso8601_valid_z_timezone() {
        assert!(is_valid_iso8601("2023-01-01T00:00:00Z"));
    }

    #[test]
    fn iso8601_valid_positive_offset() {
        assert!(is_valid_iso8601("2023-06-15T12:30:45+05:30"));
    }

    #[test]
    fn iso8601_valid_negative_offset() {
        assert!(is_valid_iso8601("2023-06-15T12:30:45-07:00"));
    }

    #[test]
    fn iso8601_valid_zero_offset() {
        assert!(is_valid_iso8601("2020-12-31T23:59:59+00:00"));
    }

    #[test]
    fn iso8601_invalid_too_short_no_timezone() {
        // 19 chars — no timezone designator
        assert!(!is_valid_iso8601("2023-01-01T00:00:00"));
    }

    #[test]
    fn iso8601_invalid_space_instead_of_t() {
        assert!(!is_valid_iso8601("2023-01-01 00:00:00Z"));
    }

    #[test]
    fn iso8601_invalid_missing_date_dashes() {
        assert!(!is_valid_iso8601("20230101T00:00:00Z"));
    }

    #[test]
    fn iso8601_invalid_missing_time_colons() {
        assert!(!is_valid_iso8601("2023-01-01T000000Z"));
    }

    #[test]
    fn iso8601_invalid_non_digit_year() {
        assert!(!is_valid_iso8601("XXXX-01-01T00:00:00Z"));
    }

    #[test]
    fn iso8601_invalid_empty_string() {
        assert!(!is_valid_iso8601(""));
    }

    #[test]
    fn iso8601_invalid_offset_too_short() {
        // 24 chars — offset missing last digit
        assert!(!is_valid_iso8601("2023-01-01T00:00:00+07:0"));
    }

    #[test]
    fn iso8601_invalid_unknown_tz_char() {
        assert!(!is_valid_iso8601("2023-01-01T00:00:00X"));
    }

    // -----------------------------------------------------------------------
    // ConformanceViolation variant tests
    // -----------------------------------------------------------------------

    #[test]
    fn missing_root_attribute_variant_carries_name() {
        let v = ConformanceViolation::MissingRootAttribute {
            name: String::from("session_start_time"),
        };
        match v {
            ConformanceViolation::MissingRootAttribute { ref name } => {
                assert_eq!(name, "session_start_time");
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn invalid_root_attribute_value_variant_carries_detail() {
        let v = ConformanceViolation::InvalidRootAttributeValue {
            name: String::from("session_start_time"),
            detail: String::from("not ISO 8601"),
        };
        match &v {
            ConformanceViolation::InvalidRootAttributeValue { name, detail } => {
                assert_eq!(name, "session_start_time");
                assert!(detail.contains("ISO"), "detail must mention ISO: {detail}");
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn missing_required_group_variant_carries_path() {
        let v = ConformanceViolation::MissingRequiredGroup {
            path: String::from("acquisition"),
        };
        match v {
            ConformanceViolation::MissingRequiredGroup { ref path } => {
                assert_eq!(path, "acquisition");
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn timeseries_missing_data_variant_carries_group_path() {
        let v = ConformanceViolation::TimeSeriesMissingData {
            group_path: String::from("acquisition/my_ts"),
        };
        match v {
            ConformanceViolation::TimeSeriesMissingData { ref group_path } => {
                assert!(
                    group_path.contains("my_ts"),
                    "must contain group name: {group_path}"
                );
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // NwbConformanceReport tests
    // -----------------------------------------------------------------------

    #[test]
    fn conformance_report_new_is_conformant() {
        let report = NwbConformanceReport::new();
        assert!(report.is_conformant());
        assert!(report.violations().is_empty());
    }

    #[test]
    fn conformance_report_default_is_conformant() {
        let report = NwbConformanceReport::default();
        assert!(report.is_conformant());
    }

    #[test]
    fn conformance_report_with_one_violation_is_not_conformant() {
        let mut report = NwbConformanceReport::new();
        report.push(ConformanceViolation::MissingRootAttribute {
            name: String::from("foo"),
        });
        assert!(!report.is_conformant());
        assert_eq!(report.violations().len(), 1);
    }

    #[test]
    fn conformance_report_into_result_ok_when_clean() {
        let report = NwbConformanceReport::new();
        assert!(report.into_result().is_ok());
    }

    #[test]
    fn conformance_report_into_result_err_when_violations_present() {
        let mut report = NwbConformanceReport::new();
        report.push(ConformanceViolation::MissingRequiredGroup {
            path: String::from("acquisition"),
        });
        let err = report.into_result().unwrap_err();
        match err {
            Error::InvalidFormat { ref message } => {
                assert!(
                    message.contains('1'),
                    "message must contain violation count: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn conformance_report_collects_multiple_violations_without_short_circuit() {
        let mut report = NwbConformanceReport::new();
        report.push(ConformanceViolation::MissingRootAttribute {
            name: String::from("identifier"),
        });
        report.push(ConformanceViolation::MissingRequiredGroup {
            path: String::from("acquisition"),
        });
        report.push(ConformanceViolation::MissingRequiredGroup {
            path: String::from("analysis"),
        });
        assert_eq!(report.violations().len(), 3);
        assert!(!report.is_conformant());
    }

    #[test]
    fn conformance_report_clone_and_eq() {
        let mut report = NwbConformanceReport::new();
        report.push(ConformanceViolation::MissingRootAttribute {
            name: String::from("x"),
        });
        let cloned = report.clone();
        assert_eq!(report, cloned);
    }

    // ── Extended session attribute tests (M-048) ─────────────────────────────

    #[test]
    fn check_root_session_attrs_passes_with_timestamps_reference_time() {
        // Build minimal HDF5 with all 5 session attrs + timestamps_reference_time
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id1"),
            ("session_description", "desc"),
            ("session_start_time", "2023-06-15T09:30:00+05:30"),
            ("timestamps_reference_time", "2023-06-15T09:30:00+05:30"),
            ("file_create_date", "2023-06-15T09:30:00+05:30"),
        ] {
            let len = value.len().max(1);
            let dt = Datatype::FixedString {
                length: len,
                encoding: StringEncoding::Ascii,
            };
            let mut raw = value.as_bytes().to_vec();
            while raw.len() < len {
                raw.push(0u8);
            }
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::default();
        check_root_session_attrs(&file, &mut report).unwrap();
        let missing: alloc::vec::Vec<_> = report.violations().iter()
            .filter(|v| matches!(v, ConformanceViolation::MissingRootAttribute { name } if name == "timestamps_reference_time"))
            .collect();
        assert!(
            missing.is_empty(),
            "should not report missing timestamps_reference_time: {:?}",
            report.violations()
        );
        let invalid: alloc::vec::Vec<_> = report.violations().iter()
            .filter(|v| matches!(v, ConformanceViolation::InvalidRootAttributeValue { name, .. } if name == "timestamps_reference_time"))
            .collect();
        assert!(
            invalid.is_empty(),
            "should not report invalid timestamps_reference_time: {:?}",
            report.violations()
        );
    }

    #[test]
    fn check_root_session_attrs_reports_missing_timestamps_reference_time() {
        // NWB file without timestamps_reference_time should report MissingRootAttribute
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id1"),
            ("session_description", "desc"),
            ("session_start_time", "2023-01-01T00:00:00Z"),
            ("file_create_date", "2023-01-01T00:00:00Z"),
            // timestamps_reference_time intentionally omitted
        ] {
            let len = value.len().max(1);
            let dt = Datatype::FixedString {
                length: len,
                encoding: StringEncoding::Ascii,
            };
            let mut raw = value.as_bytes().to_vec();
            while raw.len() < len {
                raw.push(0u8);
            }
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::default();
        check_root_session_attrs(&file, &mut report).unwrap();
        assert!(
            report.violations().iter().any(|v| matches!(v,
                ConformanceViolation::MissingRootAttribute { name } if name == "timestamps_reference_time"
            )),
            "expected MissingRootAttribute(timestamps_reference_time): {:?}", report.violations()
        );
    }

    #[test]
    fn check_root_session_attrs_reports_invalid_timestamps_reference_time() {
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id1"),
            ("session_description", "desc"),
            ("session_start_time", "2023-01-01T00:00:00Z"),
            ("timestamps_reference_time", "not-a-date"), // invalid
            ("file_create_date", "2023-01-01T00:00:00Z"),
        ] {
            let len = value.len().max(1);
            let dt = Datatype::FixedString {
                length: len,
                encoding: StringEncoding::Ascii,
            };
            let mut raw = value.as_bytes().to_vec();
            while raw.len() < len {
                raw.push(0u8);
            }
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::default();
        check_root_session_attrs(&file, &mut report).unwrap();
        assert!(
            report.violations().iter().any(|v| matches!(v,
                ConformanceViolation::InvalidRootAttributeValue { name, .. } if name == "timestamps_reference_time"
            )),
            "expected InvalidRootAttributeValue(timestamps_reference_time): {:?}", report.violations()
        );
    }

    #[test]
    fn check_root_session_attrs_reports_missing_file_create_date() {
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id1"),
            ("session_description", "desc"),
            ("session_start_time", "2023-01-01T00:00:00Z"),
            ("timestamps_reference_time", "2023-01-01T00:00:00Z"),
            // file_create_date intentionally omitted
        ] {
            let len = value.len().max(1);
            let dt = Datatype::FixedString {
                length: len,
                encoding: StringEncoding::Ascii,
            };
            let mut raw = value.as_bytes().to_vec();
            while raw.len() < len {
                raw.push(0u8);
            }
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::default();
        check_root_session_attrs(&file, &mut report).unwrap();
        assert!(
            report.violations().iter().any(|v| matches!(v,
                ConformanceViolation::MissingRootAttribute { name } if name == "file_create_date"
            )),
            "expected MissingRootAttribute(file_create_date): {:?}",
            report.violations()
        );
    }

    #[test]
    fn check_root_session_attrs_passes_valid_file_create_date_scalar() {
        // file_create_date stored as scalar FixedString (single entry)
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id1"),
            ("session_description", "desc"),
            ("session_start_time", "2023-01-01T00:00:00Z"),
            ("timestamps_reference_time", "2023-01-01T00:00:00Z"),
            ("file_create_date", "2023-01-01T00:00:00Z"),
        ] {
            let len = value.len().max(1);
            let dt = Datatype::FixedString {
                length: len,
                encoding: StringEncoding::Ascii,
            };
            let mut raw = value.as_bytes().to_vec();
            while raw.len() < len {
                raw.push(0u8);
            }
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::default();
        check_root_session_attrs(&file, &mut report).unwrap();
        assert!(
            !report.violations().iter().any(|v| matches!(v,
                ConformanceViolation::MissingRootAttribute { name }
                | ConformanceViolation::InvalidRootAttributeValue { name, .. }
                if name == "file_create_date"
            )),
            "unexpected file_create_date violation: {:?}",
            report.violations()
        );
    }

    #[test]
    fn check_root_session_attrs_reports_invalid_file_create_date_scalar() {
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id1"),
            ("session_description", "desc"),
            ("session_start_time", "2023-01-01T00:00:00Z"),
            ("timestamps_reference_time", "2023-01-01T00:00:00Z"),
            ("file_create_date", "bad-date"), // invalid ISO 8601
        ] {
            let len = value.len().max(1);
            let dt = Datatype::FixedString {
                length: len,
                encoding: StringEncoding::Ascii,
            };
            let mut raw = value.as_bytes().to_vec();
            while raw.len() < len {
                raw.push(0u8);
            }
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::default();
        check_root_session_attrs(&file, &mut report).unwrap();
        assert!(
            report.violations().iter().any(|v| matches!(v,
                ConformanceViolation::InvalidRootAttributeValue { name, .. } if name == "file_create_date"
            )),
            "expected InvalidRootAttributeValue(file_create_date): {:?}", report.violations()
        );
    }
}
