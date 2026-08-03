#[cfg(feature = "alloc")]
use alloc::string::String;

use super::basic::is_valid_iso8601;
use consus_core::{AttributeValue, Error, Result};
use consus_hdf5::file::Hdf5File;
use consus_io::ReadAt;

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
    /// A column named in a DynamicTable's `colnames` attribute has no corresponding
    /// child dataset in the group.
    DynamicTableColumnMissing {
        /// HDF5 path to the DynamicTable group (relative to root).
        group_path: String,
        /// Column name that is listed in `colnames` but absent as a child.
        column_name: String,
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
/// [`crate::validation::validate_root_attributes`] to avoid redundant I/O on the
/// normal open path.
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

/// Layer 6: For each DynamicTable group that has a `colnames` attribute,
/// verify that each named column has a corresponding child dataset.
///
/// ## Invariant
/// ∀ DynamicTable group G with `colnames` = [c₁, c₂, …, cₙ]:
///   ∀ cᵢ ∈ colnames: ∃ child dataset named cᵢ in G
///
/// ## Column Name Encoding
/// `colnames` may be stored as:
/// - `AttributeValue::String(s)`: comma-separated (e.g. "location,group_name")
/// - `AttributeValue::StringArray(names)`: 1-D array of strings
///
/// Both formats are handled. Unknown formats are skipped.
#[cfg(feature = "alloc")]
pub fn check_dynamic_table_column_content<R: ReadAt + Sync>(
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

        // Only process DynamicTable groups
        let is_dynamic_table = attrs.iter().any(|a| {
            a.name == "neurodata_type_def"
                && matches!(
                    a.decode_value(),
                    Ok(AttributeValue::String(ref s)) if s == "DynamicTable"
                )
        });
        if !is_dynamic_table {
            continue;
        }

        // Get colnames attribute; skip if absent (layer 5 already reported it)
        let colnames_attr = match attrs.iter().find(|a| a.name == "colnames") {
            Some(a) => a,
            None => continue,
        };

        // Decode column names from colnames attribute
        let col_names: alloc::vec::Vec<String> = match colnames_attr.decode_value() {
            Ok(AttributeValue::StringArray(names)) => names,
            Ok(AttributeValue::String(s)) => {
                // comma-separated scalar form
                s.split(',')
                    .map(|c| String::from(c.trim()))
                    .filter(|c| !c.is_empty())
                    .collect()
            }
            _ => continue, // unknown encoding: skip
        };

        if col_names.is_empty() {
            continue;
        }

        // List children of this DynamicTable group
        let table_children = match file.list_group_at(*addr) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let child_names: alloc::collections::BTreeSet<&str> =
            table_children.iter().map(|(n, _, _)| n.as_str()).collect();

        // Report any column name that has no corresponding child
        for col in &col_names {
            if !child_names.contains(col.as_str()) {
                report.push(ConformanceViolation::DynamicTableColumnMissing {
                    group_path: name.clone(),
                    column_name: col.clone(),
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

    // -----------------------------------------------------------------------
    // Layer 6: DynamicTable column-content consistency (M-051)
    // -----------------------------------------------------------------------

    #[test]
    fn dynamic_table_column_missing_variant_carries_fields() {
        // Theorem: DynamicTableColumnMissing stores group_path and column_name.
        let v = ConformanceViolation::DynamicTableColumnMissing {
            group_path: String::from("electrodes"),
            column_name: String::from("location"),
        };
        match v {
            ConformanceViolation::DynamicTableColumnMissing {
                ref group_path,
                ref column_name,
            } => {
                assert_eq!(group_path, "electrodes");
                assert_eq!(column_name, "location");
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn check_dynamic_table_column_content_passes_when_all_columns_present() {
        // Theorem: DynamicTable with colnames="x" and child dataset "x" produces
        // no DynamicTableColumnMissing violation.
        use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{
            ChildDatasetSpec, DatasetCreationProps, FileCreationProps, Hdf5FileBuilder,
        };
        use core::num::NonZeroUsize;

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        let ndt_dt = Datatype::FixedString {
            length: 12,
            encoding: StringEncoding::Ascii,
        };
        let ndt_raw = b"DynamicTable".to_vec();

        // colnames="x" stored as scalar FixedString (comma-separated form, single name)
        let col_dt = Datatype::FixedString {
            length: 1,
            encoding: StringEncoding::Ascii,
        };
        let col_raw = b"x".to_vec();

        // child dataset "x": single f64 element
        let f64_dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let x_shape = Shape::fixed(&[1]);
        let x_raw = 1.0f64.to_le_bytes().to_vec();

        builder
            .add_group_with_attributes(
                "my_table",
                &[
                    ("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw),
                    ("colnames", &col_dt, &scalar, &col_raw),
                ],
                &[ChildDatasetSpec {
                    name: "x",
                    datatype: &f64_dt,
                    shape: &x_shape,
                    raw_data: &x_raw,
                    dcpl: DatasetCreationProps::default(),
                    attributes: &[],
                }],
            )
            .unwrap();

        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::new();
        check_dynamic_table_column_content(&file, &mut report).unwrap();

        let missing: alloc::vec::Vec<_> = report
            .violations()
            .iter()
            .filter(|v| matches!(v, ConformanceViolation::DynamicTableColumnMissing { .. }))
            .collect();
        assert!(
            missing.is_empty(),
            "must not report DynamicTableColumnMissing when column 'x' is present: {:?}",
            report.violations()
        );
    }

    #[test]
    fn check_dynamic_table_column_content_reports_missing_column() {
        // Theorem: DynamicTable with colnames="x,y" but only child "x" reports
        // DynamicTableColumnMissing for column "y".
        use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{
            ChildDatasetSpec, DatasetCreationProps, FileCreationProps, Hdf5FileBuilder,
        };
        use core::num::NonZeroUsize;

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        let ndt_dt = Datatype::FixedString {
            length: 12,
            encoding: StringEncoding::Ascii,
        };
        let ndt_raw = b"DynamicTable".to_vec();

        // colnames="x,y" as scalar FixedString (3 bytes)
        let col_dt = Datatype::FixedString {
            length: 3,
            encoding: StringEncoding::Ascii,
        };
        let col_raw = b"x,y".to_vec();

        // Only child dataset "x" is written; "y" is intentionally absent.
        let f64_dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let x_shape = Shape::fixed(&[1]);
        let x_raw = 1.0f64.to_le_bytes().to_vec();

        builder
            .add_group_with_attributes(
                "my_table",
                &[
                    ("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw),
                    ("colnames", &col_dt, &scalar, &col_raw),
                ],
                &[ChildDatasetSpec {
                    name: "x",
                    datatype: &f64_dt,
                    shape: &x_shape,
                    raw_data: &x_raw,
                    dcpl: DatasetCreationProps::default(),
                    attributes: &[],
                }],
            )
            .unwrap();

        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::new();
        check_dynamic_table_column_content(&file, &mut report).unwrap();

        let y_missing = report.violations().iter().any(|v| {
            matches!(
                v,
                ConformanceViolation::DynamicTableColumnMissing {
                    group_path,
                    column_name,
                } if group_path == "my_table" && column_name == "y"
            )
        });
        assert!(
            y_missing,
            "expected DynamicTableColumnMissing(my_table, y): {:?}",
            report.violations()
        );
        // "x" must NOT be reported missing (it is present as a child)
        let x_missing = report.violations().iter().any(|v| {
            matches!(
                v,
                ConformanceViolation::DynamicTableColumnMissing {
                    group_path,
                    column_name,
                } if group_path == "my_table" && column_name == "x"
            )
        });
        assert!(
            !x_missing,
            "must not report DynamicTableColumnMissing for column 'x' which is present: {:?}",
            report.violations()
        );
    }

    #[test]
    fn check_dynamic_table_column_content_skips_group_without_colnames() {
        // Theorem: a DynamicTable group that has no `colnames` attribute is
        // skipped by layer 6 — no DynamicTableColumnMissing is recorded.
        // (Layer 5 would have reported GroupMissingAttribute, but not layer 6.)
        use consus_core::{Datatype, Shape, StringEncoding};
        use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        let ndt_dt = Datatype::FixedString {
            length: 12,
            encoding: StringEncoding::Ascii,
        };
        let ndt_raw = b"DynamicTable".to_vec();

        // DynamicTable group with neurodata_type_def but NO colnames
        builder
            .add_group_with_attributes(
                "bare_table",
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[],
            )
            .unwrap();

        let bytes = builder.finish().unwrap();
        let reader = consus_io::SliceReader::new(&bytes);
        let file = consus_hdf5::file::Hdf5File::open(reader).unwrap();
        let mut report = NwbConformanceReport::new();
        check_dynamic_table_column_content(&file, &mut report).unwrap();

        let col_missing: alloc::vec::Vec<_> = report
            .violations()
            .iter()
            .filter(|v| matches!(v, ConformanceViolation::DynamicTableColumnMissing { .. }))
            .collect();
        assert!(
            col_missing.is_empty(),
            "layer 6 must not emit DynamicTableColumnMissing for group with no colnames: {:?}",
            report.violations()
        );
    }

    // ── proptest harnesses (M-052) ─────────────────────────────────────────

    #[cfg(test)]
    mod proptest_harnesses {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Safety invariant: `is_valid_iso8601` never panics on any input.
            #[test]
            fn is_valid_iso8601_never_panics(s in ".*") {
                let _ = is_valid_iso8601(&s);
            }

            /// Structural invariant: strings shorter than 20 chars can never be
            /// valid ISO 8601 (minimum: YYYY-MM-DDTHH:MM:SSZ = 20 chars).
            #[test]
            fn is_valid_iso8601_returns_false_for_short_strings(
                s in ".{0,18}"
            ) {
                if s.len() < 20 {
                    assert!(
                        !is_valid_iso8601(&s),
                        "string shorter than 20 chars must not be valid ISO 8601: {:?}",
                        s
                    );
                }
            }
        }

        proptest! {
            /// Completeness invariant: all analytically-constructed valid ISO 8601
            /// strings with Z timezone are accepted.
            #[test]
            fn is_valid_iso8601_accepts_generated_valid_z_strings(
                year in 1000u32..=9999,
                month in 1u32..=12,
                day in 1u32..=28,   // conservative: avoids month-length edge cases
                hour in 0u32..=23,
                minute in 0u32..=59,
                second in 0u32..=59,
            ) {
                let s = alloc::format!(
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                    year, month, day, hour, minute, second
                );
                assert!(
                    is_valid_iso8601(&s),
                    "analytically valid ISO 8601 must be accepted: {:?}",
                    s
                );
            }

            /// Completeness invariant: all analytically-constructed valid ISO 8601
            /// strings with ±HH:MM offset timezone are accepted.
            #[test]
            fn is_valid_iso8601_accepts_generated_valid_offset_strings(
                year in 1000u32..=9999,
                month in 1u32..=12,
                day in 1u32..=28,
                hour in 0u32..=23,
                minute in 0u32..=59,
                second in 0u32..=59,
                sign in 0u32..=1u32,
                tz_hour in 0u32..=14,
                tz_minute in 0u32..=59,
            ) {
                let sign_char = if sign == 0 { '+' } else { '-' };
                let s = alloc::format!(
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}{:02}:{:02}",
                    year, month, day, hour, minute, second,
                    sign_char, tz_hour, tz_minute
                );
                assert!(
                    is_valid_iso8601(&s),
                    "analytically valid ISO 8601 with offset must be accepted: {:?}",
                    s
                );
            }
        }
    }
}
