use super::*;
use crate::model::TimeSeries;
use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
use consus_hdf5::file::writer::{
    ChildDatasetSpec, ChildGroupSpec, DatasetCreationProps, FileCreationProps, Hdf5FileBuilder,
};
use consus_hdf5::file::Hdf5File;
use consus_io::SliceReader;
use core::num::NonZeroUsize;

fn fixed_string_dt(value: &str) -> (Datatype, alloc::vec::Vec<u8>) {
    let len = value.len().max(1);
    let dt = Datatype::FixedString {
        length: len,
        encoding: StringEncoding::Ascii,
    };
    let mut raw = value.as_bytes().to_vec();
    while raw.len() < len {
        raw.push(0u8);
    }
    (dt, raw)
}

fn make_minimal_nwb(id: &str, desc: &str, ts: &str) -> alloc::vec::Vec<u8> {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();

    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", id),
        ("session_description", desc),
        ("session_start_time", ts),
        ("timestamps_reference_time", ts),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    // file_create_date as 1-D array (matches NwbFileBuilder output).
    let (fcd_dt, fcd_raw) = fixed_string_dt(ts);
    let fcd_shape = Shape::fixed(&[1]);
    builder
        .add_root_attribute("file_create_date", &fcd_dt, &fcd_shape, &fcd_raw)
        .unwrap();

    builder.finish().unwrap()
}

fn make_nwb_with_timeseries(
    id: &str,
    ts_name: &str,
    data: &[f64],
    timestamps: &[f64],
) -> alloc::vec::Vec<u8> {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();

    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", id),
        ("session_description", "test"),
        ("session_start_time", "2023-01-01T00:00:00+00:00"),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }

    let f64_dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let data_raw: alloc::vec::Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ts_raw: alloc::vec::Vec<u8> = timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();
    let data_shape = Shape::fixed(&[data.len()]);
    let ts_shape = Shape::fixed(&[timestamps.len()]);
    let (ndt_dt, ndt_raw) = fixed_string_dt("TimeSeries");

    builder
        .add_group_with_attributes(
            ts_name,
            &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
            &[
                ChildDatasetSpec {
                    name: "data",
                    datatype: &f64_dt,
                    shape: &data_shape,
                    raw_data: &data_raw,
                    dcpl: DatasetCreationProps::default(),
                    attributes: &[],
                },
                ChildDatasetSpec {
                    name: "timestamps",
                    datatype: &f64_dt,
                    shape: &ts_shape,
                    raw_data: &ts_raw,
                    dcpl: DatasetCreationProps::default(),
                    attributes: &[],
                },
            ],
        )
        .unwrap();

    builder.finish().unwrap()
}

#[test]
fn open_valid_nwb_file_succeeds() {
    let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
    let nwb = NwbFile::open(&bytes).unwrap();
    let version = nwb.nwb_version().unwrap();
    assert_eq!(version.as_str(), "2.7");
}

#[test]
fn open_non_hdf5_bytes_returns_error() {
    let bytes = b"not hdf5".to_vec();
    let err = NwbFile::open(&bytes).unwrap_err();
    assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
}

#[test]
fn open_hdf5_without_neurodata_type_def_returns_invalid_format() {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    let (dt, raw) = fixed_string_dt("2.7.0");
    builder
        .add_root_attribute("nwb_version", &dt, &scalar, &raw)
        .unwrap();
    let bytes = builder.finish().unwrap();
    let err = NwbFile::open(&bytes).unwrap_err();
    assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
}

#[test]
fn open_hdf5_without_nwb_version_returns_not_found() {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    let (dt, raw) = fixed_string_dt("NWBFile");
    builder
        .add_root_attribute("neurodata_type_def", &dt, &scalar, &raw)
        .unwrap();
    let bytes = builder.finish().unwrap();
    let err = NwbFile::open(&bytes).unwrap_err();
    assert!(matches!(err, consus_core::Error::NotFound { .. }));
}

#[test]
fn nwb_version_returns_v2_7_for_2_7_0() {
    let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
    let nwb = NwbFile::open(&bytes).unwrap();
    let version = nwb.nwb_version().unwrap();
    assert_eq!(version.as_str(), "2.7");
}

#[test]
fn session_metadata_returns_correct_fields() {
    let bytes = make_minimal_nwb("session-123", "desc", "2023-01-01T12:34:56+00:00");
    let nwb = NwbFile::open(&bytes).unwrap();
    let meta = nwb.session_metadata().unwrap();
    assert_eq!(meta.identifier(), "session-123");
    assert_eq!(meta.session_description(), "desc");
    assert_eq!(meta.session_start_time(), "2023-01-01T12:34:56+00:00");
}

#[test]
fn session_metadata_missing_identifier_returns_not_found() {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("session_description", "desc"),
        ("session_start_time", "2023-01-01T00:00:00+00:00"),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    let bytes = builder.finish().unwrap();
    let nwb = NwbFile::open(&bytes).unwrap();
    let err = nwb.session_metadata().unwrap_err();
    assert!(matches!(err, consus_core::Error::NotFound { .. }));
}

#[test]
fn time_series_reads_data_and_timestamps() {
    let bytes = make_nwb_with_timeseries("id", "ts", &[1.0, 2.0, 3.0], &[0.0, 0.1, 0.2]);
    let nwb = NwbFile::open(&bytes).unwrap();
    let ts = nwb.time_series("ts").unwrap();
    assert_eq!(ts.name(), "ts");
    assert_eq!(ts.data(), &[1.0, 2.0, 3.0]);
    assert_eq!(ts.timestamps(), Some([0.0, 0.1, 0.2].as_slice()));
}

#[test]
fn time_series_validates_length_invariant() {
    let ts = TimeSeries::from_parts("bad", vec![1.0, 2.0], Some(vec![0.0]), None, None);
    let err = ts.validate().unwrap_err();
    assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
}

#[test]
fn time_series_missing_group_returns_not_found() {
    let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
    let nwb = NwbFile::open(&bytes).unwrap();
    let err = nwb.time_series("missing").unwrap_err();
    assert!(matches!(err, consus_core::Error::NotFound { .. }));
}

#[test]
fn time_series_name_derived_from_last_path_component() {
    let mut builder =
        NwbFileBuilder::new("2.7.0", "id", "desc", "2023-01-01T00:00:00+00:00").unwrap();
    let ts = TimeSeries::with_timestamps("my_ts", vec![1.0], vec![0.0]);
    builder.write_time_series(&ts).unwrap();
    let bytes = builder.finish().unwrap();
    let nwb = NwbFile::open(&bytes).unwrap();
    let ts = nwb.time_series("my_ts").unwrap();
    assert_eq!(ts.name(), "my_ts");
}

#[test]
fn time_series_reads_starting_time_and_rate_from_dataset() {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", "id"),
        ("session_description", "desc"),
        ("session_start_time", "2023-01-01T00:00:00+00:00"),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    let f64_dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let f32_dt = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let (ndt_dt, ndt_raw) = fixed_string_dt("TimeSeries");
    let data = [1.0f64, 2.0f64];
    let st = [0.5f64];
    let rate = [2.0f32];
    builder
        .add_group_with_attributes(
            "rate_ts",
            &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
            &[
                ChildDatasetSpec {
                    name: "data",
                    datatype: &f64_dt,
                    shape: &Shape::fixed(&[data.len()]),
                    raw_data: &data
                        .iter()
                        .copied()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<alloc::vec::Vec<u8>>(),
                    dcpl: DatasetCreationProps::default(),
                    attributes: &[],
                },
                ChildDatasetSpec {
                    name: "starting_time",
                    datatype: &f64_dt,
                    shape: &scalar,
                    raw_data: &st
                        .iter()
                        .copied()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<alloc::vec::Vec<u8>>(),
                    dcpl: DatasetCreationProps::default(),
                    attributes: &[(
                        "rate",
                        &f32_dt,
                        &scalar,
                        &rate
                            .iter()
                            .copied()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<alloc::vec::Vec<u8>>(),
                    )],
                },
            ],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let nwb = NwbFile::open(&bytes).unwrap();
    let ts = nwb.time_series("rate_ts").unwrap();
    assert_eq!(ts.starting_time(), Some(0.5));
    assert_eq!(ts.rate(), Some(2.0));
}

#[test]
fn time_series_with_timestamps_does_not_read_starting_time() {
    let bytes = make_nwb_with_timeseries("id", "ts", &[1.0, 2.0], &[0.0, 0.1]);
    let nwb = NwbFile::open(&bytes).unwrap();
    let ts = nwb.time_series("ts").unwrap();
    assert_eq!(ts.timestamps(), Some([0.0, 0.1].as_slice()));
    assert!(ts.starting_time().is_none());
    assert!(ts.rate().is_none());
}

// Additional tests for list_time_series, subject, builder roundtrips, units table,
// electrode table, proptests, and negative paths should be restored alongside the
// corresponding model and storage modules.

// ── list_specifications ───────────────────────────────────────────────

#[test]
fn list_specifications_returns_empty_when_group_absent() {
    let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
    let nwb = NwbFile::open(&bytes).unwrap();
    let names = nwb.list_specifications().unwrap();
    assert!(
        names.is_empty(),
        "no /specifications/ group should yield empty list, got {:?}",
        names
    );
}

#[test]
fn list_specifications_returns_namespace_names_after_write() {
    use crate::namespace::NwbNamespaceSpec;
    let mut builder =
        NwbFileBuilder::new("2.8.0", "spec-list", "test", "2024-01-01T00:00:00").unwrap();
    let specs = alloc::vec![
        NwbNamespaceSpec {
            name: alloc::string::String::from("core"),
            version: alloc::string::String::from("2.8.0"),
            doc_url: None,
            neurodata_types: alloc::vec![],
        },
        NwbNamespaceSpec {
            name: alloc::string::String::from("hdmf-common"),
            version: alloc::string::String::from("1.8.0"),
            doc_url: None,
            neurodata_types: alloc::vec![],
        },
    ];
    builder.write_namespace_specs(&specs).unwrap();
    let bytes = builder.finish().unwrap();
    let nwb = NwbFile::open(&bytes).unwrap();
    let mut names = nwb.list_specifications().unwrap();
    names.sort();
    assert_eq!(names, &["core", "hdmf-common"]);
}

// ── read_specification ────────────────────────────────────────────────

#[test]
fn read_specification_returns_not_found_when_absent() {
    let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
    let nwb = NwbFile::open(&bytes).unwrap();
    let err = nwb
        .read_specification("core", "2.8.0")
        .expect_err("absent spec must return error");
    assert!(
        matches!(err, consus_core::Error::NotFound { .. }),
        "expected NotFound, got {:?}",
        err
    );
}

#[test]
fn read_specification_roundtrip_core_spec_with_neurodata_types() {
    use crate::namespace::NwbNamespaceSpec;
    let original = NwbNamespaceSpec {
        name: alloc::string::String::from("core"),
        version: alloc::string::String::from("2.8.0"),
        doc_url: Some(alloc::string::String::from(
            "https://nwb-schema.readthedocs.io/en/latest/",
        )),
        neurodata_types: alloc::vec![
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("TimeSeries"),
                neurodata_type_inc: None,
            },
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("ElectricalSeries"),
                neurodata_type_inc: None,
            },
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("SpatialSeries"),
                neurodata_type_inc: None,
            },
        ],
    };
    let mut builder =
        NwbFileBuilder::new("2.8.0", "spec-rt", "desc", "2024-01-01T00:00:00").unwrap();
    builder
        .write_namespace_specs(core::slice::from_ref(&original))
        .unwrap();
    let bytes = builder.finish().unwrap();
    let nwb = NwbFile::open(&bytes).unwrap();
    let restored = nwb.read_specification("core", "2.8.0").unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0], original);
}

#[test]
fn read_specification_roundtrip_hdmf_common_spec() {
    use crate::namespace::NwbNamespaceSpec;
    let original = NwbNamespaceSpec {
        name: alloc::string::String::from("hdmf-common"),
        version: alloc::string::String::from("1.8.0"),
        doc_url: None,
        neurodata_types: alloc::vec![
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("VectorData"),
                neurodata_type_inc: None,
            },
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("DynamicTable"),
                neurodata_type_inc: None,
            },
        ],
    };
    let mut builder =
        NwbFileBuilder::new("2.8.0", "spec-hdmf", "desc", "2024-01-01T00:00:00").unwrap();
    builder
        .write_namespace_specs(core::slice::from_ref(&original))
        .unwrap();
    let bytes = builder.finish().unwrap();
    let nwb = NwbFile::open(&bytes).unwrap();
    let restored = nwb.read_specification("hdmf-common", "1.8.0").unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0], original);
}

#[test]
fn write_namespace_specs_empty_slice_is_noop() {
    let mut builder =
        NwbFileBuilder::new("2.8.0", "spec-empty", "desc", "2024-01-01T00:00:00").unwrap();
    builder.write_namespace_specs(&[]).unwrap();
    let bytes = builder.finish().unwrap();
    let nwb = NwbFile::open(&bytes).unwrap();
    let names = nwb.list_specifications().unwrap();
    assert!(names.is_empty());
}

#[test]
fn nwb_version_returns_v2_8_for_2_8_0() {
    use crate::version::NwbVersion;
    let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
    // Build a file with version "2.8.0" to verify V2_8 parsing.
    let builder =
        NwbFileBuilder::new("2.8.0", "id-v28", "desc", "2024-01-01T00:00:00").unwrap();
    let bytes28 = builder.finish().unwrap();
    let nwb = NwbFile::open(&bytes28).unwrap();
    assert_eq!(nwb.nwb_version().unwrap(), NwbVersion::V2_8);
    let _ = bytes; // suppress unused warning
}

// -----------------------------------------------------------------------
// validate_conformance tests
// -----------------------------------------------------------------------

/// A file with only root attributes (no required groups) must report
/// MissingRequiredGroup violations for all 5 mandatory groups.
#[test]
fn validate_conformance_reports_missing_required_groups() {
    let bytes = make_minimal_nwb("test-id", "session desc", "2023-01-01T00:00:00Z");
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    assert!(
        !report.is_conformant(),
        "minimal file must have group violations"
    );
    let missing: alloc::vec::Vec<&str> = report
        .violations()
        .iter()
        .filter_map(|v| match v {
            crate::validation::ConformanceViolation::MissingRequiredGroup { path } => {
                Some(path.as_str())
            }
            _ => None,
        })
        .collect();
    assert!(
        missing.contains(&"acquisition"),
        "acquisition must be reported missing: {missing:?}"
    );
    assert!(
        missing.contains(&"analysis"),
        "analysis must be reported missing: {missing:?}"
    );
    assert!(
        missing.contains(&"processing"),
        "processing must be reported missing: {missing:?}"
    );
}

/// All 5 required groups must be reported when none are present.
#[test]
fn validate_conformance_collects_all_five_missing_groups() {
    let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00Z");
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    let missing_count = report
        .violations()
        .iter()
        .filter(|v| {
            matches!(
                v,
                crate::validation::ConformanceViolation::MissingRequiredGroup { .. }
            )
        })
        .count();
    assert_eq!(
        missing_count,
        5,
        "all 5 required groups must be reported; got: {:?}",
        report.violations()
    );
}

/// A fully conformant file (all required groups present, valid session
/// attributes) must pass validation with zero violations.
#[test]
fn validate_conformance_passes_with_all_required_groups() {
    let mut builder = NwbFileBuilder::new(
        "2.7.0",
        "test-id",
        "test description",
        "2023-01-01T00:00:00Z",
    )
    .unwrap();
    for group in &[
        "acquisition",
        "analysis",
        "processing",
        "stimulus",
        "general",
    ] {
        builder.write_empty_group(group).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    assert!(
        report.is_conformant(),
        "file with all required groups must be conformant: {:?}",
        report.violations()
    );
}

/// A file with a bad `session_start_time` format must report exactly one
/// InvalidRootAttributeValue violation for that attribute.
#[test]
fn validate_conformance_reports_bad_session_start_time_format() {
    // NwbFile::open only checks neurodata_type_def + nwb_version, so a file
    // with a bad timestamp format opens successfully but fails full conformance.
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", "id-001"),
        ("session_description", "desc"),
        ("session_start_time", "not-a-date"),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    for group in &[
        "acquisition",
        "analysis",
        "processing",
        "stimulus",
        "general",
    ] {
        builder.add_group_with_attributes(group, &[], &[]).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    assert!(!report.is_conformant());
    let bad_ts_violations: alloc::vec::Vec<_> = report
        .violations()
        .iter()
        .filter(|v| {
            matches!(v,
                crate::validation::ConformanceViolation::InvalidRootAttributeValue { name, .. }
                if name == "session_start_time"
            )
        })
        .collect();
    assert_eq!(
        bad_ts_violations.len(),
        1,
        "exactly one bad-format violation expected: {:?}",
        report.violations()
    );
}

/// A TimeSeries group under /acquisition that is missing a `data` dataset
/// must produce a TimeSeriesMissingData violation naming the group path.
#[test]
fn validate_conformance_reports_timeseries_missing_data() {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", "id-001"),
        ("session_description", "desc"),
        ("session_start_time", "2023-01-01T00:00:00Z"),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    // Add the 4 non-acquisition required groups directly.
    for group in &["analysis", "processing", "stimulus", "general"] {
        builder.add_group_with_attributes(group, &[], &[]).unwrap();
    }
    // Build /acquisition with a TimeSeries child that has NO `data` dataset.
    let (ndt_dt, ndt_raw) = fixed_string_dt("TimeSeries");
    let ts_group = ChildGroupSpec {
        name: "test_ts",
        attributes: &[("neurodata_type_def", &ndt_dt, &scalar, ndt_raw.as_slice())],
        datasets: &[],
        sub_groups: &[],
    };
    builder
        .add_group_with_children("acquisition", &[], &[], &[ts_group])
        .unwrap();
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    let missing_data: alloc::vec::Vec<_> = report
        .violations()
        .iter()
        .filter(|v| {
            matches!(
                v,
                crate::validation::ConformanceViolation::TimeSeriesMissingData { .. }
            )
        })
        .collect();
    assert!(
        !missing_data.is_empty(),
        "missing-data violation must be present: {:?}",
        report.violations()
    );
    match &missing_data[0] {
        crate::validation::ConformanceViolation::TimeSeriesMissingData { group_path } => {
            assert!(
                group_path.contains("test_ts"),
                "violation must name the group: {group_path}"
            );
        }
        other => panic!("unexpected variant: {:?}", other),
    }
}

/// validate_conformance must not short-circuit: a file missing all 5
/// required groups must report exactly 5 MissingRequiredGroup violations,
/// not stop after the first.
#[test]
fn validate_conformance_does_not_short_circuit() {
    let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00Z");
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    // Layer 2 finds no session-attribute violations (they are all valid).
    let session_violations = report
        .violations()
        .iter()
        .filter(|v| {
            matches!(
                v,
                crate::validation::ConformanceViolation::MissingRootAttribute { .. }
                    | crate::validation::ConformanceViolation::InvalidRootAttributeValue { .. }
            )
        })
        .count();
    assert_eq!(
        session_violations,
        0,
        "no session-attr violations expected: {:?}",
        report.violations()
    );
    // Layer 3 must find 5 missing groups — not short-circuit after 1.
    let group_violations = report
        .violations()
        .iter()
        .filter(|v| {
            matches!(
                v,
                crate::validation::ConformanceViolation::MissingRequiredGroup { .. }
            )
        })
        .count();
    assert_eq!(
        group_violations,
        5,
        "all 5 missing-group violations must be collected: {:?}",
        report.violations()
    );
}

// -----------------------------------------------------------------------
// M-048: NWB extended conformance — timestamps_reference_time,
// file_create_date, and DynamicTable colnames.
// -----------------------------------------------------------------------

/// NwbFileBuilder::new must write timestamps_reference_time as a scalar
/// FixedString equal to session_start_time.
#[test]
fn nwb_file_builder_writes_timestamps_reference_time() {
    let ts = "2024-03-15T08:00:00Z";
    let bytes = NwbFileBuilder::new("2.7.0", "uid1", "desc", ts)
        .unwrap()
        .finish()
        .unwrap();
    // NwbFile::hdf5 is private; open via Hdf5File directly for raw attr access.
    let hdf5 = Hdf5File::open(SliceReader::new(&bytes)).unwrap();
    let root_addr = hdf5.superblock().root_group_address;
    let attrs = hdf5.attributes_at(root_addr).unwrap();
    let trt_attr = attrs.iter().find(|a| a.name == "timestamps_reference_time");
    assert!(
        trt_attr.is_some(),
        "timestamps_reference_time attribute must be written"
    );
    match trt_attr.unwrap().decode_value().unwrap() {
        consus_core::AttributeValue::String(ref s) => {
            assert_eq!(
                s.as_str(),
                ts,
                "timestamps_reference_time must equal session_start_time"
            );
        }
        other => panic!("expected String, got {:?}", other),
    }
}

/// NwbFileBuilder::new must write file_create_date as a 1-D FixedString
/// array with exactly one entry equal to session_start_time.
#[test]
fn nwb_file_builder_writes_file_create_date() {
    let ts = "2024-03-15T08:00:00Z";
    let bytes = NwbFileBuilder::new("2.7.0", "uid2", "desc", ts)
        .unwrap()
        .finish()
        .unwrap();
    let hdf5 = Hdf5File::open(SliceReader::new(&bytes)).unwrap();
    let root_addr = hdf5.superblock().root_group_address;
    let attrs = hdf5.attributes_at(root_addr).unwrap();
    let fcd_attr = attrs.iter().find(|a| a.name == "file_create_date");
    assert!(
        fcd_attr.is_some(),
        "file_create_date attribute must be written"
    );
    match fcd_attr.unwrap().decode_value().unwrap() {
        consus_core::AttributeValue::String(ref s) => {
            assert_eq!(s.as_str(), ts);
        }
        consus_core::AttributeValue::StringArray(ref v) => {
            assert_eq!(v.len(), 1, "file_create_date must have exactly 1 entry");
            assert_eq!(v[0].as_str(), ts);
        }
        other => panic!("expected String or StringArray, got {:?}", other),
    }
}

/// A file built with NwbFileBuilder::new and all required groups must
/// report no violations for timestamps_reference_time or file_create_date.
#[test]
fn validate_conformance_passes_with_all_extended_attrs() {
    let ts = "2023-07-04T12:00:00Z";
    let mut builder = NwbFileBuilder::new("2.7.0", "full-id", "full desc", ts).unwrap();
    for group in &[
        "acquisition",
        "analysis",
        "processing",
        "stimulus",
        "general",
    ] {
        builder.write_empty_group(group).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    let trt_violations: alloc::vec::Vec<_> = report
        .violations()
        .iter()
        .filter(|v| {
            matches!(
                v,
                crate::validation::ConformanceViolation::MissingRootAttribute { name }
                | crate::validation::ConformanceViolation::InvalidRootAttributeValue {
                    name, ..
                } if name == "timestamps_reference_time" || name == "file_create_date"
            )
        })
        .collect();
    assert!(
        trt_violations.is_empty(),
        "unexpected timestamps_reference_time or file_create_date violations: {:?}",
        report.violations()
    );
}

/// A file whose root group lacks timestamps_reference_time must report
/// MissingRootAttribute for that attribute.
#[test]
fn validate_conformance_reports_missing_timestamps_reference_time() {
    let ts = "2023-01-01T00:00:00Z";
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", "id"),
        ("session_description", "desc"),
        ("session_start_time", ts),
        // timestamps_reference_time intentionally omitted
        ("file_create_date", ts),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    for group in &[
        "acquisition",
        "analysis",
        "processing",
        "stimulus",
        "general",
    ] {
        builder.add_group_with_attributes(group, &[], &[]).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    assert!(
        report.violations().iter().any(|v| matches!(
            v,
            crate::validation::ConformanceViolation::MissingRootAttribute { name }
                if name == "timestamps_reference_time"
        )),
        "expected MissingRootAttribute(timestamps_reference_time): {:?}",
        report.violations()
    );
}

/// A file whose root group lacks file_create_date must report
/// MissingRootAttribute for that attribute.
#[test]
fn validate_conformance_reports_missing_file_create_date() {
    let ts = "2023-01-01T00:00:00Z";
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", "id"),
        ("session_description", "desc"),
        ("session_start_time", ts),
        ("timestamps_reference_time", ts),
        // file_create_date intentionally omitted
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    for group in &[
        "acquisition",
        "analysis",
        "processing",
        "stimulus",
        "general",
    ] {
        builder.add_group_with_attributes(group, &[], &[]).unwrap();
    }
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    assert!(
        report.violations().iter().any(|v| matches!(
            v,
            crate::validation::ConformanceViolation::MissingRootAttribute { name }
                if name == "file_create_date"
        )),
        "expected MissingRootAttribute(file_create_date): {:?}",
        report.violations()
    );
}

/// A DynamicTable root-level group without a `colnames` attribute must
/// produce a GroupMissingAttribute violation for that group.
///
/// The file is built manually using Hdf5FileBuilder because NwbFileBuilder
/// does not expose its internal hdf5 field (private invariant).
#[test]
fn validate_conformance_reports_dynamic_table_missing_colnames() {
    let ts = "2023-01-01T00:00:00Z";
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    // Write all 7 required root attributes.
    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", "id"),
        ("session_description", "desc"),
        ("session_start_time", ts),
        ("timestamps_reference_time", ts),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    // file_create_date as 1-D array.
    let (fcd_dt, fcd_raw) = fixed_string_dt(ts);
    let fcd_shape = Shape::fixed(&[1]);
    builder
        .add_root_attribute("file_create_date", &fcd_dt, &fcd_shape, &fcd_raw)
        .unwrap();
    // Required top-level groups.
    for group in &[
        "acquisition",
        "analysis",
        "processing",
        "stimulus",
        "general",
    ] {
        builder.add_group_with_attributes(group, &[], &[]).unwrap();
    }
    // DynamicTable group without colnames — layer 5 must report this.
    let (ndt_dt, ndt_raw) = fixed_string_dt("DynamicTable");
    builder
        .add_group_with_attributes(
            "my_table",
            &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    assert!(
        report.violations().iter().any(|v| matches!(
            v,
            crate::validation::ConformanceViolation::GroupMissingAttribute {
                group_path,
                attr_name,
            } if group_path == "my_table" && attr_name == "colnames"
        )),
        "expected GroupMissingAttribute(my_table, colnames): {:?}",
        report.violations()
    );
}

/// Layer 6 must report DynamicTableColumnMissing when a DynamicTable
/// declares a column in `colnames` that has no corresponding child dataset.
#[test]
fn validate_conformance_reports_dynamic_table_column_missing() {
    let ts = "2023-01-01T00:00:00Z";
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let scalar = Shape::scalar();
    for (name, value) in &[
        ("neurodata_type_def", "NWBFile"),
        ("nwb_version", "2.7.0"),
        ("identifier", "id"),
        ("session_description", "desc"),
        ("session_start_time", ts),
        ("timestamps_reference_time", ts),
    ] {
        let (dt, raw) = fixed_string_dt(value);
        builder
            .add_root_attribute(name, &dt, &scalar, &raw)
            .unwrap();
    }
    let (fcd_dt, fcd_raw) = fixed_string_dt(ts);
    let fcd_shape = Shape::fixed(&[1]);
    builder
        .add_root_attribute("file_create_date", &fcd_dt, &fcd_shape, &fcd_raw)
        .unwrap();
    for group in &[
        "acquisition",
        "analysis",
        "processing",
        "stimulus",
        "general",
    ] {
        builder.add_group_with_attributes(group, &[], &[]).unwrap();
    }
    // DynamicTable with colnames="x" but no child dataset "x"
    let (ndt_dt, ndt_raw) = fixed_string_dt("DynamicTable");
    let (col_dt, col_raw) = fixed_string_dt("x");
    builder
        .add_group_with_attributes(
            "ghost_table",
            &[
                ("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw),
                ("colnames", &col_dt, &scalar, &col_raw),
            ],
            &[], // no child datasets — column "x" is missing
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    assert!(
        report.violations().iter().any(|v| matches!(
            v,
            crate::validation::ConformanceViolation::DynamicTableColumnMissing {
                group_path,
                column_name,
            } if group_path == "ghost_table" && column_name == "x"
        )),
        "expected DynamicTableColumnMissing(ghost_table, x): {:?}",
        report.violations()
    );
}

/// Layer 6 must NOT report DynamicTableColumnMissing for a properly-written
/// electrode table where all colnames columns are present as child datasets.
#[test]
fn validate_conformance_passes_electrode_table_column_content() {
    let ts = "2023-01-01T00:00:00Z";
    let mut builder = NwbFileBuilder::new("2.7.0", "uid1", "desc", ts).unwrap();
    for group in &[
        "acquisition",
        "analysis",
        "processing",
        "stimulus",
        "general",
    ] {
        builder.write_empty_group(group).unwrap();
    }
    // Write a 1-row electrode table: colnames="location,group_name" with
    // child datasets "id", "location", and "group_name" all present.
    let table = crate::model::electrode::ElectrodeTable::from_rows(alloc::vec![
        crate::model::electrode::ElectrodeRow {
            id: 0,
            location: alloc::string::String::from("CA1"),
            group_name: alloc::string::String::from("tetrode1"),
        },
    ]);
    builder.write_electrode_table(&table).unwrap();
    let bytes = builder.finish().unwrap();
    let file = NwbFile::open(&bytes).unwrap();
    let report = file.validate_conformance().unwrap();
    // No DynamicTableColumnMissing for "location" or "group_name"
    let col_missing: alloc::vec::Vec<_> = report
        .violations()
        .iter()
        .filter(|v| {
            matches!(
                v,
                crate::validation::ConformanceViolation::DynamicTableColumnMissing { .. }
            )
        })
        .collect();
    assert!(
        col_missing.is_empty(),
        "electrode table with all colnames columns present must not trigger Layer 6: {:?}",
        col_missing
    );
    // The entire report must be conformant (no violations at all)
    assert!(
        report.is_conformant(),
        "complete electrode table file must be fully conformant: {:?}",
        report.violations()
    );
}
