use super::*;

#[test]
fn classify_nwbfile_returns_nwbfile_variant() {
    assert_eq!(classify_neurodata_type("NWBFile"), NeuroDataType::NwbFile);
}

#[test]
fn classify_timeseries_returns_timeseries_variant() {
    assert_eq!(
        classify_neurodata_type("TimeSeries"),
        NeuroDataType::TimeSeries
    );
}

#[test]
fn classify_electricalseries_returns_electricalseries_variant() {
    assert_eq!(
        classify_neurodata_type("ElectricalSeries"),
        NeuroDataType::ElectricalSeries
    );
}

#[test]
fn classify_spikeeventseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("SpikeEventSeries"),
        NeuroDataType::SpikeEventSeries
    );
}

#[test]
fn classify_roiresponseseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("RoiResponseSeries"),
        NeuroDataType::RoiResponseSeries
    );
}

#[test]
fn classify_spatialseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("SpatialSeries"),
        NeuroDataType::SpatialSeries
    );
}

#[test]
fn classify_abstractfeatureseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("AbstractFeatureSeries"),
        NeuroDataType::AbstractFeatureSeries
    );
}

#[test]
fn classify_annotationseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("AnnotationSeries"),
        NeuroDataType::AnnotationSeries
    );
}

#[test]
fn classify_intervalseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("IntervalSeries"),
        NeuroDataType::IntervalSeries
    );
}

#[test]
fn classify_decompositionseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("DecompositionSeries"),
        NeuroDataType::DecompositionSeries
    );
}

#[test]
fn classify_lfp_returns_lfpseries_variant() {
    assert_eq!(classify_neurodata_type("LFP"), NeuroDataType::LfpSeries);
}

#[test]
fn classify_imageseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("ImageSeries"),
        NeuroDataType::ImageSeries
    );
}

#[test]
fn classify_imagemaskseries_returns_correct_variant() {
    assert_eq!(
        classify_neurodata_type("ImageMaskSeries"),
        NeuroDataType::ImageMaskSeries
    );
}

#[test]
fn classify_units_returns_units_variant() {
    assert_eq!(classify_neurodata_type("Units"), NeuroDataType::Units);
}

#[test]
fn classify_electrodetable_returns_electrodetable_variant() {
    assert_eq!(
        classify_neurodata_type("ElectrodeTable"),
        NeuroDataType::ElectrodeTable
    );
}

#[test]
fn classify_subject_returns_subject_variant() {
    assert_eq!(classify_neurodata_type("Subject"), NeuroDataType::Subject);
}

#[test]
fn classify_unknown_type_returns_other_with_original_string() {
    let result = classify_neurodata_type("UnknownType");
    assert_eq!(
        result,
        NeuroDataType::Other(alloc::string::String::from("UnknownType"))
    );
}

#[test]
fn classify_empty_string_returns_other() {
    let result = classify_neurodata_type("");
    assert_eq!(
        result,
        NeuroDataType::Other(alloc::string::String::from(""))
    );
}

#[test]
fn is_timeseries_type_true_for_timeseries_def() {
    assert!(is_timeseries_type("TimeSeries", None));
}

#[test]
fn is_timeseries_type_true_for_electricalseries_def() {
    assert!(is_timeseries_type("ElectricalSeries", None));
}

#[test]
fn is_timeseries_type_true_via_type_inc() {
    assert!(is_timeseries_type("MyCustom", Some("TimeSeries")));
}

#[test]
fn is_timeseries_type_false_for_units() {
    assert!(!is_timeseries_type("Units", None));
}

#[test]
fn is_timeseries_type_false_for_nwbfile() {
    assert!(!is_timeseries_type("NWBFile", None));
}

#[test]
fn is_timeseries_type_false_for_unknown_type_with_no_inc() {
    assert!(!is_timeseries_type("UnknownType", None));
}

#[test]
fn is_timeseries_type_false_when_type_inc_is_not_timeseries() {
    assert!(!is_timeseries_type("MyCustom", Some("NWBFile")));
}

#[test]
fn is_timeseries_type_true_for_all_known_subtypes() {
    for &subtype in TIMESERIES_SUBTYPES {
        assert!(
            is_timeseries_type(subtype, None),
            "expected {subtype} to be a TimeSeries subtype"
        );
    }
}

#[test]
fn is_timeseries_type_true_for_spatialseries() {
    assert!(is_timeseries_type("SpatialSeries", None));
}

#[test]
fn is_timeseries_type_true_for_decompositionseries() {
    assert!(is_timeseries_type("DecompositionSeries", None));
}

#[test]
fn is_timeseries_type_true_via_two_level_transitivity() {
    assert!(is_timeseries_type("CustomType", Some("ElectricalSeries")));
}

#[test]
fn is_timeseries_type_false_for_unknown_type_with_non_timeseries_inc() {
    assert!(!is_timeseries_type("CustomType", Some("Units")));
}

#[test]
fn is_timeseries_type_with_specs_returns_true_when_flat_check_passes() {
    let specs: &[crate::namespace::NwbNamespaceSpec] = &[];
    assert!(is_timeseries_type_with_specs("TimeSeries", specs));
}

#[test]
fn is_timeseries_type_with_specs_returns_true_via_spec_declared_type() {
    let spec = crate::namespace::NwbNamespaceSpec {
        name: alloc::string::String::from("core"),
        version: alloc::string::String::from("2.8.0"),
        doc_url: None,
        neurodata_types: vec![crate::namespace::NwbTypeSpec {
            name: alloc::string::String::from("CustomType"),
            neurodata_type_inc: Some(alloc::string::String::from("ElectricalSeries")),
        }],
    };
    assert!(is_timeseries_type_with_specs("CustomType", &[spec]));
}

#[test]
fn is_timeseries_type_with_specs_returns_false_for_non_timeseries_inc() {
    let spec = crate::namespace::NwbNamespaceSpec {
        name: alloc::string::String::from("core"),
        version: alloc::string::String::from("2.8.0"),
        doc_url: None,
        neurodata_types: vec![crate::namespace::NwbTypeSpec {
            name: alloc::string::String::from("CustomType"),
            neurodata_type_inc: Some(alloc::string::String::from("Units")),
        }],
    };
    assert!(!is_timeseries_type_with_specs("CustomType", &[spec]));
}

#[test]
fn is_timeseries_type_with_specs_resolves_arbitrary_depth() {
    let spec = crate::namespace::NwbNamespaceSpec {
        name: alloc::string::String::from("custom"),
        version: alloc::string::String::from("1.0.0"),
        doc_url: None,
        neurodata_types: vec![
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("A"),
                neurodata_type_inc: Some(alloc::string::String::from("B")),
            },
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("B"),
                neurodata_type_inc: Some(alloc::string::String::from("C")),
            },
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("C"),
                neurodata_type_inc: Some(alloc::string::String::from("TimeSeries")),
            },
        ],
    };
    assert!(is_timeseries_type_with_specs("A", &[spec]));
}

#[test]
fn is_timeseries_type_with_specs_returns_false_for_unrelated_chain() {
    let spec = crate::namespace::NwbNamespaceSpec {
        name: alloc::string::String::from("custom"),
        version: alloc::string::String::from("1.0.0"),
        doc_url: None,
        neurodata_types: vec![
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("A"),
                neurodata_type_inc: Some(alloc::string::String::from("B")),
            },
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("B"),
                neurodata_type_inc: Some(alloc::string::String::from("C")),
            },
            crate::namespace::NwbTypeSpec {
                name: alloc::string::String::from("C"),
                neurodata_type_inc: None,
            },
        ],
    };
    assert!(!is_timeseries_type_with_specs("A", &[spec]));
}

#[test]
fn neurodata_type_clone_equals_original() {
    let original = NeuroDataType::ElectricalSeries;
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn neurodata_type_other_clone_equals_original() {
    let original = NeuroDataType::Other(alloc::string::String::from("CustomType"));
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn neurodata_type_different_variants_not_equal() {
    assert_ne!(NeuroDataType::TimeSeries, NeuroDataType::ElectricalSeries);
}
