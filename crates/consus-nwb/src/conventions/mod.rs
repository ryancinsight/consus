//! NWB namespace and neurodata type resolution.
//!
//! Resolves NWB 2.x namespace definitions and maps HDF5 group attributes
//! (`neurodata_type_def`, `neurodata_type_inc`) to semantic neurodata types.
//!
//! ## Classification
//!
//! `NeuroDataType` covers the most commonly encountered NWB 2.x types.
//! `classify_neurodata_type` maps a `neurodata_type_def` string to the
//! canonical enum variant.
//!
//! ## TimeSeries membership
//!
//! A group is a TimeSeries when:
//! - Its `neurodata_type_def` is `"TimeSeries"`.
//! - Its `neurodata_type_inc` is `"TimeSeries"` (direct single-level inheritance).
//! - Its `neurodata_type_def` is in `TIMESERIES_SUBTYPES`.
//! - Its `neurodata_type_inc` is in `TIMESERIES_SUBTYPES` (two-level transitivity).

#[cfg(feature = "alloc")]
use alloc::string::String;

// ---------------------------------------------------------------------------
// Known TimeSeries subtypes
// ---------------------------------------------------------------------------

/// Complete set of NWB 2.x core types that directly extend `TimeSeries`.
///
/// Source: NWB 2.x core specification, section 3 — neurodata type hierarchy.
/// Each entry is the value of `neurodata_type_def` that implies TimeSeries
/// membership via direct single-level inheritance.
pub const TIMESERIES_SUBTYPES: &[&str] = &[
    "ElectricalSeries",
    "SpikeEventSeries",
    "RoiResponseSeries",
    "SpatialSeries",
    "AbstractFeatureSeries",
    "AnnotationSeries",
    "IntervalSeries",
    "DecompositionSeries",
    "LFP",
    "FilteredEphys",
    "Fluorescence",
    "DfOverF",
    "ImageSeries",
    "ImageMaskSeries",
    "TwoPhotonSeries",
    "OpticalSeries",
    "IndexSeries",
];

// ---------------------------------------------------------------------------
// NeuroDataType
// ---------------------------------------------------------------------------

/// Canonical NWB 2.x neurodata type variants.
///
/// Covers the most frequently encountered types in NWB 2.x core and
/// HDMF-common. Types not enumerated here are represented as `Other`.
///
/// ## Derivation
///
/// Variants map one-to-one to `neurodata_type_def` attribute values as
/// defined in the NWB 2.x core specification. `Other` captures any type
/// not in this enumeration without data loss.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuroDataType {
    /// Top-level NWB file container (`neurodata_type_def = "NWBFile"`).
    NwbFile,
    /// Base TimeSeries type (`neurodata_type_def = "TimeSeries"`).
    TimeSeries,
    /// Extracellular electrophysiology voltage traces.
    ElectricalSeries,
    /// Spike event waveforms.
    SpikeEventSeries,
    /// ROI fluorescence response series.
    RoiResponseSeries,
    /// Spatial position or direction series.
    SpatialSeries,
    /// Abstract feature time series.
    AbstractFeatureSeries,
    /// Text annotation time series.
    AnnotationSeries,
    /// Binary interval (start/stop) time series.
    IntervalSeries,
    /// Spectral decomposition time series.
    DecompositionSeries,
    /// Local field potential series (`neurodata_type_def = "LFP"`).
    LfpSeries,
    /// Generic image time series.
    ImageSeries,
    /// Image mask overlay series.
    ImageMaskSeries,
    /// Sorted spike units table.
    Units,
    /// Electrode metadata table.
    ElectrodeTable,
    /// Subject metadata.
    Subject,
    /// Any neurodata type not enumerated above.
    Other(String),
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Map a `neurodata_type_def` string to the canonical [`NeuroDataType`] variant.
///
/// Unknown type strings map to [`NeuroDataType::Other`] without error,
/// preserving the original string for downstream inspection.
///
/// ## Mapping table
///
/// | `neurodata_type_def`  | Variant              |
/// |-----------------------|----------------------|
/// | `"NWBFile"`           | `NwbFile`            |
/// | `"TimeSeries"`        | `TimeSeries`         |
/// | `"ElectricalSeries"`  | `ElectricalSeries`   |
/// | `"SpikeEventSeries"`  | `SpikeEventSeries`   |
/// | `"RoiResponseSeries"` | `RoiResponseSeries`  |
/// | `"SpatialSeries"`     | `SpatialSeries`      |
/// | `"AbstractFeatureSeries"` | `AbstractFeatureSeries` |
/// | `"AnnotationSeries"`  | `AnnotationSeries`   |
/// | `"IntervalSeries"`    | `IntervalSeries`     |
/// | `"DecompositionSeries"` | `DecompositionSeries` |
/// | `"LFP"`               | `LfpSeries`          |
/// | `"ImageSeries"`       | `ImageSeries`        |
/// | `"ImageMaskSeries"`   | `ImageMaskSeries`    |
/// | `"Units"`             | `Units`              |
/// | `"ElectrodeTable"`    | `ElectrodeTable`     |
/// | `"Subject"`           | `Subject`            |
/// | anything else         | `Other(type_def.to_owned())` |
#[cfg(feature = "alloc")]
pub fn classify_neurodata_type(type_def: &str) -> NeuroDataType {
    match type_def {
        "NWBFile" => NeuroDataType::NwbFile,
        "TimeSeries" => NeuroDataType::TimeSeries,
        "ElectricalSeries" => NeuroDataType::ElectricalSeries,
        "SpikeEventSeries" => NeuroDataType::SpikeEventSeries,
        "RoiResponseSeries" => NeuroDataType::RoiResponseSeries,
        "SpatialSeries" => NeuroDataType::SpatialSeries,
        "AbstractFeatureSeries" => NeuroDataType::AbstractFeatureSeries,
        "AnnotationSeries" => NeuroDataType::AnnotationSeries,
        "IntervalSeries" => NeuroDataType::IntervalSeries,
        "DecompositionSeries" => NeuroDataType::DecompositionSeries,
        "LFP" => NeuroDataType::LfpSeries,
        "ImageSeries" => NeuroDataType::ImageSeries,
        "ImageMaskSeries" => NeuroDataType::ImageMaskSeries,
        "Units" => NeuroDataType::Units,
        "ElectrodeTable" => NeuroDataType::ElectrodeTable,
        "Subject" => NeuroDataType::Subject,
        other => NeuroDataType::Other(other.to_owned()),
    }
}

// ---------------------------------------------------------------------------
// TimeSeries membership
// ---------------------------------------------------------------------------

/// Returns `true` when the neurodata type is a TimeSeries or a known subtype.
///
/// Four conditions independently satisfy membership:
///
/// 1. `type_def == "TimeSeries"` — the group is the base type itself.
/// 2. `type_inc == Some("TimeSeries")` — single-level inheritance declaration.
/// 3. `type_def` is in [`TIMESERIES_SUBTYPES`] — known direct subtypes per
///    the NWB 2.x core specification.
/// 4. `type_inc` is in [`TIMESERIES_SUBTYPES`] — two-level transitivity: the
///    type extends a known TimeSeries subtype (depth-2 inheritance chain).
///    Example: `CustomType → ElectricalSeries → TimeSeries`.
///
/// ## Note on inheritance depth
///
/// This function resolves up to two levels of inheritance. Chains deeper than
/// two levels require spec-guided resolution via
/// [`is_timeseries_type_with_specs`], which consults parsed namespace
/// specifications to attempt one additional resolution step.
pub fn is_timeseries_type(type_def: &str, type_inc: Option<&str>) -> bool {
    if type_def == "TimeSeries" {
        return true;
    }
    if type_inc == Some("TimeSeries") {
        return true;
    }
    if TIMESERIES_SUBTYPES.contains(&type_def) {
        return true;
    }
    // Two-level transitivity: if `type_inc` is itself a known TimeSeries subtype,
    // then this type transitively extends `TimeSeries` (depth-2 inheritance chain).
    // Example: CustomType → ElectricalSeries → TimeSeries.
    if let Some(inc) = type_inc {
        if TIMESERIES_SUBTYPES.contains(&inc) {
            return true;
        }
    }
    false
}

/// Resolve `TimeSeries` membership using parsed namespace specifications.
///
/// Builds a parent-lookup map from all [`NwbTypeSpec`] entries across all
/// provided specs, then walks the `neurodata_type_inc` chain starting from
/// `type_name`. Returns `true` when any node in the chain satisfies
/// [`is_timeseries_type`].
///
/// ## Chain walk
///
/// 1. If `is_timeseries_type(current, None)` → `true`.
/// 2. Look up the declared `neurodata_type_inc` for `current` in the spec map.
///    If absent → `false`.
/// 3. Advance `current` to the parent and repeat.
/// 4. Cycle guard: if `current` was already visited, return `false`.
/// 5. Depth guard: after 64 steps without resolution, return `false`.
///    This bounds iteration on malformed or pathologically deep specs.
///
/// [`NwbTypeSpec`]: crate::namespace::NwbTypeSpec
/// [`NwbNamespaceSpec`]: crate::namespace::NwbNamespaceSpec
#[cfg(feature = "alloc")]
pub fn is_timeseries_type_with_specs(
    type_name: &str,
    specs: &[crate::namespace::NwbNamespaceSpec],
) -> bool {
    use alloc::collections::BTreeMap;
    use alloc::collections::BTreeSet;

    // Build parent lookup: declared type name → neurodata_type_inc.
    let mut parent_map: BTreeMap<&str, &str> = BTreeMap::new();
    for spec in specs {
        for type_spec in &spec.neurodata_types {
            if let Some(ref inc) = type_spec.neurodata_type_inc {
                parent_map.insert(type_spec.name.as_str(), inc.as_str());
            }
        }
    }

    let mut current = type_name;
    let mut visited: BTreeSet<&str> = BTreeSet::new();

    for _ in 0..64_usize {
        if is_timeseries_type(current, None) {
            return true;
        }
        if !visited.insert(current) {
            // Cycle detected.
            return false;
        }
        match parent_map.get(current) {
            Some(&parent) => current = parent,
            None => return false,
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    // ── classify_neurodata_type ───────────────────────────────────────────

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
        // "LFP" maps to LfpSeries (the enum variant name is distinct from the
        // NWB string to avoid a public API name that encodes the variation
        // dimension directly).
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

    // ── is_timeseries_type ────────────────────────────────────────────────

    #[test]
    fn is_timeseries_type_true_for_timeseries_def() {
        // Condition 1: type_def == "TimeSeries".
        assert!(is_timeseries_type("TimeSeries", None));
    }

    #[test]
    fn is_timeseries_type_true_for_electricalseries_def() {
        // Condition 3: type_def in TIMESERIES_SUBTYPES.
        assert!(is_timeseries_type("ElectricalSeries", None));
    }

    #[test]
    fn is_timeseries_type_true_via_type_inc() {
        // Condition 2: type_inc == Some("TimeSeries") — single-level inheritance.
        assert!(is_timeseries_type("MyCustom", Some("TimeSeries")));
    }

    #[test]
    fn is_timeseries_type_false_for_units() {
        // Units is not a TimeSeries.
        assert!(!is_timeseries_type("Units", None));
    }

    #[test]
    fn is_timeseries_type_false_for_nwbfile() {
        // NWBFile is not a TimeSeries.
        assert!(!is_timeseries_type("NWBFile", None));
    }

    #[test]
    fn is_timeseries_type_false_for_unknown_type_with_no_inc() {
        assert!(!is_timeseries_type("UnknownType", None));
    }

    #[test]
    fn is_timeseries_type_false_when_type_inc_is_not_timeseries() {
        // type_inc being some other type does not grant TimeSeries membership.
        assert!(!is_timeseries_type("MyCustom", Some("NWBFile")));
    }

    #[test]
    fn is_timeseries_type_true_for_all_known_subtypes() {
        // Theorem: every member of TIMESERIES_SUBTYPES is a TimeSeries.
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
        // Depth-2: CustomType has type_inc = "ElectricalSeries", which is in TIMESERIES_SUBTYPES.
        assert!(is_timeseries_type("CustomType", Some("ElectricalSeries")));
    }

    #[test]
    fn is_timeseries_type_false_for_unknown_type_with_non_timeseries_inc() {
        // type_inc is "Units" which is not a TimeSeries or known subtype.
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
        // CustomType has neurodata_type_inc = "ElectricalSeries" which is a TimeSeries subtype.
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
        // "Units" is not a TimeSeries type; chain terminates without resolution.
        assert!(!is_timeseries_type_with_specs("CustomType", &[spec]));
    }

    #[test]
    fn is_timeseries_type_with_specs_resolves_arbitrary_depth() {
        // Chain: A → B → C → TimeSeries (depth 4 from A).
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
        // Chain: A → B → C (C has no parent and is not a TimeSeries).
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

    // ── NeuroDataType — Clone / PartialEq ────────────────────────────────

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
}
