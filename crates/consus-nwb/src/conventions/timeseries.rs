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
            return false;
        }
        match parent_map.get(current) {
            Some(&parent) => current = parent,
            None => return false,
        }
    }

    false
}
