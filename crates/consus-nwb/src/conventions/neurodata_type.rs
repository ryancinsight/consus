#[cfg(feature = "alloc")]
use alloc::string::String;

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
        other => NeuroDataType::Other(String::from(other)),
    }
}
