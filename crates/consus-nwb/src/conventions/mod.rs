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

mod neurodata_type;
mod timeseries;

pub use neurodata_type::{classify_neurodata_type, NeuroDataType};
pub use timeseries::{is_timeseries_type, is_timeseries_type_with_specs, TIMESERIES_SUBTYPES};

#[cfg(all(test, feature = "alloc"))]
mod tests;
