//! Core NWB semantic model.
//!
//! Canonical types for the most commonly used NWB neurodata types.
//! This module covers `TimeSeries` — the foundational neurodata type
//! for continuously sampled or event-driven time-varying measurements.
//!
//! ## Specification
//!
//! Reference: *NWB 2.x Specification* — `TimeSeries` neurodata type.
//! <https://nwb-schema.readthedocs.io/en/latest/format.html#timeseries>
//!
//! A `TimeSeries` stores:
//!
//! | Field             | HDF5 path / attribute          | Required | Description                        |
//! |-------------------|--------------------------------|----------|------------------------------------|
//! | `data`            | `{group}/data` dataset         | Yes      | Primary measurement values         |
//! | `timestamps`      | `{group}/timestamps` dataset   | No†      | Per-sample timestamps (seconds)    |
//! | `starting_time`   | `{group}/starting_time` attr   | No†      | Start time of the first sample (s) |
//! | `rate`            | `{group}/starting_time` attr   | No†      | Sampling rate (Hz)                 |
//!
//! † Either `timestamps` OR (`starting_time` + `rate`) must be present.
//!   Files omitting both are technically non-conformant but are accepted by
//!   this reader; the conformance check is performed by the validation module.
//!
//! ## Invariants
//!
//! - `data.len() >= 0` (empty TimeSeries is structurally valid).
//! - If `timestamps` is `Some`, `timestamps.len() == data.len()`.
//! - `rate` is `None` when `timestamps` is `Some` (per NWB spec; one
//!   timing representation is canonical).
//! - `starting_time` is `None` when `timestamps` is `Some`.
//! - The model is independent of HDF5 wire encoding.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::Result;

/// Canonical representation of an NWB 2.x `TimeSeries` neurodata type.
///
/// Stores the primary measurement array and one of two mutually-exclusive
/// timing representations (either explicit per-sample timestamps or a uniform
/// sampling rate with a start time).
///
/// ## Construction
///
/// Use [`TimeSeries::with_timestamps`] when per-sample timestamps are
/// available, or [`TimeSeries::with_rate`] for uniformly sampled data.
///
/// ## Example
///
/// ```
/// # #[cfg(feature = "alloc")] {
/// use consus_nwb::model::TimeSeries;
///
/// let ts = TimeSeries::with_timestamps(
///     "lick_times",
///     vec![0.5, 1.0, 1.3, 2.1],
///     vec![0.5, 1.0, 1.3, 2.1],
/// );
/// assert_eq!(ts.name(), "lick_times");
/// assert_eq!(ts.data().len(), 4);
/// assert!(ts.timestamps().is_some());
/// assert!(ts.rate().is_none());
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSeries {
    /// Name of this TimeSeries within its parent NWB group.
    #[cfg(feature = "alloc")]
    name: String,

    /// Primary measurement values decoded as `f64`.
    ///
    /// The physical unit and conversion factor are not modelled here;
    /// they are roadmap items (`unit` attribute and `conversion` attribute).
    #[cfg(feature = "alloc")]
    data: Vec<f64>,

    /// Optional per-sample timestamps in seconds relative to the session
    /// reference frame.  Present when the HDF5 group contains a `timestamps`
    /// dataset.  Mutually exclusive with `starting_time` + `rate`.
    #[cfg(feature = "alloc")]
    timestamps: Option<Vec<f64>>,

    /// Optional start time of the first sample in seconds relative to the
    /// session reference frame.  Present when the HDF5 group contains a
    /// `starting_time` dataset or attribute and no `timestamps` dataset.
    starting_time: Option<f64>,

    /// Optional uniform sampling rate in Hz.  Present when the HDF5 group
    /// stores a `rate` attribute on the `starting_time` dataset.
    rate: Option<f64>,
}

#[cfg(feature = "alloc")]
impl TimeSeries {
    /// Construct a `TimeSeries` with explicit per-sample timestamps.
    ///
    /// ## Invariant
    ///
    /// `timestamps.len()` should equal `data.len()`.  This invariant is
    /// stated but not enforced at construction time; it is checked by the
    /// validation module.
    ///
    /// `starting_time` and `rate` are set to `None` because the timestamp
    /// representation is canonical when explicit timestamps are present.
    #[must_use]
    pub fn with_timestamps(name: impl Into<String>, data: Vec<f64>, timestamps: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            data,
            timestamps: Some(timestamps),
            starting_time: None,
            rate: None,
        }
    }

    /// Construct a `TimeSeries` with a uniform sampling rate representation.
    ///
    /// `timestamps` is set to `None` because `starting_time` + `rate` is
    /// the canonical representation for uniformly sampled data.
    ///
    /// ## Arguments
    ///
    /// - `name`           — name of the TimeSeries group
    /// - `data`           — primary measurement values
    /// - `starting_time`  — time of the first sample in seconds
    /// - `rate`           — sampling rate in Hz (must be > 0.0 for valid NWB)
    #[must_use]
    pub fn with_rate(
        name: impl Into<String>,
        data: Vec<f64>,
        starting_time: f64,
        rate: f64,
    ) -> Self {
        Self {
            name: name.into(),
            data,
            timestamps: None,
            starting_time: Some(starting_time),
            rate: Some(rate),
        }
    }

    /// Construct a `TimeSeries` with no timing information.
    ///
    /// Produces a structurally valid model object; conformance validation
    /// will flag missing timing as a warning for NWB 2.x strict mode.
    #[must_use]
    pub fn without_timing(name: impl Into<String>, data: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            data,
            timestamps: None,
            starting_time: None,
            rate: None,
        }
    }

    /// Build a `TimeSeries` from its constituent parts.
    ///
    /// Low-level constructor used by the file reader.  Callers are
    /// responsible for ensuring the invariant
    /// `timestamps.as_ref().map(|t| t.len()) == Some(data.len())` when
    /// `timestamps` is `Some`.
    #[must_use]
    pub fn from_parts(
        name: impl Into<String>,
        data: Vec<f64>,
        timestamps: Option<Vec<f64>>,
        starting_time: Option<f64>,
        rate: Option<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            data,
            timestamps,
            starting_time,
            rate,
        }
    }

    /// Return the name of this TimeSeries.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the primary measurement values as a slice.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Return the number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the TimeSeries contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow the optional per-sample timestamps slice.
    ///
    /// Returns `None` when the file uses a uniform sampling rate instead.
    #[must_use]
    pub fn timestamps(&self) -> Option<&[f64]> {
        self.timestamps.as_deref()
    }

    /// Return the optional start time of the first sample in seconds.
    ///
    /// Present only when the file uses a uniform sampling rate representation.
    #[must_use]
    pub fn starting_time(&self) -> Option<f64> {
        self.starting_time
    }

    /// Return the optional sampling rate in Hz.
    ///
    /// Present only when the file uses a uniform sampling rate representation.
    #[must_use]
    pub fn rate(&self) -> Option<f64> {
        self.rate
    }

    /// Returns `true` when the timing representation uses explicit timestamps.
    #[must_use]
    pub fn has_timestamps(&self) -> bool {
        self.timestamps.is_some()
    }

    /// Returns `true` when the timing representation uses a uniform rate.
    #[must_use]
    pub fn has_rate(&self) -> bool {
        self.rate.is_some()
    }

    /// Validate the structural invariants of this TimeSeries.
    ///
    /// Checks:
    /// 1. If `timestamps` is `Some`, its length equals `data.len()`.
    ///
    /// ## Errors
    ///
    /// Returns `Err(Error::InvalidFormat)` when an invariant is violated.
    pub fn validate(&self) -> Result<()> {
        if let Some(ts) = &self.timestamps {
            if ts.len() != self.data.len() {
                return Err(consus_core::Error::InvalidFormat {
                    message: alloc::format!(
                        "NWB TimeSeries '{}': timestamps.len()={} != data.len()={}",
                        self.name,
                        ts.len(),
                        self.data.len()
                    ),
                });
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    // ── with_timestamps ───────────────────────────────────────────────────

    #[test]
    fn with_timestamps_stores_name_data_and_timestamps() {
        let ts = TimeSeries::with_timestamps("lick", vec![0.1, 0.2, 0.3], vec![1.0, 2.0, 3.0]);
        assert_eq!(ts.name(), "lick");
        assert_eq!(ts.data(), &[0.1f64, 0.2, 0.3]);
        assert_eq!(ts.timestamps(), Some([1.0f64, 2.0, 3.0].as_slice()));
        assert!(ts.starting_time().is_none());
        assert!(ts.rate().is_none());
    }

    #[test]
    fn with_timestamps_len_matches_data_len() {
        let ts = TimeSeries::with_timestamps("t", vec![1.0, 2.0], vec![0.0, 0.1]);
        assert_eq!(ts.len(), 2);
        assert!(!ts.is_empty());
    }

    #[test]
    fn with_timestamps_empty_data_and_empty_timestamps() {
        let ts = TimeSeries::with_timestamps("empty", vec![], vec![]);
        assert!(ts.is_empty());
        assert_eq!(ts.len(), 0);
        assert_eq!(ts.timestamps(), Some([].as_slice()));
    }

    // ── with_rate ─────────────────────────────────────────────────────────

    #[test]
    fn with_rate_stores_name_data_starting_time_and_rate() {
        let ts = TimeSeries::with_rate("lfp", vec![0.0, 1.0, -1.0], 0.0, 1000.0);
        assert_eq!(ts.name(), "lfp");
        assert_eq!(ts.data(), &[0.0f64, 1.0, -1.0]);
        assert_eq!(ts.starting_time(), Some(0.0f64));
        assert_eq!(ts.rate(), Some(1000.0f64));
        assert!(ts.timestamps().is_none());
    }

    #[test]
    fn with_rate_has_rate_returns_true() {
        let ts = TimeSeries::with_rate("s", vec![1.0], 0.0, 30_000.0);
        assert!(ts.has_rate());
        assert!(!ts.has_timestamps());
    }

    // ── without_timing ────────────────────────────────────────────────────

    #[test]
    fn without_timing_has_no_timestamps_or_rate() {
        let ts = TimeSeries::without_timing("bare", vec![42.0]);
        assert!(ts.timestamps().is_none());
        assert!(ts.starting_time().is_none());
        assert!(ts.rate().is_none());
        assert!(!ts.has_timestamps());
        assert!(!ts.has_rate());
    }

    // ── from_parts ────────────────────────────────────────────────────────

    #[test]
    fn from_parts_roundtrips_all_fields() {
        let ts = TimeSeries::from_parts("fp", vec![0.5, -0.5], Some(vec![0.0, 0.001]), None, None);
        assert_eq!(ts.name(), "fp");
        assert_eq!(ts.data(), &[0.5f64, -0.5]);
        assert_eq!(ts.timestamps(), Some([0.0f64, 0.001].as_slice()));
        assert!(ts.starting_time().is_none());
        assert!(ts.rate().is_none());
    }

    // ── validate ──────────────────────────────────────────────────────────

    #[test]
    fn validate_ok_when_timestamps_len_matches_data_len() {
        let ts = TimeSeries::with_timestamps("v", vec![1.0, 2.0], vec![0.0, 0.1]);
        ts.validate().unwrap();
    }

    #[test]
    fn validate_ok_when_no_timestamps() {
        let ts = TimeSeries::with_rate("v", vec![1.0, 2.0], 0.0, 100.0);
        ts.validate().unwrap();
    }

    #[test]
    fn validate_error_when_timestamps_len_differs_from_data_len() {
        let ts = TimeSeries::from_parts(
            "bad",
            vec![1.0, 2.0, 3.0],
            Some(vec![0.0, 0.1]), // only 2 timestamps for 3 data points
            None,
            None,
        );
        let err = ts.validate().unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { message } => {
                assert!(
                    message.contains("bad"),
                    "error message should contain TimeSeries name: {message}"
                );
                assert!(
                    message.contains("2") && message.contains("3"),
                    "error message should contain lengths: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    // ── Clone and PartialEq ───────────────────────────────────────────────

    #[test]
    fn clone_produces_equal_independent_copy() {
        let original = TimeSeries::with_timestamps("c", vec![1.0], vec![0.5]);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn equality_fails_on_differing_data() {
        let a = TimeSeries::with_timestamps("t", vec![1.0], vec![0.0]);
        let b = TimeSeries::with_timestamps("t", vec![2.0], vec![0.0]);
        assert_ne!(a, b);
    }

    #[test]
    fn equality_fails_on_differing_name() {
        let a = TimeSeries::without_timing("ts-a", vec![1.0]);
        let b = TimeSeries::without_timing("ts-b", vec![1.0]);
        assert_ne!(a, b);
    }
}
