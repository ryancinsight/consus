//! NWB 2.x `Units` DynamicTable model.
//!
//! The `Units` group stores sorted spike unit data in the HDMF VectorData +
//! VectorIndex representation:
//!
//! | HDF5 dataset              | Type    | Description                              |
//! |---------------------------|---------|------------------------------------------|
//! | `Units/spike_times`       | f64[]   | All spike times, flat concatenated       |
//! | `Units/spike_times_index` | u64[]   | Cumulative end offsets (one per unit)    |
//! | `Units/id`                | u64[]   | Optional integer unit identifiers        |
//!
//! ## VectorIndex Invariant
//!
//! For unit `i` (0-indexed):
//!
//! ```text
//! start = (i == 0) ? 0 : spike_times_index[i - 1]
//! end   = spike_times_index[i]
//! spike_times[start..end] → spike times for unit i
//! ```
//!
//! `spike_times_index` is monotonically non-decreasing and its last element
//! equals `spike_times.len()`.

#[cfg(feature = "alloc")]
use alloc::{format, vec::Vec};

use consus_core::{Error, Result};

/// NWB 2.x sorted spike units, decoded from the HDMF VectorData + VectorIndex
/// representation.
///
/// `spike_times_per_unit[i]` contains all spike times (seconds) for unit `i`.
/// The list can be empty (zero units) or contain units with zero spikes.
///
/// ## Invariants
///
/// - If `ids` is `Some`, `ids.len() == spike_times_per_unit.len()`.
/// - All spike times are finite `f64` values (this is not enforced at
///   construction time but is a domain invariant of the NWB specification).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct UnitsTable {
    /// Per-unit spike time arrays (seconds relative to session start).
    spike_times_per_unit: Vec<Vec<f64>>,
    /// Optional integer unit identifiers from `Units/id`.
    ids: Option<Vec<u64>>,
}

#[cfg(feature = "alloc")]
impl UnitsTable {
    /// Construct a `UnitsTable` from per-unit spike time arrays with no IDs.
    #[must_use]
    pub fn new(spike_times_per_unit: Vec<Vec<f64>>) -> Self {
        Self {
            spike_times_per_unit,
            ids: None,
        }
    }

    /// Construct a `UnitsTable` from per-unit spike time arrays and optional IDs.
    ///
    /// ## Errors
    ///
    /// Returns `Err(Error::InvalidFormat)` when `ids` is `Some` and its length
    /// differs from `spike_times_per_unit.len()`.
    pub fn from_parts(spike_times_per_unit: Vec<Vec<f64>>, ids: Option<Vec<u64>>) -> Result<Self> {
        if let Some(ref id_vec) = ids {
            if id_vec.len() != spike_times_per_unit.len() {
                return Err(Error::InvalidFormat {
                    message: format!(
                        "UnitsTable: ids.len()={} does not match spike_times_per_unit.len()={}",
                        id_vec.len(),
                        spike_times_per_unit.len()
                    ),
                });
            }
        }
        Ok(Self {
            spike_times_per_unit,
            ids,
        })
    }

    /// Decode a `UnitsTable` from the flat HDMF VectorData + VectorIndex representation.
    ///
    /// ## Arguments
    ///
    /// - `flat_times` — `Units/spike_times` dataset values (all spike times concatenated).
    /// - `index`      — `Units/spike_times_index` dataset values (cumulative end offsets,
    ///   one per unit; last element must equal `flat_times.len()`).
    /// - `ids`        — Optional `Units/id` dataset (one per unit; `None` when absent).
    ///
    /// ## Errors
    ///
    /// - `Error::InvalidFormat` when:
    ///   - `index` is non-empty and its last element ≠ `flat_times.len()`.
    ///   - `index` is not monotonically non-decreasing.
    ///   - `ids` is `Some` and its length ≠ `index.len()`.
    pub fn from_vectordata(
        flat_times: Vec<f64>,
        index: Vec<u64>,
        ids: Option<Vec<u64>>,
    ) -> Result<Self> {
        // Invariant: last cumulative index == total spike count.
        if let Some(&last) = index.last() {
            if last as usize != flat_times.len() {
                return Err(Error::InvalidFormat {
                    message: format!(
                        "UnitsTable: last spike_times_index {} != spike_times.len() {}",
                        last,
                        flat_times.len()
                    ),
                });
            }
        }

        // Invariant: index is monotonically non-decreasing.
        for i in 1..index.len() {
            if index[i] < index[i - 1] {
                return Err(Error::InvalidFormat {
                    message: format!(
                        "UnitsTable: spike_times_index is not monotone at position {}: \
                         index[{}]={} < index[{}]={}",
                        i,
                        i,
                        index[i],
                        i - 1,
                        index[i - 1]
                    ),
                });
            }
        }

        // Invariant: ids length must match unit count when present.
        if let Some(ref id_vec) = ids {
            if id_vec.len() != index.len() {
                return Err(Error::InvalidFormat {
                    message: format!(
                        "UnitsTable: ids.len()={} does not match index.len()={}",
                        id_vec.len(),
                        index.len()
                    ),
                });
            }
        }

        // Decode VectorIndex pattern into per-unit spike time arrays.
        let n_units = index.len();
        let mut spike_times_per_unit = Vec::with_capacity(n_units);
        let mut prev: usize = 0;
        for &end in &index {
            let end = end as usize;
            spike_times_per_unit.push(flat_times[prev..end].to_vec());
            prev = end;
        }

        Ok(Self {
            spike_times_per_unit,
            ids,
        })
    }

    /// Return the number of units.
    #[must_use]
    pub fn num_units(&self) -> usize {
        self.spike_times_per_unit.len()
    }

    /// Return `true` when there are no units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spike_times_per_unit.is_empty()
    }

    /// Borrow the per-unit spike time arrays.
    #[must_use]
    pub fn spike_times_per_unit(&self) -> &[Vec<f64>] {
        &self.spike_times_per_unit
    }

    /// Borrow the optional unit ID array.
    #[must_use]
    pub fn ids(&self) -> Option<&[u64]> {
        self.ids.as_deref()
    }

    /// Flatten all per-unit spike times into one contiguous array.
    ///
    /// Returns the values in unit order (all spikes of unit 0, then unit 1, …).
    /// This is the wire format for `Units/spike_times`.
    #[must_use]
    pub fn flat_spike_times(&self) -> Vec<f64> {
        self.spike_times_per_unit
            .iter()
            .flatten()
            .copied()
            .collect()
    }

    /// Compute the cumulative end-offset array for `Units/spike_times_index`.
    ///
    /// `result[i]` is the exclusive end index of unit `i` in the flat array.
    /// `result[i] = sum of spike_times_per_unit[0..=i].len()`.
    #[must_use]
    pub fn cumulative_index(&self) -> Vec<u64> {
        let mut idx = Vec::with_capacity(self.spike_times_per_unit.len());
        let mut cumulative: u64 = 0;
        for times in &self.spike_times_per_unit {
            cumulative += times.len() as u64;
            idx.push(cumulative);
        }
        idx
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    // ── new ──────────────────────────────────────────────────────────────

    #[test]
    fn new_stores_spike_times_and_no_ids() {
        let t = UnitsTable::new(vec![vec![0.1, 0.2], vec![1.0]]);
        assert_eq!(t.num_units(), 2);
        assert_eq!(t.spike_times_per_unit()[0], &[0.1f64, 0.2]);
        assert_eq!(t.spike_times_per_unit()[1], &[1.0f64]);
        assert!(t.ids().is_none());
    }

    #[test]
    fn new_empty_table_is_empty() {
        let t = UnitsTable::new(vec![]);
        assert!(t.is_empty());
        assert_eq!(t.num_units(), 0);
    }

    // ── from_parts ───────────────────────────────────────────────────────

    #[test]
    fn from_parts_stores_ids_when_lengths_match() {
        let t = UnitsTable::from_parts(
            vec![vec![0.5f64], vec![1.5f64, 2.0f64]],
            Some(vec![10u64, 11u64]),
        )
        .unwrap();
        assert_eq!(t.ids(), Some([10u64, 11u64].as_slice()));
    }

    #[test]
    fn from_parts_rejects_ids_length_mismatch() {
        let err = UnitsTable::from_parts(
            vec![vec![0.5f64], vec![1.5f64]],
            Some(vec![10u64]), // 1 id for 2 units
        )
        .unwrap_err();
        match err {
            Error::InvalidFormat { message } => {
                assert!(message.contains("ids.len()=1"), "got: {message}");
                assert!(
                    message.contains("spike_times_per_unit.len()=2"),
                    "got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    // ── from_vectordata ──────────────────────────────────────────────────

    #[test]
    fn from_vectordata_decodes_two_units_correctly() {
        // Unit 0: [0.1, 0.2], unit 1: [0.5, 0.6, 0.7]
        let flat = vec![0.1f64, 0.2, 0.5, 0.6, 0.7];
        let index = vec![2u64, 5u64];
        let t = UnitsTable::from_vectordata(flat, index, None).unwrap();
        assert_eq!(t.num_units(), 2);
        assert_eq!(t.spike_times_per_unit()[0], &[0.1f64, 0.2]);
        assert_eq!(t.spike_times_per_unit()[1], &[0.5f64, 0.6, 0.7]);
    }

    #[test]
    fn from_vectordata_three_units_with_ids() {
        let flat = vec![0.1f64, 0.2, 0.3, 1.0, 1.1];
        // unit 0: [0.1,0.2], unit 1: [0.3], unit 2: [1.0,1.1]
        let index = vec![2u64, 3u64, 5u64];
        let ids = Some(vec![7u64, 8u64, 9u64]);
        let t = UnitsTable::from_vectordata(flat, index, ids).unwrap();
        assert_eq!(t.num_units(), 3);
        assert_eq!(t.ids(), Some([7u64, 8u64, 9u64].as_slice()));
        assert_eq!(t.spike_times_per_unit()[0], &[0.1f64, 0.2]);
        assert_eq!(t.spike_times_per_unit()[1], &[0.3f64]);
        assert_eq!(t.spike_times_per_unit()[2], &[1.0f64, 1.1]);
    }

    #[test]
    fn from_vectordata_empty_index_empty_flat_produces_zero_units() {
        let t = UnitsTable::from_vectordata(vec![], vec![], None).unwrap();
        assert_eq!(t.num_units(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn from_vectordata_unit_with_zero_spikes() {
        // A unit with no spikes: index entry equal to previous.
        let flat = vec![0.1f64];
        // unit 0: [0.1], unit 1: [] (index[1] == index[0])
        let index = vec![1u64, 1u64];
        let t = UnitsTable::from_vectordata(flat, index, None).unwrap();
        assert_eq!(t.num_units(), 2);
        assert_eq!(t.spike_times_per_unit()[0], &[0.1f64]);
        assert_eq!(t.spike_times_per_unit()[1], &[] as &[f64]);
    }

    #[test]
    fn from_vectordata_rejects_last_index_mismatch() {
        let err = UnitsTable::from_vectordata(
            vec![0.1f64, 0.2], // 2 times
            vec![3u64],        // last index 3 != 2
            None,
        )
        .unwrap_err();
        match err {
            Error::InvalidFormat { message } => {
                assert!(message.contains("3"), "got: {message}");
                assert!(message.contains("2"), "got: {message}");
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn from_vectordata_rejects_non_monotone_index() {
        let err = UnitsTable::from_vectordata(
            vec![0.1f64, 0.2, 0.3],
            vec![2u64, 1u64, 3u64], // index[1]=1 < index[0]=2 → not monotone
            None,
        )
        .unwrap_err();
        match err {
            Error::InvalidFormat { message } => {
                assert!(
                    message.contains("monotone") || message.contains("index"),
                    "got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn from_vectordata_rejects_ids_length_mismatch() {
        let err = UnitsTable::from_vectordata(
            vec![0.1f64, 0.2],
            vec![2u64],             // 1 unit
            Some(vec![0u64, 1u64]), // 2 ids for 1 unit
        )
        .unwrap_err();
        match err {
            Error::InvalidFormat { message } => {
                assert!(message.contains("ids.len()=2"), "got: {message}");
                assert!(message.contains("index.len()=1"), "got: {message}");
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    // ── flat_spike_times + cumulative_index ───────────────────────────────

    #[test]
    fn flat_spike_times_produces_unit_ordered_flat_array() {
        let t = UnitsTable::new(vec![vec![0.1f64, 0.2], vec![1.0f64]]);
        assert_eq!(t.flat_spike_times(), vec![0.1f64, 0.2, 1.0]);
    }

    #[test]
    fn flat_spike_times_empty_table_returns_empty() {
        let t = UnitsTable::new(vec![]);
        assert_eq!(t.flat_spike_times(), vec![] as Vec<f64>);
    }

    #[test]
    fn cumulative_index_produces_correct_end_offsets() {
        // 2 spikes, 3 spikes, 1 spike
        let t = UnitsTable::new(vec![
            vec![0.1f64, 0.2],
            vec![1.0f64, 1.1, 1.2],
            vec![2.0f64],
        ]);
        assert_eq!(t.cumulative_index(), vec![2u64, 5u64, 6u64]);
    }

    #[test]
    fn cumulative_index_empty_unit_contributes_zero_delta() {
        let t = UnitsTable::new(vec![vec![0.1f64], vec![], vec![2.0f64]]);
        assert_eq!(t.cumulative_index(), vec![1u64, 1u64, 2u64]);
    }

    // ── VectorData/VectorIndex roundtrip ─────────────────────────────────

    /// Invariant: from_vectordata(flat_spike_times(), cumulative_index(), ids) == original.
    #[test]
    fn from_vectordata_roundtrips_from_flat_and_index() {
        let original = UnitsTable::from_parts(
            vec![vec![0.1f64, 0.2, 0.3], vec![1.0f64], vec![2.0f64, 2.1]],
            Some(vec![0u64, 1u64, 2u64]),
        )
        .unwrap();
        let flat = original.flat_spike_times();
        let index = original.cumulative_index();
        let ids = original.ids().map(|s| s.to_vec());
        let restored = UnitsTable::from_vectordata(flat, index, ids).unwrap();
        assert_eq!(original, restored);
    }

    // ── Clone + PartialEq ────────────────────────────────────────────────

    #[test]
    fn clone_produces_equal_independent_copy() {
        let t = UnitsTable::new(vec![vec![0.1f64, 0.2], vec![1.0f64]]);
        let cloned = t.clone();
        assert_eq!(t, cloned);
    }

    #[test]
    fn two_tables_with_different_spikes_are_not_equal() {
        let a = UnitsTable::new(vec![vec![0.1f64]]);
        let b = UnitsTable::new(vec![vec![0.2f64]]);
        assert_ne!(a, b);
    }
}
