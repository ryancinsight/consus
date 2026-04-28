//! NWB 2.x `electrodes` DynamicTable model.
//!
//! The `electrodes` table stores per-electrode metadata.  Each row has a
//! unique integer ID, a brain-region location string, and an electrode
//! group name string.
//!
//! ## NWB Specification
//!
//! Reference: NWB 2.x core specification — `electrodes` DynamicTable.
//! Required columns: `id`, `location`, `group_name`.
//!
//! | HDF5 path                 | Type      | Description                    |
//! |---------------------------|-----------|--------------------------------|
//! | `electrodes/id`           | u64[]     | Electrode integer IDs          |
//! | `electrodes/location`     | string[]  | Brain region / channel label   |
//! | `electrodes/group_name`   | string[]  | Electrode group name           |

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec::Vec};

use consus_core::{Error, Result};

/// Single row of the NWB 2.x `electrodes` DynamicTable.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElectrodeRow {
    /// Integer electrode identifier (from `electrodes/id`).
    pub id: u64,
    /// Brain region or channel location (e.g., `"CA1"`, `"prefrontal cortex"`).
    pub location: String,
    /// Name of the electrode group this electrode belongs to
    /// (e.g., `"tetrode1"`, `"shank0"`).
    pub group_name: String,
}

/// NWB 2.x `electrodes` DynamicTable.
///
/// Stores per-electrode metadata as an ordered list of [`ElectrodeRow`]s.
/// Rows are in the order they appear in the HDF5 table (row index 0-based).
///
/// ## Invariants
///
/// - `id` values need not be contiguous but must be non-negative (u64).
/// - `location` and `group_name` are arbitrary non-empty-by-convention UTF-8
///   strings; the model does not enforce non-empty.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElectrodeTable {
    rows: Vec<ElectrodeRow>,
}

#[cfg(feature = "alloc")]
impl ElectrodeTable {
    /// Construct an `ElectrodeTable` from a vector of rows.
    #[must_use]
    pub fn from_rows(rows: Vec<ElectrodeRow>) -> Self {
        Self { rows }
    }

    /// Construct an empty `ElectrodeTable`.
    #[must_use]
    pub fn empty() -> Self {
        Self { rows: Vec::new() }
    }

    /// Build an `ElectrodeTable` from three parallel arrays of equal length.
    ///
    /// ## Errors
    ///
    /// Returns `Err(Error::InvalidFormat)` when `ids`, `locations`, and
    /// `group_names` do not have equal lengths.
    pub fn from_columns(
        ids: Vec<u64>,
        locations: Vec<String>,
        group_names: Vec<String>,
    ) -> Result<Self> {
        let n = ids.len();
        if locations.len() != n || group_names.len() != n {
            return Err(Error::InvalidFormat {
                message: format!(
                    "ElectrodeTable: column length mismatch: ids={}, location={}, group_name={}",
                    n,
                    locations.len(),
                    group_names.len()
                ),
            });
        }
        let rows = ids
            .into_iter()
            .zip(locations)
            .zip(group_names)
            .map(|((id, location), group_name)| ElectrodeRow {
                id,
                location,
                group_name,
            })
            .collect();
        Ok(Self { rows })
    }

    /// Return the number of electrode rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Return `true` when the table contains no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Borrow all rows as a slice.
    #[must_use]
    pub fn rows(&self) -> &[ElectrodeRow] {
        &self.rows
    }

    /// Return a reference to the row at index `i`, or `None` when out of bounds.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&ElectrodeRow> {
        self.rows.get(i)
    }

    /// Return an iterator over electrode IDs in row order.
    ///
    /// Column representation without heap allocation beyond existing storage.
    pub fn id_column(&self) -> impl Iterator<Item = u64> + '_ {
        self.rows.iter().map(|r| r.id)
    }

    /// Return an iterator over location strings.
    pub fn location_column(&self) -> impl Iterator<Item = &str> {
        self.rows.iter().map(|r| r.location.as_str())
    }

    /// Return an iterator over group_name strings.
    pub fn group_name_column(&self) -> impl Iterator<Item = &str> {
        self.rows.iter().map(|r| r.group_name.as_str())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    fn row(id: u64, loc: &str, grp: &str) -> ElectrodeRow {
        ElectrodeRow {
            id,
            location: loc.to_owned(),
            group_name: grp.to_owned(),
        }
    }

    // ── from_rows ────────────────────────────────────────────────────────

    #[test]
    fn from_rows_stores_rows_in_order() {
        let tbl = ElectrodeTable::from_rows(vec![row(0, "CA1", "tet0"), row(1, "DG", "tet1")]);
        assert_eq!(tbl.len(), 2);
        assert_eq!(tbl.rows()[0].id, 0);
        assert_eq!(tbl.rows()[0].location, "CA1");
        assert_eq!(tbl.rows()[1].group_name, "tet1");
    }

    // ── empty ────────────────────────────────────────────────────────────

    #[test]
    fn empty_table_is_empty() {
        let tbl = ElectrodeTable::empty();
        assert!(tbl.is_empty());
        assert_eq!(tbl.len(), 0);
    }

    // ── from_columns ─────────────────────────────────────────────────────

    #[test]
    fn from_columns_builds_rows_from_parallel_arrays() {
        let tbl = ElectrodeTable::from_columns(
            vec![0u64, 1u64, 2u64],
            vec!["CA1".to_owned(), "CA1".to_owned(), "DG".to_owned()],
            vec!["tet0".to_owned(), "tet1".to_owned(), "tet2".to_owned()],
        )
        .unwrap();
        assert_eq!(tbl.len(), 3);
        assert_eq!(tbl.rows()[2].id, 2);
        assert_eq!(tbl.rows()[2].location, "DG");
        assert_eq!(tbl.rows()[2].group_name, "tet2");
    }

    #[test]
    fn from_columns_rejects_length_mismatch() {
        let err = ElectrodeTable::from_columns(
            vec![0u64, 1u64],                           // 2 ids
            vec!["CA1".to_owned()],                     // 1 location
            vec!["tet0".to_owned(), "tet1".to_owned()], // 2 group_names
        )
        .unwrap_err();
        match err {
            Error::InvalidFormat { message } => {
                assert!(message.contains("ids=2"), "got: {message}");
                assert!(message.contains("location=1"), "got: {message}");
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    // ── get ──────────────────────────────────────────────────────────────

    #[test]
    fn get_returns_correct_row_at_valid_index() {
        let tbl = ElectrodeTable::from_rows(vec![row(42, "PFC", "shank0")]);
        let r = tbl.get(0).unwrap();
        assert_eq!(r.id, 42);
        assert_eq!(r.location, "PFC");
        assert_eq!(r.group_name, "shank0");
    }

    #[test]
    fn get_returns_none_for_out_of_bounds_index() {
        let tbl = ElectrodeTable::empty();
        assert!(tbl.get(0).is_none());
    }

    // ── column iterators ─────────────────────────────────────────────────

    #[test]
    fn id_column_iterates_ids_in_order() {
        let tbl = ElectrodeTable::from_rows(vec![row(0, "A", "g"), row(5, "B", "g")]);
        let ids: Vec<u64> = tbl.id_column().collect();
        assert_eq!(ids, vec![0u64, 5u64]);
    }

    #[test]
    fn location_column_iterates_locations_in_order() {
        let tbl = ElectrodeTable::from_rows(vec![row(0, "CA1", "t0"), row(1, "DG", "t1")]);
        let locs: Vec<&str> = tbl.location_column().collect();
        assert_eq!(locs, vec!["CA1", "DG"]);
    }

    #[test]
    fn group_name_column_iterates_group_names_in_order() {
        let tbl = ElectrodeTable::from_rows(vec![row(0, "A", "tet0"), row(1, "B", "tet1")]);
        let grps: Vec<&str> = tbl.group_name_column().collect();
        assert_eq!(grps, vec!["tet0", "tet1"]);
    }

    // ── Clone + PartialEq ────────────────────────────────────────────────

    #[test]
    fn clone_produces_equal_independent_copy() {
        let original = ElectrodeTable::from_rows(vec![row(0, "CA1", "tet0")]);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn tables_with_different_rows_are_not_equal() {
        let a = ElectrodeTable::from_rows(vec![row(0, "CA1", "tet0")]);
        let b = ElectrodeTable::from_rows(vec![row(1, "DG", "tet1")]);
        assert_ne!(a, b);
    }

    // ── ElectrodeRow — Debug ──────────────────────────────────────────────

    #[test]
    fn electrode_row_debug_contains_id_and_location() {
        let r = row(7, "CA3", "shank1");
        let dbg = format!("{:?}", r);
        assert!(dbg.contains("7"), "got: {dbg}");
        assert!(dbg.contains("CA3"), "got: {dbg}");
    }
}
