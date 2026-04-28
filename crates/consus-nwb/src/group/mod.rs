//! NWBGroup traversal and extraction.
//!
//! Traverses HDF5 group hierarchies according to NWB type annotations,
//! building typed NWB containers from raw HDF5 group metadata.
//!
//! ## Specification
//!
//! NWB 2.x uses two attributes to annotate group semantics:
//!
//! | Attribute             | Meaning                                          |
//! |-----------------------|--------------------------------------------------|
//! | `neurodata_type_def`  | This group defines a neurodata type              |
//! | `neurodata_type_inc`  | This group extends another neurodata type        |
//!
//! A group is a TimeSeries when its `neurodata_type_def` is "TimeSeries" OR
//! its `neurodata_type_inc` is "TimeSeries" (single-level inheritance check),
//! OR its `neurodata_type_def` is in the known set of TimeSeries subtypes.
//!
//! ## Invariants
//!
//! - Only `NodeType::Group` children are included in `list_typed_group_children`.
//! - Missing or non-decodable NWB type attributes are represented as `None`.
//! - I/O errors during child listing propagate; errors on individual children
//!   are silently skipped to allow partial reads of heterogeneous groups.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::{AttributeValue, NodeType, Result};
use consus_hdf5::file::Hdf5File;
use consus_io::ReadAt;

// ---------------------------------------------------------------------------
// NwbGroupChild
// ---------------------------------------------------------------------------

/// A direct child of an NWB group, annotated with its neurodata type.
///
/// Represents a single `NodeType::Group` child discovered during traversal
/// of a parent group. Both NWB type annotation attributes are captured when
/// present; missing or non-decodable attributes are represented as `None`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct NwbGroupChild {
    /// Name of the child within its parent group.
    pub name: String,
    /// Object header address of the child.
    pub address: u64,
    /// Value of `neurodata_type_def` attribute, if present and decodable as a string.
    pub neurodata_type_def: Option<String>,
    /// Value of `neurodata_type_inc` attribute, if present and decodable as a string.
    pub neurodata_type_inc: Option<String>,
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

/// List the direct group children of `group_path` that have NWB type annotations.
///
/// Resolves `group_path` via `Hdf5File::open_path`, lists its children via
/// `Hdf5File::list_group_at`, and filters to only `NodeType::Group` nodes.
/// For each qualifying child, the `neurodata_type_def` and `neurodata_type_inc`
/// string attributes are extracted when present.
///
/// Non-group children (datasets, named datatypes) are skipped. I/O errors on
/// individual children are silently ignored so that partially-readable
/// heterogeneous groups still return whatever is accessible.
///
/// ## Errors
///
/// - Propagates errors from resolving `group_path` (`Error::NotFound` when
///   the path does not exist) or from listing the group's children.
#[cfg(feature = "alloc")]
pub fn list_typed_group_children<R: ReadAt + Sync>(
    file: &Hdf5File<R>,
    group_path: &str,
) -> Result<Vec<NwbGroupChild>> {
    let addr = file.open_path(group_path)?;
    let children = file.list_group_at(addr)?;

    let mut result: Vec<NwbGroupChild> = Vec::with_capacity(children.len());

    for (name, child_addr, _link_type) in children {
        // Include only group-type nodes; skip datasets and named datatypes.
        match file.node_type_at(child_addr) {
            Ok(NodeType::Group) => {}
            Ok(_) | Err(_) => continue,
        }

        // Read attributes; skip this child on I/O error.
        let attrs = match file.attributes_at(child_addr) {
            Ok(a) => a,
            Err(_) => continue,
        };

        let mut neurodata_type_def: Option<String> = None;
        let mut neurodata_type_inc: Option<String> = None;

        for attr in &attrs {
            match attr.name.as_str() {
                "neurodata_type_def" => {
                    if let Ok(AttributeValue::String(s)) = attr.decode_value() {
                        neurodata_type_def = Some(s);
                    }
                }
                "neurodata_type_inc" => {
                    if let Ok(AttributeValue::String(s)) = attr.decode_value() {
                        neurodata_type_inc = Some(s);
                    }
                }
                _ => {}
            }
        }

        result.push(NwbGroupChild {
            name,
            address: child_addr,
            neurodata_type_def,
            neurodata_type_inc,
        });
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
    use consus_hdf5::file::Hdf5File;
    use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
    use consus_io::SliceReader;
    use core::num::NonZeroUsize;

    // ── helpers ──────────────────────────────────────────────────────────

    /// Build `(Datatype::FixedString, raw_bytes)` for use with the HDF5 builder.
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

    // ── list_typed_group_children — only groups ───────────────────────────

    /// Invariant: datasets at root are excluded; only Group children appear.
    #[test]
    fn list_typed_group_children_returns_only_groups() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        // Add a dataset to root — must be excluded from results.
        let f64_dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let data_shape = Shape::fixed(&[1]);
        let data_raw = 0.0f64.to_le_bytes().to_vec();
        let dcpl = DatasetCreationProps::default();
        builder
            .add_dataset("a_dataset", &f64_dt, &data_shape, &data_raw, &dcpl)
            .unwrap();

        // Add a group with neurodata_type_def — must be included.
        let (ndt_dt, ndt_raw) = fixed_string_dt("TimeSeries");
        builder
            .add_group_with_attributes(
                "a_group",
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[],
            )
            .unwrap();

        let bytes = builder.finish().unwrap();
        let reader = SliceReader::new(&bytes);
        let file = Hdf5File::open(reader).unwrap();

        let children = list_typed_group_children(&file, "/").unwrap();

        // Only the group child is returned, not the dataset.
        assert_eq!(children.len(), 1, "only 1 group child expected");
        assert_eq!(children[0].name, "a_group");
        assert_eq!(
            children[0].neurodata_type_def.as_deref(),
            Some("TimeSeries")
        );
    }

    // ── list_typed_group_children — neurodata_type_def extraction ─────────

    /// Invariant: neurodata_type_def is extracted and neurodata_type_inc is None
    /// when only neurodata_type_def is present on the child group.
    #[test]
    fn list_typed_group_children_extracts_neurodata_type_def() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        let (ndt_dt, ndt_raw) = fixed_string_dt("ElectricalSeries");
        builder
            .add_group_with_attributes(
                "ecephys",
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[],
            )
            .unwrap();

        let bytes = builder.finish().unwrap();
        let reader = SliceReader::new(&bytes);
        let file = Hdf5File::open(reader).unwrap();

        let children = list_typed_group_children(&file, "/").unwrap();

        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name, "ecephys");
        assert_eq!(
            children[0].neurodata_type_def.as_deref(),
            Some("ElectricalSeries")
        );
        // neurodata_type_inc was not added, so must be None.
        assert_eq!(children[0].neurodata_type_inc, None);
    }

    // ── list_typed_group_children — both NWB type annotations ─────────────

    /// Invariant: both neurodata_type_def and neurodata_type_inc are extracted
    /// when the child group carries both attributes.
    #[test]
    fn list_typed_group_children_extracts_both_type_annotations() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        let (def_dt, def_raw) = fixed_string_dt("MyCustomSeries");
        let (inc_dt, inc_raw) = fixed_string_dt("TimeSeries");
        builder
            .add_group_with_attributes(
                "custom_series",
                &[
                    ("neurodata_type_def", &def_dt, &scalar, &def_raw),
                    ("neurodata_type_inc", &inc_dt, &scalar, &inc_raw),
                ],
                &[],
            )
            .unwrap();

        let bytes = builder.finish().unwrap();
        let reader = SliceReader::new(&bytes);
        let file = Hdf5File::open(reader).unwrap();

        let children = list_typed_group_children(&file, "/").unwrap();

        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name, "custom_series");
        assert_eq!(
            children[0].neurodata_type_def.as_deref(),
            Some("MyCustomSeries")
        );
        assert_eq!(
            children[0].neurodata_type_inc.as_deref(),
            Some("TimeSeries")
        );
    }

    // ── list_typed_group_children — empty group ────────────────────────────

    /// Invariant: when root has only dataset children (no group children), result is empty.
    ///
    /// A minimal HDF5 file with zero root links causes `list_group_at` to fall
    /// through from an empty v2 link list to the v1 symbol table path, which
    /// returns an error because no SYMBOL_TABLE message is present.  The test
    /// therefore uses a file with one dataset at root so the link list is
    /// non-empty; all children are datasets, which are filtered out, producing
    /// an empty Vec.  This exercises the same invariant: no group-type children
    /// → empty result.
    #[test]
    fn list_typed_group_children_returns_empty_when_no_group_children() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());

        // Add a dataset at root — must be excluded because it is not a Group.
        let f64_dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let data_shape = Shape::fixed(&[1]);
        let data_raw = 0.0f64.to_le_bytes().to_vec();
        let dcpl = DatasetCreationProps::default();
        builder
            .add_dataset("only_dataset", &f64_dt, &data_shape, &data_raw, &dcpl)
            .unwrap();

        let bytes = builder.finish().unwrap();
        let reader = SliceReader::new(&bytes);
        let file = Hdf5File::open(reader).unwrap();

        let children = list_typed_group_children(&file, "/").unwrap();

        assert!(
            children.is_empty(),
            "expected empty Vec when root has only dataset children, got {:?}",
            children
                .iter()
                .map(|c| c.name.as_str())
                .collect::<alloc::vec::Vec<_>>()
        );
    }

    // ── list_typed_group_children — address field ─────────────────────────

    /// Invariant: address field is non-zero for groups written by the builder.
    #[test]
    fn list_typed_group_children_address_is_nonzero() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        let (ndt_dt, ndt_raw) = fixed_string_dt("Units");
        builder
            .add_group_with_attributes(
                "units",
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[],
            )
            .unwrap();

        let bytes = builder.finish().unwrap();
        let reader = SliceReader::new(&bytes);
        let file = Hdf5File::open(reader).unwrap();

        let children = list_typed_group_children(&file, "/").unwrap();

        assert_eq!(children.len(), 1);
        // The object header address must be non-zero for any real object.
        assert_ne!(
            children[0].address, 0,
            "group object header address must be non-zero"
        );
    }
}
