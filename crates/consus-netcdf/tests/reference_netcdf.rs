//! netCDF-4 reference model validation tests.
//!
//! ## Specification Reference
//!
//! Tests validate model compliance with:
//! - Unidata netCDF-4 specification (dimension, variable, group structure)
//! - CF (Climate and Forecast) conventions (attribute naming)
//! - HDF5 compatibility layer (bridge mapping descriptors)
//!
//! ## Coverage
//!
//! - Reference model construction and validation
//! - netCDF-4 dimension handling (fixed and unlimited)
//! - Variable descriptor correctness
//! - Group hierarchy validation
//! - HDF5 bridge mapping derivation
//! - Convention constant enforcement

use consus_core::{Compression, Datatype, Shape};
use consus_netcdf::{
    NETCDF_CONVENTIONS_VALUE, NetcdfDimension, NetcdfGroup, NetcdfModel, NetcdfVariable,
    ROOT_GROUP_NAME, is_cf_attribute_name, is_recognized_convention, is_valid_name,
};

// ---------------------------------------------------------------------------
// Reference Model Construction Tests
// ---------------------------------------------------------------------------

/// Test: Construct and validate a model representing a small grid dataset.
///
/// ## Spec Compliance
///
/// netCDF-4 files must have a root group with dimensions and variables.
/// Dimensions are declared at group scope and referenced by name in variables.
#[test]
fn construct_small_grid_model() {
    let mut root = NetcdfGroup::new(String::from("/"));

    // Add spatial dimensions
    root.dimensions
        .push(NetcdfDimension::new(String::from("lat"), 180));
    root.dimensions
        .push(NetcdfDimension::new(String::from("lon"), 360));

    // Add coordinate variables
    let lat_var = NetcdfVariable::new(
        String::from("lat"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat")],
    )
    .coordinate_variable(true);

    let lon_var = NetcdfVariable::new(
        String::from("lon"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lon")],
    )
    .coordinate_variable(true);

    // Add data variable
    let temp_var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat"), String::from("lon")],
    );

    root.variables.push(lat_var);
    root.variables.push(lon_var);
    root.variables.push(temp_var);

    let model = NetcdfModel { root };
    model.validate().unwrap();
}

/// Test: Validate dimensions in a constructed model.
///
/// ## Spec Compliance
///
/// Dimensions must have non-empty names and unique names within a group.
#[test]
fn validate_model_dimensions() {
    let mut root = NetcdfGroup::new(String::from("/"));

    root.dimensions
        .push(NetcdfDimension::new(String::from("x"), 10));
    root.dimensions
        .push(NetcdfDimension::new(String::from("y"), 20));

    assert!(!root.dimensions.is_empty(), "model must have dimensions");

    for dim in &root.dimensions {
        assert!(dim.size > 0, "dimension {} must have size > 0", dim.name);
        assert!(!dim.is_unlimited());
        assert!(is_valid_name(&dim.name));
    }

    root.validate().unwrap();
}

/// Test: Validate variables reference declared dimensions.
///
/// ## Spec Compliance
///
/// Variable dimension names should reference dimensions declared in the group.
/// Variable shape must match referenced dimension order.
#[test]
fn validate_model_variables() {
    let mut root = NetcdfGroup::new(String::from("/"));

    root.dimensions
        .push(NetcdfDimension::new(String::from("time"), 100));
    root.dimensions
        .push(NetcdfDimension::new(String::from("station"), 50));

    let var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("time"), String::from("station")],
    )
    .with_shape(Shape::fixed(&[100, 50]));

    root.variables.push(var);

    root.validate().unwrap();

    // Verify variable references valid dimensions
    let v = root.variable("temperature").unwrap();
    assert_eq!(v.rank(), 2);
    for dim_name in &v.dimensions {
        assert!(
            root.dimension(dim_name).is_some(),
            "variable must reference declared dimensions"
        );
    }
}

/// Test: Validate variable datatype categories.
///
/// ## Spec Compliance
///
/// netCDF-4 supports Integer, Float, Boolean, and String datatypes among others.
#[test]
fn validate_variable_datatypes() {
    let float_var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x")],
    );

    assert!(matches!(float_var.datatype, Datatype::Float { .. }));

    let int_var = NetcdfVariable::new(
        String::from("count"),
        Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: true,
        },
        vec![String::from("x")],
    );

    assert!(matches!(int_var.datatype, Datatype::Integer { .. }));

    let bool_var = NetcdfVariable::new(
        String::from("mask"),
        Datatype::Boolean,
        vec![String::from("x")],
    );

    assert!(matches!(bool_var.datatype, Datatype::Boolean));

    let string_var = NetcdfVariable::new(
        String::from("label"),
        Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        },
        vec![String::from("x")],
    );

    assert!(matches!(
        string_var.datatype,
        Datatype::VariableString { .. }
    ));
}

// ---------------------------------------------------------------------------
// HDF5 Bridge Mapping Tests
// ---------------------------------------------------------------------------

/// Test: Map dimensions to HDF5 dimension-scale descriptors.
///
/// ## Spec Compliance
///
/// netCDF-4 dimensions are represented as HDF5 dimension scales.
/// The bridge module derives HDF5 mapping descriptors from netCDF dimensions.
#[test]
fn bridge_dimension_mapping() {
    let dim = NetcdfDimension::new(String::from("time"), 100);
    let mapping = consus_netcdf::model::bridge::map_dimension(&dim).unwrap();

    assert_eq!(mapping.name, "time");
    assert!(!mapping.unlimited);
    assert!(!mapping.object_path.is_empty());
}

/// Test: Map unlimited dimension to HDF5.
///
/// ## Spec Compliance
///
/// Unlimited dimensions must be marked as such in the HDF5 mapping.
#[test]
fn bridge_unlimited_dimension_mapping() {
    let dim = NetcdfDimension::unlimited(String::from("record"), 0);
    let mapping = consus_netcdf::model::bridge::map_dimension(&dim).unwrap();

    assert_eq!(mapping.name, "record");
    assert!(mapping.unlimited);
}

/// Test: Map variables to HDF5 dataset descriptors.
///
/// ## Spec Compliance
///
/// netCDF-4 variables map to HDF5 datasets.
#[test]
fn bridge_variable_mapping() {
    let var = NetcdfVariable::new(
        String::from("pressure"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("time"), String::from("level")],
    );

    let mapping = consus_netcdf::model::bridge::map_variable(&var).unwrap();

    assert_eq!(mapping.name, "pressure");
    assert_eq!(mapping.dimensions, vec!["time", "level"]);
    assert!(!mapping.dataset_path.is_empty());
}

/// Test: Map group to HDF5 group descriptor.
///
/// ## Spec Compliance
///
/// netCDF-4 groups map to HDF5 groups.
#[test]
fn bridge_group_mapping() {
    let group = NetcdfGroup::new(String::from("observations"));
    let mapping = consus_netcdf::model::bridge::map_group(&group).unwrap();

    assert_eq!(mapping.name, "observations");
    assert!(!mapping.group_path.is_empty());
}

/// Test: Validate full model via bridge.
///
/// ## Spec Compliance
///
/// The bridge validates the entire netCDF model can be mapped to HDF5.
#[test]
fn bridge_validate_model() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 10));
    model.root.variables.push(NetcdfVariable::new(
        String::from("data"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x")],
    ));

    consus_netcdf::model::bridge::validate_model(&model).unwrap();
}

/// Test: Bridge mapping rejects invalid dimension (empty name).
///
/// ## Spec Compliance
///
/// The bridge validates dimensions before producing HDF5 mappings.
#[test]
fn bridge_rejects_invalid_dimension() {
    let dim = NetcdfDimension::new(String::from(""), 10);
    assert!(
        consus_netcdf::model::bridge::map_dimension(&dim).is_err(),
        "bridge must reject dimension with empty name"
    );
}

/// Test: Bridge mapping rejects invalid variable (empty name).
///
/// ## Spec Compliance
///
/// The bridge validates variables before producing HDF5 mappings.
#[test]
fn bridge_rejects_invalid_variable() {
    let var = NetcdfVariable::new(String::from(""), Datatype::Boolean, vec![String::from("x")]);
    assert!(
        consus_netcdf::model::bridge::map_variable(&var).is_err(),
        "bridge must reject variable with empty name"
    );
}

// ---------------------------------------------------------------------------
// Unlimited Dimension Tests
// ---------------------------------------------------------------------------

/// Test: Unlimited dimension handling in model.
///
/// ## Spec Compliance
///
/// netCDF-4 unlimited dimensions:
/// - Can grow via resize
/// - Are marked with unlimited flag
/// - Have a current size
#[test]
fn unlimited_dimension_model() {
    let mut root = NetcdfGroup::new(String::from("/"));

    let time_dim = NetcdfDimension::unlimited(String::from("time"), 3);
    root.dimensions.push(time_dim);

    let var = NetcdfVariable::new(
        String::from("data"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("time")],
    )
    .unlimited(true);

    root.variables.push(var);

    let model = NetcdfModel { root };
    model.validate().unwrap();

    // Verify unlimited dimension
    let dim = model.root.dimension("time").unwrap();
    assert!(dim.is_unlimited());
    assert_eq!(dim.size, 3);
}

/// Test: Unlimited dimension resize semantics.
///
/// ## Spec Compliance
///
/// Unlimited dimensions can grow; fixed dimensions reject size changes.
#[test]
fn unlimited_dimension_resize() {
    let mut dim = NetcdfDimension::unlimited(String::from("time"), 0);
    assert_eq!(dim.size, 0);

    dim.resize(5).unwrap();
    assert_eq!(dim.size, 5);

    dim.resize(100).unwrap();
    assert_eq!(dim.size, 100);

    // Fixed dimension rejects resize
    let mut fixed = NetcdfDimension::new(String::from("x"), 10);
    assert!(fixed.resize(20).is_err());
    assert_eq!(fixed.size, 10);

    // Fixed dimension accepts same-size resize
    assert!(fixed.resize(10).is_ok());
    assert_eq!(fixed.size, 10);
}

// ---------------------------------------------------------------------------
// Group Hierarchy Tests
// ---------------------------------------------------------------------------

/// Test: Nested group hierarchy validation.
///
/// ## Spec Compliance
///
/// netCDF-4 supports hierarchical groups. Child groups contain their own
/// dimensions and variables.
#[test]
fn group_hierarchy_validation() {
    let mut root = NetcdfGroup::new(String::from("/"));

    let mut child = NetcdfGroup::new(String::from("observations"));
    child
        .dimensions
        .push(NetcdfDimension::new(String::from("station"), 5));
    child.variables.push(NetcdfVariable::new(
        String::from("pressure"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("station")],
    ));

    root.groups.push(child);

    let model = NetcdfModel { root };
    model.validate().unwrap();

    assert!(!model.root.groups.is_empty());
    assert_eq!(model.root.groups[0].name, "observations");
}

/// Test: Duplicate dimension names rejected in same scope.
///
/// ## Spec Compliance
///
/// Dimension names must be unique within a group scope.
#[test]
fn reject_duplicate_dimensions() {
    let mut root = NetcdfGroup::new(String::from("/"));
    root.dimensions
        .push(NetcdfDimension::new(String::from("x"), 10));
    root.dimensions
        .push(NetcdfDimension::new(String::from("x"), 20));

    assert!(
        root.validate().is_err(),
        "duplicate dimension names must be rejected"
    );
}

/// Test: Group dimension and variable lookup methods.
///
/// ## Spec Compliance
///
/// Groups provide lookup by name for dimensions and variables.
#[test]
fn group_lookup_methods() {
    let mut group = NetcdfGroup::new(String::from("root"));
    group
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 10));
    group
        .dimensions
        .push(NetcdfDimension::new(String::from("y"), 20));
    group.variables.push(NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x"), String::from("y")],
    ));

    assert!(group.dimension("x").is_some());
    assert!(group.dimension("y").is_some());
    assert!(group.dimension("z").is_none());
    assert!(group.variable("temperature").is_some());
    assert!(group.variable("pressure").is_none());

    let x = group.dimension("x").unwrap();
    assert_eq!(x.size, 10);

    let v = group.variable("temperature").unwrap();
    assert_eq!(v.rank(), 2);
}

/// Test: Empty group reports is_empty correctly.
///
/// ## Spec Compliance
///
/// A group with no dimensions, variables, or child groups is empty.
#[test]
fn empty_group_detection() {
    let empty = NetcdfGroup::new(String::from("empty"));
    assert!(empty.is_empty());

    let mut nonempty = NetcdfGroup::new(String::from("nonempty"));
    nonempty
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 1));
    assert!(!nonempty.is_empty());
}

// ---------------------------------------------------------------------------
// Convention Compliance Tests
// ---------------------------------------------------------------------------

/// Test: Convention attribute name compliance.
///
/// ## Spec Compliance
///
/// Global attributes follow naming conventions. CF attributes use specific keys.
#[test]
fn convention_attribute_compliance() {
    // Standard CF attribute names
    assert!(is_cf_attribute_name("standard_name"));
    assert!(is_cf_attribute_name("units"));
    assert!(is_cf_attribute_name("long_name"));

    // Conventions value
    assert!(is_recognized_convention(NETCDF_CONVENTIONS_VALUE));

    // Root group name
    assert_eq!(ROOT_GROUP_NAME, "/");
    let model = NetcdfModel::default();
    assert_eq!(model.root.name, ROOT_GROUP_NAME);
}

/// Test: Model with compression on variables.
///
/// ## Spec Compliance
///
/// netCDF-4 variables can carry compression policy descriptors.
#[test]
fn model_with_compressed_variable() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 1000));

    let var = NetcdfVariable::new(
        String::from("data"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x")],
    )
    .with_compression(Compression::Deflate { level: 4 });

    model.root.variables.push(var);
    model.validate().unwrap();

    let v = model.root.variable("data").unwrap();
    assert!(v.compression.is_some());
    match &v.compression {
        Some(Compression::Deflate { level }) => assert_eq!(*level, 4),
        other => panic!("expected Deflate {{ level: 4 }}, got {:?}", other),
    }
}

/// Test: Model with coordinate variable validation.
///
/// ## Spec Compliance
///
/// Coordinate variables must be rank-1 with name matching their single dimension.
#[test]
fn model_coordinate_variable_validation() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("lat"), 180));

    // Valid coordinate variable
    let valid_coord = NetcdfVariable::new(
        String::from("lat"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat")],
    )
    .coordinate_variable(true);

    model.root.variables.push(valid_coord);
    model.validate().unwrap();

    // Invalid coordinate variable: name mismatch
    let mut bad_model = NetcdfModel::default();
    bad_model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("lat"), 180));

    let bad_coord = NetcdfVariable::new(
        String::from("latitude"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat")],
    )
    .coordinate_variable(true);

    bad_model.root.variables.push(bad_coord);
    assert!(
        bad_model.validate().is_err(),
        "coordinate variable name must match its dimension"
    );
}

/// Test: Model with fill value on variable.
///
/// ## Spec Compliance
///
/// Fill values are raw-byte encodings attached to variable descriptors.
#[test]
fn model_variable_fill_value() {
    let fill_bytes = (-9999.0f64).to_le_bytes().to_vec();

    let var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("time")],
    )
    .with_fill_value(fill_bytes.clone());

    var.validate().unwrap();

    let stored = var.fill_value.as_ref().unwrap();
    let restored = f64::from_le_bytes(stored[..8].try_into().unwrap());
    assert_eq!(restored, -9999.0f64);
}
