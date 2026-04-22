//! netCDF-4 model round-trip validation tests.
//!
//! ## Specification Reference
//!
//! Tests validate that the netCDF model layer correctly represents:
//! - Dimension definitions and lookup
//! - Variable descriptors with shape consistency
//! - Group hierarchy with nested scopes
//! - Nested-group dimension inheritance and shadowing
//! - HDF5 bridge mapping derivation
//! - Unlimited dimension semantics
//! - Compression policy attachment
//!
//! These tests exercise construct → validate → inspect round-trips
//! on the in-memory descriptor model, verifying invariant preservation.

use consus_core::{Compression, Datatype, Shape};
use consus_netcdf::{NetcdfDimension, NetcdfGroup, NetcdfModel, NetcdfVariable, ROOT_GROUP_NAME};

// ---------------------------------------------------------------------------
// Basic Descriptor Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test: Construct a variable with fill value and verify the bytes round-trip.
///
/// ## Invariant
///
/// Fill value bytes attached via builder must be retrievable and decodable
/// to the original value.
#[test]
fn scalar_fill_value_roundtrip() {
    let original_value: i32 = 42;
    let fill_bytes = original_value.to_le_bytes().to_vec();

    let var = NetcdfVariable::new(
        String::from("experiment_id"),
        Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: true,
        },
        vec![String::from("x")],
    )
    .with_fill_value(fill_bytes);

    var.validate().unwrap();

    let stored = var.fill_value.as_ref().expect("fill value must be present");
    let restored = i32::from_le_bytes(stored[0..4].try_into().unwrap());
    assert_eq!(restored, 42, "fill value must round-trip exactly");
}

/// Test: Construct a 1D variable descriptor and verify shape consistency.
///
/// ## Invariant
///
/// Variable rank and shape must agree with the declared dimensions.
#[test]
fn fixed_1d_variable_roundtrip() {
    let var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x")],
    )
    .with_shape(Shape::fixed(&[5]));

    var.validate().unwrap();

    assert_eq!(var.rank(), 1);
    assert_eq!(var.dimensions, vec!["x"]);
    let shape = var.shape.as_ref().expect("shape must be present");
    assert_eq!(shape.rank(), 1);
    assert_eq!(shape.num_elements(), 5);
}

/// Test: Construct a 2D variable descriptor and verify shape consistency.
///
/// ## Invariant
///
/// 2D variable with shape [M, N] must have rank 2 and M*N elements.
#[test]
fn fixed_2d_variable_roundtrip() {
    let var = NetcdfVariable::new(
        String::from("matrix"),
        Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: true,
        },
        vec![String::from("rows"), String::from("cols")],
    )
    .with_shape(Shape::fixed(&[3, 4]));

    var.validate().unwrap();

    assert_eq!(var.rank(), 2);
    let shape = var.shape.as_ref().unwrap();
    assert_eq!(shape.rank(), 2);
    assert_eq!(shape.num_elements(), 12);
}

// ---------------------------------------------------------------------------
// Unlimited Dimension Tests
// ---------------------------------------------------------------------------

/// Test: Construct a variable with unlimited flag and verify semantics.
///
/// ## Spec Compliance
///
/// Unlimited variables must reference at least one dimension.
/// The unlimited flag is preserved through construction.
#[test]
fn unlimited_dimension_roundtrip() {
    let mut group = NetcdfGroup::new(String::from("/"));

    group
        .dimensions
        .push(NetcdfDimension::unlimited(String::from("time"), 2));
    group
        .dimensions
        .push(NetcdfDimension::new(String::from("station"), 2));

    let var = NetcdfVariable::new(
        String::from("time_series"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("time"), String::from("station")],
    )
    .unlimited(true)
    .with_shape(Shape::fixed(&[2, 2]));

    group.variables.push(var);

    group.validate().unwrap();

    // Verify unlimited dimension
    let dim = group.dimension("time").unwrap();
    assert!(dim.is_unlimited());
    assert_eq!(dim.size, 2);

    // Verify variable unlimited flag
    let v = group.variable("time_series").unwrap();
    assert!(v.unlimited);
    assert_eq!(v.rank(), 2);
}

// ---------------------------------------------------------------------------
// Group Hierarchy Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test: Construct nested group hierarchy and verify lookup.
///
/// ## Spec Compliance
///
/// netCDF-4 supports hierarchical groups. Nested groups contain their own
/// dimensions and variables.
#[test]
fn group_hierarchy_roundtrip() {
    let mut root = NetcdfGroup::new(String::from("/"));

    let mut observations = NetcdfGroup::new(String::from("observations"));
    let mut station1 = NetcdfGroup::new(String::from("station1"));

    station1
        .dimensions
        .push(NetcdfDimension::new(String::from("level"), 10));
    station1.variables.push(NetcdfVariable::new(
        String::from("pressure"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("level")],
    ));

    observations.groups.push(station1);
    root.groups.push(observations);

    let model = NetcdfModel { root };
    model.validate().unwrap();

    // Verify hierarchy traversal
    assert_eq!(model.root.groups.len(), 1);
    assert_eq!(model.root.groups[0].name, "observations");
    assert_eq!(model.root.groups[0].groups.len(), 1);
    assert_eq!(model.root.groups[0].groups[0].name, "station1");
    assert!(
        model.root.groups[0].groups[0]
            .variable("pressure")
            .is_some()
    );
}

/// Test: Nested groups inherit dimensions declared in ancestor scopes.
///
/// ## Invariant
///
/// A variable declared in a child group may reference dimensions declared
/// in any ancestor group. Validation must succeed when the referenced
/// dimensions are available through lexical scope inheritance.
#[test]
fn nested_group_dimension_inheritance_roundtrip() {
    let mut root = NetcdfGroup::new(String::from("/"));
    root.dimensions
        .push(NetcdfDimension::new(String::from("time"), 4));

    let mut observations = NetcdfGroup::new(String::from("observations"));
    observations.variables.push(
        NetcdfVariable::new(
            String::from("temperature"),
            Datatype::Float {
                bits: core::num::NonZeroUsize::new(32).unwrap(),
                byte_order: consus_core::ByteOrder::LittleEndian,
            },
            vec![String::from("time")],
        )
        .with_shape(Shape::fixed(&[4])),
    );

    root.groups.push(observations);

    let model = NetcdfModel { root };
    model.validate().unwrap();

    let child = model
        .root
        .group("observations")
        .expect("child group must exist");
    let variable = child
        .variable("temperature")
        .expect("variable must exist in child group");

    assert_eq!(variable.dimensions, vec!["time"]);
    assert_eq!(variable.rank(), 1);
    assert_eq!(
        model
            .root
            .dimension("time")
            .expect("root dimension must exist")
            .size,
        4
    );
}

/// Test: Nested groups prefer locally declared dimensions over inherited ones.
///
/// ## Invariant
///
/// If a child group declares a dimension with the same name as an ancestor
/// dimension, variables in the child scope resolve that name to the nearest
/// declaration. Validation must still succeed because shadowing is legal
/// across scopes.
#[test]
fn nested_group_dimension_shadowing_roundtrip() {
    let mut root = NetcdfGroup::new(String::from("/"));
    root.dimensions
        .push(NetcdfDimension::new(String::from("time"), 4));

    let mut observations = NetcdfGroup::new(String::from("observations"));
    observations
        .dimensions
        .push(NetcdfDimension::new(String::from("time"), 2));
    observations.variables.push(
        NetcdfVariable::new(
            String::from("temperature"),
            Datatype::Float {
                bits: core::num::NonZeroUsize::new(32).unwrap(),
                byte_order: consus_core::ByteOrder::LittleEndian,
            },
            vec![String::from("time")],
        )
        .with_shape(Shape::fixed(&[2])),
    );

    root.groups.push(observations);

    let model = NetcdfModel { root };
    model.validate().unwrap();

    let child = model
        .root
        .group("observations")
        .expect("child group must exist");
    let local_dim = child.dimension("time").expect("child dimension must exist");
    let variable = child
        .variable("temperature")
        .expect("variable must exist in child group");

    assert_eq!(
        local_dim.size, 2,
        "child scope must retain its local dimension"
    );
    assert_eq!(variable.dimensions, vec!["time"]);
    assert_eq!(
        variable
            .shape
            .as_ref()
            .expect("shape must be present")
            .num_elements(),
        2
    );
}

// ---------------------------------------------------------------------------
// Variable Attribute Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test: Variable fill value bytes round-trip for multiple types.
///
/// ## Invariant
///
/// Fill value attached to a variable descriptor must preserve exact bytes,
/// including special values such as NaN.
#[test]
fn variable_fill_value_roundtrip() {
    let fill_bytes = f64::NAN.to_le_bytes().to_vec();

    let var = NetcdfVariable::new(
        String::from("data"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x")],
    )
    .with_fill_value(fill_bytes.clone());

    var.validate().unwrap();

    let stored = var.fill_value.as_ref().unwrap();
    assert_eq!(stored.len(), 8);
    let restored = f64::from_le_bytes(stored[..8].try_into().unwrap());
    assert!(restored.is_nan(), "NaN fill value must round-trip");
}

/// Test: Model with convention-compliant root group.
///
/// ## Spec Compliance
///
/// The root group is named "/" per netCDF-4 convention.
#[test]
fn root_group_convention_roundtrip() {
    let model = NetcdfModel::default();
    assert_eq!(model.root.name, ROOT_GROUP_NAME);
    assert!(model.root.is_empty());
    model.validate().unwrap();

    // Add content and re-validate
    let mut model2 = NetcdfModel::default();
    model2
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 100));
    model2.root.variables.push(NetcdfVariable::new(
        String::from("data"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x")],
    ));
    model2.validate().unwrap();
    assert!(!model2.root.is_empty());
}

// ---------------------------------------------------------------------------
// Compression Configuration Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test: Compression policy attached to variable descriptor round-trips.
///
/// ## Invariant
///
/// Compression configuration must be preserved exactly.
#[test]
fn compression_policy_roundtrip() {
    let var = NetcdfVariable::new(
        String::from("compressed_data"),
        Datatype::Integer {
            bits: core::num::NonZeroUsize::new(8).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: false,
        },
        vec![String::from("x")],
    )
    .with_compression(Compression::Deflate { level: 6 });

    var.validate().unwrap();

    match &var.compression {
        Some(Compression::Deflate { level }) => assert_eq!(*level, 6),
        other => panic!("expected Deflate {{ level: 6 }}, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Shape Consistency Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test: Shape rank must match dimension count for valid variables.
///
/// ## Invariant
///
/// validate() rejects shape-dimension rank mismatch.
#[test]
fn shape_rank_mismatch_rejected() {
    let var = NetcdfVariable::new(
        String::from("matrix"),
        Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: true,
        },
        vec![String::from("rows"), String::from("cols")],
    )
    .with_shape(Shape::fixed(&[4, 6]));

    // Valid: rank matches
    var.validate().unwrap();

    // Invalid: shape rank != dimension count
    let bad_var = NetcdfVariable::new(
        String::from("bad"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x"), String::from("y")],
    )
    .with_shape(Shape::fixed(&[10]));

    assert!(
        bad_var.validate().is_err(),
        "shape rank mismatch must be rejected"
    );
}

/// Test: HDF5 bridge mapping for complete model.
///
/// ## Invariant
///
/// Bridge mappings must preserve dimension, variable, and group names.
#[test]
fn bridge_mapping_roundtrip() {
    let mut model = NetcdfModel::default();

    let dim = NetcdfDimension::new(String::from("x"), 100);
    model.root.dimensions.push(dim);

    let var = NetcdfVariable::new(
        String::from("array"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x")],
    );
    model.root.variables.push(var);

    // Validate model
    consus_netcdf::model::bridge::validate_model(&model).unwrap();

    // Map dimension
    let dim_mapping =
        consus_netcdf::model::bridge::map_dimension(&model.root.dimensions[0]).unwrap();
    assert_eq!(dim_mapping.name, "x");
    assert!(!dim_mapping.unlimited);

    // Map variable
    let var_mapping = consus_netcdf::model::bridge::map_variable(&model.root.variables[0]).unwrap();
    assert_eq!(var_mapping.name, "array");
    assert_eq!(var_mapping.dimensions, vec!["x"]);
}
