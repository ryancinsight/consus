//! netCDF-4 HDF5 write path round-trip integration tests.
//!
//! ## Specification
//!
//! These tests validate the full write → read round-trip:
//!   `NetcdfWriter::write_model(&model)` → HDF5 bytes
//!   → `Hdf5File::open` → `extract_group` → `NetcdfGroup`
//!
//! Each test asserts on computed values: dimension names, sizes, variable
//! names, shapes, dimension-binding correctness, and attribute values.
//!
//! ## Invariants under test
//!
//! - An empty model produces valid HDF5 bytes that parse to an empty group.
//! - Dimension name and size are preserved through write → extract_group.
//! - `DIMENSION_LIST` addresses resolve correctly to dimension scale names.
//! - Multi-dimension, multi-variable bindings are axis-order preserving.
//! - `_nc_properties` root attribute is present with the canonical value.
//! - Scalar variables (rank 0) carry no `DIMENSION_LIST` attribute.
//! - String-valued CF attributes (e.g. `units`) are preserved round-trip.

use consus_core::{AttributeValue, ByteOrder, Datatype, Shape};
use consus_hdf5::file::Hdf5File;
use consus_io::SliceReader;
use consus_netcdf::{
    NetcdfDimension, NetcdfGroup, NetcdfModel, NetcdfVariable, NetcdfWriter, extract_group,
};
use core::num::NonZeroUsize;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn f32_dt() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

fn i32_dt() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    }
}

/// Write `model` → HDF5 bytes → open → extract root group.
///
/// Panics with a descriptive message if any step fails.
fn write_and_extract(model: &NetcdfModel) -> NetcdfGroup {
    let bytes = NetcdfWriter::new()
        .write_model(model)
        .expect("NetcdfWriter::write_model must succeed");
    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("Hdf5File::open must succeed");
    let root = file.root_group();
    extract_group(&file, root.path, root.object_header_address).expect("extract_group must succeed")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// An empty model produces valid HDF5 bytes that can be opened and extracted.
///
/// ## Invariant
///
/// `NetcdfModel::default()` → write → extract yields an empty group:
/// zero dimensions, zero variables, zero sub-groups.
#[test]
fn write_empty_model_produces_valid_hdf5() {
    let model = NetcdfModel::default();
    let group = write_and_extract(&model);

    assert!(
        group.dimensions.is_empty(),
        "empty model must yield zero dimensions; got {:?}",
        group.dimensions
    );
    assert!(
        group.variables.is_empty(),
        "empty model must yield zero variables; got {:?}",
        group.variables
    );
    assert!(
        group.groups.is_empty(),
        "empty model must yield zero sub-groups; got {:?}",
        group.groups
    );
}

/// A single dimension round-trips with the correct name and size.
///
/// ## Invariant
///
/// Dimension name and size are preserved through `write_model` →
/// `extract_group`: `dim.name == "time"`, `dim.size == 10`.
#[test]
fn write_single_dimension_roundtrip() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("time"), 10));

    let group = write_and_extract(&model);

    assert_eq!(
        group.dimensions.len(),
        1,
        "must extract exactly one dimension"
    );
    assert_eq!(
        group.dimensions[0].name, "time",
        "dimension name must match"
    );
    assert_eq!(group.dimensions[0].size, 10, "dimension size must match");
    assert_eq!(group.variables.len(), 0, "no variables were written");
}

/// One dimension + one variable: DIMENSION_LIST resolves to the correct name.
///
/// ## Invariant
///
/// The variable's `dimensions` field equals `["time"]` after extraction,
/// confirming that the DIMENSION_LIST object-reference address was resolved
/// to the written dimension scale.
#[test]
fn write_dimension_and_variable_roundtrip() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("time"), 5));
    model.root.variables.push(
        NetcdfVariable::new(
            String::from("temperature"),
            f32_dt(),
            vec![String::from("time")],
        )
        .with_shape(Shape::fixed(&[5])),
    );

    let group = write_and_extract(&model);

    assert_eq!(group.dimensions.len(), 1, "one dimension");
    assert_eq!(group.dimensions[0].name, "time");
    assert_eq!(group.dimensions[0].size, 5);

    assert_eq!(group.variables.len(), 1, "one variable");
    assert_eq!(group.variables[0].name, "temperature");
    assert_eq!(
        group.variables[0].dimensions,
        vec![String::from("time")],
        "DIMENSION_LIST must resolve to 'time'"
    );

    let shape = group.variables[0]
        .shape
        .as_ref()
        .expect("variable shape must be set after extraction");
    assert_eq!(shape.rank(), 1, "variable rank must be 1");
    assert_eq!(
        shape.current_dims()[0],
        5,
        "variable leading dimension must be 5"
    );
}

/// Two dimensions + two variables with distinct axis bindings round-trip correctly.
///
/// ## Invariant
///
/// `temperature` references `["time", "station"]`; `pressure` references
/// `["time"]`.  Both bindings must be preserved through the round-trip.
#[test]
fn write_two_dimensions_two_variables_roundtrip() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("time"), 4));
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("station"), 3));

    model.root.variables.push(
        NetcdfVariable::new(
            String::from("temperature"),
            f32_dt(),
            vec![String::from("time"), String::from("station")],
        )
        .with_shape(Shape::fixed(&[4, 3])),
    );
    model.root.variables.push(
        NetcdfVariable::new(
            String::from("pressure"),
            f32_dt(),
            vec![String::from("time")],
        )
        .with_shape(Shape::fixed(&[4])),
    );

    let group = write_and_extract(&model);

    assert_eq!(
        group.dimensions.len(),
        2,
        "two dimensions must be extracted"
    );
    let dim_names: Vec<&str> = group.dimensions.iter().map(|d| d.name.as_str()).collect();
    assert!(
        dim_names.contains(&"time"),
        "time dimension must be present; got {dim_names:?}"
    );
    assert!(
        dim_names.contains(&"station"),
        "station dimension must be present; got {dim_names:?}"
    );

    assert_eq!(group.variables.len(), 2, "two variables must be extracted");

    let temp = group
        .variables
        .iter()
        .find(|v| v.name == "temperature")
        .expect("temperature variable must be present");
    assert_eq!(
        temp.dimensions,
        vec![String::from("time"), String::from("station")],
        "temperature DIMENSION_LIST must bind [time, station] in axis order"
    );

    let pressure = group
        .variables
        .iter()
        .find(|v| v.name == "pressure")
        .expect("pressure variable must be present");
    assert_eq!(
        pressure.dimensions,
        vec![String::from("time")],
        "pressure DIMENSION_LIST must bind [time]"
    );
}

/// Every file emitted by `NetcdfWriter` carries a `_nc_properties` root attribute.
///
/// ## Invariant
///
/// `_nc_properties` is present in the root group attributes with the value
/// `"version=2,netcdf=4.x.x"`, which is `NC_PROPERTIES_VALUE`.
#[test]
fn write_nc_properties_root_attribute_present() {
    let model = NetcdfModel::default();
    let group = write_and_extract(&model);

    let nc_props = group.attributes.iter().find(|(k, _)| k == "_nc_properties");

    assert!(
        nc_props.is_some(),
        "_nc_properties attribute must be present in root group attributes; got {:?}",
        group.attributes
    );

    match &nc_props.unwrap().1 {
        AttributeValue::String(s) => {
            assert_eq!(
                s.as_str(),
                consus_netcdf::NC_PROPERTIES_VALUE,
                "_nc_properties value must equal NC_PROPERTIES_VALUE"
            );
        }
        other => panic!(
            "_nc_properties must be a String attribute value, got {:?}",
            other
        ),
    }
}

/// A scalar variable (rank 0) round-trips with an empty dimension list.
///
/// ## Invariant
///
/// Scalar variables carry no DIMENSION_LIST attribute.  After extraction,
/// `var.dimensions` is empty and `var.is_scalar()` is true.
#[test]
fn write_scalar_variable_roundtrip() {
    let mut model = NetcdfModel::default();
    model.root.variables.push(
        NetcdfVariable::new(String::from("global_constant"), i32_dt(), vec![])
            .with_shape(Shape::scalar()),
    );

    let group = write_and_extract(&model);

    assert_eq!(group.variables.len(), 1, "one variable must be extracted");
    let var = &group.variables[0];
    assert_eq!(var.name, "global_constant", "variable name must match");
    assert!(
        var.dimensions.is_empty(),
        "scalar variable must have no dimensions; got {:?}",
        var.dimensions
    );
    assert!(
        var.is_scalar(),
        "is_scalar() must be true for rank-0 variable"
    );
}

/// A string-valued CF attribute is preserved through write → extract_group.
///
/// ## Invariant
///
/// `AttributeValue::String("meters")` on variable `units` attribute decodes
/// back to the same string value after round-trip.
#[test]
fn write_cf_string_attribute_preserved() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 3));
    model.root.variables.push(
        NetcdfVariable::new(String::from("depth"), f32_dt(), vec![String::from("x")])
            .with_shape(Shape::fixed(&[3]))
            .with_attributes(vec![(
                String::from("units"),
                AttributeValue::String(String::from("meters")),
            )]),
    );

    let group = write_and_extract(&model);

    let var = group
        .variables
        .iter()
        .find(|v| v.name == "depth")
        .expect("depth variable must be present");

    let units_attr = var.attributes.iter().find(|(k, _)| k == "units");
    assert!(
        units_attr.is_some(),
        "units attribute must be present; variable attributes: {:?}",
        var.attributes
    );

    match &units_attr.unwrap().1 {
        AttributeValue::String(s) => {
            assert_eq!(
                s.as_str(),
                "meters",
                "units attribute value must round-trip as 'meters'"
            );
        }
        other => panic!("units attribute must be a String value, got {:?}", other),
    }
}

/// An `AttributeValue::Int` CF attribute is preserved through write → extract_group.
///
/// ## Invariant
///
/// `AttributeValue::Int(-42)` on attribute `"valid_min"` decodes back to the
/// same `Int(-42)` value after round-trip.
#[test]
fn write_int_cf_attribute_preserved() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 3));
    model.root.variables.push(
        NetcdfVariable::new(String::from("depth"), f32_dt(), vec![String::from("x")])
            .with_shape(Shape::fixed(&[3]))
            .with_attributes(vec![(String::from("valid_min"), AttributeValue::Int(-42))]),
    );

    let group = write_and_extract(&model);

    let var = group
        .variables
        .iter()
        .find(|v| v.name == "depth")
        .expect("depth variable must be present");

    let attr = var
        .attributes
        .iter()
        .find(|(k, _)| k == "valid_min")
        .expect("valid_min attribute must be present");

    assert_eq!(
        attr.1,
        AttributeValue::Int(-42),
        "valid_min must round-trip as Int(-42)"
    );
}

/// An `AttributeValue::Uint` CF attribute is preserved through write → extract_group.
///
/// ## Invariant
///
/// `AttributeValue::Uint(255)` on attribute `"flag_mask"` decodes back to
/// `Uint(255)` after round-trip.
#[test]
fn write_uint_cf_attribute_preserved() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 2));
    model.root.variables.push(
        NetcdfVariable::new(String::from("flags"), i32_dt(), vec![String::from("x")])
            .with_shape(Shape::fixed(&[2]))
            .with_attributes(vec![(String::from("flag_mask"), AttributeValue::Uint(255))]),
    );

    let group = write_and_extract(&model);

    let var = group
        .variables
        .iter()
        .find(|v| v.name == "flags")
        .expect("flags variable must be present");

    let attr = var
        .attributes
        .iter()
        .find(|(k, _)| k == "flag_mask")
        .expect("flag_mask attribute must be present");

    assert_eq!(
        attr.1,
        AttributeValue::Uint(255),
        "flag_mask must round-trip as Uint(255)"
    );
}

/// An `AttributeValue::Float` CF attribute is preserved through write → extract_group.
///
/// ## Invariant
///
/// `AttributeValue::Float(1.5)` on attribute `"scale_factor"` decodes back to
/// the exact same f64 bit pattern after round-trip.
#[test]
fn write_float_cf_attribute_preserved() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("t"), 4));
    model.root.variables.push(
        NetcdfVariable::new(String::from("signal"), f32_dt(), vec![String::from("t")])
            .with_shape(Shape::fixed(&[4]))
            .with_attributes(vec![(
                String::from("scale_factor"),
                AttributeValue::Float(1.5),
            )]),
    );

    let group = write_and_extract(&model);

    let var = group
        .variables
        .iter()
        .find(|v| v.name == "signal")
        .expect("signal variable must be present");

    let attr = var
        .attributes
        .iter()
        .find(|(k, _)| k == "scale_factor")
        .expect("scale_factor attribute must be present");

    assert_eq!(
        attr.1,
        AttributeValue::Float(1.5),
        "scale_factor must round-trip as Float(1.5)"
    );
}

/// An `AttributeValue::IntArray` CF attribute is preserved through write → extract_group.
///
/// ## Invariant
///
/// `AttributeValue::IntArray(vec![-1, 0, 1])` on attribute `"flag_values"`
/// decodes back to `IntArray([-1, 0, 1])` after round-trip.
#[test]
fn write_int_array_cf_attribute_preserved() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 3));
    model.root.variables.push(
        NetcdfVariable::new(String::from("category"), i32_dt(), vec![String::from("x")])
            .with_shape(Shape::fixed(&[3]))
            .with_attributes(vec![(
                String::from("flag_values"),
                AttributeValue::IntArray(vec![-1, 0, 1]),
            )]),
    );

    let group = write_and_extract(&model);

    let var = group
        .variables
        .iter()
        .find(|v| v.name == "category")
        .expect("category variable must be present");

    let attr = var
        .attributes
        .iter()
        .find(|(k, _)| k == "flag_values")
        .expect("flag_values attribute must be present");

    assert_eq!(
        attr.1,
        AttributeValue::IntArray(vec![-1i64, 0i64, 1i64]),
        "flag_values must round-trip as IntArray([-1, 0, 1])"
    );
}

/// An `AttributeValue::FloatArray` CF attribute is preserved through write → extract_group.
///
/// ## Invariant
///
/// `AttributeValue::FloatArray(vec![0.1, 0.2])` on attribute `"valid_range"`
/// decodes back to `FloatArray([0.1, 0.2])` with identical f64 bit patterns
/// after round-trip.
#[test]
fn write_float_array_cf_attribute_preserved() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("z"), 2));
    model.root.variables.push(
        NetcdfVariable::new(String::from("depth"), f32_dt(), vec![String::from("z")])
            .with_shape(Shape::fixed(&[2]))
            .with_attributes(vec![(
                String::from("valid_range"),
                AttributeValue::FloatArray(vec![0.1f64, 0.2f64]),
            )]),
    );

    let group = write_and_extract(&model);

    let var = group
        .variables
        .iter()
        .find(|v| v.name == "depth")
        .expect("depth variable must be present");

    let attr = var
        .attributes
        .iter()
        .find(|(k, _)| k == "valid_range")
        .expect("valid_range attribute must be present");

    assert_eq!(
        attr.1,
        AttributeValue::FloatArray(vec![0.1f64, 0.2f64]),
        "valid_range must round-trip as FloatArray([0.1, 0.2])"
    );
}

/// A model with root dimensions and a child group containing a dimension and
/// variable round-trips correctly through the enhanced write path.
///
/// ## Invariant
///
/// Root has `"time"(3)`; child group `"measurements"` has `"x"(2)` and
/// variable `"depth"` with `dimensions == ["x"]`.
#[test]
fn write_enhanced_model_sub_group_roundtrip() {
    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("time"), 3));

    let mut measurements = NetcdfGroup::new(String::from("measurements"));
    measurements
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 2));
    measurements.variables.push(
        NetcdfVariable::new(String::from("depth"), f32_dt(), vec![String::from("x")])
            .with_shape(Shape::fixed(&[2])),
    );
    model.root.groups.push(measurements);

    let group = write_and_extract(&model);

    // Root dimension preserved.
    assert_eq!(group.dimensions.len(), 1, "root must have one dimension");
    assert_eq!(
        group.dimensions[0].name, "time",
        "root dimension must be 'time'"
    );
    assert_eq!(group.dimensions[0].size, 3, "root 'time' size must be 3");

    // Child group present.
    let meas = group
        .groups
        .iter()
        .find(|g| g.name == "measurements")
        .expect("'measurements' group must be present");

    assert_eq!(
        meas.dimensions.len(),
        1,
        "'measurements' must have one dimension"
    );
    assert_eq!(
        meas.dimensions[0].name, "x",
        "'measurements' dimension must be 'x'"
    );
    assert_eq!(meas.dimensions[0].size, 2, "'x' size must be 2");

    // Variable in child group with correct dimension binding.
    let depth = meas
        .variables
        .iter()
        .find(|v| v.name == "depth")
        .expect("'depth' variable must be present in 'measurements'");

    assert_eq!(
        depth.dimensions,
        vec![String::from("x")],
        "'depth' DIMENSION_LIST must resolve to ['x']"
    );
}

/// A two-level nested sub-group model round-trips: root → level1 → level2
/// with a dimension and variable in level2.
///
/// ## Invariant
///
/// Navigating `group.groups["level1"].groups["level2"]` yields the expected
/// dimension `"n"(4)` and variable `"data"` with `dimensions == ["n"]`.
#[test]
fn write_nested_two_level_sub_group_roundtrip() {
    let mut model = NetcdfModel::default();

    let mut level2 = NetcdfGroup::new(String::from("level2"));
    level2
        .dimensions
        .push(NetcdfDimension::new(String::from("n"), 4));
    level2.variables.push(
        NetcdfVariable::new(String::from("data"), f32_dt(), vec![String::from("n")])
            .with_shape(Shape::fixed(&[4])),
    );

    let mut level1 = NetcdfGroup::new(String::from("level1"));
    level1.groups.push(level2);
    model.root.groups.push(level1);

    let group = write_and_extract(&model);

    let level1_g = group
        .groups
        .iter()
        .find(|g| g.name == "level1")
        .expect("'level1' must be present at root");

    let level2_g = level1_g
        .groups
        .iter()
        .find(|g| g.name == "level2")
        .expect("'level2' must be present inside 'level1'");

    assert_eq!(
        level2_g.dimensions.len(),
        1,
        "'level2' must have one dimension"
    );
    assert_eq!(
        level2_g.dimensions[0].name, "n",
        "'level2' dimension must be 'n'"
    );
    assert_eq!(level2_g.dimensions[0].size, 4, "'n' size must be 4");

    let data_var = level2_g
        .variables
        .iter()
        .find(|v| v.name == "data")
        .expect("'data' variable must be present in 'level2'");

    assert_eq!(
        data_var.dimensions,
        vec![String::from("n")],
        "'data' DIMENSION_LIST must resolve to ['n']"
    );
}
