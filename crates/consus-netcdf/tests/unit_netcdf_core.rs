//! Unit tests for netCDF-4 core constructs: dimensions, variables, and conventions.
//!
//! ## Scope
//!
//! These tests validate the netCDF descriptor model API:
//! - Dimension creation, field access, validation, and resize semantics
//! - Variable creation, builder methods, validation, and rank
//! - Convention constants and validation functions
//! - Group and model construction and validation
//!
//! ## Specification Reference
//!
//! The descriptor model follows netCDF-4 semantics over HDF5:
//! - Unidata netCDF-4 specification
//! - CF (Climate and Forecast) conventions for metadata naming

use consus_core::{Compression, Datatype};
use consus_netcdf::{
    AXIS_ATTR, BOUNDS_ATTR, CELL_METHODS_ATTR, CONVENTIONS_ATTR, COORDINATES_ATTR,
    DIMENSION_SCALE_CLASS, DIMENSION_SCALE_VALUE, FILL_VALUE_ATTR, GRID_MAPPING_ATTR,
    LONG_NAME_ATTR, NETCDF_CONVENTIONS_VALUE, NetcdfDimension, NetcdfVariable, ROOT_GROUP_NAME,
    STANDARD_NAME_ATTR, UNITS_ATTR, is_cf_attribute_name, is_dimension_scale_marker,
    is_recognized_convention, is_valid_name,
};

// ---------------------------------------------------------------------------
// Dimension Tests
// ---------------------------------------------------------------------------

/// Test dimension creation with fixed size.
///
/// ## Spec Compliance
///
/// NetCDF-4 dimension size must be positive integer or NC_UNLIMITED.
/// Fixed dimensions have size > 0.
#[test]
fn dimension_fixed_size() {
    let dim = NetcdfDimension::new(String::from("time"), 100);

    assert_eq!(dim.name, "time");
    assert_eq!(dim.size, 100);
    assert_eq!(dim.len(), 100);
    assert!(!dim.is_unlimited(), "fixed dimension must not be unlimited");
    assert!(!dim.is_empty());
}

/// Test unlimited dimension (record dimension).
///
/// ## Spec Compliance
///
/// NC_UNLIMITED indicates the dimension can grow.
/// Unlimited dimensions have a current size and can be resized.
#[test]
fn dimension_unlimited() {
    let dim = NetcdfDimension::unlimited(String::from("time"), 0);

    assert_eq!(dim.name, "time");
    assert!(
        dim.is_unlimited(),
        "unlimited dimension must report is_unlimited"
    );
    assert_eq!(dim.size, 0, "unlimited dimension starts at size 0");
    assert!(dim.is_empty());
}

/// Test dimension equality and ordering.
///
/// ## Invariant
///
/// Dimensions are identified by name, size, and unlimited flag.
/// Two dimensions with same fields are equal via PartialEq.
#[test]
fn dimension_identity() {
    let dim1 = NetcdfDimension::new(String::from("x"), 50);
    let dim2 = NetcdfDimension::new(String::from("x"), 50);
    let dim3 = NetcdfDimension::new(String::from("y"), 50);
    let dim4 = NetcdfDimension::new(String::from("x"), 100);

    assert_eq!(dim1, dim2, "identical dimensions must be equal");
    assert_ne!(dim1, dim3, "different names are different dimensions");
    assert_ne!(dim1, dim4, "same name different size is mismatch");
}

/// Test dimension name validation.
///
/// ## Spec Compliance
///
/// NetCDF dimension names must be non-empty. The `is_valid_name` function
/// validates naming rules (non-empty, no NUL, no `/`).
/// The `validate()` method on `NetcdfDimension` rejects empty names.
#[test]
fn dimension_name_valid() {
    // Valid names per is_valid_name
    assert!(is_valid_name("time"));
    assert!(is_valid_name("_private"));
    assert!(is_valid_name("lat2"));
    assert!(is_valid_name("Temperature"));

    // Invalid: empty name
    assert!(!is_valid_name(""), "empty name must be invalid");

    // is_valid_name rejects names containing '/' or '\0'
    assert!(!is_valid_name("a/b"), "slash in name must be invalid");
    assert!(!is_valid_name("a\0b"), "NUL in name must be invalid");

    // validate() on dimension rejects empty name
    let empty_dim = NetcdfDimension::new(String::from(""), 1);
    assert!(
        empty_dim.validate().is_err(),
        "empty dimension name must fail validation"
    );

    // validate() on dimension accepts valid name
    let valid_dim = NetcdfDimension::new(String::from("time"), 10);
    assert!(valid_dim.validate().is_ok());
}

/// Test dimension size constraints.
///
/// ## Invariant
///
/// NetCDF-4 dimension constructors accept any usize value.
/// Unlimited dimensions can be resized; fixed dimensions reject size changes.
#[test]
fn dimension_size_bounds() {
    // Valid sizes
    let dim1 = NetcdfDimension::new(String::from("x"), 1);
    assert_eq!(dim1.size, 1);
    let dim2 = NetcdfDimension::new(String::from("x"), 1_000_000);
    assert_eq!(dim2.size, 1_000_000);

    // Zero-size fixed dimension is representable
    let dim0 = NetcdfDimension::new(String::from("x"), 0);
    assert_eq!(dim0.size, 0);
    assert!(dim0.is_empty());

    // Fixed dimension rejects resize to different size
    let mut fixed = NetcdfDimension::new(String::from("x"), 10);
    assert!(
        fixed.resize(20).is_err(),
        "fixed dimension cannot be resized to different size"
    );
    assert_eq!(fixed.size, 10, "size unchanged after failed resize");

    // Unlimited dimension allows resize
    let mut unlimited = NetcdfDimension::unlimited(String::from("y"), 5);
    assert!(unlimited.resize(10).is_ok());
    assert_eq!(unlimited.size, 10);
}

// ---------------------------------------------------------------------------
// Convention and Attribute-Name Tests
// ---------------------------------------------------------------------------

/// Test convention constants for netCDF-4 dimension scale markers.
///
/// ## Spec Compliance
///
/// HDF5-based netCDF-4 uses the CLASS attribute with value DIMENSION_SCALE.
#[test]
fn convention_dimension_scale_marker() {
    assert_eq!(DIMENSION_SCALE_CLASS, "CLASS");
    assert_eq!(DIMENSION_SCALE_VALUE, "DIMENSION_SCALE");
    assert!(is_dimension_scale_marker(
        DIMENSION_SCALE_CLASS,
        DIMENSION_SCALE_VALUE
    ));
    assert!(!is_dimension_scale_marker("CLASS", "OTHER"));
    assert!(!is_dimension_scale_marker("OTHER", DIMENSION_SCALE_VALUE));
}

/// Test convention constants for CF attributes.
///
/// ## Spec Compliance
///
/// CF conventions define standard attribute names for metadata:
/// units, long_name, standard_name, coordinates, cell_methods, bounds, axis, grid_mapping.
#[test]
fn convention_cf_attribute_names() {
    assert!(is_cf_attribute_name(STANDARD_NAME_ATTR));
    assert!(is_cf_attribute_name(UNITS_ATTR));
    assert!(is_cf_attribute_name(LONG_NAME_ATTR));
    assert!(is_cf_attribute_name(COORDINATES_ATTR));
    assert!(is_cf_attribute_name(CELL_METHODS_ATTR));
    assert!(is_cf_attribute_name(BOUNDS_ATTR));
    assert!(is_cf_attribute_name(AXIS_ATTR));
    assert!(is_cf_attribute_name(GRID_MAPPING_ATTR));

    // Non-CF attributes
    assert!(!is_cf_attribute_name(CONVENTIONS_ATTR));
    assert!(!is_cf_attribute_name(FILL_VALUE_ATTR));
    assert!(!is_cf_attribute_name("random_attribute"));
}

/// Test convention recognition for netCDF convention strings.
///
/// ## Spec Compliance
///
/// The implementation recognizes CF-prefixed convention strings.
#[test]
fn convention_recognized_strings() {
    assert!(is_recognized_convention(NETCDF_CONVENTIONS_VALUE));
    assert!(is_recognized_convention("CF-1.8"));
    assert!(is_recognized_convention("CF-1.7"));
    assert!(is_recognized_convention("CF-2.0"));
    assert!(!is_recognized_convention("HDF5"));
    assert!(!is_recognized_convention("netCDF-3"));
}

/// Test name validation rules.
///
/// ## Spec Compliance
///
/// NetCDF names must be non-empty UTF-8 strings without NUL or '/'.
#[test]
fn convention_name_validation() {
    assert!(is_valid_name("units"));
    assert!(is_valid_name("_FillValue"));
    assert!(is_valid_name("missing_value"));
    assert!(is_valid_name("Temperature"));
    assert!(is_valid_name("data_2"));

    assert!(!is_valid_name(""));
    assert!(!is_valid_name("path/name"));
    assert!(!is_valid_name("null\0byte"));
}

/// Test convention constant values match expected strings.
///
/// ## Spec Compliance
///
/// Exact string values are mandated by netCDF-4 and CF specifications.
#[test]
fn convention_constant_values() {
    assert_eq!(CONVENTIONS_ATTR, "Conventions");
    assert_eq!(FILL_VALUE_ATTR, "_FillValue");
    assert_eq!(STANDARD_NAME_ATTR, "standard_name");
    assert_eq!(UNITS_ATTR, "units");
    assert_eq!(LONG_NAME_ATTR, "long_name");
    assert_eq!(COORDINATES_ATTR, "coordinates");
    assert_eq!(CELL_METHODS_ATTR, "cell_methods");
    assert_eq!(BOUNDS_ATTR, "bounds");
    assert_eq!(AXIS_ATTR, "axis");
    assert_eq!(GRID_MAPPING_ATTR, "grid_mapping");
    assert_eq!(ROOT_GROUP_NAME, "/");
    assert_eq!(NETCDF_CONVENTIONS_VALUE, "CF-1.8");
}

// ---------------------------------------------------------------------------
// Variable Tests
// ---------------------------------------------------------------------------

/// Test variable creation with single dimension.
///
/// ## Spec Compliance
///
/// NetCDF variables have a name, type, and ordered dimension name list.
#[test]
fn variable_single_dimension() {
    let var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("time")],
    );

    assert_eq!(var.name, "temperature");
    assert_eq!(var.rank(), 1);
    assert!(!var.is_scalar());
    assert_eq!(var.dimensions, vec!["time"]);
}

/// Test variable creation with multiple dimensions.
///
/// ## Spec Compliance
///
/// Dimension order is significant (row-major, C ordering).
#[test]
fn variable_multi_dimension() {
    let var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![
            String::from("time"),
            String::from("lat"),
            String::from("lon"),
        ],
    );

    assert_eq!(var.rank(), 3);
    assert_eq!(var.dimensions, vec!["time", "lat", "lon"]);
    assert!(!var.is_scalar());
}

/// Test coordinate variable identification.
///
/// ## Spec Compliance
///
/// Coordinate variable: one-dimensional variable with same name as dimension.
/// The `coordinate_variable` builder flag marks this, and `validate()` enforces
/// that coordinate variables are rank-1 with matching dimension name.
#[test]
fn variable_coordinate_identification() {
    let lat_var = NetcdfVariable::new(
        String::from("lat"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat")],
    )
    .coordinate_variable(true);

    assert!(
        lat_var.coordinate_variable,
        "lat(var) with lat(dim) is coordinate"
    );
    // Validation succeeds when name matches the single dimension
    lat_var.validate().unwrap();

    // Mismatched coordinate variable: name != dimension
    let temp_var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat")],
    )
    .coordinate_variable(true);

    assert!(
        temp_var.validate().is_err(),
        "coordinate variable name must match its single dimension"
    );
}

/// Test variable with unlimited dimension flag.
///
/// ## Spec Compliance
///
/// The unlimited builder marks a variable as growable along one or more
/// unlimited dimensions.
#[test]
fn variable_unlimited_dimension_first() {
    let var = NetcdfVariable::new(
        String::from("observations"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("time"), String::from("station")],
    )
    .unlimited(true);

    assert!(var.unlimited, "must have unlimited flag set");
    assert_eq!(var.rank(), 2);
    var.validate().unwrap();

    // Unlimited scalar is rejected by validate()
    let scalar_unlimited =
        NetcdfVariable::new(String::from("scalar"), Datatype::Boolean, Vec::new()).unlimited(true);
    assert!(
        scalar_unlimited.validate().is_err(),
        "unlimited scalar must fail validation"
    );
}

/// Test variable with compression configuration.
///
/// ## Spec Compliance
///
/// NetCDF-4 (HDF5) supports compression filters.
/// The `.with_compression()` builder attaches a compression policy.
#[test]
fn variable_chunking() {
    let var = NetcdfVariable::new(
        String::from("data"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![
            String::from("time"),
            String::from("lat"),
            String::from("lon"),
        ],
    )
    .with_compression(Compression::Deflate { level: 6 });

    assert!(var.compression.is_some());
    assert_eq!(var.compression, Some(Compression::Deflate { level: 6 }));
    assert_eq!(var.rank(), 3);
}

/// Test variable compression configuration.
///
/// ## Spec Compliance
///
/// NetCDF-4 supports HDF5 compression filters including Deflate and Zstd.
#[test]
fn variable_compression() {
    let var = NetcdfVariable::new(
        String::from("large_array"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat"), String::from("lon")],
    )
    .with_compression(Compression::Deflate { level: 6 });

    assert!(var.compression.is_some());
    match &var.compression {
        Some(Compression::Deflate { level }) => assert_eq!(*level, 6),
        other => panic!("expected Deflate {{ level: 6 }}, got {:?}", other),
    }
}

/// Test variable fill value configuration.
///
/// ## Spec Compliance
///
/// Variables commonly have fill values for missing data.
/// The `.with_fill_value()` builder attaches a raw-byte fill value.
#[test]
fn variable_with_fill_value() {
    let fill_bytes = (-999.0f32).to_le_bytes().to_vec();
    let var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("time")],
    )
    .with_fill_value(fill_bytes.clone());

    assert!(var.fill_value.is_some());
    let stored = var.fill_value.as_ref().unwrap();
    let restored = f32::from_le_bytes(stored[..4].try_into().unwrap());
    assert_eq!(restored, -999.0f32);
}

/// Test scalar variable (no dimensions).
///
/// ## Spec Compliance
///
/// NetCDF supports scalar variables (0-dimensional).
#[test]
fn variable_scalar() {
    let var = NetcdfVariable::new(
        String::from("title"),
        Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        },
        Vec::new(),
    );

    assert_eq!(var.rank(), 0);
    assert!(var.is_scalar());
    var.validate().unwrap();
}

// ---------------------------------------------------------------------------
// Dimension and Variable Integration Tests
// ---------------------------------------------------------------------------

/// Test dimension sharing across variables.
///
/// ## Invariant
///
/// Multiple variables can reference the same dimension names.
#[test]
fn dimension_sharing() {
    let temp_var = NetcdfVariable::new(
        String::from("temperature"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat"), String::from("lon")],
    );

    let pres_var = NetcdfVariable::new(
        String::from("pressure"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("lat"), String::from("lon")],
    );

    assert_eq!(
        temp_var.dimensions, pres_var.dimensions,
        "shared dimension names"
    );
    assert_eq!(temp_var.rank(), pres_var.rank());
}

/// Test dimension ordering significance.
///
/// ## Invariant
///
/// Dimension order determines memory layout. Different order = different variable.
#[test]
fn dimension_order_matters() {
    let var_xy = NetcdfVariable::new(
        String::from("data"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("x"), String::from("y")],
    );

    let var_yx = NetcdfVariable::new(
        String::from("data"),
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        },
        vec![String::from("y"), String::from("x")],
    );

    assert_ne!(
        var_xy.dimensions, var_yx.dimensions,
        "different dimension order = different variable layout"
    );
    assert_eq!(var_xy.dimensions, vec!["x", "y"]);
    assert_eq!(var_yx.dimensions, vec!["y", "x"]);
}
