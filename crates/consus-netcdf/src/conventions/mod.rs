//! netCDF-4 and CF convention constants and validation.
//!
//! ## Specification
//!
//! netCDF-4 is an HDF5-based container with domain conventions that
//! identify dimensions, coordinate variables, fill values, and CF
//! metadata. This module centralizes the canonical strings and the
//! lightweight validation rules used by higher-level netCDF logic.
//!
//! ## Invariants
//!
//! - Dimension scales are represented by the HDF5 `CLASS` attribute
//!   with value `DIMENSION_SCALE`.
//! - netCDF metadata uses the `Conventions` attribute.
//! - CF metadata keys use lower-case canonical names.
//! - Validation is structural: it checks naming and compatibility rules,
//!   not full semantic interpretation of physical units or CF tables.

/// HDF5 attribute name marking a dataset as a netCDF-4 dimension scale.
pub const DIMENSION_SCALE_CLASS: &str = "CLASS";

/// Expected value of the `CLASS` attribute for dimension scales.
pub const DIMENSION_SCALE_VALUE: &str = "DIMENSION_SCALE";

/// netCDF-4 conventions attribute name.
pub const CONVENTIONS_ATTR: &str = "Conventions";

/// Fill value attribute name.
pub const FILL_VALUE_ATTR: &str = "_FillValue";

/// Standard name attribute (CF conventions).
pub const STANDARD_NAME_ATTR: &str = "standard_name";

/// Units attribute (CF conventions).
pub const UNITS_ATTR: &str = "units";

/// Long name attribute.
pub const LONG_NAME_ATTR: &str = "long_name";

/// Coordinate variable attribute (CF conventions).
pub const COORDINATES_ATTR: &str = "coordinates";

/// Cell methods attribute (CF conventions).
pub const CELL_METHODS_ATTR: &str = "cell_methods";

/// Bounds attribute (CF conventions).
pub const BOUNDS_ATTR: &str = "bounds";

/// Axis attribute (CF conventions).
pub const AXIS_ATTR: &str = "axis";

/// Grid mapping attribute (CF conventions).
pub const GRID_MAPPING_ATTR: &str = "grid_mapping";

/// Canonical netCDF version tag for this implementation.
pub const NETCDF_CONVENTIONS_VALUE: &str = "CF-1.8";

/// Canonical group name used for the root container in netCDF models.
pub const ROOT_GROUP_NAME: &str = "/";

/// Validate whether a name is a legal netCDF object name.
///
/// netCDF names are treated as UTF-8 labels that must not be empty and
/// must not contain the NUL byte or `/` path separator.
///
/// This is a structural validity check used before mapping names into
/// HDF5 groups, datasets, and attributes.
#[must_use]
pub fn is_valid_name(name: &str) -> bool {
    !name.is_empty() && !name.contains('\0') && !name.contains('/')
}

/// Validate whether an attribute value can be used as a dimension scale marker.
///
/// The HDF5 convention requires the `CLASS` attribute to match exactly
/// `DIMENSION_SCALE`.
#[must_use]
pub fn is_dimension_scale_marker(class_name: &str, class_value: &str) -> bool {
    class_name == DIMENSION_SCALE_CLASS && class_value == DIMENSION_SCALE_VALUE
}

/// Validate whether a convention string is recognized by this crate.
///
/// This implementation recognizes canonical CF metadata strings and
/// the netCDF umbrella tag. The check is structural and does not parse
/// version tuples beyond exact comparison.
#[must_use]
pub fn is_recognized_convention(value: &str) -> bool {
    value == NETCDF_CONVENTIONS_VALUE || value.starts_with("CF-")
}

/// Validate whether a string name is a canonical CF attribute key.
///
/// CF uses lower-case names for standard metadata fields.
#[must_use]
pub fn is_cf_attribute_name(name: &str) -> bool {
    matches!(
        name,
        STANDARD_NAME_ATTR
            | UNITS_ATTR
            | LONG_NAME_ATTR
            | COORDINATES_ATTR
            | CELL_METHODS_ATTR
            | BOUNDS_ATTR
            | AXIS_ATTR
            | GRID_MAPPING_ATTR
    )
}

/// Return the canonical root group identifier for netCDF models.
///
/// netCDF files conceptually have a root group at `/`.
#[must_use]
pub const fn root_group_name() -> &'static str {
    ROOT_GROUP_NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_names() {
        assert!(is_valid_name("temperature"));
        assert!(is_valid_name("group_1"));
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("a/b"));
        assert!(!is_valid_name("a\0b"));
    }

    #[test]
    fn validate_dimension_scale_marker() {
        assert!(is_dimension_scale_marker(
            DIMENSION_SCALE_CLASS,
            DIMENSION_SCALE_VALUE
        ));
        assert!(!is_dimension_scale_marker("CLASS", "OTHER"));
        assert!(!is_dimension_scale_marker("OTHER", DIMENSION_SCALE_VALUE));
    }

    #[test]
    fn validate_conventions() {
        assert!(is_recognized_convention("CF-1.8"));
        assert!(is_recognized_convention("CF-1.7"));
        assert!(is_recognized_convention("CF-2.0"));
        assert!(is_recognized_convention(NETCDF_CONVENTIONS_VALUE));
        assert!(!is_recognized_convention("HDF5"));
    }

    #[test]
    fn validate_cf_attribute_names() {
        assert!(is_cf_attribute_name(STANDARD_NAME_ATTR));
        assert!(is_cf_attribute_name(UNITS_ATTR));
        assert!(is_cf_attribute_name(LONG_NAME_ATTR));
        assert!(is_cf_attribute_name(COORDINATES_ATTR));
        assert!(is_cf_attribute_name(CELL_METHODS_ATTR));
        assert!(is_cf_attribute_name(BOUNDS_ATTR));
        assert!(is_cf_attribute_name(AXIS_ATTR));
        assert!(is_cf_attribute_name(GRID_MAPPING_ATTR));
        assert!(!is_cf_attribute_name("Conventions"));
    }

    #[test]
    fn root_name_is_slash() {
        assert_eq!(root_group_name(), "/");
    }
}
