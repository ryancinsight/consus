//! netCDF-4 and CF convention constants and validation.
//!
//! ## CF Conventions
//!
//! Reference: <http://cfconventions.org/>
//!
//! Key attributes recognized by this module:
//! - `_FillValue`: default fill value for missing data
//! - `units`: physical units string
//! - `standard_name`: CF standard name table entry
//! - `long_name`: human-readable variable description
//! - `coordinates`: space-separated list of coordinate variable names
//! - `cell_methods`: description of statistical operations

/// HDF5 attribute name marking a dataset as a netCDF-4 dimension scale.
pub const DIMENSION_SCALE_CLASS: &str = "CLASS";

/// Expected value of the CLASS attribute for dimension scales.
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
