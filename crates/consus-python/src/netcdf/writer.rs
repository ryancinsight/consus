//! `PyNetcdfWriter`: netCDF-4 model builder and serializer.

extern crate alloc;

use alloc::{string::String, string::ToString, vec::Vec};

use consus_core::Shape;
use consus_netcdf::{NetcdfDimension, NetcdfGroup, NetcdfModel, NetcdfVariable, NetcdfWriter};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::sync::{Mutex, MutexGuard};

use super::codec::encode_bytes;
use super::dtype::parse_dtype;
use crate::error::from_consus;
use crate::locking;

/// netCDF-4 file builder.
///
/// Add dimensions and variables, then call :meth:`write` to produce a
/// self-contained HDF5/netCDF-4 byte string.
///
/// Example:
///
/// ```python
/// w = PyNetcdfWriter()
/// w.add_dimension("x", 5)
/// w.add_variable("temperature", "f32", ["x"], [1.0, 2.0, 3.0, 4.0, 5.0])
/// raw = w.write()
/// ```
///
/// One writer may be shared between threads: each call is atomic, and a
/// variable may only reference dimensions added before its call.
#[pyclass(frozen)]
pub struct PyNetcdfWriter {
    group: Mutex<NetcdfGroup>,
}

impl PyNetcdfWriter {
    fn group(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NetcdfGroup>> {
        locking::lock(&self.group, py, "NetcdfWriter")
    }
}

#[pymethods]
impl PyNetcdfWriter {
    /// Create an empty writer.
    #[new]
    pub fn new() -> Self {
        Self {
            group: Mutex::new(NetcdfGroup::new(String::from("/"))),
        }
    }

    /// Add a fixed-size dimension with `name` and `size`.
    ///
    /// Raises `ValueError` if a dimension with this name already exists.
    pub fn add_dimension(&self, py: Python<'_>, name: &str, size: usize) -> PyResult<()> {
        let mut group = self.group(py)?;
        if group.dimensions.iter().any(|d| d.name == name) {
            return Err(PyValueError::new_err(format!(
                "dimension '{name}' already defined"
            )));
        }
        group
            .dimensions
            .push(NetcdfDimension::new(name.to_string(), size));
        Ok(())
    }

    /// Add a variable with `name`, element dtype, ordered dimension names,
    /// and a flat row-major data list.
    ///
    /// `dtype` must be one of: f32, f64, i8, u8, i16, u16, i32, u32, i64, u64.
    ///
    /// `dims` must reference dimensions already added via :meth:`add_dimension`.
    ///
    /// `data` must be a flat Python list whose length equals the product of
    /// the sizes of the listed dimensions.
    ///
    /// Raises `ValueError` on dtype, dim-reference, or length mismatch.
    pub fn add_variable(
        &self,
        py: Python<'_>,
        name: &str,
        dtype: &str,
        dims: Vec<String>,
        data: &Bound<'_, PyList>,
    ) -> PyResult<()> {
        let check_unique = |group: &NetcdfGroup| -> PyResult<()> {
            if group.variables.iter().any(|v| v.name == name) {
                return Err(PyValueError::new_err(format!(
                    "variable '{name}' already defined"
                )));
            }
            Ok(())
        };
        let guard = self.group(py)?;
        check_unique(&guard)?;
        drop(guard);

        let dt = parse_dtype(dtype)?;

        // Resolve shape from declared dimensions.
        let shape_dims: Vec<usize> = {
            let group = self.group(py)?;
            dims.iter()
                .map(|d| {
                    group
                        .dimensions
                        .iter()
                        .find(|dim| &dim.name == d)
                        .map(|dim| dim.size)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "dimension '{d}' not found; add it with add_dimension() first"
                            ))
                        })
                })
                .collect::<PyResult<Vec<_>>>()?
        };

        let expected = shape_dims.iter().product::<usize>();
        if data.len() != expected {
            return Err(PyValueError::new_err(format!(
                "data length {} does not match shape product {} for variable '{name}'",
                data.len(),
                expected
            )));
        }

        let raw = encode_bytes(data, &dt)?;
        let shape = if shape_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::fixed(&shape_dims)
        };

        let var = NetcdfVariable::new(name.to_string(), dt, dims)
            .with_shape(shape)
            .with_data(raw);

        // Element conversion above may run Python code, so the lock is taken
        // again here and uniqueness re-checked against concurrent callers.
        let mut group = self.group(py)?;
        check_unique(&group)?;
        group.variables.push(var);
        Ok(())
    }

    /// Serialise the model to netCDF-4/HDF5 bytes.
    pub fn write(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let model = NetcdfModel {
            root: self.group(py)?.clone(),
        };
        let bytes = py
            .detach(|| NetcdfWriter::new().write_model(&model))
            .map_err(from_consus)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}
