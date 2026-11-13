//! `PyNetcdfFile`: read-only access to netCDF-4/HDF5 bytes.

extern crate alloc;

use alloc::{string::String, string::ToString, vec, vec::Vec};

use consus_hdf5::file::Hdf5File;
use consus_io::MemCursor;
use consus_netcdf::{read_model, read_variable_bytes, NetcdfGroup, NetcdfModel, NetcdfVariable};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use super::codec::decode_bytes;
use super::dtype::dtype_tag;
use crate::error::from_consus;

/// In-memory netCDF-4 file reader.
///
/// Accepts raw HDF5/netCDF-4 bytes and exposes the root-group structure plus
/// variable data decoding.
///
/// Example:
///
/// ```python
/// with open("data.nc", "rb") as f:
///     nf = PyNetcdfFile(f.read())
/// print(nf.variable_names())
/// print(nf.read_variable("temperature"))
/// ```
#[pyclass]
pub struct PyNetcdfFile {
    data: Vec<u8>,
}

#[pymethods]
impl PyNetcdfFile {
    /// Construct from raw netCDF-4/HDF5 bytes.
    #[new]
    pub fn new(data: &[u8]) -> PyResult<Self> {
        // Validate the bytes open as a valid HDF5 file before storing.
        let cursor = MemCursor::from_bytes(data.to_vec());
        Hdf5File::open(cursor).map_err(from_consus)?;
        Ok(Self {
            data: data.to_vec(),
        })
    }

    /// Names of all root-group variables (non-dimension-scale datasets).
    pub fn variable_names(&self) -> PyResult<Vec<String>> {
        let model = self.open_model()?;
        Ok(model
            .root
            .variables
            .iter()
            .map(|v| v.name.clone())
            .collect())
    }

    /// Names of all root-group dimensions.
    pub fn dimension_names(&self) -> PyResult<Vec<String>> {
        let model = self.open_model()?;
        Ok(model
            .root
            .dimensions
            .iter()
            .map(|d| d.name.clone())
            .collect())
    }

    /// Dict-like list of `(name, size)` pairs for root-group dimensions.
    pub fn dimension_sizes(&self) -> PyResult<Vec<(String, usize)>> {
        let model = self.open_model()?;
        Ok(model
            .root
            .dimensions
            .iter()
            .map(|d| (d.name.clone(), d.size))
            .collect())
    }

    /// Dtype string ("f32", "f64", "i32", …) and dimension names for a variable.
    ///
    /// Returns `(dtype_str, [dim_names...])`.
    pub fn variable_info(&self, name: &str) -> PyResult<(String, Vec<String>)> {
        let model = self.open_model()?;
        let var = find_variable(&model.root, name)?;
        Ok((dtype_tag(&var.datatype).to_string(), var.dimensions.clone()))
    }

    /// Read the flat (row-major) data of a root-group variable as a Python list.
    ///
    /// Returns a list of `int` or `float` values.
    pub fn read_variable(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let cursor = MemCursor::from_bytes(self.data.clone());
        let file = Hdf5File::open(cursor).map_err(from_consus)?;
        let model = read_model(&file).map_err(from_consus)?;
        let var = find_variable(&model.root, name)?;
        let raw = read_variable_bytes(&file, var).map_err(from_consus)?;
        decode_bytes(py, &raw, &var.datatype)
    }

    /// Shape (list of dimension sizes in axis order) of a root-group variable.
    pub fn variable_shape(&self, name: &str) -> PyResult<Vec<usize>> {
        let model = self.open_model()?;
        let var = find_variable(&model.root, name)?;
        match &var.shape {
            Some(s) => Ok(s.current_dims().as_slice().to_vec()),
            None => Ok(vec![]),
        }
    }

    /// Group-level attributes as a list of `(name, value)` pairs.
    ///
    /// Numeric attributes are returned as `int` or `float`; string attributes
    /// as `str`; array attributes as a Python list.
    pub fn group_attributes(&self, py: Python<'_>) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let model = self.open_model()?;
        attrs_to_py(py, &model.root.attributes)
    }
}

impl PyNetcdfFile {
    fn open_model(&self) -> PyResult<NetcdfModel> {
        let cursor = MemCursor::from_bytes(self.data.clone());
        let file = Hdf5File::open(cursor).map_err(from_consus)?;
        read_model(&file).map_err(from_consus)
    }
}

fn find_variable<'g>(group: &'g NetcdfGroup, name: &str) -> PyResult<&'g NetcdfVariable> {
    group
        .variables
        .iter()
        .find(|v| v.name == name)
        .ok_or_else(|| PyValueError::new_err(format!("variable '{name}' not found in root group")))
}

fn attrs_to_py<'py>(
    py: Python<'py>,
    attrs: &[(String, consus_core::AttributeValue)],
) -> PyResult<Vec<(String, Py<PyAny>)>> {
    use consus_core::AttributeValue;
    let mut out = Vec::new();
    for (k, v) in attrs {
        let obj: Py<PyAny> = match v {
            AttributeValue::Int(n) => (*n).into_pyobject(py)?.into_any().unbind(),
            AttributeValue::Uint(n) => (*n).into_pyobject(py)?.into_any().unbind(),
            AttributeValue::Float(f) => (*f).into_pyobject(py)?.into_any().unbind(),
            AttributeValue::String(s) => s.into_pyobject(py)?.into_any().unbind(),
            AttributeValue::IntArray(arr) => {
                let list = PyList::new(py, arr.iter())?;
                list.into_any().unbind()
            }
            AttributeValue::UintArray(arr) => {
                let list = PyList::new(py, arr.iter())?;
                list.into_any().unbind()
            }
            AttributeValue::FloatArray(arr) => {
                let list = PyList::new(py, arr.iter())?;
                list.into_any().unbind()
            }
            AttributeValue::StringArray(arr) => {
                let list = PyList::new(py, arr.iter())?;
                list.into_any().unbind()
            }
            AttributeValue::Bytes(b) => PyBytes::new(py, b).into_any().unbind(),
        };
        out.push((k.clone(), obj));
    }
    Ok(out)
}
