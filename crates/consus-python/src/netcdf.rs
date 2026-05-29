//! netCDF-4 Python bindings: `PyNetcdfFile`, `PyNetcdfWriter`.
//!
//! `PyNetcdfFile` reads a netCDF-4/HDF5 file from raw bytes, exposes the
//! root-group structure, and decodes variable data.
//!
//! `PyNetcdfWriter` builds a `NetcdfModel` with typed data payloads and
//! serialises it to HDF5 bytes via `NetcdfWriter`.

extern crate alloc;

use core::num::NonZeroUsize;

use alloc::{string::String, string::ToString, vec, vec::Vec};

use consus_core::{ByteOrder, Datatype, Shape};
use consus_hdf5::file::Hdf5File;
use consus_io::MemCursor;
use consus_netcdf::{
    NetcdfDimension, NetcdfGroup, NetcdfModel, NetcdfVariable, NetcdfWriter, read_model,
    read_variable_bytes,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use crate::error::from_consus;

// ---------------------------------------------------------------------------
// Dtype helpers
// ---------------------------------------------------------------------------

/// Parse a dtype string ("f32", "f64", "i8", "u8", "i16", "u16",
/// "i32", "u32", "i64", "u64") to a little-endian `Datatype`.
fn parse_dtype(dtype: &str) -> PyResult<Datatype> {
    match dtype {
        "f32" => Ok(Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        }),
        "f64" => Ok(Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        }),
        "i8" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        "u8" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        "i16" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        "u16" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        "i32" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        "u32" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        "i64" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        "u64" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        other => Err(PyValueError::new_err(format!(
            "unsupported dtype '{other}'; expected one of: \
             f32, f64, i8, u8, i16, u16, i32, u32, i64, u64"
        ))),
    }
}

fn dtype_tag(dt: &Datatype) -> &'static str {
    match dt {
        Datatype::Float { bits, .. } if bits.get() == 32 => "f32",
        Datatype::Float { bits, .. } if bits.get() == 64 => "f64",
        Datatype::Integer { bits, signed: true, .. } if bits.get() == 8 => "i8",
        Datatype::Integer { bits, signed: false, .. } if bits.get() == 8 => "u8",
        Datatype::Integer { bits, signed: true, .. } if bits.get() == 16 => "i16",
        Datatype::Integer { bits, signed: false, .. } if bits.get() == 16 => "u16",
        Datatype::Integer { bits, signed: true, .. } if bits.get() == 32 => "i32",
        Datatype::Integer { bits, signed: false, .. } if bits.get() == 32 => "u32",
        Datatype::Integer { bits, signed: true, .. } if bits.get() == 64 => "i64",
        Datatype::Integer { bits, signed: false, .. } if bits.get() == 64 => "u64",
        Datatype::FixedString { .. } => "str",
        _ => "opaque",
    }
}

// ---------------------------------------------------------------------------
// Byte ↔ Python list conversion
// ---------------------------------------------------------------------------

/// Decode a raw byte buffer into a Python list of numeric values according to
/// `dt`.  Handles both little-endian and big-endian variants.
fn decode_bytes(py: Python<'_>, data: &[u8], dt: &Datatype) -> PyResult<PyObject> {
    let nbytes = match dt.element_size() {
        Some(n) if n > 0 => n,
        _ => {
            return Err(PyValueError::new_err(format!(
                "cannot decode datatype '{}'",
                dtype_tag(dt)
            )));
        }
    };
    if data.len() % nbytes != 0 {
        return Err(PyValueError::new_err(format!(
            "data length {} is not a multiple of element size {}",
            data.len(),
            nbytes
        )));
    }
    let n = data.len() / nbytes;
    let list = PyList::empty_bound(py);
    for i in 0..n {
        let s = &data[i * nbytes..(i + 1) * nbytes];
        match dt {
            Datatype::Float { bits, byte_order } if bits.get() == 32 => {
                let arr: [u8; 4] = s.try_into().unwrap();
                let v = match byte_order {
                    ByteOrder::LittleEndian => f32::from_le_bytes(arr),
                    ByteOrder::BigEndian => f32::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Float { bits, byte_order } if bits.get() == 64 => {
                let arr: [u8; 8] = s.try_into().unwrap();
                let v = match byte_order {
                    ByteOrder::LittleEndian => f64::from_le_bytes(arr),
                    ByteOrder::BigEndian => f64::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer { bits, signed: true, byte_order } if bits.get() == 8 => {
                let v = s[0] as i8;
                list.append(v)?;
            }
            Datatype::Integer { bits, signed: false, .. } if bits.get() == 8 => {
                list.append(s[0])?;
            }
            Datatype::Integer { bits, signed: true, byte_order } if bits.get() == 16 => {
                let arr: [u8; 2] = s.try_into().unwrap();
                let v = match byte_order {
                    ByteOrder::LittleEndian => i16::from_le_bytes(arr),
                    ByteOrder::BigEndian => i16::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer { bits, signed: false, byte_order } if bits.get() == 16 => {
                let arr: [u8; 2] = s.try_into().unwrap();
                let v = match byte_order {
                    ByteOrder::LittleEndian => u16::from_le_bytes(arr),
                    ByteOrder::BigEndian => u16::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer { bits, signed: true, byte_order } if bits.get() == 32 => {
                let arr: [u8; 4] = s.try_into().unwrap();
                let v = match byte_order {
                    ByteOrder::LittleEndian => i32::from_le_bytes(arr),
                    ByteOrder::BigEndian => i32::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer { bits, signed: false, byte_order } if bits.get() == 32 => {
                let arr: [u8; 4] = s.try_into().unwrap();
                let v = match byte_order {
                    ByteOrder::LittleEndian => u32::from_le_bytes(arr),
                    ByteOrder::BigEndian => u32::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer { bits, signed: true, byte_order } if bits.get() == 64 => {
                let arr: [u8; 8] = s.try_into().unwrap();
                let v = match byte_order {
                    ByteOrder::LittleEndian => i64::from_le_bytes(arr),
                    ByteOrder::BigEndian => i64::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            Datatype::Integer { bits, signed: false, byte_order } if bits.get() == 64 => {
                let arr: [u8; 8] = s.try_into().unwrap();
                let v = match byte_order {
                    ByteOrder::LittleEndian => u64::from_le_bytes(arr),
                    ByteOrder::BigEndian => u64::from_be_bytes(arr),
                };
                list.append(v)?;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported datatype variant '{}'",
                    dtype_tag(dt)
                )));
            }
        }
    }
    Ok(list.into_any().unbind())
}

/// Encode a Python list of numbers into little-endian raw bytes for `dt`.
fn encode_bytes(data: &Bound<'_, PyList>, dt: &Datatype) -> PyResult<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::with_capacity(data.len() * dt.element_size().unwrap_or(0));
    for item in data {
        match dt {
            Datatype::Float { bits, .. } if bits.get() == 32 => {
                let v: f32 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Float { bits, .. } if bits.get() == 64 => {
                let v: f64 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer { bits, signed: true, .. } if bits.get() == 8 => {
                let v: i8 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer { bits, signed: false, .. } if bits.get() == 8 => {
                let v: u8 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer { bits, signed: true, .. } if bits.get() == 16 => {
                let v: i16 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer { bits, signed: false, .. } if bits.get() == 16 => {
                let v: u16 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer { bits, signed: true, .. } if bits.get() == 32 => {
                let v: i32 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer { bits, signed: false, .. } if bits.get() == 32 => {
                let v: u32 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer { bits, signed: true, .. } if bits.get() == 64 => {
                let v: i64 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Datatype::Integer { bits, signed: false, .. } if bits.get() == 64 => {
                let v: u64 = item.extract()?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported datatype '{}' for encoding",
                    dtype_tag(dt)
                )));
            }
        }
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// PyNetcdfFile
// ---------------------------------------------------------------------------

/// In-memory netCDF-4 file reader.
///
/// Accepts raw HDF5/netCDF-4 bytes and exposes the root-group structure plus
/// variable data decoding.
///
/// Example::
///
///     with open("data.nc", "rb") as f:
///         nf = PyNetcdfFile(f.read())
///     print(nf.variable_names())
///     print(nf.read_variable("temperature"))
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
        Ok(Self { data: data.to_vec() })
    }

    /// Names of all root-group variables (non-dimension-scale datasets).
    pub fn variable_names(&self) -> PyResult<Vec<String>> {
        let model = self.open_model()?;
        Ok(model.root.variables.iter().map(|v| v.name.clone()).collect())
    }

    /// Names of all root-group dimensions.
    pub fn dimension_names(&self) -> PyResult<Vec<String>> {
        let model = self.open_model()?;
        Ok(model.root.dimensions.iter().map(|d| d.name.clone()).collect())
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
    pub fn read_variable(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
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
    pub fn group_attributes(&self, py: Python<'_>) -> PyResult<Vec<(String, PyObject)>> {
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
) -> PyResult<Vec<(String, PyObject)>> {
    use consus_core::AttributeValue;
    let mut out = Vec::new();
    for (k, v) in attrs {
        let obj: PyObject = match v {
            AttributeValue::Int(n) => (*n).to_object(py),
            AttributeValue::Uint(n) => (*n).to_object(py),
            AttributeValue::Float(f) => (*f).to_object(py),
            AttributeValue::String(s) => s.to_object(py),
            AttributeValue::IntArray(arr) => {
                let list = PyList::new_bound(py, arr.iter());
                list.into_any().unbind()
            }
            AttributeValue::UintArray(arr) => {
                let list = PyList::new_bound(py, arr.iter());
                list.into_any().unbind()
            }
            AttributeValue::FloatArray(arr) => {
                let list = PyList::new_bound(py, arr.iter());
                list.into_any().unbind()
            }
            AttributeValue::StringArray(arr) => {
                let list = PyList::new_bound(py, arr.iter());
                list.into_any().unbind()
            }
            AttributeValue::Bytes(b) => {
                PyBytes::new_bound(py, b).into_any().unbind()
            }
        };
        out.push((k.clone(), obj));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// PyNetcdfWriter
// ---------------------------------------------------------------------------

/// netCDF-4 file builder.
///
/// Add dimensions and variables, then call :meth:`write` to produce a
/// self-contained HDF5/netCDF-4 byte string.
///
/// Example::
///
///     w = PyNetcdfWriter()
///     w.add_dimension("x", 5)
///     w.add_variable("temperature", "f32", ["x"], [1.0, 2.0, 3.0, 4.0, 5.0])
///     raw = w.write()
#[pyclass]
pub struct PyNetcdfWriter {
    group: NetcdfGroup,
}

#[pymethods]
impl PyNetcdfWriter {
    /// Create an empty writer.
    #[new]
    pub fn new() -> Self {
        Self {
            group: NetcdfGroup::new(String::from("/")),
        }
    }

    /// Add a fixed-size dimension with `name` and `size`.
    ///
    /// Raises `ValueError` if a dimension with this name already exists.
    pub fn add_dimension(&mut self, name: &str, size: usize) -> PyResult<()> {
        if self.group.dimensions.iter().any(|d| d.name == name) {
            return Err(PyValueError::new_err(format!(
                "dimension '{name}' already defined"
            )));
        }
        self.group
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
        &mut self,
        name: &str,
        dtype: &str,
        dims: Vec<String>,
        data: &Bound<'_, PyList>,
    ) -> PyResult<()> {
        if self.group.variables.iter().any(|v| v.name == name) {
            return Err(PyValueError::new_err(format!(
                "variable '{name}' already defined"
            )));
        }

        let dt = parse_dtype(dtype)?;

        // Resolve shape from declared dimensions.
        let shape_dims: Vec<usize> = dims
            .iter()
            .map(|d| {
                self.group
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
            .collect::<PyResult<Vec<_>>>()?;

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

        self.group.variables.push(var);
        Ok(())
    }

    /// Serialise the model to netCDF-4/HDF5 bytes.
    pub fn write(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let model = NetcdfModel {
            root: self.group.clone(),
        };
        let bytes = NetcdfWriter::new()
            .write_model(&model)
            .map_err(from_consus)?;
        Ok(PyBytes::new_bound(py, &bytes).unbind())
    }
}
