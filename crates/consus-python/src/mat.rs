//! MATLAB `.mat` Python bindings: `MatFile`, `MatVariable`.

use consus_mat::{loadmat_bytes as consus_loadmat_bytes, MatArray, MatNumericClass};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::error::from_consus_mat;

// ---------------------------------------------------------------------------
// MatVariable — single variable from a .mat file
// ---------------------------------------------------------------------------

/// A single variable loaded from a MATLAB `.mat` file.
#[pyclass]
#[derive(Clone)]
pub struct PyMatVariable {
    /// Variable name as stored in the file.
    #[pyo3(get)]
    pub name: String,
    /// Top-level class: ``"numeric"``, ``"char"``, ``"logical"``,
    /// ``"sparse"``, ``"cell"``, or ``"struct"``.
    #[pyo3(get)]
    pub array_class: String,
    /// Array shape (list of dimension sizes).
    #[pyo3(get)]
    pub shape: Vec<usize>,
    /// For numeric arrays: MATLAB numeric class string (``"double"``,
    /// ``"single"``, ``"int32"``, etc.).  Empty string for non-numeric.
    #[pyo3(get)]
    pub numeric_class: String,
    /// Whether a numeric array has an imaginary component.
    #[pyo3(get)]
    pub is_complex: bool,
    /// Text content of a character array.  Empty string for non-char.
    #[pyo3(get)]
    pub text: String,
    /// Boolean elements of a logical array.  Empty for non-logical.
    #[pyo3(get)]
    pub bools: Vec<bool>,
    /// Field names of a struct array.  Empty for non-struct.
    #[pyo3(get)]
    pub field_names: Vec<String>,
    // Raw bytes stored for read_data().
    raw_data: Vec<u8>,
}

#[pymethods]
impl PyMatVariable {
    /// Raw little-endian bytes of the real part of a numeric array.
    ///
    /// Returns empty bytes for non-numeric variables.
    fn read_data<'py>(&self, py: Python<'py>) -> Py<PyBytes> {
        PyBytes::new_bound(py, &self.raw_data).unbind()
    }
}

// ---------------------------------------------------------------------------
// MatFile — parsed .mat file
// ---------------------------------------------------------------------------

/// A parsed MATLAB `.mat` file.
///
/// Obtain via :func:`loadmat_bytes`.
#[pyclass]
pub struct PyMatFile {
    /// Detected container version: ``"v4"``, ``"v5"``, or ``"v7.3"``.
    #[pyo3(get)]
    pub version: String,
    // Stored as plain structs; Python objects created on demand.
    variables: Vec<PyMatVariable>,
}

#[pymethods]
impl PyMatFile {
    /// List all top-level variable names in file order.
    fn variable_names(&self) -> Vec<String> {
        self.variables.iter().map(|v| v.name.clone()).collect()
    }

    /// Retrieve a variable by name.  Returns ``None`` when not found.
    fn get_variable(&self, name: &str, py: Python<'_>) -> PyResult<Option<Py<PyMatVariable>>> {
        match self.variables.iter().find(|v| v.name == name) {
            Some(v) => Ok(Some(Py::new(py, v.clone())?)),
            None => Ok(None),
        }
    }

    /// All variables as a list of :class:`MatVariable` objects.
    fn variables(&self, py: Python<'_>) -> PyResult<Vec<Py<PyMatVariable>>> {
        self.variables
            .iter()
            .map(|v| Py::new(py, v.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// loadmat_bytes — module-level function
// ---------------------------------------------------------------------------

/// Parse a MATLAB `.mat` file from raw bytes.
///
/// Supports MAT v4, v5, and v7.3 (HDF5-backed) containers.  Returns a
/// :class:`MatFile` with all top-level variables.
#[pyfunction]
pub fn loadmat_bytes(data: &[u8]) -> PyResult<PyMatFile> {
    let mat = consus_loadmat_bytes(data).map_err(from_consus_mat)?;
    let version = match mat.version {
        consus_mat::MatVersion::V4 => "v4",
        consus_mat::MatVersion::V5 => "v5",
        consus_mat::MatVersion::V73 => "v7.3",
    }
    .to_string();
    let variables: Vec<PyMatVariable> = mat
        .variables
        .into_iter()
        .map(|(name, array)| build_var(name, array))
        .collect();
    Ok(PyMatFile { version, variables })
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn numeric_class_str(class: MatNumericClass) -> &'static str {
    class.as_str()
}

fn build_var(name: String, array: MatArray) -> PyMatVariable {
    match array {
        MatArray::Numeric(n) => {
            let shape = n.shape.clone();
            let numeric_class = numeric_class_str(n.class).to_string();
            let is_complex = n.is_complex();
            let raw_data = n.real_data.clone();
            PyMatVariable {
                name,
                array_class: "numeric".into(),
                shape,
                numeric_class,
                is_complex,
                text: String::new(),
                bools: Vec::new(),
                field_names: Vec::new(),
                raw_data,
            }
        }
        MatArray::Char(c) => {
            let shape = c.shape.clone();
            let text = c.data.clone();
            PyMatVariable {
                name,
                array_class: "char".into(),
                shape,
                numeric_class: String::new(),
                is_complex: false,
                text,
                bools: Vec::new(),
                field_names: Vec::new(),
                raw_data: Vec::new(),
            }
        }
        MatArray::Logical(l) => {
            let shape = l.shape.clone();
            let bools = l.data.clone();
            PyMatVariable {
                name,
                array_class: "logical".into(),
                shape,
                numeric_class: String::new(),
                is_complex: false,
                text: String::new(),
                bools,
                field_names: Vec::new(),
                raw_data: Vec::new(),
            }
        }
        MatArray::Sparse(s) => {
            let shape = vec![s.nrows, s.ncols];
            PyMatVariable {
                name,
                array_class: "sparse".into(),
                shape,
                numeric_class: String::new(),
                is_complex: s.is_complex(),
                text: String::new(),
                bools: Vec::new(),
                field_names: Vec::new(),
                raw_data: Vec::new(),
            }
        }
        MatArray::Cell(c) => {
            let shape = c.shape().to_vec();
            PyMatVariable {
                name,
                array_class: "cell".into(),
                shape,
                numeric_class: String::new(),
                is_complex: false,
                text: String::new(),
                bools: Vec::new(),
                field_names: Vec::new(),
                raw_data: Vec::new(),
            }
        }
        MatArray::Struct(s) => {
            let shape = s.shape.clone();
            let field_names: Vec<String> = s.field_names().map(|f| f.to_string()).collect();
            PyMatVariable {
                name,
                array_class: "struct".into(),
                shape,
                numeric_class: String::new(),
                is_complex: false,
                text: String::new(),
                bools: Vec::new(),
                field_names,
                raw_data: Vec::new(),
            }
        }
    }
}
