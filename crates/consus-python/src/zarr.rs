//! Zarr v2 in-memory array Python bindings.
//!
//! Exposes `ZarrArray` which wraps an `InMemoryStore` + `ArrayMetadata`.

use consus_zarr::{
    ArrayMetadata, ArrayMetadataV2, ChunkError, ChunkKeySeparator, InMemoryStore, Store,
    read_chunk, write_chunk,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::error::from_consus;

// ---------------------------------------------------------------------------
// Helper: build .zarray JSON
// ---------------------------------------------------------------------------

fn make_zarray_json(
    shape: &[u64],
    chunks: &[u64],
    dtype: &str,
    compressor: &str,
) -> String {
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let chunks_str = chunks
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let compressor_json: &str = match compressor {
        "none" | "None" => "null",
        "gzip" => r#"{"id":"gzip","level":6}"#,
        _ => "null",
    };
    format!(
        r#"{{"zarr_format":2,"shape":[{shape_str}],"chunks":[{chunks_str}],"dtype":"{dtype}","compressor":{compressor_json},"fill_value":0,"order":"C","filters":null}}"#
    )
}

// ---------------------------------------------------------------------------
// ZarrArray
// ---------------------------------------------------------------------------

/// In-memory Zarr v2 array.
///
/// Create with :meth:`ZarrArray.__init__`, write chunks via :meth:`write_chunk`,
/// and read them back via :meth:`read_chunk`. Export to a zarr-python
/// compatible store dict via :meth:`to_store`, or import from one via
/// :meth:`ZarrArray.from_store`.
#[pyclass]
pub struct PyZarrArray {
    store: InMemoryStore,
    meta: ArrayMetadata,
}

#[pymethods]
impl PyZarrArray {
    /// Create a new in-memory Zarr v2 array.
    ///
    /// Parameters
    /// ----------
    /// shape : list[int]
    ///     Array dimensions.
    /// chunks : list[int]
    ///     Chunk dimensions (must have the same rank as ``shape``).
    /// dtype : str
    ///     Zarr dtype string, e.g. ``"<f8"``, ``"<i4"``.
    /// compressor : str, optional
    ///     Compression codec: ``"none"`` (default) or ``"gzip"``.
    #[new]
    #[pyo3(signature = (shape, chunks, dtype, compressor = "none"))]
    fn new(
        shape: Vec<u64>,
        chunks: Vec<u64>,
        dtype: &str,
        compressor: &str,
    ) -> PyResult<Self> {
        let json = make_zarray_json(&shape, &chunks, dtype, compressor);
        let mut store = InMemoryStore::new();
        store
            .set(".zarray", json.as_bytes())
            .map_err(from_consus)?;
        let v2 = ArrayMetadataV2::parse(&json).map_err(|e| {
            PyRuntimeError::new_err(format!("invalid zarr metadata: {e:?}"))
        })?;
        let meta = v2.to_canonical();
        Ok(Self { store, meta })
    }

    /// Array shape.
    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.meta.shape.iter().map(|&d| d as u64).collect()
    }

    /// Chunk shape.
    #[getter]
    fn chunks(&self) -> Vec<u64> {
        self.meta.chunks.iter().map(|&d| d as u64).collect()
    }

    /// Element dtype string.
    #[getter]
    fn dtype(&self) -> &str {
        self.meta.dtype.as_str()
    }

    /// Total number of chunks across all dimensions.
    fn num_chunks(&self) -> usize {
        self.meta
            .shape
            .iter()
            .zip(self.meta.chunks.iter())
            .map(|(&s, &c)| s.div_ceil(c))
            .product()
    }

    /// Canonical key string for the chunk at the given coordinates.
    fn chunk_key(&self, coords: Vec<u64>) -> String {
        let usize_coords: Vec<usize> = coords.iter().map(|&c| c as usize).collect();
        consus_zarr::chunk_key(&usize_coords, ChunkKeySeparator::Dot)
    }

    /// Write raw `data` bytes for the chunk at `coords`.
    fn write_chunk(&mut self, coords: Vec<u64>, data: &[u8]) -> PyResult<()> {
        write_chunk(&mut self.store, "", &coords, &self.meta, data)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Read the raw bytes for the chunk at `coords`.
    fn read_chunk(&self, coords: Vec<u64>, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let data = read_chunk(&self.store, "", &coords, &self.meta)
            .map_err(|e: ChunkError| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(PyBytes::new_bound(py, &data).unbind())
    }

    /// Export the in-memory store as a ``dict[str, bytes]``.
    ///
    /// The returned dict is directly compatible with zarr-python's
    /// ``MemoryStore`` after wrapping values in
    /// ``zarr.core.buffer.cpu.Buffer.from_bytes(v)``.
    fn to_store(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        for key in self.store.keys() {
            let value = self.store.get(&key).map_err(from_consus)?;
            dict.set_item(key, PyBytes::new_bound(py, &value))?;
        }
        Ok(dict.unbind())
    }

    /// Construct a ``ZarrArray`` from a ``dict[str, bytes]`` store.
    ///
    /// Parameters
    /// ----------
    /// entries : dict[str, bytes]
    ///     Raw store entries. The dict must contain a ``.zarray`` key.
    /// prefix : str, optional
    ///     Key prefix (default ``""``).
    #[staticmethod]
    #[pyo3(signature = (entries, prefix = ""))]
    fn from_store(entries: &Bound<'_, PyDict>, prefix: &str) -> PyResult<Self> {
        let mut store = InMemoryStore::new();
        for (k, v) in entries.iter() {
            let key: String = k.extract()?;
            let val: &[u8] = v.extract()?;
            store.set(&key, val).map_err(from_consus)?;
        }
        let zarray_key = if prefix.is_empty() {
            ".zarray".to_string()
        } else {
            format!("{prefix}/.zarray")
        };
        let raw = store.get(&zarray_key).map_err(|_| {
            PyValueError::new_err(format!("store missing key: {zarray_key}"))
        })?;
        let json = core::str::from_utf8(&raw)
            .map_err(|_| PyRuntimeError::new_err(".zarray is not valid UTF-8"))?;
        let v2 = ArrayMetadataV2::parse(json).map_err(|e| {
            PyRuntimeError::new_err(format!("invalid .zarray JSON: {e:?}"))
        })?;
        let meta = v2.to_canonical();
        Ok(Self { store, meta })
    }
}
