//! HDF5 Python bindings: `Hdf5File`, `FileBuilder`, `DatasetInfo`.
//!
//! Uses `MemCursor` for in-memory round-trips so no filesystem access is required.

use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Compression, Datatype, Shape};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use consus_hdf5::file::Hdf5File;
use consus_hdf5::property_list::DatasetLayout;
use consus_io::MemCursor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::error::from_consus;

// ---------------------------------------------------------------------------
// DatasetInfo — read-only view of dataset metadata
// ---------------------------------------------------------------------------

/// Metadata for a single HDF5 dataset.
#[pyclass]
pub struct PyDatasetInfo {
    #[pyo3(get)]
    pub address: u64,
    #[pyo3(get)]
    pub dtype: String,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub layout: String,
    #[pyo3(get)]
    pub chunk_shape: Option<Vec<usize>>,
    #[pyo3(get)]
    pub filters: Vec<u16>,
}

fn dtype_str(dt: &Datatype) -> String {
    match dt {
        Datatype::Boolean => "bool".into(),
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            let order = if *byte_order == ByteOrder::LittleEndian {
                "<"
            } else {
                ">"
            };
            let sign = if *signed { "i" } else { "u" };
            format!("{order}{sign}{}", bits.get() / 8)
        }
        Datatype::Float { bits, byte_order } => {
            let order = if *byte_order == ByteOrder::LittleEndian {
                "<"
            } else {
                ">"
            };
            format!("{order}f{}", bits.get() / 8)
        }
        Datatype::FixedString { length, .. } => format!("S{length}"),
        Datatype::VariableString { .. } => "object".into(),
        _ => "opaque".into(),
    }
}

fn layout_str(l: StorageLayout) -> &'static str {
    match l {
        StorageLayout::Contiguous => "contiguous",
        StorageLayout::Chunked => "chunked",
        StorageLayout::Compact => "compact",
        StorageLayout::Virtual => "virtual",
    }
}

// ---------------------------------------------------------------------------
// Hdf5File — in-memory HDF5 reader
// ---------------------------------------------------------------------------

/// In-memory HDF5 file reader.
///
/// Instantiate via :meth:`Hdf5File.open_path` with the raw file bytes.
#[pyclass]
pub struct PyHdf5File {
    data: Vec<u8>,
}

#[pymethods]
impl PyHdf5File {
    /// Construct from raw HDF5 bytes.
    #[staticmethod]
    fn open_path(data: &[u8]) -> PyResult<Self> {
        Ok(Self {
            data: data.to_vec(),
        })
    }

    /// List direct children of the root group. Returns a list of names.
    fn list_root_group(&self) -> PyResult<Vec<String>> {
        let cursor = MemCursor::from_bytes(self.data.clone());
        let file = Hdf5File::open(cursor).map_err(from_consus)?;
        let children = file.list_root_group().map_err(from_consus)?;
        Ok(children.into_iter().map(|(name, _, _)| name).collect())
    }

    /// List children of the group at `path`. Returns a list of names.
    fn list_group_at(&self, path: &str) -> PyResult<Vec<String>> {
        let cursor = MemCursor::from_bytes(self.data.clone());
        let file = Hdf5File::open(cursor).map_err(from_consus)?;
        let address = file.open_path(path).map_err(from_consus)?;
        let children = file.list_group_at(address).map_err(from_consus)?;
        Ok(children.into_iter().map(|(name, _, _)| name).collect())
    }

    /// Return dataset metadata for the object at `path`.
    fn dataset_at(&self, path: &str) -> PyResult<PyDatasetInfo> {
        let cursor = MemCursor::from_bytes(self.data.clone());
        let file = Hdf5File::open(cursor).map_err(from_consus)?;
        let address = file.open_path(path).map_err(from_consus)?;
        let ds = file.dataset_at(address).map_err(from_consus)?;
        let shape: Vec<usize> = ds.shape.current_dims().to_vec();
        let chunk_shape = ds.chunk_shape.as_ref().map(|cs| cs.dims().to_vec());
        Ok(PyDatasetInfo {
            address,
            dtype: dtype_str(&ds.datatype),
            shape,
            layout: layout_str(ds.layout).into(),
            chunk_shape,
            filters: ds.filters,
        })
    }

    /// Read the raw byte payload of the dataset at `path`.
    fn read_dataset(&self, path: &str, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let cursor = MemCursor::from_bytes(self.data.clone());
        let file = Hdf5File::open(cursor).map_err(from_consus)?;
        let address = file.open_path(path).map_err(from_consus)?;
        let ds = file.dataset_at(address).map_err(from_consus)?;
        let bytes = match ds.layout {
            StorageLayout::Chunked => file
                .read_chunked_dataset_all_bytes(address)
                .map_err(from_consus)?,
            StorageLayout::Contiguous | StorageLayout::Compact => {
                let elem_size = ds.datatype.element_size().ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "variable-length datasets are not supported by read_dataset",
                    )
                })?;
                let total = ds.shape.num_elements() * elem_size;
                let data_address = ds.data_address.ok_or_else(|| {
                    PyRuntimeError::new_err("contiguous dataset has no data address")
                })?;
                let mut buf = vec![0u8; total];
                file.read_contiguous_dataset_bytes(data_address, 0, &mut buf)
                    .map_err(from_consus)?;
                buf
            }
            StorageLayout::Virtual => {
                return Err(PyRuntimeError::new_err(
                    "virtual datasets are not supported",
                ));
            }
        };
        Ok(PyBytes::new_bound(py, &bytes).unbind())
    }

    /// Return attributes of the object at `path` as a ``dict[str, bytes]``.
    fn attributes_at(&self, path: &str, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let cursor = MemCursor::from_bytes(self.data.clone());
        let file = Hdf5File::open(cursor).map_err(from_consus)?;
        let address = file.open_path(path).map_err(from_consus)?;
        let attrs = file.attributes_at(address).map_err(from_consus)?;
        let dict = PyDict::new_bound(py);
        for attr in attrs {
            let value = PyBytes::new_bound(py, &attr.raw_data);
            dict.set_item(&attr.name, value)?;
        }
        Ok(dict.unbind())
    }
}

// ---------------------------------------------------------------------------
// FileBuilder — HDF5 file writer
// ---------------------------------------------------------------------------

fn parse_dtype(dtype: &str) -> PyResult<Datatype> {
    let s = dtype.trim();
    let (order, rest) = if let Some(rest) = s.strip_prefix('<') {
        (ByteOrder::LittleEndian, rest)
    } else if let Some(rest) = s.strip_prefix('>') {
        (ByteOrder::BigEndian, rest)
    } else {
        (ByteOrder::LittleEndian, s)
    };
    match rest {
        "f2" => Ok(Datatype::Float {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: order,
        }),
        "f4" => Ok(Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: order,
        }),
        "f8" => Ok(Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: order,
        }),
        "i1" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: order,
            signed: true,
        }),
        "i2" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: order,
            signed: true,
        }),
        "i4" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: order,
            signed: true,
        }),
        "i8" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: order,
            signed: true,
        }),
        "u1" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: order,
            signed: false,
        }),
        "u2" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: order,
            signed: false,
        }),
        "u4" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: order,
            signed: false,
        }),
        "u8" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: order,
            signed: false,
        }),
        "bool" | "bool_" => Ok(Datatype::Boolean),
        _ => Err(PyValueError::new_err(format!("unsupported dtype: {dtype}"))),
    }
}

fn parse_compression(compression: &str) -> PyResult<Compression> {
    match compression {
        "none" | "None" => Ok(Compression::None),
        "gzip" | "deflate" => Ok(Compression::Gzip { level: 6 }),
        "lzf" | "lz4" => Ok(Compression::Lz4),
        _ => Err(PyValueError::new_err(format!(
            "unsupported compression: {compression}"
        ))),
    }
}

/// HDF5 file writer.
///
/// Build datasets and call :meth:`FileBuilder.finish` to obtain the HDF5 bytes.
#[pyclass]
pub struct PyFileBuilder {
    builder: Option<Hdf5FileBuilder>,
}

#[pymethods]
impl PyFileBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: Some(Hdf5FileBuilder::new(FileCreationProps::default())),
        }
    }

    /// Add a dataset to the root group.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Dataset name within the root group.
    /// dtype : str
    ///     Dtype string e.g. ``"<f4"``, ``"<i4"``, ``">u2"``, ``"bool"``.
    /// shape : list[int]
    ///     Dataset shape as a list of dimension sizes.
    /// data : bytes
    ///     Raw dataset payload bytes (must match dtype × product(shape)).
    /// layout : str, optional
    ///     Storage layout: ``"contiguous"`` (default) or ``"chunked"``.
    /// chunk_dims : list[int] | None, optional
    ///     Chunk dimensions (required when ``layout="chunked"``).
    /// compression : str, optional
    ///     Compression codec: ``"none"`` (default), ``"gzip"``, ``"lzf"``.
    /// layout_version : int | None, optional
    ///     Layout message version (3 or 4). ``None`` selects version 3.
    #[pyo3(signature = (name, dtype, shape, data, layout="contiguous", chunk_dims=None, compression="none", layout_version=None))]
    fn add_dataset(
        &mut self,
        name: &str,
        dtype: &str,
        shape: Vec<u64>,
        data: &[u8],
        layout: &str,
        chunk_dims: Option<Vec<u64>>,
        compression: &str,
        layout_version: Option<u8>,
    ) -> PyResult<()> {
        let dt = parse_dtype(dtype)?;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let sh = Shape::fixed(&dims);
        let comp = parse_compression(compression)?;
        let dl = match layout {
            "contiguous" => DatasetLayout::Contiguous,
            "chunked" => DatasetLayout::Chunked,
            "compact" => DatasetLayout::Compact,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported layout: {layout}"
                )))
            }
        };
        let chunk_dims_usize = chunk_dims.map(|v| v.into_iter().map(|d| d as usize).collect());
        let dcpl = DatasetCreationProps {
            layout: dl,
            chunk_dims: chunk_dims_usize,
            compression: comp,
            layout_version,
            ..Default::default()
        };
        self.builder
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("FileBuilder already consumed by finish()"))?
            .add_dataset(name, &dt, &sh, data, &dcpl)
            .map_err(from_consus)?;
        Ok(())
    }

    /// Finalize the HDF5 file and return the raw bytes.
    fn finish(&mut self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let builder = self
            .builder
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("FileBuilder already consumed by finish()"))?;
        let bytes = builder.finish().map_err(from_consus)?;
        Ok(PyBytes::new_bound(py, &bytes).unbind())
    }
}
