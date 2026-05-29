use consus_core::LinkType;
use consus_hdf5::{
    dataset::{
        layout::DataLayout,
        StorageLayout,
    },
    file::{reader, Hdf5File},
    object_header::message_types,
};
use consus_io::MemCursor;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{dtype::dtype_to_str, error::from_consus};

use super::info::PyDatasetInfo;

/// An open HDF5 file backed by an in-memory byte buffer.
#[pyclass(name = "Hdf5File")]
pub struct PyHdf5File {
    inner: Hdf5File<MemCursor>,
}

#[pymethods]
impl PyHdf5File {
    /// Construct an `Hdf5File` from raw HDF5 bytes.
    #[new]
    fn new(data: &[u8]) -> PyResult<Self> {
        from_bytes(data)
    }

    /// List the root group.
    ///
    /// Returns `[(name, address, link_type)]` where `link_type` is
    /// one of `"hard"`, `"soft"`, or `"external"`.
    fn list_root_group(&self) -> PyResult<Vec<(String, u64, String)>> {
        self.inner
            .list_root_group()
            .map_err(from_consus)
            .map(|v| v.into_iter().map(|(n, a, lt)| (n, a, link_type_str(lt))).collect())
    }

    /// List the group at `addr`.
    fn list_group_at(&self, addr: u64) -> PyResult<Vec<(String, u64, String)>> {
        self.inner
            .list_group_at(addr)
            .map_err(from_consus)
            .map(|v| v.into_iter().map(|(n, a, lt)| (n, a, link_type_str(lt))).collect())
    }

    /// Resolve a `/`-separated path to an object header address.
    fn open_path(&self, path: &str) -> PyResult<u64> {
        self.inner.open_path(path).map_err(from_consus)
    }

    /// Return `DatasetInfo` for the dataset at `addr`.
    fn dataset_at(&self, addr: u64) -> PyResult<PyDatasetInfo> {
        self.inner.dataset_at(addr).map(PyDatasetInfo::from).map_err(from_consus)
    }

    /// Read all raw bytes for the dataset at `addr`.
    ///
    /// Dispatches automatically based on the storage layout
    /// (contiguous, compact, or chunked).
    fn read_dataset(&self, addr: u64) -> PyResult<Vec<u8>> {
        let ds = self.inner.dataset_at(addr).map_err(from_consus)?;
        match ds.layout {
            StorageLayout::Contiguous => {
                let data_addr = ds.data_address.ok_or_else(|| {
                    PyRuntimeError::new_err("contiguous dataset has no data_address")
                })?;
                let element_bytes = ds.datatype.element_size().ok_or_else(|| {
                    PyRuntimeError::new_err("variable-length datatype not supported for read_dataset")
                })?;
                let total = ds.shape.num_elements() * element_bytes;
                let mut buf = vec![0u8; total];
                self.inner
                    .read_contiguous_dataset_bytes(data_addr, 0, &mut buf)
                    .map_err(from_consus)?;
                Ok(buf)
            }
            StorageLayout::Compact => {
                let header = reader::read_object_header(
                    self.inner.source(),
                    addr,
                    self.inner.context(),
                )
                .map_err(from_consus)?;
                let msg = reader::find_message(&header, message_types::DATA_LAYOUT)
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("compact dataset missing DATA_LAYOUT message")
                    })?;
                let layout =
                    DataLayout::parse(&msg.data, self.inner.context()).map_err(from_consus)?;
                layout.compact_data.ok_or_else(|| {
                    PyRuntimeError::new_err("compact_data is absent in compact dataset")
                })
            }
            StorageLayout::Chunked => {
                self.inner.read_chunked_dataset_all_bytes(addr).map_err(from_consus)
            }
            StorageLayout::Virtual => Err(PyRuntimeError::new_err(
                "virtual datasets cannot be read with read_dataset",
            )),
        }
    }

    /// Read attributes at `addr`.
    ///
    /// Returns `[(name, dtype_str, raw_bytes)]`.
    fn attributes_at(&self, addr: u64) -> PyResult<Vec<(String, String, Vec<u8>)>> {
        self.inner
            .attributes_at(addr)
            .map_err(from_consus)
            .map(|v| {
                v.into_iter()
                    .map(|attr| (attr.name, dtype_to_str(&attr.datatype), attr.raw_data))
                    .collect()
            })
    }
}

pub fn from_bytes(data: &[u8]) -> PyResult<PyHdf5File> {
    let cursor = MemCursor::from_bytes(data.to_vec());
    let inner = Hdf5File::open(cursor).map_err(from_consus)?;
    Ok(PyHdf5File { inner })
}

fn link_type_str(lt: LinkType) -> String {
    match lt {
        LinkType::Hard => "hard".to_owned(),
        LinkType::Soft => "soft".to_owned(),
        LinkType::External => "external".to_owned(),
    }
}
