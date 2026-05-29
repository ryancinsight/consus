use consus_hdf5::dataset::{Hdf5Dataset, StorageLayout};
use pyo3::prelude::*;

use crate::dtype::dtype_to_str;

/// Metadata for an HDF5 dataset.
#[pyclass(name = "DatasetInfo")]
pub struct PyDatasetInfo {
    #[pyo3(get)]
    pub dtype: String,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub layout: String,
    #[pyo3(get)]
    pub filters: Vec<u16>,
    #[pyo3(get)]
    pub chunk_shape: Option<Vec<usize>>,
    #[pyo3(get)]
    pub address: u64,
}

#[pymethods]
impl PyDatasetInfo {
    fn __repr__(&self) -> String {
        format!(
            "DatasetInfo(dtype={:?}, shape={:?}, layout={:?})",
            self.dtype, self.shape, self.layout
        )
    }
}

impl From<Hdf5Dataset> for PyDatasetInfo {
    fn from(ds: Hdf5Dataset) -> Self {
        let dtype = dtype_to_str(&ds.datatype);
        let shape = ds.shape.current_dims().to_vec();
        let layout = layout_str(ds.layout);
        let chunk_shape = ds.chunk_shape.map(|cs| cs.dims().to_vec());
        Self {
            dtype,
            shape,
            layout,
            filters: ds.filters,
            chunk_shape,
            address: ds.object_header_address,
        }
    }
}

fn layout_str(layout: StorageLayout) -> String {
    match layout {
        StorageLayout::Compact => "compact".to_owned(),
        StorageLayout::Contiguous => "contiguous".to_owned(),
        StorageLayout::Chunked => "chunked".to_owned(),
        StorageLayout::Virtual => "virtual".to_owned(),
    }
}
