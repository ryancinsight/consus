use consus_core::{Compression, Shape};
use consus_hdf5::{
    file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder},
    property_list::DatasetLayout,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{dtype::parse_dtype, error::from_consus};

/// Parse a compression specifier string into a `Compression` value.
///
/// Accepted: `"none"`, `"lz4"`, `"deflate"`, `"deflate:<level>"`,
/// `"zstd"`, `"zstd:<level>"`.
fn parse_compression(s: &str) -> PyResult<Compression> {
    if s == "none" {
        return Ok(Compression::None);
    }
    if s == "lz4" {
        return Ok(Compression::Lz4);
    }
    if let Some(rest) = s.strip_prefix("deflate") {
        let level = if rest.is_empty() {
            6u32
        } else if let Some(n) = rest.strip_prefix(':') {
            n.parse::<u32>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid deflate level in {s:?}"
                ))
            })?
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid compression spec {s:?}"
            )));
        };
        return Ok(Compression::Deflate { level });
    }
    if let Some(rest) = s.strip_prefix("zstd") {
        let level = if rest.is_empty() {
            3i32
        } else if let Some(n) = rest.strip_prefix(':') {
            n.parse::<i32>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid zstd level in {s:?}"
                ))
            })?
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid compression spec {s:?}"
            )));
        };
        return Ok(Compression::Zstd { level });
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "unsupported compression {s:?}; accepted: none, deflate[:N], zstd[:N], lz4"
    )))
}

fn dims_to_shape(dims: Vec<usize>) -> Shape {
    if dims.is_empty() { Shape::scalar() } else { Shape::fixed(&dims) }
}

/// Builder for writing new HDF5 files.
///
/// ```python
/// builder = consus.FileBuilder()
/// builder.add_dataset("x", "<f4", [1000], raw_bytes)
/// hdf5_bytes = builder.finish()
/// ```
#[pyclass(name = "FileBuilder")]
pub struct PyFileBuilder {
    inner: Option<Hdf5FileBuilder>,
}

#[pymethods]
impl PyFileBuilder {
    #[new]
    fn new() -> Self {
        Self { inner: Some(Hdf5FileBuilder::new(FileCreationProps::default())) }
    }

    /// Add a dataset to the root group.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Dataset name within the root group.
    /// dtype : str
    ///     Dtype string e.g. `"<f4"`, `"<i4"`, `">u2"`, `"bool"`.
    /// shape : list[int]
    ///     Dimension sizes.  Empty list produces a scalar dataset.
    /// data : bytes
    ///     Raw data bytes in the dataset's native byte order.
    /// layout : str, optional
    ///     `"contiguous"` (default), `"compact"`, or `"chunked"`.
    /// chunk_dims : list[int] | None, optional
    ///     Chunk shape; required when `layout="chunked"`.
    /// compression : str, optional
    ///     `"none"` (default), `"deflate"`, `"deflate:<level>"`,
    ///     `"zstd"`, `"zstd:<level>"`, or `"lz4"`.
    /// layout_version : int | None, optional
    ///     Layout message version (3 or 4); `None` selects version 3.
    #[pyo3(signature = (name, dtype, shape, data, layout="contiguous", chunk_dims=None, compression="none", layout_version=None))]
    fn add_dataset(
        &mut self,
        name: &str,
        dtype: &str,
        shape: Vec<usize>,
        data: &[u8],
        layout: &str,
        chunk_dims: Option<Vec<usize>>,
        compression: &str,
        layout_version: Option<u8>,
    ) -> PyResult<()> {
        let dt = parse_dtype(dtype)?;
        let sh = dims_to_shape(shape);
        let compression = parse_compression(compression)?;

        let (dl, cdims) = match layout {
            "contiguous" => (DatasetLayout::Contiguous, None),
            "compact" => (DatasetLayout::Compact, None),
            "chunked" => {
                let cdims = chunk_dims.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "chunk_dims is required when layout=\"chunked\"",
                    )
                })?;
                (DatasetLayout::Chunked, Some(cdims))
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown layout {other:?}; accepted: contiguous, compact, chunked"
                )))
            }
        };

        let dcpl = DatasetCreationProps {
            layout: dl,
            chunk_dims: cdims,
            compression,
            layout_version,
            ..Default::default()
        };

        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("FileBuilder already consumed by finish()"))?
            .add_dataset(name, &dt, &sh, data, &dcpl)
            .map_err(from_consus)?;
        Ok(())
    }

    /// Finalise the file and return its bytes.
    ///
    /// The builder is consumed; calling any method after `finish()`
    /// raises `RuntimeError`.
    fn finish(&mut self) -> PyResult<Vec<u8>> {
        self.inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("FileBuilder already consumed by finish()"))?
            .finish()
            .map_err(from_consus)
    }
}
