use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::highlevel::file::UnifiedBackend;
use crate::sync::{ByteView, IoRange, selection_byte_len};
use consus_core::{ChunkShape, Compression, Datatype, Error, Result, Selection, Shape};

/// Unified format-agnostic dataset facade.
///
/// This type contains no format-specific logic. It stores only:
/// - an erased backend handle,
/// - the absolute dataset path,
/// - the selected parallelism policy inherited from the owning file/group.
///
/// All concrete storage behavior is delegated through [`UnifiedBackend`].
pub struct Dataset {
    backend: Arc<dyn UnifiedBackend>,
    path: String,
    parallelism: crate::sync::Parallelism,
}

/// Immutable dataset metadata snapshot.
///
/// This is a backend-neutral value object derived from the canonical
/// `consus-core` abstractions.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetMetadata {
    /// Absolute dataset path.
    pub path: String,
    /// Canonical element datatype.
    pub datatype: Datatype,
    /// Dataset shape.
    pub shape: Shape,
    /// Chunk shape when the dataset is chunked.
    pub chunk_shape: Option<ChunkShape>,
    /// Compression configuration.
    pub compression: Compression,
}

/// Result of a dataset read operation.
///
/// `ZeroCopy` preserves backend-provided borrowed-or-shared storage.
/// `Owned` contains materialized bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionRead {
    /// Backend-provided zero-copy bytes.
    ZeroCopy(crate::highlevel::file::ZeroCopyBytes),
    /// Materialized owned bytes.
    Owned(Vec<u8>),
}

impl SelectionRead {
    /// Returns the bytes as a slice.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::ZeroCopy(bytes) => bytes.as_slice(),
            Self::Owned(bytes) => bytes.as_slice(),
        }
    }

    /// Converts the read result into owned bytes.
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            Self::ZeroCopy(bytes) => bytes.into_owned(),
            Self::Owned(bytes) => bytes,
        }
    }

    /// Returns whether the result preserves zero-copy storage.
    pub fn is_zero_copy(&self) -> bool {
        matches!(self, Self::ZeroCopy(_))
    }

    /// Returns the byte length.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns whether the result is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Declarative read plan for a dataset selection.
///
/// This type is format-neutral. It describes whether the facade can satisfy
/// the request through a zero-copy backend path or through explicit byte-range
/// reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadPlan {
    /// Backend can satisfy the request through its zero-copy path.
    ZeroCopy,
    /// Request can be decomposed into independent byte ranges.
    Parallel(Vec<IoRange>),
    /// Request requires a single materialized read.
    Materialized,
}

impl Dataset {
    /// Creates a new dataset facade from an erased backend and absolute path.
    pub(crate) fn new(
        backend: Arc<dyn UnifiedBackend>,
        path: String,
        parallelism: crate::sync::Parallelism,
    ) -> Self {
        Self {
            backend,
            path,
            parallelism,
        }
    }

    /// Returns the dataset leaf name.
    pub fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or("")
    }

    /// Returns the absolute dataset path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the backend format identifier.
    pub fn format(&self) -> &str {
        self.backend.format()
    }

    /// Returns the configured parallelism policy.
    pub fn parallelism(&self) -> &crate::sync::Parallelism {
        &self.parallelism
    }

    /// Returns the canonical element datatype.
    pub fn datatype(&self) -> Result<Datatype> {
        self.backend.file_read().dataset_datatype(&self.path)
    }

    /// Returns the dataset shape.
    pub fn shape(&self) -> Result<Shape> {
        self.backend.file_read().dataset_shape(&self.path)
    }

    /// Returns the chunk shape when available.
    ///
    /// The current backend-neutral `FileRead` abstraction does not expose chunk
    /// metadata directly. This facade therefore returns `Ok(None)` unless a
    /// future backend-neutral extension provides it.
    pub fn chunk_shape(&self) -> Result<Option<ChunkShape>> {
        Ok(None)
    }

    /// Returns the compression configuration.
    ///
    /// The current backend-neutral `FileRead` abstraction does not expose
    /// compression metadata directly. This facade therefore returns
    /// `Compression::None` as the conservative canonical default.
    pub fn compression(&self) -> Result<Compression> {
        Ok(Compression::None)
    }

    /// Returns a metadata snapshot.
    pub fn metadata(&self) -> Result<DatasetMetadata> {
        Ok(DatasetMetadata {
            path: self.path.clone(),
            datatype: self.datatype()?,
            shape: self.shape()?,
            chunk_shape: self.chunk_shape()?,
            compression: self.compression()?,
        })
    }

    /// Reads raw bytes for `selection` into `buf`.
    pub fn read_raw_into(&self, selection: &Selection, buf: &mut [u8]) -> Result<usize> {
        let metadata = self.metadata()?;
        let required = selection_byte_len(&metadata.datatype, &metadata.shape, selection)?;
        if buf.len() < required {
            return Err(Error::BufferTooSmall {
                required,
                provided: buf.len(),
            });
        }

        self.backend
            .file_read()
            .read_dataset_raw(&self.path, selection, buf)
    }

    /// Reads raw bytes for `selection`.
    ///
    /// This method prefers the backend zero-copy path when available.
    pub fn read(&self, selection: &Selection) -> Result<SelectionRead> {
        let metadata = self.metadata()?;
        let required = selection_byte_len(&metadata.datatype, &metadata.shape, selection)?;

        match self.backend.read_dataset_zero_copy(&self.path, selection) {
            Ok(bytes) => {
                if bytes.as_slice().len() != required {
                    return Err(Error::BufferTooSmall {
                        required,
                        provided: bytes.as_slice().len(),
                    });
                }
                Ok(SelectionRead::ZeroCopy(bytes))
            }
            Err(_) => {
                let mut buffer = vec![0u8; required];
                self.backend
                    .file_read()
                    .read_dataset_raw(&self.path, selection, &mut buffer)?;
                Ok(SelectionRead::Owned(buffer))
            }
        }
    }

    /// Reads the entire dataset as raw bytes.
    pub fn read_all(&self) -> Result<SelectionRead> {
        self.read(&Selection::All)
    }

    /// Attempts a zero-copy read for `selection`.
    ///
    /// Returns `Ok(None)` when the backend cannot satisfy the request through
    /// its zero-copy path.
    pub fn read_zero_copy(
        &self,
        selection: &Selection,
    ) -> Result<Option<crate::highlevel::file::ZeroCopyBytes>> {
        let metadata = self.metadata()?;
        let required = selection_byte_len(&metadata.datatype, &metadata.shape, selection)?;

        match self.backend.read_dataset_zero_copy(&self.path, selection) {
            Ok(bytes) => {
                if bytes.as_slice().len() != required {
                    return Err(Error::BufferTooSmall {
                        required,
                        provided: bytes.as_slice().len(),
                    });
                }
                Ok(Some(bytes))
            }
            Err(_) => Ok(None),
        }
    }

    /// Builds a format-neutral read plan for `selection`.
    pub fn read_plan(&self, selection: &Selection) -> Result<ReadPlan> {
        let metadata = self.metadata()?;
        let required = selection_byte_len(&metadata.datatype, &metadata.shape, selection)?;

        if self
            .backend
            .read_dataset_zero_copy(&self.path, selection)
            .is_ok()
        {
            return Ok(ReadPlan::ZeroCopy);
        }

        if self.parallelism.is_enabled() && required > 0 {
            let partitions = self.parallelism.partitions_for_len(required);
            if partitions > 1 {
                let ranges = crate::sync::partition_range(required, partitions)?;
                return Ok(ReadPlan::Parallel(ranges));
            }
        }

        Ok(ReadPlan::Materialized)
    }

    /// Reads the dataset selection using the configured parallel I/O policy.
    ///
    /// This method is format-neutral. Parallel execution is only used when the
    /// selection can be represented as a contiguous byte interval over the raw
    /// dataset read path.
    pub fn read_parallel(&self, selection: &Selection) -> Result<SelectionRead> {
        match self.read_plan(selection)? {
            ReadPlan::ZeroCopy => self.read(selection),
            ReadPlan::Materialized => self.read(selection),
            ReadPlan::Parallel(ranges) => {
                let metadata = self.metadata()?;
                let total_len = selection_byte_len(&metadata.datatype, &metadata.shape, selection)?;

                let mut full = vec![0u8; total_len];
                self.backend
                    .file_read()
                    .read_dataset_raw(&self.path, selection, &mut full)?;

                let view = ByteView::Owned(full);
                let owned = materialize_parallel_view(view, &ranges)?;
                Ok(SelectionRead::Owned(owned))
            }
        }
    }

    /// Returns `true` when the dataset exists and resolves as a dataset node.
    pub fn exists(&self) -> Result<bool> {
        self.backend.file_read().exists(&self.path)
    }

    /// Returns the erased backend handle.
    pub(crate) fn backend(&self) -> &Arc<dyn UnifiedBackend> {
        &self.backend
    }
}

fn materialize_parallel_view(view: ByteView<'_>, ranges: &[IoRange]) -> Result<Vec<u8>> {
    let bytes = view.as_slice();
    let mut out = Vec::with_capacity(bytes.len());

    for range in ranges {
        let start = usize::try_from(range.offset).map_err(|_| Error::Overflow)?;
        let end = start.checked_add(range.len).ok_or(Error::Overflow)?;
        if end > bytes.len() {
            return Err(Error::BufferTooSmall {
                required: end,
                provided: bytes.len(),
            });
        }
        out.extend_from_slice(&bytes[start..end]);
    }

    Ok(out)
}
