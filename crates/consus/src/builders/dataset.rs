use alloc::string::String;
use alloc::vec::Vec;

use crate::highlevel::dataset::Dataset;
use crate::highlevel::file::{DatasetCreateSpec, File};
use crate::{Compression, Datatype, Error, Result, Shape};

/// Canonical fluent dataset creation builder.
///
/// This builder is format-agnostic. It captures only backend-neutral dataset
/// creation intent and delegates materialization to the selected backend
/// through the owning [`File`] facade.
///
/// # Invariants
///
/// - `shape` must be specified before `create` or `write`.
/// - `datatype` must be specified explicitly or inferred from `write`.
/// - `chunk_dims.len() == shape.rank()` when chunking is configured.
/// - All chunk dimensions are strictly positive.
/// - Typed writes require a fixed-size canonical datatype.
/// - Typed write payload size must equal
///   `shape.num_elements() * datatype.element_size()`.
#[derive(Debug, Clone)]
pub struct DatasetBuilder<'a> {
    file: &'a File,
    path: String,
    datatype: Option<Datatype>,
    shape: Option<Shape>,
    chunk_dims: Option<Vec<usize>>,
    compression: Compression,
}

/// Resolved canonical dataset creation specification emitted by
/// [`DatasetBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetBuilderSpec {
    /// Absolute dataset path.
    pub path: String,
    /// Canonical element datatype.
    pub datatype: Datatype,
    /// Dataset shape.
    pub shape: Shape,
    /// Chunk dimensions when chunked storage is requested.
    pub chunk_dims: Option<Vec<usize>>,
    /// Compression configuration.
    pub compression: Compression,
}

impl<'a> DatasetBuilder<'a> {
    /// Creates a new builder bound to `file` and the target absolute path.
    pub(crate) fn new(file: &'a File, path: impl Into<String>) -> Self {
        Self {
            file,
            path: normalize_absolute_path(path.into()),
            datatype: None,
            shape: None,
            chunk_dims: None,
            compression: Compression::None,
        }
    }

    /// Sets the canonical dataset datatype explicitly.
    pub fn datatype(mut self, datatype: Datatype) -> Self {
        self.datatype = Some(datatype);
        self
    }

    /// Sets the dataset shape from fixed extents.
    pub fn shape(mut self, dims: &[usize]) -> Self {
        self.shape = Some(Shape::fixed(dims));
        self
    }

    /// Sets the dataset shape from an already constructed canonical shape.
    pub fn shape_from_shape(mut self, shape: Shape) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Configures chunked storage.
    ///
    /// The chunk rank must match the dataset rank.
    pub fn chunks(mut self, dims: &[usize]) -> Self {
        self.chunk_dims = Some(dims.to_vec());
        self
    }

    /// Configures chunked storage from an optional chunk vector.
    pub fn maybe_chunks(mut self, dims: Option<Vec<usize>>) -> Self {
        self.chunk_dims = dims;
        self
    }

    /// Configures dataset compression.
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Returns the configured absolute dataset path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the configured datatype, if present.
    pub fn configured_datatype(&self) -> Option<&Datatype> {
        self.datatype.as_ref()
    }

    /// Returns the configured shape, if present.
    pub fn configured_shape(&self) -> Option<&Shape> {
        self.shape.as_ref()
    }

    /// Returns the configured chunk dimensions, if present.
    pub fn configured_chunks(&self) -> Option<&[usize]> {
        self.chunk_dims.as_deref()
    }

    /// Returns the configured compression policy.
    pub fn configured_compression(&self) -> &Compression {
        &self.compression
    }

    /// Resolves the builder into a canonical specification without performing
    /// I/O.
    pub fn build(self) -> Result<DatasetBuilderSpec> {
        let datatype = self.datatype.ok_or_else(missing_datatype)?;
        let shape = self.shape.ok_or_else(missing_shape)?;

        validate_chunk_dims(&shape, self.chunk_dims.as_deref())?;

        Ok(DatasetBuilderSpec {
            path: self.path,
            datatype,
            shape,
            chunk_dims: self.chunk_dims,
            compression: self.compression,
        })
    }

    /// Creates dataset metadata without writing payload bytes.
    pub fn create(self) -> Result<Dataset> {
        let DatasetBuilder {
            file,
            path,
            datatype,
            shape,
            chunk_dims,
            compression,
        } = self;
        let datatype = datatype.ok_or_else(missing_datatype)?;
        let shape = shape.ok_or_else(missing_shape)?;
        validate_chunk_dims(&shape, chunk_dims.as_deref())?;
        let create_spec = DatasetCreateSpec {
            path,
            datatype,
            shape,
            chunk_shape: chunk_dims,
            compression,
        };

        file.create_dataset_from_builder(create_spec)
    }

    /// Creates the dataset and writes a typed contiguous payload.
    ///
    /// If no datatype was configured explicitly, the builder infers one from
    /// `T` using the canonical `consus-core` datatype model.
    pub fn write<T>(mut self, data: &[T]) -> Result<Dataset>
    where
        T: Copy + 'static,
    {
        if self.datatype.is_none() {
            self.datatype = Some(infer_datatype::<T>()?);
        }

        let DatasetBuilder {
            file,
            path,
            datatype,
            shape,
            chunk_dims,
            compression,
        } = self;
        let datatype = datatype.ok_or_else(missing_datatype)?;
        let shape = shape.ok_or_else(missing_shape)?;
        validate_chunk_dims(&shape, chunk_dims.as_deref())?;

        let element_size = datatype
            .element_size()
            .ok_or_else(variable_length_write_unsupported)?;

        let expected_len = shape
            .num_elements()
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;

        let actual_len = core::mem::size_of_val(data);
        if actual_len != expected_len {
            return Err(Error::ShapeError {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "dataset payload byte length mismatch: expected {expected_len}, found {actual_len}"
                ),
            });
        }

        let bytes = typed_slice_as_bytes(data);
        let create_spec = DatasetCreateSpec {
            path,
            datatype,
            shape,
            chunk_shape: chunk_dims,
            compression,
        };

        file.create_dataset_and_write_from_builder(create_spec, bytes)
    }
}

fn normalize_absolute_path(path: String) -> String {
    if path.is_empty() {
        String::from("/")
    } else if path.starts_with('/') {
        path
    } else {
        alloc::format!("/{path}")
    }
}

fn validate_chunk_dims(shape: &Shape, chunk_dims: Option<&[usize]>) -> Result<()> {
    let Some(chunk_dims) = chunk_dims else {
        return Ok(());
    };

    if chunk_dims.contains(&0) {
        return Err(Error::ShapeError {
            #[cfg(feature = "alloc")]
            message: String::from("chunk dimensions must be strictly positive"),
        });
    }

    if chunk_dims.len() != shape.rank() {
        return Err(Error::ShapeError {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "chunk rank mismatch: dataset rank {}, chunk rank {}",
                shape.rank(),
                chunk_dims.len()
            ),
        });
    }

    Ok(())
}

fn missing_datatype() -> Error {
    Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: String::from("dataset datatype must be specified before creation"),
    }
}

fn missing_shape() -> Error {
    Error::ShapeError {
        #[cfg(feature = "alloc")]
        message: String::from("dataset shape must be specified before creation"),
    }
}

fn variable_length_write_unsupported() -> Error {
    Error::UnsupportedFeature {
        #[cfg(feature = "alloc")]
        feature: String::from("typed slice writes require fixed-size element datatypes"),
    }
}

fn infer_datatype<T>() -> Result<Datatype>
where
    T: 'static,
{
    use core::any::TypeId;
    use core::num::NonZeroUsize;

    let type_id = TypeId::of::<T>();
    let little = crate::core::ByteOrder::LittleEndian;

    let datatype = if type_id == TypeId::of::<u8>() {
        Datatype::Integer {
            bits: NonZeroUsize::new(8).expect("8 is non-zero"),
            byte_order: little,
            signed: false,
        }
    } else if type_id == TypeId::of::<i8>() {
        Datatype::Integer {
            bits: NonZeroUsize::new(8).expect("8 is non-zero"),
            byte_order: little,
            signed: true,
        }
    } else if type_id == TypeId::of::<u16>() {
        Datatype::Integer {
            bits: NonZeroUsize::new(16).expect("16 is non-zero"),
            byte_order: little,
            signed: false,
        }
    } else if type_id == TypeId::of::<i16>() {
        Datatype::Integer {
            bits: NonZeroUsize::new(16).expect("16 is non-zero"),
            byte_order: little,
            signed: true,
        }
    } else if type_id == TypeId::of::<u32>() {
        Datatype::Integer {
            bits: NonZeroUsize::new(32).expect("32 is non-zero"),
            byte_order: little,
            signed: false,
        }
    } else if type_id == TypeId::of::<i32>() {
        Datatype::Integer {
            bits: NonZeroUsize::new(32).expect("32 is non-zero"),
            byte_order: little,
            signed: true,
        }
    } else if type_id == TypeId::of::<u64>() {
        Datatype::Integer {
            bits: NonZeroUsize::new(64).expect("64 is non-zero"),
            byte_order: little,
            signed: false,
        }
    } else if type_id == TypeId::of::<i64>() {
        Datatype::Integer {
            bits: NonZeroUsize::new(64).expect("64 is non-zero"),
            byte_order: little,
            signed: true,
        }
    } else if type_id == TypeId::of::<f32>() {
        Datatype::Float {
            bits: NonZeroUsize::new(32).expect("32 is non-zero"),
            byte_order: little,
        }
    } else if type_id == TypeId::of::<f64>() {
        Datatype::Float {
            bits: NonZeroUsize::new(64).expect("64 is non-zero"),
            byte_order: little,
        }
    } else if type_id == TypeId::of::<bool>() {
        Datatype::Boolean
    } else {
        return Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: alloc::format!(
                "automatic datatype inference is unavailable for {}",
                core::any::type_name::<T>()
            ),
        });
    };

    Ok(datatype)
}

fn typed_slice_as_bytes<T>(data: &[T]) -> &[u8] {
    let len = core::mem::size_of_val(data);
    let ptr = data.as_ptr().cast::<u8>();

    // SAFETY:
    // - `u8` has alignment 1, so any `T` pointer is valid for `u8`.
    // - The resulting slice covers exactly the initialized memory of `data`.
    // - The returned slice borrows from `data` and cannot outlive it.
    unsafe { core::slice::from_raw_parts(ptr, len) }
}
