use crate::builders::dataset::DatasetBuilder;
use crate::highlevel::group::Group;
use crate::{Compression, Datatype, Result, Shape};

/// README-compatible group-scoped dataset builder wrapper.
///
/// This type contains no format-specific logic. It exists to preserve the
/// fluent quick-start flow:
/// `group.create_dataset("name").shape(...).chunks(...).compression(...).write(&data)`.
#[derive(Clone)]
pub struct GroupBuilder<'a> {
    group: &'a Group,
    dataset_path: String,
    datatype: Option<Datatype>,
    shape: Option<Shape>,
    chunk_dims: Option<Vec<usize>>,
    compression: Compression,
}

impl<'a> GroupBuilder<'a> {
    /// Creates a new group-scoped builder for a child dataset.
    pub(crate) fn new(group: &'a Group, dataset_name: impl AsRef<str>) -> Self {
        Self {
            group,
            dataset_path: join_group_child_path(group.path(), dataset_name.as_ref()),
            datatype: None,
            shape: None,
            chunk_dims: None,
            compression: Compression::None,
        }
    }

    /// Sets the dataset shape from fixed extents.
    pub fn shape(mut self, dims: &[usize]) -> Self {
        self.shape = Some(Shape::fixed(dims));
        self
    }

    /// Sets the dataset chunk shape.
    pub fn chunks(mut self, dims: &[usize]) -> Self {
        self.chunk_dims = Some(dims.to_vec());
        self
    }

    /// Sets the dataset compression policy.
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Sets the canonical dataset datatype explicitly.
    pub fn datatype(mut self, datatype: Datatype) -> Self {
        self.datatype = Some(datatype);
        self
    }

    /// Creates the dataset metadata without writing payload bytes.
    pub fn create(self) -> Result<crate::Dataset> {
        let file = self.group.file();
        let mut builder = DatasetBuilder::new(&file, self.dataset_path);

        if let Some(datatype) = self.datatype {
            builder = builder.datatype(datatype);
        }

        if let Some(shape) = self.shape {
            builder = builder.shape_from_shape(shape);
        }

        if let Some(chunk_dims) = self.chunk_dims {
            builder = builder.chunks(&chunk_dims);
        }

        builder.compression(self.compression).create()
    }

    /// Creates the dataset and writes a typed contiguous payload.
    pub fn write<T>(self, data: &[T]) -> Result<crate::Dataset>
    where
        T: Copy + 'static,
    {
        let file = self.group.file();
        let mut builder = DatasetBuilder::new(&file, self.dataset_path);

        if let Some(datatype) = self.datatype {
            builder = builder.datatype(datatype);
        }

        if let Some(shape) = self.shape {
            builder = builder.shape_from_shape(shape);
        }

        if let Some(chunk_dims) = self.chunk_dims {
            builder = builder.chunks(&chunk_dims);
        }

        builder.compression(self.compression).write(data)
    }

    /// Returns the owning group.
    pub fn group(&self) -> &Group {
        self.group
    }

    /// Returns the resolved absolute dataset path.
    pub fn dataset_path(&self) -> &str {
        &self.dataset_path
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
}

fn join_group_child_path(group_path: &str, child_name: &str) -> String {
    let trimmed = child_name.trim_matches('/');

    if group_path == "/" {
        format!("/{trimmed}")
    } else {
        format!("{}/{}", group_path.trim_end_matches('/'), trimmed)
    }
}
