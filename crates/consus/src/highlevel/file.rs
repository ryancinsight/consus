use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::builders::dataset::DatasetBuilder;
use crate::highlevel::dataset::Dataset;
use crate::highlevel::group::Group;
use crate::sync::Parallelism;
use consus_core::{
    Compression, Datatype, Error, FileRead, FileWrite, NodeType, Result, Selection, Shape,
};

/// Unified high-level file facade.
///
/// This type contains no format-specific logic. It delegates all storage
/// operations to a backend implementing [`UnifiedBackend`]. Backend selection
/// is performed by a registry supplied by the caller or by the crate-level
/// default registry.
pub struct File {
    backend: Arc<dyn UnifiedBackend>,
    parallelism: Parallelism,
}

impl core::fmt::Debug for File {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("File")
            .field("format", &self.backend.format())
            .field("path", &self.backend.path())
            .field("parallelism", &self.parallelism)
            .finish()
    }
}

/// Backend-neutral file open/create options.
///
/// These options are interpreted by the selected backend implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOptions {
    create: bool,
    truncate: bool,
    read: bool,
    write: bool,
}

impl Default for FileOptions {
    fn default() -> Self {
        Self {
            create: false,
            truncate: false,
            read: true,
            write: false,
        }
    }
}

impl FileOptions {
    /// Create default read-only open options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable create semantics.
    pub fn create(mut self, enabled: bool) -> Self {
        self.create = enabled;
        self
    }

    /// Enable truncation semantics.
    pub fn truncate(mut self, enabled: bool) -> Self {
        self.truncate = enabled;
        self
    }

    /// Enable read access.
    pub fn read(mut self, enabled: bool) -> Self {
        self.read = enabled;
        self
    }

    /// Enable write access.
    pub fn write(mut self, enabled: bool) -> Self {
        self.write = enabled;
        self
    }

    /// Canonical create-new writable options.
    pub fn create_new() -> Self {
        Self::new().create(true).truncate(true).write(true)
    }

    /// Canonical read-write open options.
    pub fn read_write() -> Self {
        Self::new().read(true).write(true)
    }

    /// Whether create is enabled.
    pub fn is_create(&self) -> bool {
        self.create
    }

    /// Whether truncate is enabled.
    pub fn is_truncate(&self) -> bool {
        self.truncate
    }

    /// Whether read is enabled.
    pub fn is_read(&self) -> bool {
        self.read
    }

    /// Whether write is enabled.
    pub fn is_write(&self) -> bool {
        self.write
    }
}

/// Backend-neutral dataset creation specification.
///
/// This is the resolved specification emitted by fluent builders.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetCreateSpec {
    pub path: String,
    pub datatype: Datatype,
    pub shape: Shape,
    pub chunk_shape: Option<Vec<usize>>,
    pub compression: Compression,
}

/// Zero-copy read result.
///
/// `Borrowed` indicates the backend can expose a stable byte slice without
/// materialization. `Owned` indicates the backend had to allocate or decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZeroCopyBytes {
    Borrowed(Arc<[u8]>),
    Owned(Vec<u8>),
}

impl ZeroCopyBytes {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
        }
    }

    pub fn into_owned(self) -> Vec<u8> {
        match self {
            Self::Borrowed(bytes) => bytes.as_ref().to_vec(),
            Self::Owned(bytes) => bytes,
        }
    }

    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }
}

/// Backend-neutral file backend contract.
///
/// Implementations live in format crates or adapter crates. This facade crate
/// depends only on the abstraction.
pub trait UnifiedBackend: Send + Sync {
    fn format(&self) -> &str;
    fn path(&self) -> &str;
    fn file_read(&self) -> &dyn FileRead;
    fn file_write(&self) -> Option<&dyn FileWrite>;
    fn create_group(&self, path: &str) -> Result<()>;
    fn create_dataset(&self, spec: &DatasetCreateSpec) -> Result<()>;
    fn read_dataset_zero_copy(&self, path: &str, selection: &Selection) -> Result<ZeroCopyBytes>;
    fn clone_boxed(&self) -> Arc<dyn UnifiedBackend>;

    /// Flush pending writes through interior mutability.
    ///
    /// Concrete backends that support writes must override this. The default
    /// returns [`Error::ReadOnly`].
    fn flush(&self) -> Result<()> {
        Err(Error::ReadOnly)
    }

    /// Write raw bytes to a dataset through interior mutability.
    ///
    /// Concrete backends that support writes must override this. The default
    /// returns [`Error::ReadOnly`].
    fn write_dataset_raw(&self, _path: &str, _selection: &Selection, _data: &[u8]) -> Result<()> {
        Err(Error::ReadOnly)
    }
}

/// Backend factory.
///
/// Factories are registered in a registry and queried in order.
pub trait BackendFactory: Send + Sync {
    fn name(&self) -> &str;
    fn can_open(&self, path: &str) -> bool;
    fn can_create(&self, path: &str) -> bool;
    fn open(&self, path: &str, options: &FileOptions) -> Result<Arc<dyn UnifiedBackend>>;
    fn create(&self, path: &str, options: &FileOptions) -> Result<Arc<dyn UnifiedBackend>>;
}

/// Ordered backend registry.
///
/// The registry contains no format-specific logic. It only stores factories and
/// delegates selection.
#[derive(Default)]
pub struct BackendRegistry {
    factories: Vec<Arc<dyn BackendFactory>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
        }
    }

    pub fn register<F>(mut self, factory: F) -> Self
    where
        F: BackendFactory + 'static,
    {
        self.factories.push(Arc::new(factory));
        self
    }

    pub fn push_arc(&mut self, factory: Arc<dyn BackendFactory>) {
        self.factories.push(factory);
    }

    pub fn factories(&self) -> &[Arc<dyn BackendFactory>] {
        &self.factories
    }

    pub fn open(&self, path: &str, options: &FileOptions) -> Result<Arc<dyn UnifiedBackend>> {
        for factory in &self.factories {
            if factory.can_open(path) {
                return factory.open(path, options);
            }
        }

        Err(Error::UnsupportedFeature {
            feature: unsupported_backend_message("open", path),
        })
    }

    pub fn create(&self, path: &str, options: &FileOptions) -> Result<Arc<dyn UnifiedBackend>> {
        for factory in &self.factories {
            if factory.can_create(path) {
                return factory.create(path, options);
            }
        }

        Err(Error::UnsupportedFeature {
            feature: unsupported_backend_message("create", path),
        })
    }
}

impl File {
    /// Open a file using the crate default registry.
    pub fn open(path: impl AsRef<str>) -> Result<Self> {
        Self::open_with_registry(path, &crate::default_backend_registry(), FileOptions::new())
    }

    /// Create a file using the crate default registry.
    pub fn create(path: impl AsRef<str>) -> Result<Self> {
        Self::create_with_registry(
            path,
            &crate::default_backend_registry(),
            FileOptions::create_new(),
        )
    }

    /// Open a file using an explicit registry.
    pub fn open_with_registry(
        path: impl AsRef<str>,
        registry: &BackendRegistry,
        options: FileOptions,
    ) -> Result<Self> {
        let backend = registry.open(path.as_ref(), &options)?;
        Ok(Self {
            backend,
            parallelism: Parallelism::default(),
        })
    }

    /// Create a file using an explicit registry.
    pub fn create_with_registry(
        path: impl AsRef<str>,
        registry: &BackendRegistry,
        options: FileOptions,
    ) -> Result<Self> {
        let backend = registry.create(path.as_ref(), &options)?;
        Ok(Self {
            backend,
            parallelism: Parallelism::default(),
        })
    }

    /// Construct a file facade from an already selected backend.
    pub fn from_backend(backend: Arc<dyn UnifiedBackend>) -> Self {
        Self {
            backend,
            parallelism: Parallelism::default(),
        }
    }

    /// Return the backend format identifier.
    pub fn format(&self) -> &str {
        self.backend.format()
    }

    /// Return the underlying path.
    pub fn path(&self) -> &str {
        self.backend.path()
    }

    /// Return the configured parallelism policy.
    pub fn parallelism(&self) -> &Parallelism {
        &self.parallelism
    }

    /// Set the parallelism policy for subsequent high-level operations.
    pub fn with_parallelism(mut self, parallelism: Parallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    /// Whether an absolute path exists.
    pub fn exists(&self, path: &str) -> Result<bool> {
        self.backend.file_read().exists(path)
    }

    /// Return the node type at an absolute path.
    pub fn node_type_at(&self, path: &str) -> Result<NodeType> {
        self.backend.file_read().node_type_at(path)
    }

    /// Open a group handle.
    pub fn group(&self, path: impl AsRef<str>) -> Result<Group> {
        let path = normalize_absolute_path(path.as_ref());
        match self.backend.file_read().node_type_at(&path)? {
            NodeType::Group => Ok(Group::new(
                self.backend.clone_boxed(),
                path,
                self.parallelism.clone(),
            )),
            other => Err(Error::InvalidFormat {
                message: format!("path is not a group: {other:?}"),
            }),
        }
    }

    /// Open a dataset handle.
    pub fn dataset(&self, path: impl AsRef<str>) -> Result<Dataset> {
        let path = normalize_absolute_path(path.as_ref());
        match self.backend.file_read().node_type_at(&path)? {
            NodeType::Dataset => Ok(Dataset::new(
                self.backend.clone_boxed(),
                path,
                self.parallelism.clone(),
            )),
            other => Err(Error::InvalidFormat {
                message: format!("path is not a dataset: {other:?}"),
            }),
        }
    }

    /// Create a group and return its handle.
    pub fn create_group(&self, path: impl AsRef<str>) -> Result<Group> {
        let path = normalize_absolute_path(path.as_ref());
        self.backend.create_group(&path)?;
        Ok(Group::new(
            self.backend.clone_boxed(),
            path,
            self.parallelism.clone(),
        ))
    }

    /// Start fluent dataset creation under this file.
    pub fn create_dataset(&self, path: impl AsRef<str>) -> DatasetBuilder<'_> {
        DatasetBuilder::new(self, normalize_absolute_path(path.as_ref()))
    }

    /// Creates a dataset from a resolved builder specification.
    pub(crate) fn create_dataset_from_builder(&self, spec: DatasetCreateSpec) -> Result<Dataset> {
        self.backend.create_dataset(&spec)?;
        Ok(Dataset::new(
            self.backend.clone_boxed(),
            spec.path,
            self.parallelism.clone(),
        ))
    }

    /// Creates a dataset and writes raw bytes from a resolved builder specification.
    pub(crate) fn create_dataset_and_write_from_builder(
        &self,
        spec: DatasetCreateSpec,
        data: &[u8],
    ) -> Result<Dataset> {
        self.backend.create_dataset(&spec)?;
        let dataset = Dataset::new(
            self.backend.clone_boxed(),
            spec.path.clone(),
            self.parallelism.clone(),
        );

        // Write the data through the backend's interior-mutability path.
        self.backend
            .write_dataset_raw(&spec.path, &Selection::All, data)?;

        Ok(dataset)
    }

    /// Flush pending writes if the backend is writable.
    pub fn flush(&self) -> Result<()> {
        self.backend.flush()
    }
}

fn normalize_absolute_path(path: &str) -> String {
    if path.is_empty() {
        "/".to_string()
    } else if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

fn unsupported_backend_message(operation: &str, path: &str) -> String {
    format!("no registered backend can {operation} path `{path}`")
}
