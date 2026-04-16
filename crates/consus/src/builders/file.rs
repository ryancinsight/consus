use alloc::string::String;

use crate::highlevel::file::{BackendRegistry, File, FileOptions};
use crate::{Error, Result};

/// Format-agnostic builder for opening or creating a unified [`File`].
///
/// This builder contains no format-specific logic. It only assembles validated
/// backend-neutral options and delegates backend resolution to a
/// [`BackendRegistry`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileBuilder {
    path: Option<String>,
    options: FileOptions,
}

impl Default for FileBuilder {
    fn default() -> Self {
        Self {
            path: None,
            options: FileOptions::new(),
        }
    }
}

/// Validated file open/create request emitted by [`FileBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOpenOptions {
    /// Target path passed to backend resolution.
    pub path: String,
    /// Backend-neutral open/create options.
    pub options: FileOptions,
}

impl FileBuilder {
    /// Creates a new builder with conservative read-only defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the target path.
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Enables or disables create semantics.
    pub fn create(mut self, enabled: bool) -> Self {
        self.options = self.options.create(enabled);
        self
    }

    /// Enables or disables truncation semantics.
    pub fn truncate(mut self, enabled: bool) -> Self {
        self.options = self.options.truncate(enabled);
        self
    }

    /// Enables or disables read access.
    pub fn read(mut self, enabled: bool) -> Self {
        self.options = self.options.read(enabled);
        self
    }

    /// Enables or disables write access.
    pub fn write(mut self, enabled: bool) -> Self {
        self.options = self.options.write(enabled);
        self
    }

    /// Builds validated backend-neutral open options.
    pub fn build(self) -> Result<FileOpenOptions> {
        let path = self.path.ok_or_else(missing_path)?;
        validate_options(&self.options)?;

        Ok(FileOpenOptions {
            path,
            options: self.options,
        })
    }

    /// Opens a file using the provided backend registry.
    pub fn open_with_registry(self, registry: &BackendRegistry) -> Result<File> {
        let request = self.build()?;
        File::open_with_registry(request.path, registry, request.options)
    }

    /// Creates a file using the provided backend registry.
    pub fn create_with_registry(self, registry: &BackendRegistry) -> Result<File> {
        let request = self.create(true).write(true).build()?;
        File::create_with_registry(request.path, registry, request.options)
    }

    /// Opens a file using the crate default backend registry.
    #[cfg(all(feature = "alloc", feature = "std"))]
    pub fn open(self) -> Result<File> {
        let registry = crate::default_backend_registry();
        self.open_with_registry(&registry)
    }

    /// Creates a file using the crate default backend registry.
    #[cfg(all(feature = "alloc", feature = "std"))]
    pub fn create_file(self) -> Result<File> {
        let registry = crate::default_backend_registry();
        self.create_with_registry(&registry)
    }

    /// Returns the configured path, if present.
    pub fn configured_path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Returns the configured backend-neutral options.
    pub fn configured_options(&self) -> &FileOptions {
        &self.options
    }
}

fn validate_options(options: &FileOptions) -> Result<()> {
    if options.is_truncate() && !options.is_write() {
        return Err(Error::ReadOnly);
    }

    if options.is_create() && !options.is_write() {
        return Err(Error::ReadOnly);
    }

    if !options.is_read() && !options.is_write() {
        return Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: String::from("file access mode must enable read and/or write"),
        });
    }

    Ok(())
}

fn missing_path() -> Error {
    Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: String::from("file builder requires a path"),
    }
}
