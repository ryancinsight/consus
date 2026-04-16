use alloc::string::{String, ToString};
use alloc::sync::Arc;

use crate::Result;
use crate::builders::GroupBuilder;
use crate::highlevel::dataset::Dataset;
use crate::highlevel::file::UnifiedBackend;
use consus_core::NodeType;

/// Unified format-agnostic group facade.
///
/// This type contains no format-specific logic. It stores only the erased
/// backend handle, the absolute group path, and the selected parallelism
/// policy inherited from the owning [`crate::File`].
#[derive(Clone)]
pub struct Group {
    backend: Arc<dyn UnifiedBackend>,
    path: String,
    parallelism: crate::sync::Parallelism,
}

impl Group {
    /// Creates a new group facade from an erased backend and absolute path.
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

    /// Returns the leaf name of this group.
    pub fn name(&self) -> &str {
        if self.path == "/" {
            "/"
        } else {
            self.path.rsplit('/').next().unwrap_or("/")
        }
    }

    /// Returns the absolute path of this group.
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

    /// Returns the number of direct children.
    pub fn num_children(&self) -> Result<usize> {
        self.backend.file_read().num_children_at(&self.path)
    }

    /// Returns whether a direct child with `name` exists.
    pub fn contains(&self, name: &str) -> Result<bool> {
        let child_path = self.child_path(name);
        self.backend.file_read().exists(&child_path)
    }

    /// Returns the node type of a direct child.
    pub fn child_node_type(&self, name: &str) -> Result<NodeType> {
        let child_path = self.child_path(name);
        self.backend.file_read().node_type_at(&child_path)
    }

    /// Opens a direct child group.
    pub fn group(&self, name: &str) -> Result<Self> {
        let child_path = self.child_path(name);
        match self.backend.file_read().node_type_at(&child_path)? {
            NodeType::Group => Ok(Self::new(
                self.backend.clone_boxed(),
                child_path,
                self.parallelism.clone(),
            )),
            other => Err(consus_core::Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!("path is not a group: {other:?}"),
            }),
        }
    }

    /// Opens a direct child dataset.
    pub fn dataset(&self, name: &str) -> Result<Dataset> {
        let child_path = self.child_path(name);
        match self.backend.file_read().node_type_at(&child_path)? {
            NodeType::Dataset => Ok(Dataset::new(
                self.backend.clone_boxed(),
                child_path,
                self.parallelism.clone(),
            )),
            other => Err(consus_core::Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!("path is not a dataset: {other:?}"),
            }),
        }
    }

    /// Creates a direct child group and returns its unified facade.
    pub fn create_group(&self, name: &str) -> Result<Self> {
        let child_path = self.child_path(name);
        self.backend.create_group(&child_path)?;
        Ok(Self::new(
            self.backend.clone_boxed(),
            child_path,
            self.parallelism.clone(),
        ))
    }

    /// Starts fluent dataset creation under this group.
    pub fn create_dataset(&self, name: impl AsRef<str>) -> GroupBuilder<'_> {
        GroupBuilder::new(self, name.as_ref())
    }

    /// Returns a lightweight file view over the same backend.
    pub(crate) fn file(&self) -> crate::highlevel::file::File {
        crate::highlevel::file::File::from_backend(self.backend.clone_boxed())
            .with_parallelism(self.parallelism.clone())
    }

    fn child_path(&self, name: &str) -> String {
        if self.path == "/" {
            format!("/{}", trim_relative_component(name))
        } else {
            format!("{}/{}", self.path, trim_relative_component(name))
        }
    }
}

fn trim_relative_component(name: &str) -> String {
    name.trim_matches('/').to_string()
}
