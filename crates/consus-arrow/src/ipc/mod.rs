//! Arrow IPC and stream descriptors.
//!
//! This module defines the canonical metadata model for IPC-style
//! materialization in `consus-arrow`. It does not implement the Arrow
//! wire format. It describes record batch boundaries, stream framing,
//! dictionary handling, and zero-copy constraints.
//!
//! ## Invariants
//!
//! - IPC descriptors preserve schema identity.
//! - Record batch boundaries are explicit.
//! - Dictionary batches are tracked separately from data batches.
//! - Stream and file framing are modeled independently.
//! - The module remains dependency-light and crate-local.
//!
//! ## Architecture
//!
//! ```text
//! ipc/
//! ├── framing        # file/stream framing descriptors
//! ├── batch          # record batch metadata
//! ├── dictionary     # dictionary batch metadata
//! └── plan           # IPC execution and materialization plan
//! ```
//!
//! ## Notes
//!
//! This module is intentionally descriptive. A future implementation
//! can map these descriptors onto the Arrow Rust ecosystem without
//! changing the public semantic model.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::fmt;

/// Arrow IPC framing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IpcFraming {
    /// Contiguous stream framing.
    Stream,
    /// Random-access file framing.
    File,
}

/// Arrow IPC batch kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchKind {
    /// Main data batch.
    RecordBatch,
    /// Dictionary batch.
    DictionaryBatch,
    /// Delta dictionary batch.
    DeltaDictionaryBatch,
}

/// Arrow IPC buffer ownership model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferOwnership {
    /// Borrowed from an existing source.
    Borrowed,
    /// Owned by the IPC materialization layer.
    Owned,
    /// Zero-copy eligible if alignment and lifetime conditions hold.
    ZeroCopyEligible,
}

/// Metadata for one record batch.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordBatchDescriptor {
    /// Batch identifier.
    pub batch_id: u64,
    /// Number of rows in the batch.
    pub row_count: usize,
    /// Number of serialized buffers.
    pub buffer_count: usize,
    /// Optional byte range inside the source.
    pub byte_range: Option<(u64, u64)>,
    /// Ownership semantics.
    pub ownership: BufferOwnership,
}

#[cfg(feature = "alloc")]
impl RecordBatchDescriptor {
    /// Create a record batch descriptor.
    #[must_use]
    pub fn new(
        batch_id: u64,
        row_count: usize,
        buffer_count: usize,
        byte_range: Option<(u64, u64)>,
        ownership: BufferOwnership,
    ) -> Self {
        Self {
            batch_id,
            row_count,
            buffer_count,
            byte_range,
            ownership,
        }
    }

    /// Returns `true` if the batch can be exposed zero-copy.
    #[must_use]
    pub fn is_zero_copy_eligible(&self) -> bool {
        matches!(self.ownership, BufferOwnership::ZeroCopyEligible)
    }
}

/// Metadata for one dictionary batch.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryBatchDescriptor {
    /// Dictionary identifier.
    pub dictionary_id: u64,
    /// Number of encoded values.
    pub value_count: usize,
    /// Optional byte range inside the source.
    pub byte_range: Option<(u64, u64)>,
    /// Whether this is a delta dictionary batch.
    pub is_delta: bool,
}

#[cfg(feature = "alloc")]
impl DictionaryBatchDescriptor {
    /// Create a dictionary batch descriptor.
    #[must_use]
    pub fn new(
        dictionary_id: u64,
        value_count: usize,
        byte_range: Option<(u64, u64)>,
        is_delta: bool,
    ) -> Self {
        Self {
            dictionary_id,
            value_count,
            byte_range,
            is_delta,
        }
    }
}

/// IPC framing and batch layout for a complete stream or file.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcPlan {
    /// Framing mode.
    pub framing: IpcFraming,
    /// Ordered record batch descriptors.
    pub record_batches: Vec<RecordBatchDescriptor>,
    /// Ordered dictionary batch descriptors.
    pub dictionary_batches: Vec<DictionaryBatchDescriptor>,
}

#[cfg(feature = "alloc")]
impl IpcPlan {
    /// Create an empty plan.
    #[must_use]
    pub fn new(framing: IpcFraming) -> Self {
        Self {
            framing,
            record_batches: Vec::new(),
            dictionary_batches: Vec::new(),
        }
    }

    /// Append a record batch descriptor.
    pub fn push_record_batch(&mut self, batch: RecordBatchDescriptor) {
        self.record_batches.push(batch);
    }

    /// Append a dictionary batch descriptor.
    pub fn push_dictionary_batch(&mut self, batch: DictionaryBatchDescriptor) {
        self.dictionary_batches.push(batch);
    }

    /// Total number of batches.
    #[must_use]
    pub fn total_batch_count(&self) -> usize {
        self.record_batches.len() + self.dictionary_batches.len()
    }

    /// Returns `true` if the plan contains no batches.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.record_batches.is_empty() && self.dictionary_batches.is_empty()
    }
}

/// IPC materialization policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpcMaterializationPolicy {
    /// Prefer zero-copy exposure when possible.
    pub prefer_zero_copy: bool,
    /// Preserve dictionary batches instead of expanding them.
    pub preserve_dictionaries: bool,
    /// Allow stream framing.
    pub allow_stream: bool,
    /// Allow file framing.
    pub allow_file: bool,
}

impl IpcMaterializationPolicy {
    /// Conservative default policy.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            prefer_zero_copy: true,
            preserve_dictionaries: true,
            allow_stream: true,
            allow_file: true,
        }
    }
}

impl Default for IpcMaterializationPolicy {
    fn default() -> Self {
        Self::conservative()
    }
}

/// Error raised by IPC planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcPlanError {
    /// A required batch was missing.
    MissingBatch {
        /// Batch kind.
        kind: BatchKind,
    },
    /// The requested framing mode is not allowed.
    FramingNotAllowed {
        /// Requested framing.
        framing: IpcFraming,
    },
    /// The batch layout is internally inconsistent.
    InconsistentLayout,
}

impl fmt::Display for IpcPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBatch { kind } => write!(f, "missing IPC batch: {kind:?}"),
            Self::FramingNotAllowed { framing } => write!(f, "framing not allowed: {framing:?}"),
            Self::InconsistentLayout => write!(f, "inconsistent IPC layout"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for IpcPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn plan_tracks_batches() {
        let mut plan = IpcPlan::new(IpcFraming::Stream);
        assert!(plan.is_empty());
        plan.push_record_batch(RecordBatchDescriptor::new(
            1,
            128,
            4,
            Some((0, 1024)),
            BufferOwnership::ZeroCopyEligible,
        ));
        plan.push_dictionary_batch(DictionaryBatchDescriptor::new(7, 16, None, false));
        assert_eq!(plan.total_batch_count(), 2);
        assert!(!plan.is_empty());
        assert!(plan.record_batches[0].is_zero_copy_eligible());
    }

    #[test]
    fn policy_defaults_are_conservative() {
        let policy = IpcMaterializationPolicy::default();
        assert!(policy.prefer_zero_copy);
        assert!(policy.preserve_dictionaries);
        assert!(policy.allow_stream);
        assert!(policy.allow_file);
    }

    #[test]
    fn errors_format_with_context() {
        let err = IpcPlanError::FramingNotAllowed {
            framing: IpcFraming::File,
        };
        assert!(err.to_string().contains("File"));
    }
}
