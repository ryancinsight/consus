//! Parquet schema evolution model.
//!
//! This module defines the canonical schema evolution contracts used by
//! `consus-parquet`. It does not implement Parquet wire encoding. It
//! models how schema revisions are compared, merged, and projected.
//!
//! ## Invariants
//!
//! - Field identity is preserved through a stable `FieldId` when present.
//! - Evolution steps are explicit and ordered.
//! - A projection may drop fields, but it may not fabricate fields.
//! - Compatibility checks distinguish additive, breaking, and ambiguous changes.

use core::fmt;

use crate::schema::field::{FieldDescriptor, FieldId, SchemaDescriptor};

/// Schema evolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaMergeMode {
    /// Reject any incompatible change.
    Strict,
    /// Permit additive changes and compatible type widening.
    BackwardCompatible,
    /// Permit forward-compatible field removal in projection-only flows.
    ForwardCompatible,
    /// Permit all changes and record them as explicit evolution steps.
    Permissive,
}

/// A single schema evolution step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaEvolutionStep {
    /// A new field was added.
    AddField {
        /// Added field descriptor.
        field: FieldDescriptor,
    },
    /// A field was removed.
    RemoveField {
        /// Removed field identifier, if tracked.
        field_id: Option<FieldId>,
        /// Removed field name.
        name: alloc::string::String,
    },
    /// A field was renamed.
    RenameField {
        /// Stable field identifier.
        field_id: FieldId,
        /// Previous field name.
        from: alloc::string::String,
        /// New field name.
        to: alloc::string::String,
    },
    /// A field type changed in a compatible way.
    WidenFieldType {
        /// Stable field identifier.
        field_id: FieldId,
        /// Previous descriptor.
        from: FieldDescriptor,
        /// Updated descriptor.
        to: FieldDescriptor,
    },
    /// A field nullability changed.
    NullabilityChanged {
        /// Stable field identifier.
        field_id: FieldId,
        /// Previous nullability.
        from_nullable: bool,
        /// New nullability.
        to_nullable: bool,
    },
}

/// Complete schema evolution plan between two schema revisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaEvolution {
    /// Base schema revision.
    pub base: SchemaDescriptor,
    /// Target schema revision.
    pub target: SchemaDescriptor,
    /// Ordered list of evolution steps.
    pub steps: alloc::vec::Vec<SchemaEvolutionStep>,
    /// Merge mode used to derive this plan.
    pub mode: SchemaMergeMode,
}

/// Error emitted by schema merge/projection analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaMergeError {
    /// A required field is missing in the target schema.
    MissingRequiredField {
        /// Field name.
        name: alloc::string::String,
    },
    /// A field changed incompatibly.
    IncompatibleFieldChange {
        /// Field name.
        name: alloc::string::String,
    },
    /// Two fields collide on identity or name.
    FieldCollision {
        /// Field name.
        name: alloc::string::String,
    },
    /// A projection requests an unknown field.
    UnknownField {
        /// Requested field name.
        name: alloc::string::String,
    },
}

impl fmt::Display for SchemaMergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequiredField { name } => {
                write!(f, "missing required field: {name}")
            }
            Self::IncompatibleFieldChange { name } => {
                write!(f, "incompatible field change: {name}")
            }
            Self::FieldCollision { name } => write!(f, "field collision: {name}"),
            Self::UnknownField { name } => write!(f, "unknown field: {name}"),
        }
    }
}

/// Projection of a source schema into a target field subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaProjection {
    /// Source schema.
    pub source: SchemaDescriptor,
    /// Selected field names.
    pub fields: alloc::vec::Vec<alloc::string::String>,
}

/// Error raised when building a schema projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaProjectionError {
    /// The projection is empty.
    EmptyProjection,
    /// A requested field does not exist.
    UnknownField {
        /// Requested field name.
        name: alloc::string::String,
    },
}

impl fmt::Display for SchemaProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProjection => write!(f, "empty schema projection"),
            Self::UnknownField { name } => write!(f, "unknown projection field: {name}"),
        }
    }
}

impl SchemaEvolution {
    /// Construct an empty evolution plan between two schema revisions.
    #[must_use]
    pub fn new(base: SchemaDescriptor, target: SchemaDescriptor, mode: SchemaMergeMode) -> Self {
        Self {
            base,
            target,
            steps: alloc::vec::Vec::new(),
            mode,
        }
    }

    /// Append an explicit evolution step.
    pub fn push_step(&mut self, step: SchemaEvolutionStep) {
        self.steps.push(step);
    }

    /// Whether the plan contains no steps.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.steps.is_empty() && self.base == self.target
    }

    /// Number of evolution steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

impl SchemaProjection {
    /// Build a projection from a list of field names.
    pub fn new(
        source: SchemaDescriptor,
        fields: alloc::vec::Vec<alloc::string::String>,
    ) -> Result<Self, SchemaProjectionError> {
        if fields.is_empty() {
            return Err(SchemaProjectionError::EmptyProjection);
        }

        for field in &fields {
            if source.field(field).is_none() {
                return Err(SchemaProjectionError::UnknownField {
                    name: field.clone(),
                });
            }
        }

        Ok(Self { source, fields })
    }

    /// Whether the projection retains all source fields.
    #[must_use]
    pub fn is_full_projection(&self) -> bool {
        self.fields.len() == self.source.fields().len()
    }
}

impl SchemaMergeMode {
    /// Whether the mode permits additive changes.
    #[must_use]
    pub const fn permits_additions(self) -> bool {
        matches!(
            self,
            Self::BackwardCompatible | Self::ForwardCompatible | Self::Permissive
        )
    }

    /// Whether the mode permits breaking removals.
    #[must_use]
    pub const fn permits_removals(self) -> bool {
        matches!(self, Self::ForwardCompatible | Self::Permissive)
    }
}
