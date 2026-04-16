/// Arrow compute descriptors for the Consus Arrow runtime.
///
/// This module defines the canonical compute-planning layer for Arrow-style
/// operations over `consus-core` and `consus-parquet` schema models.
///
/// ## Scope
///
/// - no wire-format encoding
/// - no dependency on the external Arrow crate
/// - compute plans, kernel descriptors, and buffer compatibility rules
/// - zero-copy eligibility for projection and cast operations
///
/// ## Invariants
///
/// - Compute plans preserve input field identity.
/// - Kernel selection is explicit and value-semantic.
/// - Projection does not fabricate fields.
/// - Casts are only accepted when the source and target types are compatible
///   under the declared conversion mode.
///
/// ## Architecture
///
/// ```text
/// compute/
/// ├── kernel        # kernel identifiers and execution modes
/// ├── cast          # cast compatibility and conversion plans
/// ├── projection    # field projection descriptors
/// └── plan          # full compute plan composition
/// ```

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use core::fmt;

use consus_core::Datatype;

use crate::field::ArrowField;

/// Execution mode for an Arrow compute plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeMode {
    /// Direct, zero-copy eligible execution.
    ZeroCopy,
    /// Buffer materialization is allowed.
    Materialize,
    /// A fallback conversion path is used.
    Convert,
}

/// Canonical compute kernel identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputeKernel {
    /// Kernel name.
    #[cfg(feature = "alloc")]
    pub name: String,
    /// Kernel version tag.
    pub version: u16,
    /// Execution mode required by the kernel.
    pub mode: ComputeMode,
}

#[cfg(feature = "alloc")]
impl ComputeKernel {
    /// Create a new compute kernel descriptor.
    #[must_use]
    pub fn new(name: String, version: u16, mode: ComputeMode) -> Self {
        Self { name, version, mode }
    }
}

/// Compatibility of a cast between two datatypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastCompatibility {
    /// Cast is exact and lossless.
    Exact,
    /// Cast is representable but may require a buffer conversion.
    Compatible,
    /// Cast is not permitted.
    Incompatible,
}

/// Conversion mode for a cast operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastMode {
    /// Direct reinterpretation when allowed by layout and type.
    Reinterpret,
    /// Conversion through an intermediate buffer.
    Convert,
    /// Strict rejection unless the cast is exact.
    Strict,
}

/// Descriptor for a cast from one datatype to another.
#[derive(Debug, Clone, PartialEq)]
pub struct CastPlan {
    /// Source datatype.
    pub source: Datatype,
    /// Target datatype.
    pub target: Datatype,
    /// Compatibility classification.
    pub compatibility: CastCompatibility,
    /// Declared cast mode.
    pub mode: CastMode,
}

impl CastPlan {
    /// Create a cast plan.
    #[must_use]
    pub fn new(
        source: Datatype,
        target: Datatype,
        compatibility: CastCompatibility,
        mode: CastMode,
    ) -> Self {
        Self {
            source,
            target,
            compatibility,
            mode,
        }
    }

    /// Returns `true` if the cast may proceed.
    #[must_use]
    pub fn permits_cast(&self) -> bool {
        !matches!(self.compatibility, CastCompatibility::Incompatible)
    }
}

/// Projection of a schema into a selected field subset.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionPlan {
    /// Selected field paths in declaration order.
    pub fields: Vec<String>,
}

#[cfg(feature = "alloc")]
impl ProjectionPlan {
    /// Create a projection plan.
    #[must_use]
    pub fn new(fields: Vec<String>) -> Self {
        Self { fields }
    }

    /// Returns `true` if the projection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Number of selected fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// Full Arrow compute plan.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ComputePlan {
    /// Input fields.
    pub input_fields: Vec<ArrowField>,
    /// Kernel selection.
    pub kernel: ComputeKernel,
    /// Optional cast operation.
    pub cast: Option<CastPlan>,
    /// Optional projection.
    pub projection: Option<ProjectionPlan>,
}

#[cfg(feature = "alloc")]
impl ComputePlan {
    /// Create a compute plan with no cast or projection.
    #[must_use]
    pub fn new(input_fields: Vec<ArrowField>, kernel: ComputeKernel) -> Self {
        Self {
            input_fields,
            kernel,
            cast: None,
            projection: None,
        }
    }

    /// Attach a cast plan.
    #[must_use]
    pub fn with_cast(mut self, cast: CastPlan) -> Self {
        self.cast = Some(cast);
        self
    }

    /// Attach a projection plan.
    #[must_use]
    pub fn with_projection(mut self, projection: ProjectionPlan) -> Self {
        self.projection = Some(projection);
        self
    }

    /// Number of input fields.
    #[must_use]
    pub fn input_field_count(&self) -> usize {
        self.input_fields.len()
    }

    /// Returns `true` if the plan permits direct zero-copy execution.
    #[must_use]
    pub fn is_zero_copy(&self) -> bool {
        matches!(self.kernel.mode, ComputeMode::ZeroCopy)
            && self.cast.as_ref().map_or(true, |cast| cast.permits_cast())
    }
}

/// Error returned when a compute plan cannot be formed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputePlanError {
    /// The requested projection is empty.
    EmptyProjection,
    /// The cast is incompatible.
    IncompatibleCast,
    /// The kernel mode is inconsistent with the selected operation.
    InvalidKernelMode,
}

impl fmt::Display for ComputePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProjection => write!(f, "empty compute projection"),
            Self::IncompatibleCast => write!(f, "incompatible compute cast"),
            Self::InvalidKernelMode => write!(f, "invalid compute kernel mode"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ComputePlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    fn sample_field(name: &str) -> ArrowField {
        ArrowField::new(name.to_owned(), Datatype::Boolean, false)
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compute_plan_tracks_inputs() {
        let kernel = ComputeKernel::new(String::from("identity"), 1, ComputeMode::ZeroCopy);
        let plan = ComputePlan::new(vec![sample_field("x"), sample_field("y")], kernel);
        assert_eq!(plan.input_field_count(), 2);
        assert!(plan.is_zero_copy());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn projection_plan_counts_fields() {
        let projection = ProjectionPlan::new(vec![String::from("x"), String::from("y")]);
        assert_eq!(projection.field_count(), 2);
        assert!(!projection.is_empty());
    }

    #[test]
    fn cast_plan_accepts_exact_casts() {
        let cast = CastPlan::new(
            Datatype::Boolean,
            Datatype::Boolean,
            CastCompatibility::Exact,
            CastMode::Strict,
        );
        assert!(cast.permits_cast());
    }

    #[test]
    fn cast_plan_rejects_incompatible_casts() {
        let cast = CastPlan::new(
            Datatype::Boolean,
            Datatype::Reference(consus_core::ReferenceType::Object),
            CastCompatibility::Incompatible,
            CastMode::Strict,
        );
        assert!(!cast.permits_cast());
    }
}
