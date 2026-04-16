//! Arrow schema model for Consus interoperability.
//!
//! This module defines the canonical schema boundary for the `consus-arrow`
//! crate. It is intentionally dependency-light and models the Arrow-facing
//! concepts needed by the rest of the workspace without binding to an
//! external Arrow implementation.
//!
//! ## Specification
//!
//! Arrow schemas are composed of ordered fields, each with a name,
//! datatype, nullability, and optional child fields.
//!
//! ## Invariants
//!
//! - Field names are non-empty.
//! - Schema field order is preserved.
//! - Nested schemas are represented recursively.
//! - Nullability is explicit and never inferred from datatype alone.
//!
//! ## Architecture
//!
//! ```text
//! schema/
//! ├── datatype/   # Arrow logical/physical datatype model
//! ├── field/      # Field descriptors and validation
//! └── schema/     # Ordered field collections and projection rules
//! ```

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::fmt;

#[cfg(feature = "alloc")]
use crate::datatype::ArrowDataType;
#[cfg(feature = "alloc")]
use crate::field::ArrowField;
use consus_core::Error;

/// Ordered Arrow schema descriptor.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowSchema {
    fields: Vec<ArrowField>,
}

#[cfg(feature = "alloc")]
impl ArrowSchema {
    /// Create an empty schema.
    #[must_use]
    pub fn empty() -> Self {
        Self { fields: Vec::new() }
    }

    /// Create a schema from ordered fields.
    #[must_use]
    pub fn new(fields: Vec<ArrowField>) -> Self {
        Self { fields }
    }

    /// Return the ordered fields.
    #[must_use]
    pub fn fields(&self) -> &[ArrowField] {
        &self.fields
    }

    /// Return the number of fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Returns `true` when the schema contains no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Find a field by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&ArrowField> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// Validate the schema.
    pub fn validate(&self) -> Result<(), ArrowSchemaError> {
        let mut i = 0;
        while i < self.fields.len() {
            self.fields[i].validate().map_err(ArrowSchemaError::from)?;
            let mut j = i + 1;
            while j < self.fields.len() {
                if self.fields[i].name == self.fields[j].name {
                    return Err(ArrowSchemaError::DuplicateFieldName {
                        name: self.fields[i].name.clone(),
                    });
                }
                j += 1;
            }
            i += 1;
        }
        Ok(())
    }

    /// Project the schema onto a subset of field names.
    pub fn project(&self, names: &[&str]) -> Result<Self, ArrowSchemaError> {
        if names.is_empty() {
            return Err(ArrowSchemaError::EmptyProjection);
        }

        let mut projected = Vec::with_capacity(names.len());
        for name in names {
            let field = self
                .field(name)
                .ok_or_else(|| ArrowSchemaError::UnknownField {
                    name: (*name).to_owned(),
                })?;
            projected.push(field.clone());
        }

        Ok(Self::new(projected))
    }

    /// Count nullable fields.
    #[must_use]
    pub fn nullable_field_count(&self) -> usize {
        self.fields.iter().filter(|field| field.is_nullable()).count()
    }

    /// Count nested fields.
    #[must_use]
    pub fn nested_field_count(&self) -> usize {
        self.fields.iter().filter(|field| field.is_nested()).count()
    }

    /// Merge two schemas under a conservative compatibility rule.
    ///
    /// The resulting schema preserves the left schema's field order and
    /// appends fields present only in `other` when they do not conflict
    /// by name. Conflicting duplicate names are rejected.
    pub fn merge(&self, other: &Self) -> Result<ArrowSchemaMergePlan, ArrowSchemaError> {
        let mut merged_fields = self.fields.clone();
        let mut steps = Vec::new();

        let mut idx = 0;
        while idx < other.fields.len() {
            let field = &other.fields[idx];
            match self.field(&field.name) {
                Some(existing) => {
                    if existing != field {
                        steps.push(ArrowSchemaMergeStep::UpdateField {
                            name: field.name.clone(),
                        });
                        merged_fields[idx] = field.clone();
                    } else {
                        steps.push(ArrowSchemaMergeStep::KeepField {
                            name: field.name.clone(),
                        });
                    }
                }
                None => {
                    steps.push(ArrowSchemaMergeStep::AddField {
                        name: field.name.clone(),
                    });
                    merged_fields.push(field.clone());
                }
            }
            idx += 1;
        }

        Ok(ArrowSchemaMergePlan {
            left: self.clone(),
            right: other.clone(),
            merged: Self::new(merged_fields),
            steps,
        })
    }
}

#[cfg(feature = "alloc")]
impl Default for ArrowSchema {
    fn default() -> Self {
        Self::empty()
    }
}

/// Error raised during Arrow schema validation or projection.
#[derive(Debug)]
pub enum ArrowSchemaError {
    /// A field name is empty.
    EmptyFieldName,
    /// Two fields in the same scope share a name.
    DuplicateFieldName {
        /// Conflicting field name.
        name: alloc::string::String,
    },
    /// A requested field does not exist.
    UnknownField {
        /// Missing field name.
        name: alloc::string::String,
    },
    /// Projection list is empty.
    EmptyProjection,
    /// Validation failed in a child descriptor or lower-level model.
    InvalidField {
        /// Source error.
        source: Error,
    },
}

impl fmt::Display for ArrowSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFieldName => write!(f, "arrow field name must not be empty"),
            Self::DuplicateFieldName { name } => {
                write!(f, "duplicate arrow field name: {name}")
            }
            Self::UnknownField { name } => write!(f, "unknown arrow field: {name}"),
            Self::EmptyProjection => write!(f, "arrow schema projection is empty"),
            Self::InvalidField { source } => write!(f, "invalid arrow field: {source}"),
        }
    }
}

impl From<Error> for ArrowSchemaError {
    fn from(source: Error) -> Self {
        Self::InvalidField { source }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ArrowSchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidField { source } => Some(source),
            _ => None,
        }
    }
}

/// A step in an Arrow schema merge plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrowSchemaMergeStep {
    /// Keep an unchanged field.
    KeepField {
        /// Field name.
        name: alloc::string::String,
    },
    /// Add a new field.
    AddField {
        /// Field name.
        name: alloc::string::String,
    },
    /// Update an existing field with a compatible replacement.
    UpdateField {
        /// Field name.
        name: alloc::string::String,
    },
}

/// Merge plan for two Arrow schemas.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowSchemaMergePlan {
    /// Left-hand schema.
    pub left: ArrowSchema,
    /// Right-hand schema.
    pub right: ArrowSchema,
    /// Merged schema.
    pub merged: ArrowSchema,
    /// Merge steps.
    pub steps: Vec<ArrowSchemaMergeStep>,
}

#[cfg(feature = "alloc")]
impl ArrowSchemaMergePlan {
    /// Returns `true` when the merge introduces no changes.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.left == self.right
            && self
                .steps
                .iter()
                .all(|step| matches!(step, ArrowSchemaMergeStep::KeepField { .. }))
    }

    /// Number of merge steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Borrow the resulting schema.
    #[must_use]
    pub fn merged_schema(&self) -> &ArrowSchema {
        &self.merged
    }
}

/// Projection plan for a schema.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaProjectionPlan {
    /// Source schema.
    pub source: ArrowSchema,
    /// Requested field names in projection order.
    pub fields: Vec<alloc::string::String>,
    /// Projected schema.
    pub projected: ArrowSchema,
}

#[cfg(feature = "alloc")]
impl SchemaProjectionPlan {
    /// Build a projection plan from a schema and field names.
    pub fn new(
        source: ArrowSchema,
        fields: Vec<alloc::string::String>,
    ) -> Result<Self, ArrowSchemaError> {
        let requested: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
        let projected = source.project(&requested)?;
        Ok(Self {
            source,
            fields,
            projected,
        })
    }

    /// Returns `true` when the projection keeps all source fields.
    #[must_use]
    pub fn is_full_projection(&self) -> bool {
        self.projected.field_count() == self.source.field_count()
    }

    /// Number of projected fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.projected.field_count()
    }
}

#[cfg(feature = "alloc")]
impl ArrowSchema {
    /// Construct a schema from Consus core datatypes and field names.
    #[must_use]
    pub fn from_pairs(fields: &[(alloc::string::String, ArrowDataType)]) -> Self {
        let mut ordered = Vec::with_capacity(fields.len());
        for (name, data_type) in fields {
            ordered.push(crate::field::ArrowField::new(
                name.clone(),
                data_type.to_consus_datatype(),
                true,
            ));
        }
        Self::new(ordered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    use crate::field::{ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind};

    #[cfg(feature = "alloc")]
    fn sample_field(name: &str) -> ArrowField {
        ArrowFieldBuilder::new(
            ArrowFieldId::new(name.len() as u32),
            name.to_owned(),
            ArrowFieldKind::Utf8,
            ArrowDataType::Utf8.to_consus_datatype(),
        )
        .nullable(false)
        .build()
        .expect("field must build")
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn empty_schema_is_valid() {
        let schema = ArrowSchema::empty();
        assert!(schema.validate().is_ok());
        assert_eq!(schema.field_count(), 0);
        assert!(schema.is_empty());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn schema_detects_duplicate_names() {
        let schema = ArrowSchema::new(vec![sample_field("x"), sample_field("x")]);
        let err = schema.validate().unwrap_err();
        match err {
            ArrowSchemaError::DuplicateFieldName { .. } => {}
            other => panic!("expected DuplicateFieldName, got {other}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn projection_preserves_requested_order() {
        let schema = ArrowSchema::new(vec![sample_field("a"), sample_field("b"), sample_field("c")]);

        let projected = schema.project(&["c", "a"]).unwrap();
        assert_eq!(projected.field_count(), 2);
        assert_eq!(projected.fields()[0].name, "c");
        assert_eq!(projected.fields()[1].name, "a");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn merge_appends_new_fields() {
        let left = ArrowSchema::new(vec![sample_field("a")]);
        let right = ArrowSchema::new(vec![sample_field("a"), sample_field("b")]);

        let plan = left.merge(&right).unwrap();
        assert_eq!(plan.step_count(), 2);
        assert_eq!(plan.merged_schema().field_count(), 2);
        assert_eq!(plan.merged_schema().fields()[1].name, "b");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn projection_plan_round_trip() {
        let source = ArrowSchema::new(vec![sample_field("x"), sample_field("y")]);
        let plan =
            SchemaProjectionPlan::new(source, vec![alloc::string::String::from("y")]).unwrap();
        assert_eq!(plan.field_count(), 1);
        assert!(!plan.is_full_projection());
    }
}
