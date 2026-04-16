//! Arrow bridge descriptors for the Consus Arrow crate.
//!
//! This module defines the canonical bridge layer between `consus-core`
//! data model types and Arrow-oriented execution descriptors.
//!
//! ## Scope
//!
//! - Descriptive bridge metadata, not wire encoding.
//! - Zero-copy eligibility as an explicit structural property.
//! - Schema and field conversion helpers from `consus-core`.
//! - Bridge plans for materialized and zero-copy execution paths.
//!
//! ## Invariants
//!
//! - Bridge descriptors preserve field identity and order.
//! - Zero-copy eligibility requires fixed-width storage and required fields.
//! - Nested fields are represented recursively.
//! - The bridge does not depend on the external Arrow crate.
//!
//! ## Architecture
//!
//! - `ArrowDataTypeHint` captures Arrow-facing logical shape.
//! - `ArrowZeroCopyConstraint` records conditions for direct exposure.
//! - `ArrowFieldDescriptor` models one schema field.
//! - `ArrowSchemaMapping` records aggregate datatype mapping.
//! - `ArrowBridgePlan` selects an execution mode for the bridge.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::Datatype;

use crate::field::{ArrowField, ArrowFieldKind};
use crate::schema::ArrowSchema;

/// Arrow data type hint derived from a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowDataTypeHint {
    /// Boolean array.
    Boolean,
    /// Signed integer array.
    Int8,
    /// Signed integer array.
    Int16,
    /// Signed integer array.
    Int32,
    /// Signed integer array.
    Int64,
    /// Unsigned integer array.
    Uint8,
    /// Unsigned integer array.
    Uint16,
    /// Unsigned integer array.
    Uint32,
    /// Unsigned integer array.
    Uint64,
    /// 32-bit floating-point array.
    Float32,
    /// 64-bit floating-point array.
    Float64,
    /// UTF-8 string array.
    Utf8,
    /// Binary array.
    Binary,
    /// Fixed-size binary array.
    FixedSizeBinary(usize),
    /// List array.
    List,
    /// Struct array.
    Struct,
    /// Dictionary-encoded array.
    Dictionary,
    /// Timestamp-like array.
    Timestamp,
}

/// Constraint describing whether a field can be exposed zero-copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrowZeroCopyConstraint {
    /// Whether nullability must match exactly.
    pub nullability_must_match: bool,
    /// Whether only required fields are eligible.
    pub required_only: bool,
    /// Whether fixed-width physical storage is required.
    pub fixed_width_required: bool,
    /// Whether offsets must be preserved.
    pub preserve_offsets: bool,
}

impl ArrowZeroCopyConstraint {
    /// Constraint set for direct zero-copy eligibility.
    #[must_use]
    pub const fn direct() -> Self {
        Self {
            nullability_must_match: true,
            required_only: true,
            fixed_width_required: true,
            preserve_offsets: true,
        }
    }

    /// Constraint set for conversion-tolerant bridging.
    #[must_use]
    pub const fn relaxed() -> Self {
        Self {
            nullability_must_match: false,
            required_only: false,
            fixed_width_required: false,
            preserve_offsets: false,
        }
    }
}

/// Descriptor for one Arrow-facing field.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowFieldDescriptor {
    /// Field name.
    pub name: String,
    /// Stable field identifier.
    pub field_id: Option<u32>,
    /// Repetition and nullability semantics.
    pub repetition: crate::field::ArrowFieldMultiplicity,
    /// Physical storage type.
    pub physical_type: ArrowFieldKind,
    /// Derived Arrow type hint.
    pub arrow_hint: ArrowDataTypeHint,
    /// Whether zero-copy materialization is permitted.
    pub zero_copy: bool,
    /// Constraints that must hold for zero-copy materialization.
    pub zero_copy_constraint: ArrowZeroCopyConstraint,
    /// Nested child descriptors.
    pub children: Vec<ArrowFieldDescriptor>,
}

#[cfg(feature = "alloc")]
impl ArrowFieldDescriptor {
    /// Build a descriptor from a field descriptor.
    #[must_use]
    pub fn from_field(field: &ArrowField) -> Self {
        let arrow_hint = arrow_hint_from_kind(field.kind);
        let fixed_width = field.is_fixed_size();
        let zero_copy = !field.is_nullable() && fixed_width;
        let zero_copy_constraint = if zero_copy {
            ArrowZeroCopyConstraint::direct()
        } else {
            ArrowZeroCopyConstraint::relaxed()
        };

        Self {
            name: field.name.clone(),
            field_id: Some(field.id.get()),
            repetition: field.semantics.multiplicity,
            physical_type: field.kind,
            arrow_hint,
            zero_copy,
            zero_copy_constraint,
            children: field.children.iter().map(Self::from_field).collect(),
        }
    }

    /// Whether the field can be bridged without allocating a new buffer.
    #[must_use]
    pub fn is_zero_copy_eligible(&self) -> bool {
        self.zero_copy
    }

    /// Whether the field is nested.
    #[must_use]
    pub fn is_nested(&self) -> bool {
        !self.children.is_empty()
    }
}

/// Mapping from a datatype to an Arrow-compatible logical type hint.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowSchemaMapping {
    /// Source datatype.
    pub datatype: Datatype,
    /// Arrow logical type hint.
    pub hint: ArrowDataTypeHint,
    /// Whether the mapping is exact.
    pub exact: bool,
}

impl ArrowSchemaMapping {
    /// Create a new schema mapping.
    #[must_use]
    pub fn new(datatype: Datatype, hint: ArrowDataTypeHint, exact: bool) -> Self {
        Self {
            datatype,
            hint,
            exact,
        }
    }
}

/// Arrow bridge execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowBridgeMode {
    /// Materialize directly from existing buffers.
    ZeroCopy,
    /// Convert with an intermediate buffer.
    Materialize,
    /// Use nested representations.
    Nested,
}

/// Full Arrow bridge plan for one schema.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowBridgePlan {
    /// Bridge descriptors for the schema.
    pub bridge: ArrowBridge,
    /// Selected bridge mode.
    pub mode: ArrowBridgeMode,
    /// Aggregate schema mapping.
    pub schema_mapping: Option<ArrowSchemaMapping>,
}

#[cfg(feature = "alloc")]
impl ArrowBridgePlan {
    /// Create a zero-copy bridge plan.
    #[must_use]
    pub fn zero_copy(schema: ArrowSchema) -> Self {
        Self {
            bridge: ArrowBridge::new(schema),
            mode: ArrowBridgeMode::ZeroCopy,
            schema_mapping: None,
        }
    }

    /// Create a materializing bridge plan.
    #[must_use]
    pub fn materialize(schema: ArrowSchema) -> Self {
        Self {
            bridge: ArrowBridge::new(schema),
            mode: ArrowBridgeMode::Materialize,
            schema_mapping: None,
        }
    }

    /// Create a nested bridge plan.
    #[must_use]
    pub fn nested(schema: ArrowSchema) -> Self {
        Self {
            bridge: ArrowBridge::new(schema),
            mode: ArrowBridgeMode::Nested,
            schema_mapping: None,
        }
    }

    /// Attach a schema mapping.
    #[must_use]
    pub fn with_schema_mapping(mut self, mapping: ArrowSchemaMapping) -> Self {
        self.schema_mapping = Some(mapping);
        self
    }

    /// Whether the plan uses direct zero-copy conversion.
    #[must_use]
    pub fn is_zero_copy(&self) -> bool {
        matches!(self.mode, ArrowBridgeMode::ZeroCopy)
    }
}

/// Canonical Arrow bridge for a schema.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowBridge {
    /// Schema descriptor being bridged.
    pub schema: ArrowSchema,
    /// Field descriptors in schema order.
    pub fields: Vec<ArrowFieldDescriptor>,
}

#[cfg(feature = "alloc")]
impl ArrowBridge {
    /// Build a bridge from a schema descriptor.
    #[must_use]
    pub fn new(schema: ArrowSchema) -> Self {
        let fields = schema
            .fields()
            .iter()
            .map(ArrowFieldDescriptor::from_field)
            .collect();
        Self { schema, fields }
    }

    /// Number of fields in the bridge.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Whether the bridge contains no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Count zero-copy eligible fields.
    #[must_use]
    pub fn zero_copy_field_count(&self) -> usize {
        self.fields.iter().filter(|field| field.zero_copy).count()
    }

    /// Borrow the descriptor for a named field.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&ArrowFieldDescriptor> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// Whether the bridge contains any nested fields.
    #[must_use]
    pub fn has_nested_fields(&self) -> bool {
        self.fields.iter().any(ArrowFieldDescriptor::is_nested)
    }
}

fn arrow_hint_from_kind(kind: ArrowFieldKind) -> ArrowDataTypeHint {
    match kind {
        ArrowFieldKind::Boolean => ArrowDataTypeHint::Boolean,
        ArrowFieldKind::Int => ArrowDataTypeHint::Int64,
        ArrowFieldKind::Uint => ArrowDataTypeHint::Uint64,
        ArrowFieldKind::Float => ArrowDataTypeHint::Float64,
        ArrowFieldKind::Utf8 => ArrowDataTypeHint::Utf8,
        ArrowFieldKind::Binary => ArrowDataTypeHint::Binary,
        ArrowFieldKind::FixedSizeBinary(width) => ArrowDataTypeHint::FixedSizeBinary(width),
        ArrowFieldKind::List => ArrowDataTypeHint::List,
        ArrowFieldKind::Struct => ArrowDataTypeHint::Struct,
        ArrowFieldKind::Dictionary => ArrowDataTypeHint::Dictionary,
        ArrowFieldKind::Timestamp => ArrowDataTypeHint::Timestamp,
        ArrowFieldKind::Duration => ArrowDataTypeHint::Timestamp,
        ArrowFieldKind::Map => ArrowDataTypeHint::Struct,
        ArrowFieldKind::Extension => ArrowDataTypeHint::Binary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    use crate::field::{ArrowField, ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind};
    #[cfg(feature = "alloc")]
    use crate::schema::ArrowSchema;

    #[test]
    fn arrow_hint_mapping_is_stable() {
        assert_eq!(
            arrow_hint_from_kind(ArrowFieldKind::Boolean),
            ArrowDataTypeHint::Boolean
        );
        assert_eq!(
            arrow_hint_from_kind(ArrowFieldKind::FixedSizeBinary(16)),
            ArrowDataTypeHint::FixedSizeBinary(16)
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn bridge_counts_zero_copy_fields() {
        let schema = ArrowSchema::new(vec![
            ArrowFieldBuilder::new(
                ArrowFieldId::new(1),
                String::from("temperature"),
                ArrowFieldKind::Float,
                consus_core::Datatype::Float {
                    bits: core::num::NonZeroUsize::new(64).expect("nonzero"),
                    byte_order: consus_core::ByteOrder::LittleEndian,
                },
            )
            .nullable(false)
            .build()
            .expect("field must build"),
            ArrowFieldBuilder::new(
                ArrowFieldId::new(2),
                String::from("quality"),
                ArrowFieldKind::Boolean,
                consus_core::Datatype::Boolean,
            )
            .nullable(true)
            .build()
            .expect("field must build"),
        ]);

        let bridge = ArrowBridge::new(schema);
        assert_eq!(bridge.field_count(), 2);
        assert_eq!(bridge.zero_copy_field_count(), 1);
        assert!(bridge.field("temperature").unwrap().is_zero_copy_eligible());
        assert!(!bridge.field("quality").unwrap().is_zero_copy_eligible());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn integration_plan_reports_mode() {
        let schema = ArrowSchema::empty();
        let plan = ArrowBridgePlan::zero_copy(schema);
        assert!(plan.is_zero_copy());
        assert_eq!(plan.bridge.field_count(), 0);
    }
}
