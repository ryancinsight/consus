//! Arrow bridge descriptors for the implemented Parquet field model.
//!
//! This module maps the authoritative `consus-parquet` field schema onto
//! Arrow-oriented execution descriptors without depending on the Arrow crate.
//!
//! ## Specification
//!
//! - The bridge is descriptive, not an encoder or decoder.
//! - Zero-copy eligibility is explicit and derived from field structure.
//! - Field identity and repetition semantics are preserved.
//! - Nested fields are represented recursively through the implemented
//!   `FieldDescriptor` tree.
//!
//! ## Invariants
//!
//! - A bridge descriptor preserves field name, id, physical type, and repetition.
//! - Zero-copy eligibility requires fixed-width storage and required repetition.
//! - Optional or repeated fields are never marked zero-copy by default.
//! - Schema order is preserved.
//!
//! ## Architecture
//!
//! ```text
//! schema/arrow
//! ├── ArrowDataTypeHint
//! ├── ArrowZeroCopyConstraint
//! ├── ArrowFieldDescriptor
//! ├── ArrowBridge
//! └── ArrowIntegrationPlan
//! ```
//!
//! This module is aligned with the implemented schema model in:
//! - `crate::schema::field`
//! - `crate::schema::logical`
//! - `crate::schema::physical`
//! - `crate::schema::evolution`
//!
//! It intentionally uses the canonical schema types and no parallel field model.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use crate::schema::{FieldDescriptor, ParquetPhysicalType, Repetition, SchemaDescriptor};

/// Arrow data type hint derived from a Parquet field.
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
    /// Nested list-like array.
    List,
    /// Nested struct-like array.
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
    pub repetition: Repetition,
    /// Parquet physical type.
    pub physical_type: ParquetPhysicalType,
    /// Derived Arrow type hint.
    pub arrow_hint: ArrowDataTypeHint,
    /// Whether zero-copy materialization is permitted.
    pub zero_copy: bool,
    /// Constraints that must hold for zero-copy materialization.
    pub zero_copy_constraint: ArrowZeroCopyConstraint,
    /// Nested child descriptors for structured fields.
    pub children: Vec<ArrowFieldDescriptor>,
}

#[cfg(feature = "alloc")]
impl ArrowFieldDescriptor {
    /// Build a descriptor from a Parquet field.
    #[must_use]
    pub fn from_field(field: &FieldDescriptor) -> Self {
        let arrow_hint = arrow_hint_for_physical_type(field.physical_type());
        let fixed_width = matches!(
            field.physical_type(),
            ParquetPhysicalType::Boolean
                | ParquetPhysicalType::Int32
                | ParquetPhysicalType::Int64
                | ParquetPhysicalType::Int96
                | ParquetPhysicalType::Float
                | ParquetPhysicalType::Double
                | ParquetPhysicalType::FixedLenByteArray(_)
        );
        let zero_copy = field.is_required() && fixed_width;
        let zero_copy_constraint = if zero_copy {
            ArrowZeroCopyConstraint::direct()
        } else {
            ArrowZeroCopyConstraint::relaxed()
        };

        Self {
            name: field.name().to_owned(),
            field_id: Some(field.id().get()),
            repetition: field.repetition(),
            physical_type: field.physical_type(),
            arrow_hint,
            zero_copy,
            zero_copy_constraint,
            children: field.children().iter().map(Self::from_field).collect(),
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

    /// Return the stable field id.
    #[must_use]
    pub fn field_id(&self) -> Option<u32> {
        self.field_id
    }
}

/// Canonical Arrow bridge plan for one schema.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowBridge {
    /// Schema descriptor being bridged.
    pub schema: SchemaDescriptor,
    /// Field descriptors in schema order.
    pub fields: Vec<ArrowFieldDescriptor>,
}

#[cfg(feature = "alloc")]
impl ArrowBridge {
    /// Build a bridge plan from a schema descriptor.
    #[must_use]
    pub fn new(schema: SchemaDescriptor) -> Self {
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

    /// Returns `true` if any field is nested.
    #[must_use]
    pub fn has_nested_fields(&self) -> bool {
        self.fields.iter().any(ArrowFieldDescriptor::is_nested)
    }
}

/// Strategy for applying an Arrow integration path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowBridgeMode {
    /// Materialize directly from the existing Parquet buffers.
    ZeroCopy,
    /// Convert with an intermediate buffer when required.
    Materialize,
    /// Use nested Arrow representations for grouped data.
    Nested,
}

/// Mapping from a Consus datatype to an Arrow-compatible logical type hint.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowSchemaMapping {
    /// Source datatype.
    pub datatype: consus_core::Datatype,
    /// Arrow logical type hint.
    pub hint: ArrowDataTypeHint,
    /// Whether the mapping is exact.
    pub exact: bool,
}

impl ArrowSchemaMapping {
    /// Create a new schema mapping.
    #[must_use]
    pub fn new(datatype: consus_core::Datatype, hint: ArrowDataTypeHint, exact: bool) -> Self {
        Self {
            datatype,
            hint,
            exact,
        }
    }
}

/// Full Arrow integration plan.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowIntegrationPlan {
    /// Bridge descriptors for the schema.
    pub bridge: ArrowBridge,
    /// Selected bridge mode.
    pub mode: ArrowBridgeMode,
    /// Aggregate mapping for the schema.
    pub schema_mapping: Option<ArrowSchemaMapping>,
}

#[cfg(feature = "alloc")]
impl ArrowIntegrationPlan {
    /// Create a direct bridge plan.
    #[must_use]
    pub fn zero_copy(schema: SchemaDescriptor) -> Self {
        Self {
            bridge: ArrowBridge::new(schema),
            mode: ArrowBridgeMode::ZeroCopy,
            schema_mapping: None,
        }
    }

    /// Create a materializing bridge plan.
    #[must_use]
    pub fn materialize(schema: SchemaDescriptor) -> Self {
        Self {
            bridge: ArrowBridge::new(schema),
            mode: ArrowBridgeMode::Materialize,
            schema_mapping: None,
        }
    }

    /// Create a nested bridge plan.
    #[must_use]
    pub fn nested(schema: SchemaDescriptor) -> Self {
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

fn arrow_hint_for_physical_type(physical_type: ParquetPhysicalType) -> ArrowDataTypeHint {
    match physical_type {
        ParquetPhysicalType::Boolean => ArrowDataTypeHint::Boolean,
        ParquetPhysicalType::Int32 => ArrowDataTypeHint::Int32,
        ParquetPhysicalType::Int64 => ArrowDataTypeHint::Int64,
        ParquetPhysicalType::Int96 => ArrowDataTypeHint::Timestamp,
        ParquetPhysicalType::Float => ArrowDataTypeHint::Float32,
        ParquetPhysicalType::Double => ArrowDataTypeHint::Float64,
        ParquetPhysicalType::ByteArray => ArrowDataTypeHint::Binary,
        ParquetPhysicalType::FixedLenByteArray(width) => ArrowDataTypeHint::FixedSizeBinary(width),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FieldDescriptor, FieldId, ParquetPhysicalType, SchemaDescriptor};

    #[test]
    fn arrow_hint_mapping_is_stable() {
        assert_eq!(
            arrow_hint_for_physical_type(ParquetPhysicalType::Boolean),
            ArrowDataTypeHint::Boolean
        );
        assert_eq!(
            arrow_hint_for_physical_type(ParquetPhysicalType::Double),
            ArrowDataTypeHint::Float64
        );
        assert_eq!(
            arrow_hint_for_physical_type(ParquetPhysicalType::FixedLenByteArray(16)),
            ArrowDataTypeHint::FixedSizeBinary(16)
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn bridge_counts_zero_copy_fields() {
        let schema = SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "temperature", ParquetPhysicalType::Double),
            FieldDescriptor::optional(
                FieldId::new(2),
                "quality",
                ParquetPhysicalType::Boolean,
                None,
            ),
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
        let schema = SchemaDescriptor::new(Vec::new());
        let plan = ArrowIntegrationPlan::zero_copy(schema);
        assert!(plan.is_zero_copy());
        assert_eq!(plan.bridge.field_count(), 0);
    }
}
