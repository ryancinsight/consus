//! Parquet field and schema primitives.
//!
//! This module defines the canonical field-level schema model used by
//! `consus-parquet`.
//!
//! ## Specification
//!
//! A Parquet schema field carries:
//! - stable identity
//! - field name
//! - repetition semantics
//! - physical storage type
//! - optional logical annotation
//! - optional nested children
//!
//! ## Invariants
//!
//! - Field names are non-empty.
//! - Field identity is stable within a schema.
//! - Nested children are only valid for group fields.
//! - Schema order is preserved.
//! - The model is independent of wire encoding.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use core::fmt;

use consus_core::Result;

use super::logical::{LogicalType, Repetition};
use super::physical::ParquetPhysicalType;

/// Stable identifier for a Parquet field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldId(u32);

impl FieldId {
    /// Create a new field identifier.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the raw identifier value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Canonical Parquet field descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDescriptor {
    id: FieldId,
    name: String,
    repetition: Repetition,
    physical_type: ParquetPhysicalType,
    logical_type: Option<LogicalType>,
    children: Vec<FieldDescriptor>,
}

impl FieldDescriptor {
    /// Create a required scalar field.
    #[must_use]
    pub fn required(
        id: FieldId,
        name: impl Into<String>,
        physical_type: ParquetPhysicalType,
    ) -> Self {
        Self::new(
            id,
            name,
            Repetition::Required,
            physical_type,
            None,
            Vec::new(),
        )
    }

    /// Create an optional scalar field.
    #[must_use]
    pub fn optional(
        id: FieldId,
        name: impl Into<String>,
        physical_type: ParquetPhysicalType,
        logical_type: Option<LogicalType>,
    ) -> Self {
        Self::new(
            id,
            name,
            Repetition::Optional,
            physical_type,
            logical_type,
            Vec::new(),
        )
    }

    /// Create a repeated scalar field.
    #[must_use]
    pub fn repeated(
        id: FieldId,
        name: impl Into<String>,
        physical_type: ParquetPhysicalType,
    ) -> Self {
        Self::new(
            id,
            name,
            Repetition::Repeated,
            physical_type,
            None,
            Vec::new(),
        )
    }

    /// Create a nested group field.
    #[must_use]
    pub fn group(
        id: FieldId,
        name: impl Into<String>,
        repetition: Repetition,
        children: Vec<FieldDescriptor>,
    ) -> Self {
        Self::new(
            id,
            name,
            repetition,
            ParquetPhysicalType::ByteArray,
            None,
            children,
        )
    }

    /// Construct a field descriptor.
    #[must_use]
    pub fn new(
        id: FieldId,
        name: impl Into<String>,
        repetition: Repetition,
        physical_type: ParquetPhysicalType,
        logical_type: Option<LogicalType>,
        children: Vec<FieldDescriptor>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            repetition,
            physical_type,
            logical_type,
            children,
        }
    }

    /// Validate the descriptor.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(consus_core::Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("parquet field name must not be empty"),
            });
        }

        if self.is_group() {
            let mut i = 0;
            while i < self.children.len() {
                self.children[i].validate()?;
                i += 1;
            }
        } else if !self.children.is_empty() {
            return Err(consus_core::Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("non-group fields cannot have children"),
            });
        }

        Ok(())
    }

    /// Return the stable field identifier.
    #[must_use]
    pub const fn id(&self) -> FieldId {
        self.id
    }

    /// Return the field name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the field repetition.
    #[must_use]
    pub const fn repetition(&self) -> Repetition {
        self.repetition
    }

    /// Return the physical type.
    #[must_use]
    pub const fn physical_type(&self) -> ParquetPhysicalType {
        self.physical_type
    }

    /// Return the logical type annotation, if any.
    #[must_use]
    pub const fn logical_type(&self) -> Option<&LogicalType> {
        self.logical_type.as_ref()
    }

    /// Return nested children.
    #[must_use]
    pub fn children(&self) -> &[FieldDescriptor] {
        &self.children
    }

    /// Returns `true` if this is a group field.
    #[must_use]
    pub const fn is_group(&self) -> bool {
        !self.children.is_empty()
    }

    /// Returns `true` if the field is required.
    #[must_use]
    pub const fn is_required(&self) -> bool {
        matches!(self.repetition, Repetition::Required)
    }

    /// Returns `true` if the field is optional.
    #[must_use]
    pub const fn is_optional(&self) -> bool {
        matches!(self.repetition, Repetition::Optional)
    }

    /// Returns `true` if the field is repeated.
    #[must_use]
    pub const fn is_repeated(&self) -> bool {
        matches!(self.repetition, Repetition::Repeated)
    }
}

/// Top-level schema descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaDescriptor {
    fields: Vec<FieldDescriptor>,
}

impl SchemaDescriptor {
    /// Create an empty schema.
    #[must_use]
    pub fn empty() -> Self {
        Self { fields: Vec::new() }
    }

    /// Create a schema from fields.
    #[must_use]
    pub fn new(fields: Vec<FieldDescriptor>) -> Self {
        Self { fields }
    }

    /// Return the number of top-level fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Returns `true` if the schema contains no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Borrow the top-level fields.
    #[must_use]
    pub fn fields(&self) -> &[FieldDescriptor] {
        &self.fields
    }

    /// Find a field by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&FieldDescriptor> {
        let mut i = 0;
        while i < self.fields.len() {
            if self.fields[i].name() == name {
                return Some(&self.fields[i]);
            }
            i += 1;
        }
        None
    }

    /// Validate the schema recursively.
    pub fn validate(&self) -> Result<()> {
        let mut i = 0;
        while i < self.fields.len() {
            self.fields[i].validate()?;
            let mut j = i + 1;
            while j < self.fields.len() {
                if self.fields[i].name() == self.fields[j].name() {
                    return Err(consus_core::Error::InvalidFormat {
                        #[cfg(feature = "alloc")]
                        message: String::from("duplicate top-level parquet field name"),
                    });
                }
                j += 1;
            }
            i += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_id_roundtrip() {
        let id = FieldId::new(7);
        assert_eq!(id.get(), 7);
        assert_eq!(id.to_string(), "7");
    }

    #[test]
    fn scalar_descriptor_validates() {
        let field = FieldDescriptor::required(
            FieldId::new(1),
            "temperature",
            ParquetPhysicalType::Double,
        );
        assert_eq!(field.name(), "temperature");
        assert!(field.is_required());
        field.validate().unwrap();
    }

    #[test]
    fn group_descriptor_validates() {
        let child = FieldDescriptor::required(FieldId::new(2), "x", ParquetPhysicalType::Int32);
        let field = FieldDescriptor::group(FieldId::new(1), "point", Repetition::Required, vec![child]);
        assert!(field.is_group());
        field.validate().unwrap();
    }

    #[test]
    fn schema_find_field_works() {
        let schema = SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "x", ParquetPhysicalType::Int32),
            FieldDescriptor::required(FieldId::new(2), "y", ParquetPhysicalType::Int32),
        ]);
        assert_eq!(schema.field_count(), 2);
        assert_eq!(schema.field("y").unwrap().id().get(), 2);
        assert!(schema.field("z").is_none());
    }

    #[test]
    fn schema_rejects_duplicates() {
        let schema = SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "x", ParquetPhysicalType::Int32),
            FieldDescriptor::required(FieldId::new(2), "x", ParquetPhysicalType::Int64),
        ]);
        let err = schema.validate().unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got {other}"),
        }
    }
}
