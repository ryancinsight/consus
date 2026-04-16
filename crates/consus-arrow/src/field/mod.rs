/// Arrow field definitions and schema-adjacent metadata.
///
/// This module provides the canonical field model for `consus-arrow`.
/// It is intentionally independent of any external Arrow crate so the
/// crate can serve as a stable Rust implementation boundary.
///
/// ## Invariants
///
/// - Field names are non-empty.
/// - Field identity is stable within a schema.
/// - Field nullability is explicit.
/// - Nested fields are represented recursively.
/// - Field ordering is preserved.
///
/// ## Architecture
///
/// - `ArrowFieldId` identifies a field within a schema.
/// - `ArrowFieldKind` classifies the logical shape of the field.
/// - `ArrowField` stores the canonical field descriptor.
/// - `ArrowFieldBuilder` constructs validated field values.
///
/// This module is designed to support:
/// - schema materialization
/// - IPC metadata translation
/// - compute planning
/// - zero-copy eligibility analysis

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use core::fmt;

use consus_core::{Datatype, Error, Result};

/// Stable identifier for an Arrow field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArrowFieldId(u32);

impl ArrowFieldId {
    /// Create a new field identifier.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the raw identifier.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ArrowFieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Field nullability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrowNullability {
    /// Field must be present in every logical record.
    Required,
    /// Field may be absent.
    Optional,
}

impl ArrowNullability {
    /// Returns `true` if the field may be null.
    #[must_use]
    pub const fn is_nullable(self) -> bool {
        matches!(self, Self::Optional)
    }
}

/// Field repetition model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrowFieldMultiplicity {
    /// Single value per row.
    Scalar,
    /// Repeated values per row.
    Repeated,
}

impl ArrowFieldMultiplicity {
    /// Returns `true` if the field can contain repeated values.
    #[must_use]
    pub const fn is_repeated(self) -> bool {
        matches!(self, Self::Repeated)
    }
}

/// Canonical Arrow field kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrowFieldKind {
    /// Boolean field.
    Boolean,
    /// Signed integer field.
    Int,
    /// Unsigned integer field.
    Uint,
    /// Floating-point field.
    Float,
    /// UTF-8 string field.
    Utf8,
    /// Binary field.
    Binary,
    /// Fixed-size binary field.
    FixedSizeBinary(usize),
    /// Nested list field.
    List,
    /// Nested struct field.
    Struct,
    /// Dictionary-encoded field.
    Dictionary,
    /// Timestamp field.
    Timestamp,
    /// Duration field.
    Duration,
    /// Map field.
    Map,
    /// Extension field.
    Extension,
}

/// Validation and conversion hints for Arrow fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArrowFieldSemantics {
    /// Nullability model.
    pub nullability: ArrowNullability,
    /// Repetition model.
    pub multiplicity: ArrowFieldMultiplicity,
}

impl ArrowFieldSemantics {
    /// Required scalar semantics.
    #[must_use]
    pub const fn required_scalar() -> Self {
        Self {
            nullability: ArrowNullability::Required,
            multiplicity: ArrowFieldMultiplicity::Scalar,
        }
    }

    /// Optional scalar semantics.
    #[must_use]
    pub const fn optional_scalar() -> Self {
        Self {
            nullability: ArrowNullability::Optional,
            multiplicity: ArrowFieldMultiplicity::Scalar,
        }
    }

    /// Required repeated semantics.
    #[must_use]
    pub const fn required_repeated() -> Self {
        Self {
            nullability: ArrowNullability::Required,
            multiplicity: ArrowFieldMultiplicity::Repeated,
        }
    }
}

/// Canonical Arrow field descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowField {
    /// Stable field identifier.
    pub id: ArrowFieldId,
    /// Field name.
    pub name: String,
    /// Canonical field kind.
    pub kind: ArrowFieldKind,
    /// Semantic flags.
    pub semantics: ArrowFieldSemantics,
    /// Logical datatype descriptor from Consus.
    pub datatype: Datatype,
    /// Child fields for nested structures.
    pub children: Vec<ArrowField>,
}

impl ArrowField {
    /// Create a new field descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, datatype: Datatype, nullable: bool) -> Self {
        let kind = kind_from_datatype(&datatype);
        Self {
            id: ArrowFieldId::new(0),
            name: name.into(),
            kind,
            semantics: if nullable {
                ArrowFieldSemantics::optional_scalar()
            } else {
                ArrowFieldSemantics::required_scalar()
            },
            datatype,
            children: Vec::new(),
        }
    }

    /// Validate a field descriptor.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` when:
    /// - the name is empty
    /// - a non-nested field has children
    /// - a repeated field is declared with incompatible structure
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("Arrow field name must not be empty"),
            });
        }

        if !self.is_nested() && !self.children.is_empty() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("non-nested Arrow fields cannot have children"),
            });
        }

        if self.is_dictionary_like()
            && !matches!(self.semantics.multiplicity, ArrowFieldMultiplicity::Scalar)
        {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("dictionary-like fields must be scalar"),
            });
        }

        let mut i = 0;
        while i < self.children.len() {
            self.children[i].validate()?;
            i += 1;
        }

        Ok(())
    }

    /// Returns `true` if the field is nested.
    #[must_use]
    pub fn is_nested(&self) -> bool {
        matches!(
            self.kind,
            ArrowFieldKind::List | ArrowFieldKind::Struct | ArrowFieldKind::Map
        )
    }

    /// Returns `true` if the field uses a fixed-size physical layout.
    #[must_use]
    pub fn is_fixed_size(&self) -> bool {
        matches!(
            self.kind,
            ArrowFieldKind::Boolean
                | ArrowFieldKind::Int
                | ArrowFieldKind::Uint
                | ArrowFieldKind::Float
                | ArrowFieldKind::Timestamp
                | ArrowFieldKind::Duration
                | ArrowFieldKind::FixedSizeBinary(_)
        )
    }

    /// Returns `true` if the field is dictionary-like.
    #[must_use]
    pub fn is_dictionary_like(&self) -> bool {
        matches!(self.kind, ArrowFieldKind::Dictionary)
    }

    /// Returns `true` if the field is nullable.
    #[must_use]
    pub fn is_nullable(&self) -> bool {
        self.semantics.nullability.is_nullable()
    }

    /// Returns `true` if the field is repeated.
    #[must_use]
    pub fn is_repeated(&self) -> bool {
        self.semantics.multiplicity.is_repeated()
    }

    /// Count the number of descendant fields.
    #[must_use]
    pub fn descendant_count(&self) -> usize {
        let mut total = 0;
        let mut i = 0;
        while i < self.children.len() {
            total += 1;
            total += self.children[i].descendant_count();
            i += 1;
        }
        total
    }

    /// Borrow the child fields.
    #[must_use]
    pub fn children(&self) -> &[ArrowField] {
        &self.children
    }
}

/// Builder for `ArrowField`.
#[derive(Debug, Clone)]
pub struct ArrowFieldBuilder {
    id: ArrowFieldId,
    name: String,
    kind: ArrowFieldKind,
    semantics: ArrowFieldSemantics,
    datatype: Datatype,
    children: Vec<ArrowField>,
}

impl ArrowFieldBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new(
        id: ArrowFieldId,
        name: impl Into<String>,
        kind: ArrowFieldKind,
        datatype: Datatype,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            semantics: ArrowFieldSemantics::required_scalar(),
            datatype,
            children: Vec::new(),
        }
    }

    /// Set nullability.
    #[must_use]
    pub fn nullable(mut self, value: bool) -> Self {
        self.semantics.nullability = if value {
            ArrowNullability::Optional
        } else {
            ArrowNullability::Required
        };
        self
    }

    /// Set multiplicity.
    #[must_use]
    pub fn repeated(mut self, value: bool) -> Self {
        self.semantics.multiplicity = if value {
            ArrowFieldMultiplicity::Repeated
        } else {
            ArrowFieldMultiplicity::Scalar
        };
        self
    }

    /// Add a child field.
    #[must_use]
    pub fn push_child(mut self, child: ArrowField) -> Self {
        self.children.push(child);
        self
    }

    /// Replace all child fields.
    #[must_use]
    pub fn children(mut self, children: Vec<ArrowField>) -> Self {
        self.children = children;
        self
    }

    /// Build the field and validate invariants.
    pub fn build(self) -> Result<ArrowField> {
        let field = ArrowField {
            id: self.id,
            name: self.name,
            kind: self.kind,
            semantics: self.semantics,
            datatype: self.datatype,
            children: self.children,
        };
        field.validate()?;
        Ok(field)
    }
}

/// Map a Consus datatype to an Arrow field kind.
#[must_use]
pub fn kind_from_datatype(datatype: &Datatype) -> ArrowFieldKind {
    match datatype {
        Datatype::Boolean => ArrowFieldKind::Boolean,
        Datatype::Integer { signed, .. } => {
            if *signed {
                ArrowFieldKind::Int
            } else {
                ArrowFieldKind::Uint
            }
        }
        Datatype::Float { .. } => ArrowFieldKind::Float,
        Datatype::FixedString { length, .. } => ArrowFieldKind::FixedSizeBinary(*length),
        Datatype::VariableString { .. } => ArrowFieldKind::Utf8,
        Datatype::Opaque { .. } => ArrowFieldKind::Binary,
        Datatype::Reference(_) => ArrowFieldKind::Binary,
        #[cfg(feature = "alloc")]
        Datatype::Compound { .. } => ArrowFieldKind::Struct,
        #[cfg(feature = "alloc")]
        Datatype::Array { .. } => ArrowFieldKind::List,
        #[cfg(feature = "alloc")]
        Datatype::Enum { .. } => ArrowFieldKind::Dictionary,
        #[cfg(feature = "alloc")]
        Datatype::VarLen { .. } => ArrowFieldKind::List,
        Datatype::Complex { .. } => ArrowFieldKind::Struct,
    }
}

/// Build a canonical Arrow field from a datatype and metadata.
#[must_use]
pub fn field_from_datatype(
    id: ArrowFieldId,
    name: impl Into<String>,
    datatype: Datatype,
    nullable: bool,
) -> ArrowField {
    let kind = kind_from_datatype(&datatype);
    let mut builder = ArrowFieldBuilder::new(id, name, kind, datatype);
    builder = builder.nullable(nullable);
    builder.build().expect("field construction must validate")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn builder_constructs_scalar_field() {
        let field = ArrowFieldBuilder::new(
            ArrowFieldId::new(1),
            String::from("temperature"),
            ArrowFieldKind::Float,
            Datatype::Float {
                bits: core::num::NonZeroUsize::new(64).expect("nonzero"),
                byte_order: consus_core::ByteOrder::LittleEndian,
            },
        )
        .nullable(false)
        .build()
        .expect("field must build");

        assert_eq!(field.id.get(), 1);
        assert_eq!(field.name, "temperature");
        assert!(field.is_fixed_size());
        assert!(!field.is_nullable());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn nested_field_counts_descendants() {
        let child = ArrowFieldBuilder::new(
            ArrowFieldId::new(2),
            String::from("x"),
            ArrowFieldKind::Int,
            Datatype::Integer {
                bits: core::num::NonZeroUsize::new(32).expect("nonzero"),
                byte_order: consus_core::ByteOrder::LittleEndian,
                signed: true,
            },
        )
        .build()
        .expect("child must build");

        let parent = ArrowFieldBuilder::new(
            ArrowFieldId::new(1),
            String::from("point"),
            ArrowFieldKind::Struct,
            Datatype::Compound {
                #[cfg(feature = "alloc")]
                fields: Vec::new(),
                #[cfg(feature = "alloc")]
                size: 0,
            },
        )
        .children(vec![child])
        .build()
        .expect("parent must build");

        assert_eq!(parent.descendant_count(), 1);
        assert!(parent.is_nested());
    }
}
