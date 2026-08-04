use consus_core::Datatype;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use crate::field::{ArrowField, ArrowFieldId, ArrowFieldSemantics, kind_from_datatype};

/// Conversion mode controlling how strictly types must match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ConversionMode {
    /// Reject any conversion that may lose information.
    #[default]
    Strict,
    /// Allow widening conversions (e.g., Int32 → Int64).
    AllowWidening,
    /// Allow lossy conversions with explicit acknowledgment.
    AllowLossy,
    /// Convert to the closest representable type.
    BestEffort,
}

/// Result of a datatype conversion analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionCompatibility {
    /// Conversion is exact and reversible.
    Exact,
    /// Conversion is lossless but may change representation.
    Lossless,
    /// Conversion may lose precision or information.
    Lossy,
    /// Conversion is not possible.
    Incompatible,
}

impl ConversionCompatibility {
    /// Returns `true` if the conversion can proceed in the given mode.
    #[must_use]
    pub const fn is_permitted(self, mode: ConversionMode) -> bool {
        match (self, mode) {
            (Self::Exact | Self::Lossless, _) => true,
            (Self::Lossy, ConversionMode::AllowLossy | ConversionMode::BestEffort) => true,
            (Self::Incompatible, ConversionMode::BestEffort) => false,
            _ => false,
        }
    }
}

/// Builder for constructing Arrow fields from Core datatypes.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ArrowFieldFromCoreBuilder {
    name: String,
    datatype: Datatype,
    nullable: bool,
    id: ArrowFieldId,
}

#[cfg(feature = "alloc")]
impl ArrowFieldFromCoreBuilder {
    /// Create a new builder for a field with the given name and Core datatype.
    #[must_use]
    pub fn new(name: String, datatype: Datatype) -> Self {
        Self {
            name,
            datatype,
            nullable: false,
            id: ArrowFieldId::new(0),
        }
    }

    /// Set the field nullability.
    #[must_use]
    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Set the field identifier.
    #[must_use]
    pub fn id(mut self, id: ArrowFieldId) -> Self {
        self.id = id;
        self
    }

    /// Build the Arrow field.
    #[must_use]
    pub fn build(self) -> ArrowField {
        let kind = kind_from_datatype(&self.datatype);

        ArrowField {
            id: self.id,
            name: self.name,
            kind,
            semantics: if self.nullable {
                ArrowFieldSemantics::optional_scalar()
            } else {
                ArrowFieldSemantics::required_scalar()
            },
            datatype: self.datatype,
            children: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_mode_permits_correctly() {
        use ConversionCompatibility::*;

        assert!(Exact.is_permitted(ConversionMode::Strict));
        assert!(Lossless.is_permitted(ConversionMode::Strict));
        assert!(!Lossy.is_permitted(ConversionMode::Strict));
        assert!(Lossy.is_permitted(ConversionMode::AllowLossy));
        assert!(!Incompatible.is_permitted(ConversionMode::AllowLossy));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn field_builder_constructs_valid_field() {
        let field = ArrowFieldFromCoreBuilder::new(
            String::from("temperature"),
            Datatype::Float {
                bits: core::num::NonZeroUsize::new(64).unwrap(),
                byte_order: consus_core::ByteOrder::LittleEndian,
            },
        )
        .nullable(true)
        .id(ArrowFieldId::new(1))
        .build();
        assert_eq!(field.name, "temperature");
        assert!(field.is_nullable());
        assert_eq!(field.id.get(), 1);
    }
}
