#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use super::{array::ZarrVersion, codec::Codec};

/// Zarr group metadata for both v2 and v3.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GroupMetadata {
    /// Zarr format version.
    pub version: ZarrVersion,
    /// Custom attributes (key-value pairs).
    pub attributes: Vec<(String, AttributeValue)>,
    /// For v3: group codec chain.
    pub codecs: Vec<Codec>,
}

#[cfg(feature = "alloc")]
impl Default for GroupMetadata {
    fn default() -> Self {
        Self {
            version: ZarrVersion::V3,
            attributes: Vec::new(),
            codecs: Vec::new(),
        }
    }
}

/// Attribute value for Zarr group attributes.
///
/// Zarr attributes are stored as JSON in `.zattrs`. Values may be scalars
/// or arrays of any JSON-compatible type.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    String(String),
    BoolArray(Vec<bool>),
    IntArray(Vec<i64>),
    UintArray(Vec<u64>),
    FloatArray(Vec<f64>),
    StringArray(Vec<String>),
}

#[cfg(feature = "alloc")]
impl AttributeValue {
    /// Number of elements (1 for scalar, length for array).
    pub fn num_elements(&self) -> usize {
        match self {
            Self::Bool(_) | Self::Int(_) | Self::Uint(_) | Self::Float(_) | Self::String(_) => 1,
            Self::BoolArray(values) => values.len(),
            Self::IntArray(values) => values.len(),
            Self::UintArray(values) => values.len(),
            Self::FloatArray(values) => values.len(),
            Self::StringArray(values) => values.len(),
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_value_num_elements() {
        assert_eq!(AttributeValue::Int(42).num_elements(), 1);
        assert_eq!(
            AttributeValue::IntArray(alloc::vec![1, 2, 3]).num_elements(),
            3
        );
    }
}
