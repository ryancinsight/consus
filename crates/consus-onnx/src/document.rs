//! Borrowed ONNX document vocabulary.

use alloc::vec::Vec;

/// ONNX tensor element type code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ElementType {
    /// IEEE binary32.
    Float,
    /// Unsigned 8-bit integer.
    Uint8,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 16-bit integer.
    Uint16,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// UTF-8 string.
    String,
    /// Boolean.
    Bool,
    /// IEEE binary16.
    Float16,
    /// IEEE binary64.
    Double,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Unsigned 64-bit integer.
    Uint64,
    /// Complex binary64.
    Complex64,
    /// Complex binary128.
    Complex128,
    /// Brain floating point 16.
    Bfloat16,
    /// Unknown extension code retained from the document.
    Unknown(i32),
}

impl From<i32> for ElementType {
    fn from(value: i32) -> Self {
        match value {
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            4 => Self::Uint16,
            5 => Self::Int16,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::Float16,
            11 => Self::Double,
            12 => Self::Uint32,
            13 => Self::Uint64,
            14 => Self::Complex64,
            15 => Self::Complex128,
            16 => Self::Bfloat16,
            code => Self::Unknown(code),
        }
    }
}

/// Tensor type and static dimensions exposed by ONNX metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Element representation.
    pub element_type: ElementType,
    /// Static dimensions. Symbolic or absent dimensions are represented by `None`.
    pub dimensions: Vec<Option<u64>>,
}

/// Named graph input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueInfo<'a> {
    /// Borrowed ONNX value name.
    pub name: &'a str,
    /// Optional tensor metadata.
    pub tensor: Option<TensorInfo>,
}

/// Graph computation node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<'a> {
    /// Borrowed node name.
    pub name: &'a str,
    /// Borrowed operator name.
    pub operation: &'a str,
    /// Borrowed operator domain.
    pub domain: &'a str,
    /// Borrowed input names in declared order.
    pub inputs: Vec<&'a str>,
    /// Borrowed output names in declared order.
    pub outputs: Vec<&'a str>,
}

/// Static tensor initializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Initializer<'a> {
    /// Borrowed tensor name.
    pub name: &'a str,
    /// Tensor element representation.
    pub element_type: ElementType,
    /// Static dimensions.
    pub dimensions: Vec<u64>,
    /// Borrowed raw little-endian payload, when encoded in `raw_data`.
    pub raw_data: &'a [u8],
}

/// Imported operator-set declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatorSet<'a> {
    /// Borrowed operator domain; empty denotes the standard ONNX domain.
    pub domain: &'a str,
    /// Operator-set version.
    pub version: i64,
}

/// Borrowed ONNX model document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDocument<'a> {
    /// ONNX IR version.
    pub ir_version: i64,
    /// Producer name.
    pub producer_name: &'a str,
    /// Producer version.
    pub producer_version: &'a str,
    /// Model domain.
    pub domain: &'a str,
    /// Model version.
    pub model_version: i64,
    /// Graph name.
    pub graph_name: &'a str,
    /// Graph inputs excluding no initializer metadata.
    pub inputs: Vec<ValueInfo<'a>>,
    /// Graph outputs.
    pub outputs: Vec<ValueInfo<'a>>,
    /// Topologically declared graph nodes.
    pub nodes: Vec<Node<'a>>,
    /// Static tensor initializers.
    pub initializers: Vec<Initializer<'a>>,
    /// Imported operator sets.
    pub operator_sets: Vec<OperatorSet<'a>>,
}
