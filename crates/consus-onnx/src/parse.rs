//! ONNX ModelProto and GraphProto decoding.

use alloc::vec::Vec;
use core::fmt;

use crate::document::{
    ElementType, Initializer, ModelDocument, Node, OperatorSet, TensorInfo, ValueInfo,
};
use crate::wire::{LENGTH_DELIMITED, Reader, VARINT};

/// Resource bounds enforced while parsing an untrusted ONNX document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseLimits {
    /// Maximum complete document size.
    pub document_bytes: usize,
    /// Maximum one length-delimited protobuf field size.
    pub field_bytes: usize,
    /// Maximum graph node count.
    pub nodes: usize,
    /// Maximum input plus output value count.
    pub values: usize,
    /// Maximum initializer count.
    pub initializers: usize,
    /// Maximum inputs plus outputs on one node.
    pub node_names: usize,
    /// Maximum dimensions on one tensor.
    pub dimensions: usize,
    /// Maximum operator-set declarations.
    pub operator_sets: usize,
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            document_bytes: 1 << 30,
            field_bytes: 1 << 29,
            nodes: 1_000_000,
            values: 1_000_000,
            initializers: 1_000_000,
            node_names: 65_536,
            dimensions: 64,
            operator_sets: 256,
        }
    }
}

/// Failure while decoding an ONNX protobuf document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Input ended within a protobuf value.
    Truncated {
        /// Byte offset of the incomplete value.
        offset: usize,
    },
    /// A protobuf varint exceeds 64 bits.
    VarintOverflow {
        /// Byte offset of the varint.
        offset: usize,
    },
    /// A length cannot be represented or added safely.
    LengthOverflow {
        /// Byte offset of the length prefix.
        offset: usize,
    },
    /// A protobuf field key is invalid.
    InvalidFieldKey {
        /// Offset after reading the key.
        offset: usize,
    },
    /// A string field is not UTF-8.
    InvalidUtf8 {
        /// Byte offset of the string field.
        offset: usize,
    },
    /// Deprecated protobuf group wire types are rejected.
    UnsupportedWireType {
        /// Wire type code.
        wire: u8,
        /// Byte offset after the key.
        offset: usize,
    },
    /// A configured resource bound was exceeded.
    LimitExceeded {
        /// Bounded resource.
        resource: &'static str,
        /// Configured maximum.
        limit: usize,
        /// Observed value.
        actual: usize,
    },
    /// ModelProto does not contain its required GraphProto.
    MissingGraph,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { offset } => write!(formatter, "protobuf value truncated at {offset}"),
            Self::VarintOverflow { offset } => write!(formatter, "varint overflows at {offset}"),
            Self::LengthOverflow { offset } => write!(formatter, "length overflows at {offset}"),
            Self::InvalidFieldKey { offset } => write!(formatter, "invalid field key at {offset}"),
            Self::InvalidUtf8 { offset } => write!(formatter, "invalid UTF-8 at {offset}"),
            Self::UnsupportedWireType { wire, offset } => {
                write!(formatter, "unsupported wire type {wire} at {offset}")
            }
            Self::LimitExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "{resource} limit exceeded: maximum {limit}, observed {actual}"
            ),
            Self::MissingGraph => formatter.write_str("ModelProto is missing GraphProto field 7"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

/// Parse one ONNX `ModelProto` while borrowing names and raw tensor data.
pub fn parse_model(bytes: &[u8], limits: ParseLimits) -> Result<ModelDocument<'_>, ParseError> {
    ensure_limit("document bytes", limits.document_bytes, bytes.len())?;
    let mut reader = Reader::new(bytes, limits.field_bytes);
    let mut document = ModelDocument {
        ir_version: 0,
        producer_name: "",
        producer_version: "",
        domain: "",
        model_version: 0,
        graph_name: "",
        inputs: Vec::new(),
        outputs: Vec::new(),
        nodes: Vec::new(),
        initializers: Vec::new(),
        operator_sets: Vec::new(),
    };
    let mut graph_seen = false;
    while let Some((field, wire)) = reader.next()? {
        match (field, wire) {
            (1, VARINT) => document.ir_version = reader.varint()? as i64,
            (2, LENGTH_DELIMITED) => document.producer_name = reader.string()?,
            (3, LENGTH_DELIMITED) => document.producer_version = reader.string()?,
            (4, LENGTH_DELIMITED) => document.domain = reader.string()?,
            (5, VARINT) => document.model_version = reader.varint()? as i64,
            (7, LENGTH_DELIMITED) => {
                parse_graph(reader.message()?, limits, &mut document)?;
                graph_seen = true;
            }
            (8, LENGTH_DELIMITED) => {
                ensure_push(
                    "operator sets",
                    limits.operator_sets,
                    document.operator_sets.len(),
                )?;
                document
                    .operator_sets
                    .push(parse_operator_set(reader.message()?)?);
            }
            _ => reader.skip(wire)?,
        }
    }
    if !graph_seen {
        return Err(ParseError::MissingGraph);
    }
    Ok(document)
}

fn parse_graph<'a>(
    mut reader: Reader<'a>,
    limits: ParseLimits,
    document: &mut ModelDocument<'a>,
) -> Result<(), ParseError> {
    while let Some((field, wire)) = reader.next()? {
        match (field, wire) {
            (1, LENGTH_DELIMITED) => {
                ensure_push("nodes", limits.nodes, document.nodes.len())?;
                document.nodes.push(parse_node(reader.message()?, limits)?);
            }
            (2, LENGTH_DELIMITED) => document.graph_name = reader.string()?,
            (5, LENGTH_DELIMITED) => {
                ensure_push(
                    "initializers",
                    limits.initializers,
                    document.initializers.len(),
                )?;
                document
                    .initializers
                    .push(parse_initializer(reader.message()?, limits)?);
            }
            (11, LENGTH_DELIMITED) => {
                ensure_push(
                    "values",
                    limits.values,
                    document.inputs.len() + document.outputs.len(),
                )?;
                document
                    .inputs
                    .push(parse_value(reader.message()?, limits)?);
            }
            (12, LENGTH_DELIMITED) => {
                ensure_push(
                    "values",
                    limits.values,
                    document.inputs.len() + document.outputs.len(),
                )?;
                document
                    .outputs
                    .push(parse_value(reader.message()?, limits)?);
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(())
}

fn parse_node<'a>(mut reader: Reader<'a>, limits: ParseLimits) -> Result<Node<'a>, ParseError> {
    let mut node = Node {
        name: "",
        operation: "",
        domain: "",
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    while let Some((field, wire)) = reader.next()? {
        match (field, wire) {
            (1, LENGTH_DELIMITED) => {
                ensure_push(
                    "node names",
                    limits.node_names,
                    node.inputs.len() + node.outputs.len(),
                )?;
                node.inputs.push(reader.string()?);
            }
            (2, LENGTH_DELIMITED) => {
                ensure_push(
                    "node names",
                    limits.node_names,
                    node.inputs.len() + node.outputs.len(),
                )?;
                node.outputs.push(reader.string()?);
            }
            (3, LENGTH_DELIMITED) => node.name = reader.string()?,
            (4, LENGTH_DELIMITED) => node.operation = reader.string()?,
            (7, LENGTH_DELIMITED) => node.domain = reader.string()?,
            _ => reader.skip(wire)?,
        }
    }
    Ok(node)
}

fn parse_initializer<'a>(
    mut reader: Reader<'a>,
    limits: ParseLimits,
) -> Result<Initializer<'a>, ParseError> {
    let mut initializer = Initializer {
        name: "",
        element_type: ElementType::Unknown(0),
        dimensions: Vec::new(),
        raw_data: &[],
    };
    while let Some((field, wire)) = reader.next()? {
        match (field, wire) {
            (1, VARINT) => {
                ensure_push(
                    "dimensions",
                    limits.dimensions,
                    initializer.dimensions.len(),
                )?;
                initializer.dimensions.push(reader.varint()?);
            }
            (1, LENGTH_DELIMITED) => {
                let mut packed = reader.message()?;
                while !packed.is_empty() {
                    ensure_push(
                        "dimensions",
                        limits.dimensions,
                        initializer.dimensions.len(),
                    )?;
                    initializer.dimensions.push(packed.varint()?);
                }
            }
            (2, VARINT) => initializer.element_type = ElementType::from(reader.varint()? as i32),
            (8, LENGTH_DELIMITED) => initializer.name = reader.string()?,
            (9, LENGTH_DELIMITED) => initializer.raw_data = reader.bytes()?,
            _ => reader.skip(wire)?,
        }
    }
    Ok(initializer)
}

fn parse_value<'a>(
    mut reader: Reader<'a>,
    limits: ParseLimits,
) -> Result<ValueInfo<'a>, ParseError> {
    let mut value = ValueInfo {
        name: "",
        tensor: None,
    };
    while let Some((field, wire)) = reader.next()? {
        match (field, wire) {
            (1, LENGTH_DELIMITED) => value.name = reader.string()?,
            (2, LENGTH_DELIMITED) => value.tensor = parse_type(reader.message()?, limits)?,
            _ => reader.skip(wire)?,
        }
    }
    Ok(value)
}

fn parse_type(
    mut reader: Reader<'_>,
    limits: ParseLimits,
) -> Result<Option<TensorInfo>, ParseError> {
    while let Some((field, wire)) = reader.next()? {
        if (field, wire) == (1, LENGTH_DELIMITED) {
            return Ok(Some(parse_tensor_type(reader.message()?, limits)?));
        }
        reader.skip(wire)?;
    }
    Ok(None)
}

fn parse_tensor_type(
    mut reader: Reader<'_>,
    limits: ParseLimits,
) -> Result<TensorInfo, ParseError> {
    let mut tensor = TensorInfo {
        element_type: ElementType::Unknown(0),
        dimensions: Vec::new(),
    };
    while let Some((field, wire)) = reader.next()? {
        match (field, wire) {
            (1, VARINT) => tensor.element_type = ElementType::from(reader.varint()? as i32),
            (2, LENGTH_DELIMITED) => {
                tensor.dimensions = parse_shape(reader.message()?, limits)?;
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(tensor)
}

fn parse_shape(
    mut reader: Reader<'_>,
    limits: ParseLimits,
) -> Result<Vec<Option<u64>>, ParseError> {
    let mut dimensions = Vec::new();
    while let Some((field, wire)) = reader.next()? {
        if (field, wire) == (1, LENGTH_DELIMITED) {
            ensure_push("dimensions", limits.dimensions, dimensions.len())?;
            dimensions.push(parse_dimension(reader.message()?)?);
        } else {
            reader.skip(wire)?;
        }
    }
    Ok(dimensions)
}

fn parse_dimension(mut reader: Reader<'_>) -> Result<Option<u64>, ParseError> {
    let mut value = None;
    while let Some((field, wire)) = reader.next()? {
        match (field, wire) {
            (1, VARINT) => value = Some(reader.varint()?),
            _ => reader.skip(wire)?,
        }
    }
    Ok(value)
}

fn parse_operator_set(mut reader: Reader<'_>) -> Result<OperatorSet<'_>, ParseError> {
    let mut operator_set = OperatorSet {
        domain: "",
        version: 0,
    };
    while let Some((field, wire)) = reader.next()? {
        match (field, wire) {
            (1, LENGTH_DELIMITED) => operator_set.domain = reader.string()?,
            (2, VARINT) => operator_set.version = reader.varint()? as i64,
            _ => reader.skip(wire)?,
        }
    }
    Ok(operator_set)
}

fn ensure_push(resource: &'static str, limit: usize, current: usize) -> Result<(), ParseError> {
    ensure_limit(resource, limit, current.saturating_add(1))
}

fn ensure_limit(resource: &'static str, limit: usize, actual: usize) -> Result<(), ParseError> {
    if actual > limit {
        Err(ParseError::LimitExceeded {
            resource,
            limit,
            actual,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn varint(mut value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }

    fn key(field: u32, wire: u8) -> Vec<u8> {
        varint(u64::from(field) << 3 | u64::from(wire))
    }

    fn integer(field: u32, value: u64) -> Vec<u8> {
        let mut encoded = key(field, VARINT);
        encoded.extend(varint(value));
        encoded
    }

    fn bytes(field: u32, value: &[u8]) -> Vec<u8> {
        let mut encoded = key(field, LENGTH_DELIMITED);
        encoded.extend(varint(value.len() as u64));
        encoded.extend_from_slice(value);
        encoded
    }

    fn value_info(name: &str, dimensions: &[u64]) -> Vec<u8> {
        let mut shape = Vec::new();
        for &dimension in dimensions {
            shape.extend(bytes(1, &integer(1, dimension)));
        }
        let mut tensor = integer(1, 1);
        tensor.extend(bytes(2, &shape));
        let ty = bytes(1, &tensor);
        let mut value = bytes(1, name.as_bytes());
        value.extend(bytes(2, &ty));
        value
    }

    fn document() -> Vec<u8> {
        let mut node = bytes(1, b"x");
        node.extend(bytes(1, b"w"));
        node.extend(bytes(2, b"y"));
        node.extend(bytes(3, b"add"));
        node.extend(bytes(4, b"Add"));

        let raw = [0_u8, 0, 128, 63];
        let mut initializer = integer(1, 1);
        initializer.extend(integer(2, 1));
        initializer.extend(bytes(8, b"w"));
        initializer.extend(bytes(9, &raw));

        let mut graph = bytes(1, &node);
        graph.extend(bytes(2, b"registration"));
        graph.extend(bytes(5, &initializer));
        graph.extend(bytes(11, &value_info("x", &[1])));
        graph.extend(bytes(12, &value_info("y", &[1])));

        let mut operator_set = bytes(1, b"");
        operator_set.extend(integer(2, 17));

        let mut model = integer(1, 9);
        model.extend(bytes(2, b"consus-test"));
        model.extend(bytes(7, &graph));
        model.extend(bytes(8, &operator_set));
        model
    }

    #[test]
    fn parses_graph_metadata_and_borrows_payload() {
        let bytes = document();
        let parsed = parse_model(&bytes, ParseLimits::default()).expect("fixture is valid");

        assert_eq!(parsed.ir_version, 9);
        assert_eq!(parsed.producer_name, "consus-test");
        assert_eq!(parsed.graph_name, "registration");
        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.nodes[0].operation, "Add");
        assert_eq!(parsed.nodes[0].inputs, vec!["x", "w"]);
        assert_eq!(
            parsed.inputs[0].tensor.as_ref().unwrap().dimensions,
            vec![Some(1)]
        );
        assert_eq!(parsed.initializers[0].raw_data, &[0, 0, 128, 63]);
        let source = bytes.as_ptr_range();
        let payload = parsed.initializers[0].raw_data.as_ptr();
        assert!(payload >= source.start && payload < source.end);
        assert_eq!(parsed.operator_sets[0].version, 17);
    }

    #[test]
    fn rejects_document_and_node_bounds() {
        let bytes = document();
        let limits = ParseLimits {
            document_bytes: bytes.len() - 1,
            ..ParseLimits::default()
        };
        assert!(matches!(
            parse_model(&bytes, limits),
            Err(ParseError::LimitExceeded {
                resource: "document bytes",
                ..
            })
        ));

        let limits = ParseLimits {
            nodes: 0,
            ..ParseLimits::default()
        };
        assert!(matches!(
            parse_model(&bytes, limits),
            Err(ParseError::LimitExceeded {
                resource: "nodes",
                ..
            })
        ));
    }

    #[test]
    fn rejects_truncated_and_missing_graph_documents() {
        let mut bytes = document();
        bytes.pop();
        assert!(matches!(
            parse_model(&bytes, ParseLimits::default()),
            Err(ParseError::Truncated { .. })
        ));
        assert_eq!(
            parse_model(&integer(1, 9), ParseLimits::default()),
            Err(ParseError::MissingGraph)
        );
    }
}
