use std::num::NonZeroUsize;

use consus_core::{
    types::{ByteOrder, StringEncoding},
    Datatype,
};
use pyo3::{exceptions::PyValueError, PyErr};

/// Parse a dtype string into a `consus_core::Datatype`.
///
/// Accepted formats (case-sensitive):
/// - NumPy-style with endian prefix: `"<i4"`, `">f8"`, `"|b1"`, `">u2"`, etc.
/// - Bare names: `"int8"` … `"int64"`, `"uint8"` … `"uint64"`,
///   `"float16"` … `"float64"`, `"bool"`, `"bool_"`
/// - Fixed-length string: `"S<n>"` e.g. `"S10"`
///
/// All integer/float types without an explicit endian prefix default to
/// little-endian byte order.
pub fn parse_dtype(s: &str) -> Result<Datatype, PyErr> {
    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("dtype bit width must be non-zero")
    }

    let (byte_order, rest) = match s.as_bytes().first() {
        Some(b'<') | Some(b'|') => (ByteOrder::LittleEndian, &s[1..]),
        Some(b'>') => (ByteOrder::BigEndian, &s[1..]),
        _ => (ByteOrder::LittleEndian, s),
    };

    match rest {
        "b1" | "bool" | "bool_" => Ok(Datatype::Boolean),

        "i1" | "int8" => Ok(Datatype::Integer { bits: nz(8), byte_order, signed: true }),
        "i2" | "int16" => Ok(Datatype::Integer { bits: nz(16), byte_order, signed: true }),
        "i4" | "int32" => Ok(Datatype::Integer { bits: nz(32), byte_order, signed: true }),
        "i8" | "int64" => Ok(Datatype::Integer { bits: nz(64), byte_order, signed: true }),

        "u1" | "uint8" => Ok(Datatype::Integer { bits: nz(8), byte_order, signed: false }),
        "u2" | "uint16" => Ok(Datatype::Integer { bits: nz(16), byte_order, signed: false }),
        "u4" | "uint32" => Ok(Datatype::Integer { bits: nz(32), byte_order, signed: false }),
        "u8" | "uint64" => Ok(Datatype::Integer { bits: nz(64), byte_order, signed: false }),

        "f2" | "float16" => Ok(Datatype::Float { bits: nz(16), byte_order }),
        "f4" | "float32" => Ok(Datatype::Float { bits: nz(32), byte_order }),
        "f8" | "float64" => Ok(Datatype::Float { bits: nz(64), byte_order }),

        other if other.starts_with('S') => {
            let len_str = &other[1..];
            let length = len_str.parse::<usize>().map_err(|_| {
                PyValueError::new_err(format!("invalid fixed-string dtype {s:?}: expected S<n>"))
            })?;
            Ok(Datatype::FixedString { length, encoding: StringEncoding::Ascii })
        }

        _ => Err(PyValueError::new_err(format!(
            "unsupported dtype {s:?}; accepted: <i1..i8 >i1..i8 <u1..u8 >u1..u8 <f2..f8 >f2..f8 bool S<n>"
        ))),
    }
}

/// Format a `consus_core::Datatype` as a compact dtype string.
///
/// Output examples: `"<int32"`, `"<float64"`, `">int16"`, `"bool"`, `"S10"`,
/// `"vlen_str"`, `"compound"`, `"reference"`.
pub fn dtype_to_str(dt: &Datatype) -> String {
    match dt {
        Datatype::Boolean => "bool".to_owned(),

        Datatype::Integer { bits, byte_order, signed } => {
            let end = endian_prefix(*byte_order);
            let kind = if *signed { "int" } else { "uint" };
            format!("{end}{kind}{}", bits.get())
        }

        Datatype::Float { bits, byte_order } => {
            let end = endian_prefix(*byte_order);
            format!("{end}float{}", bits.get())
        }

        Datatype::Complex { component_bits, byte_order } => {
            let end = endian_prefix(*byte_order);
            format!("{end}complex{}", component_bits.get() * 2)
        }

        Datatype::FixedString { length, .. } => format!("S{length}"),

        Datatype::VariableString { .. } => "vlen_str".to_owned(),

        Datatype::Opaque { size, .. } => format!("opaque{size}"),

        Datatype::Compound { .. } => "compound".to_owned(),

        Datatype::Array { base, dims } => {
            format!("array[{}]{}", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","), dtype_to_str(base))
        }

        Datatype::Enum { base, .. } => format!("enum({})", dtype_to_str(base)),

        Datatype::VarLen { base } => format!("vlen({})", dtype_to_str(base)),

        Datatype::Reference(_) => "reference".to_owned(),
    }
}

fn endian_prefix(byte_order: ByteOrder) -> &'static str {
    match byte_order {
        ByteOrder::BigEndian => ">",
        ByteOrder::LittleEndian => "<",
    }
}
