//! Dtype string parsing and tagging for netCDF variables.

use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Datatype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Parse a dtype string ("f32", "f64", "i8", "u8", "i16", "u16",
/// "i32", "u32", "i64", "u64") to a little-endian `Datatype`.
pub(super) fn parse_dtype(dtype: &str) -> PyResult<Datatype> {
    match dtype {
        "f32" => Ok(Datatype::Float {
            bits: NonZeroUsize::new(32).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
        }),
        "f64" => Ok(Datatype::Float {
            bits: NonZeroUsize::new(64).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
        }),
        "i8" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(8).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        "u8" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(8).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        "i16" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(16).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        "u16" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(16).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        "i32" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(32).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        "u32" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(32).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        "i64" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(64).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        "u64" => Ok(Datatype::Integer {
            bits: NonZeroUsize::new(64).expect("nonzero literal"),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        other => Err(PyValueError::new_err(format!(
            "unsupported dtype '{other}'; expected one of: \
             f32, f64, i8, u8, i16, u16, i32, u32, i64, u64"
        ))),
    }
}

pub(super) fn dtype_tag(dt: &Datatype) -> &'static str {
    match dt {
        Datatype::Float { bits, .. } if bits.get() == 32 => "f32",
        Datatype::Float { bits, .. } if bits.get() == 64 => "f64",
        Datatype::Integer {
            bits, signed: true, ..
        } if bits.get() == 8 => "i8",
        Datatype::Integer {
            bits,
            signed: false,
            ..
        } if bits.get() == 8 => "u8",
        Datatype::Integer {
            bits, signed: true, ..
        } if bits.get() == 16 => "i16",
        Datatype::Integer {
            bits,
            signed: false,
            ..
        } if bits.get() == 16 => "u16",
        Datatype::Integer {
            bits, signed: true, ..
        } if bits.get() == 32 => "i32",
        Datatype::Integer {
            bits,
            signed: false,
            ..
        } if bits.get() == 32 => "u32",
        Datatype::Integer {
            bits, signed: true, ..
        } if bits.get() == 64 => "i64",
        Datatype::Integer {
            bits,
            signed: false,
            ..
        } if bits.get() == 64 => "u64",
        Datatype::FixedString { .. } => "str",
        _ => "opaque",
    }
}
