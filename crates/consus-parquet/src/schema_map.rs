//! Consus ↔ Parquet schema mapping.
//!
//! ## Type Mapping
//!
//! | Consus Type | Parquet Physical | Parquet Logical |
//! |-------------|-----------------|-----------------|
//! | Boolean | BOOLEAN | - |
//! | Integer{8,signed} | INT32 | INT_8 |
//! | Integer{16,signed} | INT32 | INT_16 |
//! | Integer{32,signed} | INT32 | INT_32 |
//! | Integer{64,signed} | INT64 | INT_64 |
//! | Float{32} | FLOAT | - |
//! | Float{64} | DOUBLE | - |
//! | FixedString{n} | FIXED_LEN_BYTE_ARRAY(n) | STRING |
//! | VariableString | BYTE_ARRAY | STRING |

/// Parquet physical type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetPhysicalType {
    Boolean,
    Int32,
    Int64,
    Int96,
    Float,
    Double,
    ByteArray,
    FixedLenByteArray(usize),
}
