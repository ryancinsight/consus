//! Thrift Compact Protocol decoder for Parquet footer and page-header payloads.
//!
//! ## Specification
//!
//! Implements the read-only subset of the Thrift Compact Protocol (Apache Thrift v0.16)
//! required by the Apache Parquet format.
//!
//! Reference: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md
//!
//! ## Protocol invariants
//!
//! - zigzag_decode_i32(n: u32) = (n >> 1) as i32 ^ -((n and 1) as i32)
//! - zigzag_decode_i64(n: u64) = (n >> 1) as i64 ^ -((n and 1) as i64)
//! - Unsigned varint: bytes are 7-bit LSB-first groups; high bit = continuation.
//! - Field header encodes delta from previous field ID in upper nibble and type code
//!   in lower nibble; delta == 0 signals an absolute i16 field ID (zigzag varint) follows.
//! - Lists/sets encode short counts (<= 14) in the upper nibble of the header byte;
//!   long counts use 0xF0 | elem_type followed by a varint count.
//! - Bool fields carry their value in the type nibble (0x01 = true, 0x02 = false);
//!   no additional byte is read for booleans in struct context.
//! - A byte with value 0x00 terminates a struct (field stop).