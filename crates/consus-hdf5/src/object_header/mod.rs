//! Object header parsing.
//!
//! ## Specification
//!
//! Every HDF5 object (group or dataset) has an object header containing
//! header messages that describe the object's properties.
//!
//! ### Version 1 Object Header
//!
//! | Field | Size | Description |
//! |-------|------|-------------|
//! | Version | 1 | Always 1 |
//! | Reserved | 1 | |
//! | Number of messages | 2 | |
//! | Object reference count | 4 | |
//! | Object header size | 4 | Total size of header messages |
//!
//! ### Version 2 Object Header (signature "OHDR")
//!
//! More compact, with flags, timestamps, and a chunk-based structure.

pub mod message_types;
#[cfg(feature = "alloc")]
pub(crate) mod v1;
#[cfg(feature = "alloc")]
pub(crate) mod v2;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Object header version 2 signature.
pub const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

/// Object header continuation signature.
pub const OCHK_SIGNATURE: [u8; 4] = *b"OCHK";

/// A parsed object header.
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Header version (1 or 2).
    pub version: u8,
    /// Header messages contained in this object.
    #[cfg(feature = "alloc")]
    pub messages: Vec<HeaderMessage>,
}

/// A single header message within an object header.
///
/// Message types define dataset properties (dataspace, datatype, layout,
/// fill value, filter pipeline, attributes, etc.).
#[derive(Debug, Clone)]
pub struct HeaderMessage {
    /// Message type ID.
    ///
    /// Key types:
    /// - 0x0001: Dataspace
    /// - 0x0003: Datatype
    /// - 0x0008: Data layout
    /// - 0x000B: Filter pipeline
    /// - 0x000C: Attribute
    /// - 0x0010: Object header continuation
    /// - 0x0011: Symbol table (groups, v1)
    /// - 0x0012: Object modification time
    pub message_type: u16,

    /// Size of the message data in bytes.
    pub data_size: u16,

    /// Flags (bit 0: constant, bit 1: shared, etc.).
    pub flags: u8,

    /// Raw message data bytes.
    #[cfg(feature = "alloc")]
    pub data: Vec<u8>,
}
