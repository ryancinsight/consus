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

/// HDF5 header message type identifiers.
///
/// From the HDF5 File Format Specification, Section IV.A.2.
pub mod message_types {
    /// Dataspace message.
    pub const DATASPACE: u16 = 0x0001;
    /// Link info message.
    pub const LINK_INFO: u16 = 0x0002;
    /// Datatype message.
    pub const DATATYPE: u16 = 0x0003;
    /// Fill value (old) message.
    pub const FILL_VALUE_OLD: u16 = 0x0004;
    /// Fill value message.
    pub const FILL_VALUE: u16 = 0x0005;
    /// Link message.
    pub const LINK: u16 = 0x0006;
    /// External data files message.
    pub const EXTERNAL_FILES: u16 = 0x0007;
    /// Data layout message.
    pub const DATA_LAYOUT: u16 = 0x0008;
    /// Bogus message (testing only).
    pub const BOGUS: u16 = 0x0009;
    /// Group info message.
    pub const GROUP_INFO: u16 = 0x000A;
    /// Filter pipeline message.
    pub const FILTER_PIPELINE: u16 = 0x000B;
    /// Attribute message.
    pub const ATTRIBUTE: u16 = 0x000C;
    /// Object comment message.
    pub const COMMENT: u16 = 0x000D;
    /// Object modification time (old) message.
    pub const MODIFICATION_TIME_OLD: u16 = 0x000E;
    /// Shared message table message.
    pub const SHARED_MSG_TABLE: u16 = 0x000F;
    /// Object header continuation message.
    pub const CONTINUATION: u16 = 0x0010;
    /// Symbol table message.
    pub const SYMBOL_TABLE: u16 = 0x0011;
    /// Object modification time message.
    pub const MODIFICATION_TIME: u16 = 0x0012;
    /// B-tree 'K' values message.
    pub const BTREE_K: u16 = 0x0013;
    /// Driver info message.
    pub const DRIVER_INFO: u16 = 0x0014;
    /// Attribute info message.
    pub const ATTRIBUTE_INFO: u16 = 0x0015;
    /// Object reference count message.
    pub const REFERENCE_COUNT: u16 = 0x0016;
}
