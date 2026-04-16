//! HDF5 header message type identifiers.
//!
//! From the HDF5 File Format Specification, Section IV.A.2.

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
