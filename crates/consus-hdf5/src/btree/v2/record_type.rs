/// Testing only (type 0).
pub const TESTING: u8 = 0;
/// Shared object header message index (version 1).
pub const SHARED_MSG_V1: u8 = 1;
/// Shared object header message index (version 2).
pub const SHARED_MSG_V2: u8 = 2;
/// Unsorted, non-filtered, non-paged chunked data (v3 layout B-tree v1 replacement).
pub const CHUNK_NON_FILTERED: u8 = 3;
/// Unsorted, filtered, non-paged chunked data.
pub const CHUNK_FILTERED: u8 = 4;
/// Link name index for dense groups.
pub const LINK_NAME: u8 = 5;
/// Creation order index for dense groups.
pub const LINK_CREATION_ORDER: u8 = 6;
/// Shared header message sorted by reference count.
pub const SHARED_MSG_BY_REFCOUNT: u8 = 7;
/// Attribute name index for dense attribute storage.
pub const ATTRIBUTE_NAME: u8 = 8;
/// Attribute creation order index.
pub const ATTRIBUTE_CREATION_ORDER: u8 = 9;
/// Non-filtered chunked data, non-paged (v4 layout).
pub const CHUNK_V4_NON_FILTERED: u8 = 10;
/// Filtered chunked data, non-paged (v4 layout).
pub const CHUNK_V4_FILTERED: u8 = 11;
/// Fractal heap huge object index (record type 48).
/// Used by dense group/attribute storage to locate huge objects
/// stored outside the managed fractal heap space.
pub const HUGE_OBJECT: u8 = 48;
