use consus_core::ByteOrder;

/// Interpret the first 2 bytes of `c` as a u16 with the given byte order.
pub(super) fn read_u16(c: &[u8], bo: ByteOrder) -> u16 {
    let arr = [c[0], c[1]];
    match bo {
        ByteOrder::LittleEndian => u16::from_le_bytes(arr),
        ByteOrder::BigEndian => u16::from_be_bytes(arr),
    }
}

/// Interpret the first 2 bytes of `c` as an i16 with the given byte order.
pub(super) fn read_i16(c: &[u8], bo: ByteOrder) -> i16 {
    let arr = [c[0], c[1]];
    match bo {
        ByteOrder::LittleEndian => i16::from_le_bytes(arr),
        ByteOrder::BigEndian => i16::from_be_bytes(arr),
    }
}

/// Interpret the first 4 bytes of `c` as a u32 with the given byte order.
pub(super) fn read_u32(c: &[u8], bo: ByteOrder) -> u32 {
    let arr = [c[0], c[1], c[2], c[3]];
    match bo {
        ByteOrder::LittleEndian => u32::from_le_bytes(arr),
        ByteOrder::BigEndian => u32::from_be_bytes(arr),
    }
}

/// Interpret the first 4 bytes of `c` as an i32 with the given byte order.
pub(super) fn read_i32(c: &[u8], bo: ByteOrder) -> i32 {
    let arr = [c[0], c[1], c[2], c[3]];
    match bo {
        ByteOrder::LittleEndian => i32::from_le_bytes(arr),
        ByteOrder::BigEndian => i32::from_be_bytes(arr),
    }
}

/// Interpret the first 8 bytes of `c` as a u64 with the given byte order.
pub(super) fn read_u64(c: &[u8], bo: ByteOrder) -> u64 {
    let arr = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
    match bo {
        ByteOrder::LittleEndian => u64::from_le_bytes(arr),
        ByteOrder::BigEndian => u64::from_be_bytes(arr),
    }
}

/// Interpret the first 8 bytes of `c` as an i64 with the given byte order.
pub(super) fn read_i64(c: &[u8], bo: ByteOrder) -> i64 {
    let arr = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
    match bo {
        ByteOrder::LittleEndian => i64::from_le_bytes(arr),
        ByteOrder::BigEndian => i64::from_be_bytes(arr),
    }
}
