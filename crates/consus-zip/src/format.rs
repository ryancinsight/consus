//! ZIP record layout shared by the reader and the writer.
//!
//! Field order and widths follow PKWARE APPNOTE.TXT 6.3.10: local file
//! header (§4.3.7), central directory header (§4.3.12), end of central
//! directory record (§4.3.16), ZIP64 end record (§4.3.14), its locator
//! (§4.3.15), and the ZIP64 extended information extra field (§4.5.3).
//! Every multi-byte field is little-endian.

use consus_core::{ByteOrder, EndianScalar, read_integer, write_integer};

use crate::error::{Error, Result};

pub(crate) const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
pub(crate) const CENTRAL_HEADER_SIGNATURE: u32 = 0x0201_4b50;
pub(crate) const END_SIGNATURE: u32 = 0x0605_4b50;
pub(crate) const ZIP64_END_SIGNATURE: u32 = 0x0606_4b50;
pub(crate) const ZIP64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;

/// Fixed part of a local file header.
pub(crate) const LOCAL_HEADER_LEN: usize = 30;
/// Fixed part of a central directory header; also the minimum entry length.
pub(crate) const CENTRAL_HEADER_LEN: usize = 46;
/// End of central directory record without its comment.
pub(crate) const END_LEN: usize = 22;
/// ZIP64 end record without its extensible data sector.
pub(crate) const ZIP64_END_LEN: usize = 56;
/// "Size of zip64 end of central directory record": the record length
/// minus its leading 12 bytes (signature and this size field).
pub(crate) const ZIP64_END_RECORD_SIZE: u64 = 44;
pub(crate) const ZIP64_LOCATOR_LEN: usize = 20;
/// Largest end-record comment, bounded by its 16-bit length field.
pub(crate) const MAX_COMMENT_LEN: usize = 0xFFFF;

pub(crate) const ZIP64_EXTRA_ID: u16 = 0x0001;
/// Sentinel a 32-bit size or offset holds when ZIP64 carries the value.
pub(crate) const SENTINEL_32: u32 = u32::MAX;
/// Sentinel a 16-bit entry count holds when ZIP64 carries the value.
pub(crate) const SENTINEL_16: u16 = u16::MAX;

/// General-purpose flag bit 0: the member is encrypted.
pub(crate) const FLAG_ENCRYPTED: u16 = 1;
/// General-purpose flag bit 11: name and comment are UTF-8.
pub(crate) const FLAG_UTF8: u16 = 1 << 11;

/// Version needed to extract a DEFLATE member (APPNOTE §4.4.3.2).
pub(crate) const VERSION_DEFLATE: u16 = 20;
/// Version needed to extract a ZIP64 member (APPNOTE §4.4.3.2).
pub(crate) const VERSION_ZIP64: u16 = 45;

/// MS-DOS date for 1980-01-01, the format's earliest date: day 1 in bits
/// 0-4, month 1 in bits 5-8, year offset 0 in bits 9-15. Written members
/// carry this fixed stamp so equal inputs produce equal archives.
pub(crate) const DOS_DATE_EPOCH: u16 = (1 << 5) | 1;
pub(crate) const DOS_TIME_MIDNIGHT: u16 = 0;

/// Largest DEFLATE expansion ratio, uncompressed bytes per compressed byte.
///
/// The longest match DEFLATE encodes is 258 bytes, and a dynamic Huffman
/// block can code its length and distance symbols in one bit each, so two
/// input bits yield at most 258 output bytes: 8 / 2 × 258 = 1032. Block
/// headers only lower the ratio. Source: zlib technical details,
/// <https://zlib.net/zlib_tech.html>, "Maximum Expansion Factor".
pub(crate) const MAX_DEFLATE_RATIO: u64 = 1032;

/// Member compression method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// Method 0: bytes stored verbatim.
    Stored,
    /// Method 8: raw DEFLATE (RFC 1951).
    Deflated,
}

impl Method {
    /// The method's APPNOTE §4.4.5 code.
    pub const fn code(self) -> u16 {
        match self {
            Self::Stored => 0,
            Self::Deflated => 8,
        }
    }

    /// The method for an APPNOTE §4.4.5 code, or `None` when unsupported.
    pub const fn from_code(code: u16) -> Option<Self> {
        match code {
            0 => Some(Self::Stored),
            8 => Some(Self::Deflated),
            _ => None,
        }
    }
}

/// Bounded little-endian cursor over one in-memory record.
pub(crate) struct Fields<'a> {
    bytes: &'a [u8],
    structure: &'static str,
}

impl<'a> Fields<'a> {
    pub(crate) fn new(bytes: &'a [u8], structure: &'static str) -> Self {
        Self { bytes, structure }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let (head, tail) = self.bytes.split_at_checked(len).ok_or(Error::Truncated {
            structure: self.structure,
        })?;
        self.bytes = tail;
        Ok(head)
    }

    /// Reads one little-endian integer field.
    pub(crate) fn read<T: EndianScalar>(&mut self) -> Result<T> {
        let bytes = self.take(T::BYTE_WIDTH)?;
        read_integer(bytes, ByteOrder::LittleEndian).ok_or(Error::Truncated {
            structure: self.structure,
        })
    }

    /// Reads a signature word and rejects any other value.
    pub(crate) fn signature(&mut self, expected: u32) -> Result<()> {
        let found = self.read::<u32>()?;
        if found == expected {
            Ok(())
        } else {
            Err(Error::BadSignature {
                structure: self.structure,
                found,
            })
        }
    }
}

/// Converts an archive offset or length to a buffer index.
pub(crate) fn to_index(value: u64, structure: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::TooLarge { structure })
}

/// Converts a buffer length to an archive offset or length.
pub(crate) fn to_offset(value: usize, structure: &'static str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::TooLarge { structure })
}

/// The 32-bit field value, or the sentinel when ZIP64 must carry it.
pub(crate) fn narrow_32(value: u64) -> u32 {
    u32::try_from(value)
        .ok()
        .filter(|&narrow| narrow != SENTINEL_32)
        .unwrap_or(SENTINEL_32)
}

/// The 16-bit count value, or the sentinel when ZIP64 must carry it.
pub(crate) fn narrow_16(value: u64) -> u16 {
    u16::try_from(value)
        .ok()
        .filter(|&narrow| narrow != SENTINEL_16)
        .unwrap_or(SENTINEL_16)
}

/// Little-endian record builder.
#[derive(Default)]
pub(crate) struct Record(Vec<u8>);

impl Record {
    /// Appends one little-endian integer field.
    pub(crate) fn put<T: EndianScalar>(&mut self, value: T) -> &mut Self {
        let start = self.0.len();
        self.0.resize(start + T::BYTE_WIDTH, 0);
        write_integer(&mut self.0[start..], value, ByteOrder::LittleEndian)
            .expect("invariant: the record was just extended by the field width");
        self
    }

    pub(crate) fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.0.extend_from_slice(value);
        self
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_codes_round_trip_and_reject_others() {
        for method in [Method::Stored, Method::Deflated] {
            assert_eq!(Method::from_code(method.code()), Some(method));
        }
        assert_eq!(Method::from_code(12), None);
        assert_eq!(Method::from_code(99), None);
    }

    #[test]
    fn narrowing_uses_the_sentinel_at_and_above_the_field_maximum() {
        assert_eq!(narrow_32(0), 0);
        assert_eq!(narrow_32(u64::from(u32::MAX) - 1), u32::MAX - 1);
        assert_eq!(narrow_32(u64::from(u32::MAX)), SENTINEL_32);
        assert_eq!(narrow_32(u64::MAX), SENTINEL_32);
        assert_eq!(narrow_16(65_534), 65_534);
        assert_eq!(narrow_16(65_535), SENTINEL_16);
        assert_eq!(narrow_16(1 << 20), SENTINEL_16);
    }

    #[test]
    fn fields_reject_reads_past_the_record() {
        let mut fields = Fields::new(&[1, 0, 2], "test record");
        assert_eq!(fields.read::<u16>().unwrap(), 1);
        assert!(matches!(
            fields.read::<u16>(),
            Err(Error::Truncated {
                structure: "test record"
            })
        ));
    }

    #[test]
    fn dos_epoch_decodes_to_1980_01_01() {
        assert_eq!(DOS_DATE_EPOCH & 0x1F, 1);
        assert_eq!((DOS_DATE_EPOCH >> 5) & 0x0F, 1);
        assert_eq!(DOS_DATE_EPOCH >> 9, 0);
    }
}
