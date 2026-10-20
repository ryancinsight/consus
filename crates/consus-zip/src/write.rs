//! Streaming ZIP container writer.
//!
//! Each member is encoded in memory, so its sizes and CRC-32 are known before
//! its local header is written. The writer needs no `Seek` and never emits a
//! data descriptor. ZIP64 fields appear only when a size, offset or entry
//! count reaches its 32- or 16-bit sentinel, so small archives keep the
//! classic layout every reader accepts.

use std::collections::HashSet;
use std::io::Write;

use consus_compression::codec::raw_deflate::RawDeflateCodec;
use consus_compression::{Codec, CompressionLevel, Crc32};

use crate::error::{Error, Result};
use crate::format::{
    CENTRAL_HEADER_SIGNATURE, DOS_DATE_EPOCH, DOS_TIME_MIDNIGHT, END_SIGNATURE, FLAG_UTF8,
    LOCAL_HEADER_SIGNATURE, Method, Record, SENTINEL_16, SENTINEL_32, VERSION_DEFLATE,
    VERSION_ZIP64, ZIP64_END_RECORD_SIZE, ZIP64_END_SIGNATURE, ZIP64_EXTRA_ID,
    ZIP64_LOCATOR_SIGNATURE, narrow_16, narrow_32, to_offset,
};

/// Directory facts retained for one written member.
struct Written {
    name: String,
    flags: u16,
    method: Method,
    crc32: u32,
    compressed_size: u64,
    uncompressed_size: u64,
    local_header_offset: u64,
}

impl Written {
    /// ZIP64 values for the fields whose 32-bit form is the sentinel, in
    /// APPNOTE §4.5.3 order.
    fn zip64_values(&self) -> Vec<u64> {
        [
            self.uncompressed_size,
            self.compressed_size,
            self.local_header_offset,
        ]
        .into_iter()
        .filter(|&value| narrow_32(value) == SENTINEL_32)
        .collect()
    }
}

/// ZIP writer over any byte sink.
pub struct ZipWriter<W> {
    sink: W,
    offset: u64,
    members: Vec<Written>,
    names: HashSet<String>,
    level: CompressionLevel,
}

impl<W: Write> ZipWriter<W> {
    /// Starts an archive at the sink's current position, compressing DEFLATE
    /// members at the codec's default level.
    pub fn new(sink: W) -> Self {
        Self {
            sink,
            offset: 0,
            members: Vec::new(),
            names: HashSet::new(),
            level: CompressionLevel::default(),
        }
    }

    /// Appends a member holding `data`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidName`] for an empty name or one longer than
    /// 65,535 bytes, [`Error::DuplicateName`] when the name is already
    /// present, [`Error::Compression`] when the codec fails, and
    /// [`Error::Io`] when the sink fails.
    pub fn add_entry(&mut self, name: &str, data: &[u8], method: Method) -> Result<()> {
        let name_len = u16::try_from(name.len()).map_err(|_| Error::InvalidName {
            name: name.to_owned(),
            reason: "longer than 65535 bytes",
        })?;
        if name.is_empty() {
            return Err(Error::InvalidName {
                name: String::new(),
                reason: "empty",
            });
        }
        if self.names.contains(name) {
            return Err(Error::DuplicateName {
                name: name.to_owned(),
            });
        }

        let compressed;
        let payload = match method {
            Method::Stored => data,
            Method::Deflated => {
                compressed = RawDeflateCodec
                    .compress(data, self.level)
                    .map_err(|source| Error::Compression {
                        name: name.to_owned(),
                        source,
                    })?;
                compressed.as_slice()
            }
        };
        let member = Written {
            name: name.to_owned(),
            flags: if name.is_ascii() { 0 } else { FLAG_UTF8 },
            method,
            crc32: Crc32::compute_slice(data),
            compressed_size: to_offset(payload.len(), "member payload")?,
            uncompressed_size: to_offset(data.len(), "member output")?,
            local_header_offset: self.offset,
        };

        // The local header carries both sizes in its ZIP64 field whenever
        // either needs it (APPNOTE §4.5.3); the offset never appears there.
        let local_zip64 = narrow_32(member.uncompressed_size) == SENTINEL_32
            || narrow_32(member.compressed_size) == SENTINEL_32;
        let mut header = Record::default();
        header
            .put::<u32>(LOCAL_HEADER_SIGNATURE)
            .put::<u16>(if local_zip64 {
                VERSION_ZIP64
            } else {
                VERSION_DEFLATE
            })
            .put::<u16>(member.flags)
            .put::<u16>(method.code())
            .put::<u16>(DOS_TIME_MIDNIGHT)
            .put::<u16>(DOS_DATE_EPOCH)
            .put::<u32>(member.crc32);
        if local_zip64 {
            header
                .put::<u32>(SENTINEL_32)
                .put::<u32>(SENTINEL_32)
                .put::<u16>(name_len)
                .put::<u16>(20)
                .bytes(name.as_bytes())
                .put::<u16>(ZIP64_EXTRA_ID)
                .put::<u16>(16)
                .put::<u64>(member.uncompressed_size)
                .put::<u64>(member.compressed_size);
        } else {
            header
                .put::<u32>(narrow_32(member.compressed_size))
                .put::<u32>(narrow_32(member.uncompressed_size))
                .put::<u16>(name_len)
                .put::<u16>(0)
                .bytes(name.as_bytes());
        }
        self.emit(header.as_slice())?;
        self.emit(payload)?;
        self.names.insert(member.name.clone());
        self.members.push(member);
        Ok(())
    }

    /// Writes the central directory and end records and returns the sink.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the sink fails and [`Error::TooLarge`] when
    /// the archive passes the 64-bit offset range.
    pub fn finish(mut self) -> Result<W> {
        let directory_offset = self.offset;
        let members = std::mem::take(&mut self.members);
        for member in &members {
            let zip64 = member.zip64_values();
            let extra_len = if zip64.is_empty() {
                0
            } else {
                4 + 8 * zip64.len()
            };
            let mut header = Record::default();
            header
                .put::<u32>(CENTRAL_HEADER_SIGNATURE)
                .put::<u16>(VERSION_ZIP64)
                .put::<u16>(if zip64.is_empty() {
                    VERSION_DEFLATE
                } else {
                    VERSION_ZIP64
                })
                .put::<u16>(member.flags)
                .put::<u16>(member.method.code())
                .put::<u16>(DOS_TIME_MIDNIGHT)
                .put::<u16>(DOS_DATE_EPOCH)
                .put::<u32>(member.crc32)
                .put::<u32>(narrow_32(member.compressed_size))
                .put::<u32>(narrow_32(member.uncompressed_size))
                .put::<u16>(field_len(member.name.len())?)
                .put::<u16>(field_len(extra_len)?)
                .put::<u16>(0) // comment length
                .put::<u16>(0) // disk number start
                .put::<u16>(0) // internal attributes
                .put::<u32>(0) // external attributes
                .put::<u32>(narrow_32(member.local_header_offset))
                .bytes(member.name.as_bytes());
            if !zip64.is_empty() {
                header
                    .put::<u16>(ZIP64_EXTRA_ID)
                    .put::<u16>(field_len(8 * zip64.len())?);
                for value in zip64 {
                    header.put::<u64>(value);
                }
            }
            self.emit(header.as_slice())?;
        }
        let directory_size = self.offset - directory_offset;
        let count = to_offset(members.len(), "entry count")?;

        let zip64 = narrow_16(count) == SENTINEL_16
            || narrow_32(directory_size) == SENTINEL_32
            || narrow_32(directory_offset) == SENTINEL_32;
        let mut tail = Record::default();
        if zip64 {
            let record_offset = self.offset;
            tail.put::<u32>(ZIP64_END_SIGNATURE)
                .put::<u64>(ZIP64_END_RECORD_SIZE)
                .put::<u16>(VERSION_ZIP64)
                .put::<u16>(VERSION_ZIP64)
                .put::<u32>(0) // this disk
                .put::<u32>(0) // directory disk
                .put::<u64>(count)
                .put::<u64>(count)
                .put::<u64>(directory_size)
                .put::<u64>(directory_offset)
                .put::<u32>(ZIP64_LOCATOR_SIGNATURE)
                .put::<u32>(0) // disk holding the ZIP64 end record
                .put::<u64>(record_offset)
                .put::<u32>(1); // total disks
        }
        tail.put::<u32>(END_SIGNATURE)
            .put::<u16>(0) // this disk
            .put::<u16>(0) // directory disk
            .put::<u16>(narrow_16(count))
            .put::<u16>(narrow_16(count))
            .put::<u32>(narrow_32(directory_size))
            .put::<u32>(narrow_32(directory_offset))
            .put::<u16>(0); // comment length
        self.emit(tail.as_slice())?;
        self.sink.flush()?;
        Ok(self.sink)
    }

    fn emit(&mut self, bytes: &[u8]) -> Result<()> {
        self.sink.write_all(bytes)?;
        self.offset = self
            .offset
            .checked_add(to_offset(bytes.len(), "archive")?)
            .ok_or(Error::TooLarge {
                structure: "archive",
            })?;
        Ok(())
    }
}

/// A 16-bit length field value.
fn field_len(len: usize) -> Result<u16> {
    u16::try_from(len).map_err(|_| Error::TooLarge {
        structure: "header field length",
    })
}
