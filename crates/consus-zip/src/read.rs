//! Bounded ZIP container reader.
//!
//! Opening an archive reads only its tail and central directory; members are
//! decoded on request. Every declared offset and length is checked against
//! the region that must contain it before any read or allocation, so a
//! hostile header cannot drive an allocation larger than the input permits:
//!
//! - the central directory lies before the end record, and its entry count is
//!   at most its size divided by the 46-byte minimum header;
//! - each member's local header and payload lie before the central directory;
//! - a DEFLATE member's declared output is at most
//!   [`MAX_DEFLATE_RATIO`](crate::format::MAX_DEFLATE_RATIO) times its
//!   compressed size, and a stored member's output equals its compressed size.
//!
//! Decoded members are checked against their declared size and CRC-32.

use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom};

use consus_compression::Crc32;
use consus_compression::codec::flate::{RawDeflateFraming, flate2_decompress};
use consus_core::{EndianScalar, ParseBudget};

use crate::error::{Error, Result};
use crate::format::{
    CENTRAL_HEADER_LEN, CENTRAL_HEADER_SIGNATURE, END_LEN, END_SIGNATURE, FLAG_ENCRYPTED,
    FLAG_UTF8, Fields, LOCAL_HEADER_LEN, LOCAL_HEADER_SIGNATURE, MAX_COMMENT_LEN,
    MAX_DEFLATE_RATIO, Method, SENTINEL_16, SENTINEL_32, ZIP64_END_LEN, ZIP64_END_SIGNATURE,
    ZIP64_EXTRA_ID, ZIP64_LOCATOR_LEN, ZIP64_LOCATOR_SIGNATURE, to_index, to_offset,
};

/// One member as described by the central directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    name: String,
    method: Method,
    crc32: u32,
    compressed_size: u64,
    uncompressed_size: u64,
    local_header_offset: u64,
}

impl Entry {
    /// Member name, decoded from ASCII or flagged UTF-8.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Compression method.
    pub fn method(&self) -> Method {
        self.method
    }

    /// Declared CRC-32 of the uncompressed bytes.
    pub fn crc32(&self) -> u32 {
        self.crc32
    }

    /// Declared payload length in the archive.
    pub fn compressed_size(&self) -> u64 {
        self.compressed_size
    }

    /// Declared length after decoding.
    pub fn uncompressed_size(&self) -> u64 {
        self.uncompressed_size
    }
}

/// Extent of the central directory, from the end record or its ZIP64 form.
struct Directory {
    offset: u64,
    size: u64,
    entries: u64,
}

/// Random-access ZIP reader over a seekable source.
pub struct ZipReader<R> {
    source: R,
    entries: Vec<Entry>,
    directory_offset: u64,
}

impl<R: Read + Seek> ZipReader<R> {
    /// Opens an archive, reading its end record and central directory.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] naming the structure whose signature, length,
    /// bound, flags, method or name encoding violates the format.
    pub fn new(mut source: R) -> Result<Self> {
        let len = source.seek(SeekFrom::End(0))?;
        let directory = locate_directory(&mut source, len)?;
        let size = to_index(directory.size, "central directory")?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size).map_err(|_| Error::TooLarge {
            structure: "central directory",
        })?;
        bytes.resize(size, 0);
        source.seek(SeekFrom::Start(directory.offset))?;
        source.read_exact(&mut bytes)?;
        let entries = parse_directory(&bytes, directory.entries, directory.offset)?;
        Ok(Self {
            source,
            entries,
            directory_offset: directory.offset,
        })
    }

    /// Members in central-directory order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Decodes the member named `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingEntry`] when no member has that name, and the
    /// errors of [`ZipReader::read_entry`] otherwise.
    pub fn read_by_name(&mut self, name: &str) -> Result<Vec<u8>> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.name == name)
            .ok_or_else(|| Error::MissingEntry {
                name: name.to_owned(),
            })?;
        self.read_entry(index)
    }

    /// Decodes the member at `index` in [`ZipReader::entries`] order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingEntry`] for an index past the last member, a
    /// bound or signature error for an inconsistent local header, and
    /// [`Error::SizeMismatch`] or [`Error::Checksum`] when the decoded bytes
    /// disagree with the directory.
    pub fn read_entry(&mut self, index: usize) -> Result<Vec<u8>> {
        let entry = self.entries.get(index).ok_or_else(|| Error::MissingEntry {
            name: format!("#{index}"),
        })?;
        let payload = read_payload(&mut self.source, entry, self.directory_offset)?;
        decode(entry, payload)
    }
}

/// Finds the end record in the archive tail and resolves its ZIP64 form.
fn locate_directory<R: Read + Seek>(source: &mut R, len: u64) -> Result<Directory> {
    let window = to_offset(END_LEN + MAX_COMMENT_LEN, "end record search")?;
    let tail_start = len.saturating_sub(window);
    let mut tail = vec![0; to_index(len.saturating_sub(tail_start), "end record search")?];
    source.seek(SeekFrom::Start(tail_start))?;
    source.read_exact(&mut tail)?;

    // The end record is the last signature whose comment length reaches
    // exactly to the end of the archive. Scanning backwards skips a bare
    // signature inside the comment; a comment that embeds a whole end record
    // with a consistent length is indistinguishable and wins, as it does for
    // other readers.
    let record = (0..=tail.len().saturating_sub(END_LEN))
        .rev()
        .find_map(|at| tail.get(at..).filter(|record| is_end_record(record)))
        .ok_or(Error::MissingEndOfCentralDirectory)?;
    let end_offset = tail_start
        .checked_add(to_offset(
            tail.len().saturating_sub(record.len()),
            "end record",
        )?)
        .ok_or(Error::TooLarge {
            structure: "end record",
        })?;

    let mut fields = Fields::new(record, "end of central directory record");
    fields.signature(END_SIGNATURE)?;
    let disk = fields.read::<u16>()?;
    let directory_disk = fields.read::<u16>()?;
    let disk_entries = fields.read::<u16>()?;
    let entries = fields.read::<u16>()?;
    let size = fields.read::<u32>()?;
    let offset = fields.read::<u32>()?;
    if disk != 0 || directory_disk != 0 || disk_entries != entries {
        return Err(Error::MultiDisk);
    }

    // A sentinel defers to the ZIP64 records. An archive of exactly 65,535
    // entries may carry that count without them, so a missing locator falls
    // back to the classic fields when the size and offset are not sentinels.
    let zip64 = size == SENTINEL_32
        || offset == SENTINEL_32
        || (entries == SENTINEL_16 && has_zip64_locator(source, end_offset)?);
    let directory = if zip64 {
        locate_zip64_directory(source, end_offset)?
    } else {
        Directory {
            offset: u64::from(offset),
            size: u64::from(size),
            entries: u64::from(entries),
        }
    };
    let directory_end = directory
        .offset
        .checked_add(directory.size)
        .ok_or(Error::OutOfBounds {
            structure: "central directory",
        })?;
    if directory_end > end_offset {
        return Err(Error::OutOfBounds {
            structure: "central directory",
        });
    }
    let capacity = directory
        .size
        .checked_div(to_offset(CENTRAL_HEADER_LEN, "central directory")?)
        .expect("invariant: the minimum header length is nonzero");
    if directory.entries > capacity {
        return Err(Error::EntryCount {
            declared: directory.entries,
            capacity,
        });
    }
    Ok(directory)
}

/// Whether `record` starts with an end record whose comment fills the rest.
fn is_end_record(record: &[u8]) -> bool {
    let mut fields = Fields::new(record, "end of central directory record");
    // Signature, then 16 bytes of disk numbers, counts, size and offset.
    if fields.signature(END_SIGNATURE).is_err() || fields.take(16).is_err() {
        return false;
    }
    fields
        .read::<u16>()
        .is_ok_and(|comment_len| usize::from(comment_len) == fields.remaining())
}

/// Whether a ZIP64 locator signature precedes the end record.
fn has_zip64_locator<R: Read + Seek>(source: &mut R, end_offset: u64) -> Result<bool> {
    let Some(locator_offset) = end_offset.checked_sub(to_offset(ZIP64_LOCATOR_LEN, "locator")?)
    else {
        return Ok(false);
    };
    let mut signature = [0; 4];
    source.seek(SeekFrom::Start(locator_offset))?;
    source.read_exact(&mut signature)?;
    Ok(u32::from_le_bytes(signature) == ZIP64_LOCATOR_SIGNATURE)
}

/// Reads the ZIP64 locator before the end record, then the record it names.
fn locate_zip64_directory<R: Read + Seek>(source: &mut R, end_offset: u64) -> Result<Directory> {
    let locator_offset = end_offset
        .checked_sub(to_offset(ZIP64_LOCATOR_LEN, "ZIP64 locator")?)
        .ok_or(Error::Truncated {
            structure: "ZIP64 end of central directory locator",
        })?;
    let mut locator = [0; ZIP64_LOCATOR_LEN];
    source.seek(SeekFrom::Start(locator_offset))?;
    source.read_exact(&mut locator)?;
    let mut fields = Fields::new(&locator, "ZIP64 end of central directory locator");
    fields.signature(ZIP64_LOCATOR_SIGNATURE)?;
    let record_disk = fields.read::<u32>()?;
    let record_offset = fields.read::<u64>()?;
    let disks = fields.read::<u32>()?;
    if record_disk != 0 || disks > 1 {
        return Err(Error::MultiDisk);
    }
    let record_end = record_offset
        .checked_add(to_offset(ZIP64_END_LEN, "ZIP64 end record")?)
        .ok_or(Error::OutOfBounds {
            structure: "ZIP64 end of central directory record",
        })?;
    if record_end > locator_offset {
        return Err(Error::OutOfBounds {
            structure: "ZIP64 end of central directory record",
        });
    }

    let mut record = [0; ZIP64_END_LEN];
    source.seek(SeekFrom::Start(record_offset))?;
    source.read_exact(&mut record)?;
    let mut fields = Fields::new(&record, "ZIP64 end of central directory record");
    fields.signature(ZIP64_END_SIGNATURE)?;
    let _record_size = fields.read::<u64>()?;
    let _version_made_by = fields.read::<u16>()?;
    let _version_needed = fields.read::<u16>()?;
    let disk = fields.read::<u32>()?;
    let directory_disk = fields.read::<u32>()?;
    let disk_entries = fields.read::<u64>()?;
    let entries = fields.read::<u64>()?;
    let size = fields.read::<u64>()?;
    let offset = fields.read::<u64>()?;
    if disk != 0 || directory_disk != 0 || disk_entries != entries {
        return Err(Error::MultiDisk);
    }
    // The directory must end at or before the ZIP64 record.
    if offset
        .checked_add(size)
        .is_none_or(|end| end > record_offset)
    {
        return Err(Error::OutOfBounds {
            structure: "central directory",
        });
    }
    Ok(Directory {
        offset,
        size,
        entries,
    })
}

/// Parses `count` central directory headers that exactly fill `bytes`.
fn parse_directory(bytes: &[u8], count: u64, directory_offset: u64) -> Result<Vec<Entry>> {
    let mut fields = Fields::new(bytes, "central directory header");
    let count = to_index(count, "central directory")?;
    let mut entries = Vec::with_capacity(count);
    let mut names = HashSet::with_capacity(count);
    for _ in 0..count {
        let entry = parse_central_header(&mut fields)?;
        if !names.insert(entry.name.clone()) {
            return Err(Error::DuplicateName { name: entry.name });
        }
        let header_end = entry
            .local_header_offset
            .checked_add(to_offset(LOCAL_HEADER_LEN, "local header")?);
        if header_end.is_none_or(|end| end > directory_offset) {
            return Err(Error::OutOfBounds {
                structure: "local file header",
            });
        }
        entries.push(entry);
    }
    if fields.remaining() != 0 {
        return Err(Error::TrailingDirectoryBytes);
    }
    Ok(entries)
}

fn parse_central_header(fields: &mut Fields<'_>) -> Result<Entry> {
    fields.signature(CENTRAL_HEADER_SIGNATURE)?;
    let _version_made_by = fields.read::<u16>()?;
    let _version_needed = fields.read::<u16>()?;
    let flags = fields.read::<u16>()?;
    let method_code = fields.read::<u16>()?;
    let _time = fields.read::<u16>()?;
    let _date = fields.read::<u16>()?;
    let crc32 = fields.read::<u32>()?;
    let compressed = fields.read::<u32>()?;
    let uncompressed = fields.read::<u32>()?;
    let name_len = usize::from(fields.read::<u16>()?);
    let extra_len = usize::from(fields.read::<u16>()?);
    let comment_len = usize::from(fields.read::<u16>()?);
    let disk = fields.read::<u16>()?;
    let _internal_attributes = fields.read::<u16>()?;
    let _external_attributes = fields.read::<u32>()?;
    let offset = fields.read::<u32>()?;
    let name = decode_name(fields.take(name_len)?, flags)?;
    let extra = fields.take(extra_len)?;
    fields.take(comment_len)?;

    let mut zip64 = Zip64Fields::find(extra)?;
    let uncompressed_size = zip64.resolve(uncompressed, "uncompressed size")?;
    let compressed_size = zip64.resolve(compressed, "compressed size")?;
    let local_header_offset = zip64.resolve(offset, "local header offset")?;
    let disk = if disk == SENTINEL_16 {
        zip64.next::<u32>("disk number")?
    } else {
        u32::from(disk)
    };
    if disk != 0 {
        return Err(Error::MultiDisk);
    }

    if flags & FLAG_ENCRYPTED != 0 {
        return Err(Error::Encrypted { name });
    }
    let method = Method::from_code(method_code).ok_or_else(|| Error::UnsupportedMethod {
        name: name.clone(),
        method: method_code,
    })?;
    let admissible = match method {
        Method::Stored => uncompressed_size == compressed_size,
        Method::Deflated => uncompressed_size <= compressed_size.saturating_mul(MAX_DEFLATE_RATIO),
    };
    if !admissible {
        return Err(Error::ExpansionBound {
            name,
            compressed: compressed_size,
            uncompressed: uncompressed_size,
        });
    }
    Ok(Entry {
        name,
        method,
        crc32,
        compressed_size,
        uncompressed_size,
        local_header_offset,
    })
}

fn decode_name(raw: &[u8], flags: u16) -> Result<String> {
    if flags & FLAG_UTF8 == 0 && !raw.is_ascii() {
        return Err(Error::NameEncoding);
    }
    String::from_utf8(raw.to_vec()).map_err(|_| Error::NameEncoding)
}

/// Values of the ZIP64 extended information field, consumed in APPNOTE order.
struct Zip64Fields<'a> {
    fields: Option<Fields<'a>>,
}

impl<'a> Zip64Fields<'a> {
    /// Finds the ZIP64 block. Fewer than four trailing bytes cannot hold a
    /// block header and are padding some writers leave, so they are skipped.
    fn find(extra: &'a [u8]) -> Result<Self> {
        let mut blocks = Fields::new(extra, "extra field");
        while blocks.remaining() >= 4 {
            let id = blocks.read::<u16>()?;
            let len = usize::from(blocks.read::<u16>()?);
            let data = blocks.take(len)?;
            if id == ZIP64_EXTRA_ID {
                return Ok(Self {
                    fields: Some(Fields::new(data, "ZIP64 extra field")),
                });
            }
        }
        Ok(Self { fields: None })
    }

    /// The next field: 8 bytes for sizes and offsets, 4 for the disk number.
    fn next<T: EndianScalar>(&mut self, field: &'static str) -> Result<T> {
        let fields = self
            .fields
            .as_mut()
            .ok_or(Error::MissingZip64Field { field })?;
        fields
            .read::<T>()
            .map_err(|_| Error::MissingZip64Field { field })
    }

    /// The 64-bit value when `narrow` is the sentinel, else `narrow` itself.
    fn resolve(&mut self, narrow: u32, field: &'static str) -> Result<u64> {
        if narrow == SENTINEL_32 {
            self.next::<u64>(field)
        } else {
            Ok(u64::from(narrow))
        }
    }
}

/// Reads a member's payload after checking its local header.
fn read_payload<R: Read + Seek>(
    source: &mut R,
    entry: &Entry,
    directory_offset: u64,
) -> Result<Vec<u8>> {
    let mut header = [0; LOCAL_HEADER_LEN];
    source.seek(SeekFrom::Start(entry.local_header_offset))?;
    source.read_exact(&mut header)?;
    let mut fields = Fields::new(&header, "local file header");
    fields.signature(LOCAL_HEADER_SIGNATURE)?;
    fields.take(22)?; // version, flags, method, time, date, CRC, both sizes
    let name_len = fields.read::<u16>()?;
    let extra_len = fields.read::<u16>()?;

    let data_start = entry
        .local_header_offset
        .checked_add(to_offset(LOCAL_HEADER_LEN, "local header")?)
        .and_then(|at| at.checked_add(u64::from(name_len)))
        .and_then(|at| at.checked_add(u64::from(extra_len)));
    let data_end = data_start.and_then(|start| start.checked_add(entry.compressed_size));
    if data_end.is_none_or(|end| end > directory_offset) {
        return Err(Error::OutOfBounds {
            structure: "member payload",
        });
    }

    let mut name = vec![0; usize::from(name_len)];
    source.read_exact(&mut name)?;
    if name != entry.name.as_bytes() {
        return Err(Error::LocalHeaderMismatch {
            name: entry.name.clone(),
        });
    }
    source.seek(SeekFrom::Current(i64::from(extra_len)))?;

    let len = to_index(entry.compressed_size, "member payload")?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(len)
        .map_err(|_| Error::TooLarge {
            structure: "member payload",
        })?;
    payload.resize(len, 0);
    source.read_exact(&mut payload)?;
    Ok(payload)
}

/// Decodes a payload and checks its length and CRC-32 against the directory.
fn decode(entry: &Entry, payload: Vec<u8>) -> Result<Vec<u8>> {
    let declared = to_index(entry.uncompressed_size, "member output")?;
    let bytes = match entry.method {
        Method::Stored => payload,
        Method::Deflated => {
            // The declared size is the ceiling. The initial reservation is at
            // most the compressed size, so a hostile declaration cannot reserve
            // 1032 times the input before the stream proves it.
            let defaults = ParseBudget::default();
            let budget = ParseBudget::new(declared, defaults.max_elements, defaults.max_depth);
            let initial = declared.min(payload.len());
            flate2_decompress::<RawDeflateFraming>(&payload, initial, &budget).map_err(
                |source| Error::Compression {
                    name: entry.name.clone(),
                    source,
                },
            )?
        }
    };
    if bytes.len() != declared {
        return Err(Error::SizeMismatch {
            name: entry.name.clone(),
            declared: entry.uncompressed_size,
            actual: to_offset(bytes.len(), "member output")?,
        });
    }
    let actual = Crc32::compute_slice(&bytes);
    if actual != entry.crc32 {
        return Err(Error::Checksum {
            name: entry.name.clone(),
            expected: entry.crc32,
            actual,
        });
    }
    Ok(bytes)
}
