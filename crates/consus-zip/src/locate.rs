//! Locating the central directory from the archive tail.
//!
//! The end record is found by a bounded backwards scan of the last
//! 65,557 bytes; a sentinel in it defers to the ZIP64 locator and record.
//! The resolved directory must end before the end record, and its entry
//! count must fit its size at the 46-byte minimum header length.

use std::io::{Read, Seek, SeekFrom};

use crate::error::{Error, Result};
use crate::format::{
    CENTRAL_HEADER_LEN, END_LEN, END_SIGNATURE, Fields, MAX_COMMENT_LEN, SENTINEL_16, SENTINEL_32,
    ZIP64_END_LEN, ZIP64_END_SIGNATURE, ZIP64_LOCATOR_LEN, ZIP64_LOCATOR_SIGNATURE, to_index,
    to_offset,
};

/// Extent of the central directory, from the end record or its ZIP64 form.
pub(crate) struct Directory {
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) entries: u64,
}

/// Finds the end record in the archive tail and resolves its ZIP64 form.
pub(crate) fn locate_directory<R: Read + Seek>(source: &mut R, len: u64) -> Result<Directory> {
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
