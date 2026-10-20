//! ZIP64 sizes, offsets, entry counts and disk numbers.

use super::{archive, central_header_offset, open};
use crate::format::{
    CENTRAL_HEADER_LEN, CENTRAL_HEADER_SIGNATURE, DOS_DATE_EPOCH, END_LEN, END_SIGNATURE,
    LOCAL_HEADER_SIGNATURE, Record,
};
use crate::{Error, Method, ZipWriter};
use consus_compression::Crc32;

#[test]
fn entry_count_at_the_16_bit_sentinel_round_trips_through_zip64() {
    let mut writer = ZipWriter::new(Vec::new());
    let names: Vec<String> = (0..65_535u32).map(|i| format!("{i:05}")).collect();
    for name in &names {
        writer.add_entry(name, b"", Method::Stored).unwrap();
    }
    let bytes = writer.finish().unwrap();
    // The classic end record must carry the sentinel, deferring to ZIP64.
    let end = &bytes[bytes.len() - END_LEN..];
    assert_eq!(u16::from_le_bytes([end[10], end[11]]), u16::MAX);
    let reader = open(bytes).unwrap();
    assert_eq!(reader.entries().len(), 65_535);
    assert_eq!(reader.entries()[65_534].name(), "65534");
}

/// A stored member whose central header carries every size and its offset
/// in the ZIP64 extra field, as a writer may do below the 4 GiB limit.
fn zip64_extra_archive(central_extra_values: &[u64]) -> Vec<u8> {
    let name = b"big.npy";
    let data = b"zip64 payload";
    let crc = Crc32::compute_slice(data);
    let size = u64::try_from(data.len()).unwrap();
    let name_len = u16::try_from(name.len()).unwrap();
    let mut out = Record::default();
    out.put::<u32>(LOCAL_HEADER_SIGNATURE)
        .put::<u16>(45)
        .put::<u16>(0)
        .put::<u16>(0)
        .put::<u16>(0)
        .put::<u16>(DOS_DATE_EPOCH)
        .put::<u32>(crc)
        .put::<u32>(u32::MAX)
        .put::<u32>(u32::MAX)
        .put::<u16>(name_len)
        .put::<u16>(20)
        .bytes(name)
        .put::<u16>(1)
        .put::<u16>(16)
        .put::<u64>(size)
        .put::<u64>(size)
        .bytes(data);
    let directory_offset = u32::try_from(out.as_slice().len()).unwrap();
    let extra_len = 8 * u16::try_from(central_extra_values.len()).unwrap();
    out.put::<u32>(CENTRAL_HEADER_SIGNATURE)
        .put::<u16>(45)
        .put::<u16>(45)
        .put::<u16>(0)
        .put::<u16>(0)
        .put::<u16>(0)
        .put::<u16>(DOS_DATE_EPOCH)
        .put::<u32>(crc)
        .put::<u32>(u32::MAX)
        .put::<u32>(u32::MAX)
        .put::<u16>(name_len)
        .put::<u16>(4 + extra_len)
        .put::<u16>(0)
        .put::<u16>(0)
        .put::<u16>(0)
        .put::<u32>(0)
        .put::<u32>(u32::MAX)
        .bytes(name)
        .put::<u16>(1)
        .put::<u16>(extra_len);
    for &value in central_extra_values {
        out.put::<u64>(value);
    }
    let directory_size = u32::try_from(out.as_slice().len()).unwrap() - directory_offset;
    out.put::<u32>(END_SIGNATURE)
        .put::<u16>(0)
        .put::<u16>(0)
        .put::<u16>(1)
        .put::<u16>(1)
        .put::<u32>(directory_size)
        .put::<u32>(directory_offset)
        .put::<u16>(0);
    out.as_slice().to_vec()
}

#[test]
fn zip64_extra_field_supplies_sizes_and_offset() {
    let mut reader = open(zip64_extra_archive(&[13, 13, 0])).unwrap();
    let entry = &reader.entries()[0];
    assert_eq!(entry.uncompressed_size(), 13);
    assert_eq!(entry.compressed_size(), 13);
    assert_eq!(reader.read_by_name("big.npy").unwrap(), b"zip64 payload");
}

#[test]
fn sentinel_without_its_zip64_value_is_rejected() {
    let err = open(zip64_extra_archive(&[13, 13])).err().unwrap();
    assert!(
        matches!(
            err,
            Error::MissingZip64Field {
                field: "local header offset"
            }
        ),
        "{err:?}"
    );
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
        .collect()
}

/// A stored member whose central header sets all four ZIP64 sentinels and
/// carries the standard 28-byte extra field: both sizes, the offset, and a
/// 4-byte disk number (APPNOTE §4.5.3). Python's `zipfile` opens it.
const ZIP64_ALL_SENTINELS: &str = "504b030414000000000000002100c24124350300000003000000010000006d616263504b01022d002d000000000000002100c2412435ffffffffffffffff010020000000ffff000000000000ffffffff6d01001c0003000000000000000300000000000000000000000000000000000000504b050600000000010001004f000000220000000000";

#[test]
fn zip64_disk_number_is_four_bytes_wide() {
    let mut reader = open(hex(ZIP64_ALL_SENTINELS)).unwrap();
    assert_eq!(reader.entries()[0].name(), "m");
    assert_eq!(reader.read_entry(0).unwrap(), b"abc");
}

#[test]
fn zip64_disk_number_other_than_zero_is_rejected() {
    let mut bytes = hex(ZIP64_ALL_SENTINELS);
    // The disk number is the last four bytes of the 28-byte extra field, which
    // ends where the central directory ends, 22 bytes before the archive end.
    let disk_at = bytes.len() - END_LEN - 4;
    bytes[disk_at] = 7;
    assert!(matches!(open(bytes), Err(Error::MultiDisk)));
}

#[test]
fn trailing_extra_bytes_too_short_for_a_block_are_ignored() {
    let mut bytes = archive(&[("m", b"abc", Method::Stored)]);
    let header = central_header_offset(&bytes, "m");
    // Append two padding bytes to the central extra field and shift the end
    // record's directory size accordingly.
    let name_end = header + CENTRAL_HEADER_LEN + 1;
    bytes.splice(name_end..name_end, [0, 0]);
    bytes[header + 30] = 2;
    let end = bytes.len() - END_LEN;
    let size = u32::from_le_bytes(bytes[end + 12..end + 16].try_into().unwrap()) + 2;
    bytes[end + 12..end + 16].copy_from_slice(&size.to_le_bytes());
    assert_eq!(open(bytes).unwrap().read_entry(0).unwrap(), b"abc");
}

#[test]
fn classic_end_record_with_65535_entries_and_no_zip64_records_is_read() {
    // Write 65,534 members, then patch in one more central header by hand so
    // the classic count is exactly 0xFFFF and no ZIP64 records exist.
    let mut writer = ZipWriter::new(Vec::new());
    for i in 0..65_534u32 {
        writer
            .add_entry(&format!("{i:05}"), b"", Method::Stored)
            .unwrap();
    }
    let mut bytes = writer.finish().unwrap();
    let end = bytes.len() - END_LEN;
    let count = u16::from_le_bytes([bytes[end + 10], bytes[end + 11]]);
    assert_eq!(count, 65_534);
    let first = usize::try_from(u32::from_le_bytes(
        bytes[end + 16..end + 20].try_into().unwrap(),
    ))
    .unwrap();
    let mut extra_header = bytes[first..first + CENTRAL_HEADER_LEN + 5].to_vec();
    extra_header[CENTRAL_HEADER_LEN..].copy_from_slice(b"extra");
    extra_header[28..30].copy_from_slice(&5u16.to_le_bytes());
    let mut tail = bytes.split_off(end);
    bytes.extend_from_slice(&extra_header);
    tail[8..12].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    let size = u32::from_le_bytes(tail[12..16].try_into().unwrap())
        + u32::try_from(extra_header.len()).unwrap();
    tail[12..16].copy_from_slice(&size.to_le_bytes());
    bytes.extend_from_slice(&tail);

    let reader = open(bytes).unwrap();
    assert_eq!(reader.entries().len(), 65_535);
    assert_eq!(reader.entries()[65_534].name(), "extra");
}
