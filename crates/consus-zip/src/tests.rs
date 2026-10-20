//! Round-trip, ZIP64 and adversarial-header cases for the container.

use std::io::Cursor;

use proptest::prelude::*;

use crate::format::{
    CENTRAL_HEADER_LEN, CENTRAL_HEADER_SIGNATURE, DOS_DATE_EPOCH, END_LEN, END_SIGNATURE,
    LOCAL_HEADER_SIGNATURE, Record,
};
use crate::{Error, Method, ZipReader, ZipWriter};
use consus_compression::Crc32;

fn archive(members: &[(&str, &[u8], Method)]) -> Vec<u8> {
    let mut writer = ZipWriter::new(Vec::new());
    for &(name, data, method) in members {
        writer.add_entry(name, data, method).unwrap();
    }
    writer.finish().unwrap()
}

fn open(bytes: Vec<u8>) -> crate::Result<ZipReader<Cursor<Vec<u8>>>> {
    ZipReader::new(Cursor::new(bytes))
}

/// Offset of the only central header in a one-member archive with no comment.
fn central_header_offset(bytes: &[u8], name: &str) -> usize {
    bytes.len() - END_LEN - CENTRAL_HEADER_LEN - name.len()
}

#[test]
fn stored_and_deflated_members_round_trip_by_name_and_index() {
    let pattern: Vec<u8> = (0..10_000u32)
        .map(|i| u8::try_from(i % 251).unwrap())
        .collect();
    let bytes = archive(&[
        ("stored.txt", b"stored verbatim", Method::Stored),
        ("dir/pattern.bin", &pattern, Method::Deflated),
        ("empty", b"", Method::Deflated),
        ("ünïcode.txt", "ä".as_bytes(), Method::Stored),
    ]);
    let mut reader = open(bytes).unwrap();
    let names: Vec<&str> = reader.entries().iter().map(|e| e.name()).collect();
    assert_eq!(
        names,
        ["stored.txt", "dir/pattern.bin", "empty", "ünïcode.txt"]
    );
    let deflated = &reader.entries()[1];
    assert_eq!(deflated.method(), Method::Deflated);
    assert_eq!(deflated.uncompressed_size(), 10_000);
    assert!(deflated.compressed_size() < 10_000);
    assert_eq!(deflated.crc32(), Crc32::compute_slice(&pattern));

    assert_eq!(
        reader.read_by_name("stored.txt").unwrap(),
        b"stored verbatim"
    );
    assert_eq!(reader.read_entry(1).unwrap(), pattern);
    assert_eq!(reader.read_by_name("empty").unwrap(), b"");
    assert_eq!(reader.read_by_name("ünïcode.txt").unwrap(), "ä".as_bytes());
}

#[test]
fn equal_inputs_write_identical_archives() {
    let members = [
        ("a", &b"x"[..], Method::Deflated),
        ("b", &b"y"[..], Method::Stored),
    ];
    assert_eq!(archive(&members), archive(&members));
}

#[test]
fn empty_archive_is_one_end_record() {
    let bytes = archive(&[]);
    assert_eq!(bytes.len(), END_LEN);
    assert!(open(bytes).unwrap().entries().is_empty());
}

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

#[test]
fn end_record_signature_inside_a_comment_does_not_shadow_the_record() {
    let mut bytes = archive(&[("a", b"payload", Method::Stored)]);
    let comment: Vec<u8> = END_SIGNATURE.to_le_bytes().repeat(4);
    let len = bytes.len();
    bytes[len - 2..].copy_from_slice(&u16::try_from(comment.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&comment);
    let mut reader = open(bytes).unwrap();
    assert_eq!(reader.read_by_name("a").unwrap(), b"payload");
}

#[test]
fn truncated_archives_are_rejected_at_every_cut() {
    let bytes = archive(&[("member", &[3u8; 300], Method::Deflated)]);
    for cut in 0..bytes.len() {
        // The end record is the last 22 bytes and its comment length must
        // reach exactly to the end, so every cut removes the only valid one.
        let result = open(bytes[..cut].to_vec());
        assert!(
            matches!(result, Err(Error::MissingEndOfCentralDirectory)),
            "cut at {cut}: {:?}",
            result.err()
        );
    }
}

#[test]
fn crc_mismatch_is_reported_with_both_values() {
    let mut bytes = archive(&[("m", b"value", Method::Stored)]);
    let crc_at = central_header_offset(&bytes, "m") + 16;
    bytes[crc_at] ^= 0xFF;
    let mut reader = open(bytes).unwrap();
    let err = reader.read_entry(0).unwrap_err();
    assert!(
        matches!(err, Error::Checksum { expected, actual, .. }
            if actual == Crc32::compute_slice(b"value") && expected != actual),
        "{err:?}"
    );
}

#[test]
fn declared_deflate_expansion_beyond_1032_to_1_is_rejected_before_decoding() {
    let mut bytes = archive(&[("m", &[0u8; 64], Method::Deflated)]);
    let header = central_header_offset(&bytes, "m");
    let compressed = u32::from_le_bytes(bytes[header + 20..header + 24].try_into().unwrap());
    let declared = compressed * 1032 + 1;
    bytes[header + 24..header + 28].copy_from_slice(&declared.to_le_bytes());
    let err = open(bytes).err().unwrap();
    assert!(
        matches!(err, Error::ExpansionBound { uncompressed, .. } if uncompressed == u64::from(declared)),
        "{err:?}"
    );
}

#[test]
fn understated_deflate_size_fails_without_overrun() {
    let mut bytes = archive(&[("m", &[9u8; 4096], Method::Deflated)]);
    let header = central_header_offset(&bytes, "m");
    bytes[header + 24..header + 28].copy_from_slice(&100u32.to_le_bytes());
    let err = open(bytes).unwrap().read_entry(0).unwrap_err();
    assert!(matches!(err, Error::Compression { .. }), "{err:?}");
}

#[test]
fn stored_member_with_unequal_sizes_is_rejected() {
    let mut bytes = archive(&[("m", b"abc", Method::Stored)]);
    let header = central_header_offset(&bytes, "m");
    bytes[header + 24..header + 28].copy_from_slice(&4u32.to_le_bytes());
    assert!(matches!(open(bytes), Err(Error::ExpansionBound { .. })));
}

#[test]
fn entry_count_beyond_directory_capacity_is_rejected() {
    let mut bytes = archive(&[("m", b"abc", Method::Stored)]);
    let end = bytes.len() - END_LEN;
    bytes[end + 8..end + 12].copy_from_slice(&[2, 0, 2, 0]);
    let err = open(bytes).err().unwrap();
    assert!(
        matches!(
            err,
            Error::EntryCount {
                declared: 2,
                capacity: 1
            }
        ),
        "{err:?}"
    );
}

#[test]
fn directory_offset_past_the_end_record_is_rejected() {
    let mut bytes = archive(&[("m", b"abc", Method::Stored)]);
    let end = bytes.len() - END_LEN;
    bytes[end + 16..end + 20].copy_from_slice(&u32::MAX.wrapping_sub(1).to_le_bytes());
    assert!(matches!(
        open(bytes),
        Err(Error::OutOfBounds {
            structure: "central directory"
        })
    ));
}

#[test]
fn local_header_offset_into_the_directory_is_rejected() {
    let mut bytes = archive(&[("m", b"abc", Method::Stored)]);
    let header = central_header_offset(&bytes, "m");
    let offset = u32::try_from(header).unwrap();
    bytes[header + 42..header + 46].copy_from_slice(&offset.to_le_bytes());
    assert!(matches!(
        open(bytes),
        Err(Error::OutOfBounds {
            structure: "local file header"
        })
    ));
}

#[test]
fn encrypted_and_unsupported_members_are_rejected() {
    let mut encrypted = archive(&[("m", b"abc", Method::Stored)]);
    let header = central_header_offset(&encrypted, "m");
    encrypted[header + 8] |= 1;
    assert!(matches!(open(encrypted), Err(Error::Encrypted { .. })));

    let mut bzip2 = archive(&[("m", b"abc", Method::Stored)]);
    bzip2[header + 10] = 12;
    assert!(matches!(
        open(bzip2),
        Err(Error::UnsupportedMethod { method: 12, .. })
    ));
}

#[test]
fn non_ascii_name_without_the_utf8_flag_is_rejected() {
    let mut bytes = archive(&[("é", b"abc", Method::Stored)]);
    let header = central_header_offset(&bytes, "é");
    bytes[header + 9] &= !0x08; // clear flag bit 11
    assert!(matches!(open(bytes), Err(Error::NameEncoding)));
}

#[test]
fn renamed_local_header_is_rejected() {
    let mut bytes = archive(&[("m", b"abc", Method::Stored)]);
    bytes[30] = b'n';
    let err = open(bytes).unwrap().read_entry(0).unwrap_err();
    assert!(matches!(err, Error::LocalHeaderMismatch { .. }), "{err:?}");
}

#[test]
fn duplicate_names_are_rejected_on_write_and_read() {
    let mut writer = ZipWriter::new(Vec::new());
    writer.add_entry("m", b"1", Method::Stored).unwrap();
    assert!(matches!(
        writer.add_entry("m", b"2", Method::Stored),
        Err(Error::DuplicateName { .. })
    ));

    let mut bytes = archive(&[("a", b"1", Method::Stored), ("b", b"2", Method::Stored)]);
    let second_name = bytes.len() - END_LEN - 1;
    bytes[second_name] = b'a';
    assert!(matches!(open(bytes), Err(Error::DuplicateName { .. })));
}

#[test]
fn invalid_names_are_rejected_on_write() {
    let mut writer = ZipWriter::new(Vec::new());
    assert!(matches!(
        writer.add_entry("", b"", Method::Stored),
        Err(Error::InvalidName {
            reason: "empty",
            ..
        })
    ));
    let long = "n".repeat(65_536);
    assert!(matches!(
        writer.add_entry(&long, b"", Method::Stored),
        Err(Error::InvalidName { .. })
    ));
}

#[test]
fn missing_member_is_reported_by_name() {
    let mut reader = open(archive(&[("a", b"1", Method::Stored)])).unwrap();
    assert!(matches!(
        reader.read_by_name("b"),
        Err(Error::MissingEntry { name }) if name == "b"
    ));
    assert!(matches!(
        reader.read_entry(1),
        Err(Error::MissingEntry { .. })
    ));
}

#[test]
fn input_without_an_end_record_is_rejected() {
    assert!(matches!(
        open(vec![0; 100]),
        Err(Error::MissingEndOfCentralDirectory)
    ));
    assert!(matches!(
        open(Vec::new()),
        Err(Error::MissingEndOfCentralDirectory)
    ));
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

#[test]
fn overstated_deflate_size_within_the_bound_is_a_size_mismatch() {
    let mut bytes = archive(&[("m", &[5u8; 512], Method::Deflated)]);
    let header = central_header_offset(&bytes, "m");
    bytes[header + 24..header + 28].copy_from_slice(&600u32.to_le_bytes());
    let err = open(bytes).unwrap().read_entry(0).unwrap_err();
    assert!(
        matches!(
            err,
            Error::SizeMismatch {
                declared: 600,
                actual: 512,
                ..
            }
        ),
        "{err:?}"
    );
}

proptest! {
    #[test]
    fn arbitrary_members_round_trip(
        members in proptest::collection::btree_map(
            "[a-z/._]{1,24}",
            (proptest::collection::vec(any::<u8>(), 0..2048), any::<bool>()),
            0..12,
        )
    ) {
        let mut writer = ZipWriter::new(Vec::new());
        for (name, (data, deflate)) in &members {
            let method = if *deflate { Method::Deflated } else { Method::Stored };
            writer.add_entry(name, data, method).unwrap();
        }
        let mut reader = open(writer.finish().unwrap()).unwrap();
        prop_assert_eq!(reader.entries().len(), members.len());
        for (name, (data, _)) in &members {
            prop_assert_eq!(&reader.read_by_name(name).unwrap(), data);
        }
    }
}
