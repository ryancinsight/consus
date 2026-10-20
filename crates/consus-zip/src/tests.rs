//! Round-trip and adversarial-header cases for the container.

use std::io::Cursor;

use proptest::prelude::*;

use crate::format::{CENTRAL_HEADER_LEN, END_LEN, END_SIGNATURE};
use crate::{Error, Method, ZipReader, ZipWriter};
use consus_compression::Crc32;

mod zip64;

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
