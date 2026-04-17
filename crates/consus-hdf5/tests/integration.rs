#![cfg(feature = "alloc")]

use core::num::NonZeroUsize;

use byteorder::{ByteOrder, LittleEndian};
use consus_core::{ByteOrder as CoreByteOrder, Datatype, LinkType, Shape};
use consus_hdf5::attribute::Hdf5Attribute;
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use consus_hdf5::file::reader;
use consus_hdf5::file::writer::{
    FileCreationProps, Hdf5FileBuilder, WriteState, update_superblock_eof, write_contiguous_data,
    write_dataset_header, write_group_header, write_superblock,
};
use consus_hdf5::object_header::{HeaderMessage, ObjectHeader};
use consus_hdf5::property_list::{DatasetCreationProps, DatasetLayout, GroupCreationProps};
use consus_io::{MemCursor, WriteAt};

fn u32_le_datatype() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(32).expect("non-zero"),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    }
}

#[test]
fn v4_chunked_dataset_value_roundtrip() {
    use core::num::NonZeroUsize;

    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: true,
    };
    let shape = consus_core::Shape::fixed(&[4, 4]);
    let values: Vec<u32> = (0..16).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![2, 2]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("v4_chunked", &dt, &shape, &raw, &dcpl)
        .expect("write chunked dataset");

    let bytes = builder.finish().expect("finish file");
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");

    let datasets = file.list_root_group().expect("list root");
    let addr = datasets
        .iter()
        .find(|(name, _, _)| name == "v4_chunked")
        .map(|(_, addr, _)| *addr)
        .expect("dataset link");

    let dataset = file.dataset_at(addr).expect("dataset metadata");
    assert_eq!(dataset.layout, StorageLayout::Chunked);
    assert_eq!(dataset.shape.current_dims().as_slice(), &[4, 4]);
    assert_eq!(
        dataset.chunk_shape.as_ref().expect("chunk shape").dims(),
        &[2, 2]
    );

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read v4 chunked dataset");
    let read_values: Vec<u32> = read_buf
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    assert_eq!(read_values, values);
}

fn make_scalar_u32_attribute(name: &str, value: u32) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let datatype = vec![
        0x10, 0x00, 0x00, 0x00, // fixed-point, LE unsigned
        0x04, 0x00, 0x00, 0x00, // size = 4
        0x00, 0x00, // bit offset
        0x20, 0x00, // precision = 32
    ];
    let dataspace = vec![2, 0, 0, 0]; // scalar dataspace v2
    let mut data = vec![0u8; 9 + name_bytes.len() + datatype.len() + dataspace.len() + 4];
    data[0] = 3; // version
    data[1] = 0; // flags
    LittleEndian::write_u16(&mut data[2..4], name_bytes.len() as u16);
    LittleEndian::write_u16(&mut data[4..6], datatype.len() as u16);
    LittleEndian::write_u16(&mut data[6..8], dataspace.len() as u16);
    data[8] = 0; // ASCII

    let mut pos = 9;
    data[pos..pos + name_bytes.len()].copy_from_slice(name_bytes);
    pos += name_bytes.len();
    data[pos..pos + datatype.len()].copy_from_slice(&datatype);
    pos += datatype.len();
    data[pos..pos + dataspace.len()].copy_from_slice(&dataspace);
    pos += dataspace.len();
    LittleEndian::write_u32(&mut data[pos..pos + 4], value);
    data
}

fn make_hard_link_message(name: &str, target_address: u64) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let mut data = vec![0u8; 2 + 1 + name_bytes.len() + 8];
    data[0] = 1; // version
    data[1] = 0; // 1-byte name length, hard link default type
    data[2] = name_bytes.len() as u8;
    let mut pos = 3;
    data[pos..pos + name_bytes.len()].copy_from_slice(name_bytes);
    pos += name_bytes.len();
    LittleEndian::write_u64(&mut data[pos..pos + 8], target_address);
    data
}

fn make_soft_link_message(name: &str, target: &str) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let target_bytes = target.as_bytes();
    let mut data = vec![0u8; 2 + 1 + 1 + name_bytes.len() + 2 + target_bytes.len()];
    data[0] = 1; // version
    data[1] = 0x08; // link type present
    data[2] = 1; // soft link
    data[3] = name_bytes.len() as u8;
    let mut pos = 4;
    data[pos..pos + name_bytes.len()].copy_from_slice(name_bytes);
    pos += name_bytes.len();
    LittleEndian::write_u16(&mut data[pos..pos + 2], target_bytes.len() as u16);
    pos += 2;
    data[pos..pos + target_bytes.len()].copy_from_slice(target_bytes);
    data
}

fn make_external_link_message(name: &str, file_path: &str, object_path: &str) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let file_bytes = file_path.as_bytes();
    let object_bytes = object_path.as_bytes();

    let payload_len = 1 + file_bytes.len() + 1 + object_bytes.len() + 1;
    let mut data = vec![0u8; 2 + 1 + 1 + name_bytes.len() + 2 + payload_len];
    data[0] = 1; // version
    data[1] = 0x08; // link type present
    data[2] = 64; // external link
    data[3] = name_bytes.len() as u8;

    let mut pos = 4;
    data[pos..pos + name_bytes.len()].copy_from_slice(name_bytes);
    pos += name_bytes.len();

    LittleEndian::write_u16(&mut data[pos..pos + 2], payload_len as u16);
    pos += 2;

    data[pos] = 0;
    pos += 1;

    data[pos..pos + file_bytes.len()].copy_from_slice(file_bytes);
    pos += file_bytes.len();
    data[pos] = 0;
    pos += 1;

    data[pos..pos + object_bytes.len()].copy_from_slice(object_bytes);
    pos += object_bytes.len();
    data[pos] = 0;

    data
}

fn build_v2_object_header(messages: &[(u16, Vec<u8>)]) -> Vec<u8> {
    let chunk_data_size: usize = messages.iter().map(|(_, data)| 5 + data.len()).sum();
    let (chunk_size_width, flags) = if chunk_data_size < 256 {
        (1usize, 0u8)
    } else if chunk_data_size < 65_536 {
        (2usize, 1u8)
    } else if chunk_data_size < (1usize << 32) {
        (4usize, 2u8)
    } else {
        (8usize, 3u8)
    };

    let total = 4 + 1 + 1 + chunk_size_width + chunk_data_size + 4;
    let mut buf = vec![0u8; total];
    buf[0..4].copy_from_slice(b"OHDR");
    buf[4] = 2;
    buf[5] = flags;

    let mut pos = 6;
    match chunk_size_width {
        1 => buf[pos] = chunk_data_size as u8,
        2 => LittleEndian::write_u16(&mut buf[pos..pos + 2], chunk_data_size as u16),
        4 => LittleEndian::write_u32(&mut buf[pos..pos + 4], chunk_data_size as u32),
        8 => LittleEndian::write_u64(&mut buf[pos..pos + 8], chunk_data_size as u64),
        _ => unreachable!(),
    }
    pos += chunk_size_width;

    for (message_type, data) in messages {
        LittleEndian::write_u16(&mut buf[pos..pos + 2], *message_type);
        LittleEndian::write_u16(&mut buf[pos + 2..pos + 4], data.len() as u16);
        buf[pos + 4] = 0;
        pos += 5;
        buf[pos..pos + data.len()].copy_from_slice(data);
        pos += data.len();
    }

    let checksum = consus_compression::Crc32::compute_slice(&buf[..pos]);
    buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());
    buf
}

fn build_compact_group_file() -> (MemCursor, u64, u64) {
    let mut sink = MemCursor::new();
    let mut state = WriteState::new(FileCreationProps::default());

    let superblock_size = 12 + 4 * state.ctx.offset_bytes() + 4;
    state.eof = superblock_size as u64;

    let dataset_values = [10u32, 20, 30, 40];
    let mut dataset_bytes = vec![0u8; dataset_values.len() * 4];
    for (index, value) in dataset_values.iter().enumerate() {
        LittleEndian::write_u32(&mut dataset_bytes[index * 4..index * 4 + 4], *value);
    }

    let data_address =
        write_contiguous_data(&mut sink, &mut state, &dataset_bytes).expect("write data");
    let dataset_address = write_dataset_header(
        &mut sink,
        &mut state,
        &u32_le_datatype(),
        &Shape::fixed(&[2, 2]),
        data_address,
        &DatasetCreationProps {
            layout: DatasetLayout::Contiguous,
            ..DatasetCreationProps::default()
        },
    )
    .expect("write dataset header");

    let subgroup_address =
        write_group_header(&mut sink, &mut state, &GroupCreationProps::default()).expect("group");

    let root_messages = vec![
        (0x0006, make_hard_link_message("data", dataset_address)),
        (0x0006, make_hard_link_message("subgroup", subgroup_address)),
        (0x0006, make_soft_link_message("soft_data", "/data")),
        (
            0x0006,
            make_external_link_message("external_data", "other.h5", "/entry/data"),
        ),
        (0x000C, make_scalar_u32_attribute("answer", 42)),
    ];
    let root_header = build_v2_object_header(&root_messages);
    let root_address = state.allocate_aligned(root_header.len() as u64);
    sink.write_at(root_address, &root_header)
        .expect("write root header");

    write_superblock(&mut sink, &mut state, root_address).expect("write superblock");
    update_superblock_eof(&mut sink, &state).expect("update eof");

    (sink, root_address, dataset_address)
}

#[test]
fn file_api_opens_and_exposes_root_group() {
    let (cursor, root_address, _) = build_compact_group_file();
    let file = Hdf5File::open(cursor).expect("open file");

    assert_eq!(file.superblock().version, 2);
    assert_eq!(file.superblock().root_group_address, root_address);
    assert_eq!(file.context().offset_size, 8);
    assert_eq!(file.context().length_size, 8);

    let root = file.root_group();
    assert_eq!(root.path, "/");
    assert_eq!(root.object_header_address, root_address);

    let root_type = file.root_node_type().expect("classify root");
    assert_eq!(root_type, consus_core::NodeType::Group);
}

#[test]
fn list_root_group_returns_hard_soft_and_external_links() {
    let (cursor, dataset_address, subgroup_address) = {
        let (cursor, root_address, dataset_address) = build_compact_group_file();
        let file = Hdf5File::open(cursor.clone()).expect("open");
        let children = file.list_root_group().expect("list root");
        let subgroup_address = children
            .iter()
            .find(|(name, _, _)| name == "subgroup")
            .map(|(_, address, _)| *address)
            .expect("subgroup link");
        assert_eq!(file.superblock().root_group_address, root_address);
        (cursor, dataset_address, subgroup_address)
    };

    let file = Hdf5File::open(cursor).expect("open file");
    let children = file.list_root_group().expect("list root");
    assert_eq!(children.len(), 4);

    assert!(children.iter().any(|(name, address, link_type)| {
        name == "data" && *address == dataset_address && *link_type == LinkType::Hard
    }));
    assert!(children.iter().any(|(name, address, link_type)| {
        name == "subgroup" && *address == subgroup_address && *link_type == LinkType::Hard
    }));
    assert!(children.iter().any(|(name, address, link_type)| {
        name == "soft_data"
            && *address == consus_hdf5::constants::UNDEFINED_ADDRESS
            && *link_type == LinkType::Soft
    }));
    assert!(children.iter().any(|(name, address, link_type)| {
        name == "external_data"
            && *address == consus_hdf5::constants::UNDEFINED_ADDRESS
            && *link_type == LinkType::External
    }));
}

#[test]
fn dataset_metadata_and_contiguous_bytes_roundtrip() {
    let (cursor, _, dataset_address) = build_compact_group_file();
    let file = Hdf5File::open(cursor).expect("open file");

    let dataset = file.dataset_at(dataset_address).expect("dataset metadata");
    assert_eq!(dataset.object_header_address, dataset_address);
    assert_eq!(
        dataset.layout,
        consus_hdf5::dataset::StorageLayout::Contiguous
    );
    assert_eq!(dataset.shape, Shape::fixed(&[2, 2]));
    assert_eq!(dataset.chunk_shape, None);
    assert!(dataset.filters.is_empty());

    match dataset.datatype {
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            assert_eq!(bits.get(), 32);
            assert_eq!(byte_order, CoreByteOrder::LittleEndian);
            assert!(!signed);
        }
        other => panic!("unexpected datatype: {other:?}"),
    }

    let data_address = dataset.data_address.expect("contiguous address");
    let mut raw = [0u8; 16];
    file.read_contiguous_dataset_bytes(data_address, 0, &mut raw)
        .expect("read raw bytes");

    let values = [
        LittleEndian::read_u32(&raw[0..4]),
        LittleEndian::read_u32(&raw[4..8]),
        LittleEndian::read_u32(&raw[8..12]),
        LittleEndian::read_u32(&raw[12..16]),
    ];
    assert_eq!(values, [10, 20, 30, 40]);
}

#[test]
fn attributes_are_parsed_from_root_object_header() {
    let (cursor, root_address, _) = build_compact_group_file();
    let file = Hdf5File::open(cursor).expect("open file");

    let attrs = file.attributes_at(root_address).expect("attributes");
    assert_eq!(attrs.len(), 1);

    let attr = &attrs[0];
    assert_eq!(attr.name, "answer");
    assert_eq!(attr.shape, Shape::scalar());
    assert_eq!(attr.name_encoding, 0);
    assert_eq!(attr.creation_order, None);
    assert_eq!(attr.raw_data.len(), 4);
    assert_eq!(LittleEndian::read_u32(&attr.raw_data), 42);

    match attr.datatype {
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            assert_eq!(bits.get(), 32);
            assert_eq!(byte_order, CoreByteOrder::LittleEndian);
            assert!(!signed);
        }
        ref other => panic!("unexpected attribute datatype: {other:?}"),
    }
}

#[test]
fn reader_helpers_extract_messages_and_attributes() {
    let (_, _, _) = build_compact_group_file();
    let header = ObjectHeader {
        version: 2,
        messages: vec![
            HeaderMessage {
                message_type: 0x0006,
                data_size: 0,
                flags: 0,
                data: make_hard_link_message("child", 0x1234),
            },
            HeaderMessage {
                message_type: 0x000C,
                data_size: 0,
                flags: 0,
                data: make_scalar_u32_attribute("units", 7),
            },
        ],
    };

    let link = reader::find_message(&header, 0x0006).expect("link message");
    assert_eq!(link.message_type, 0x0006);

    let attrs = reader::read_attributes(
        &MemCursor::from_bytes(vec![]),
        &header,
        &consus_hdf5::address::ParseContext::new(8, 8),
    )
    .expect("read attributes");
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "units");
    assert_eq!(LittleEndian::read_u32(&attrs[0].raw_data), 7);
}

#[test]
fn attribute_parser_roundtrips_scalar_u32_payload() {
    let raw = make_scalar_u32_attribute("scale", 99);
    let attr = Hdf5Attribute::parse(&raw, &consus_hdf5::address::ParseContext::new(8, 8))
        .expect("parse attribute");

    assert_eq!(attr.name, "scale");
    assert_eq!(attr.shape, Shape::scalar());
    assert_eq!(attr.raw_data.len(), 4);
    assert_eq!(LittleEndian::read_u32(&attr.raw_data), 99);
}

#[test]
fn hdf5_file_builder_produces_openable_file() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, NodeType, Shape};
    use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
    use core::num::NonZeroUsize;

    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[4]);
    let raw: Vec<u8> = [1u32, 2, 3, 4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "values",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");
    let bytes = builder.finish().expect("finish");

    let cursor = MemCursor::from_bytes(bytes);
    let file = Hdf5File::open(cursor).expect("open builder-produced file");

    assert_eq!(file.superblock().version, 2);
    let root_type = file.root_node_type().expect("classify root");
    assert_eq!(root_type, NodeType::Group);

    let children = file.list_root_group().expect("list root");
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].0, "values");
}

#[test]
fn hdf5_file_builder_with_root_attributes() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};
    use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
    use core::num::NonZeroUsize;

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let attr_dt = Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let attr_raw = 100u64.to_le_bytes();
    builder
        .add_root_attribute("version", &attr_dt, &Shape::scalar(), &attr_raw)
        .expect("add root attribute");
    let bytes = builder.finish().expect("finish");

    let cursor = MemCursor::from_bytes(bytes);
    let file = Hdf5File::open(cursor).expect("open");

    let root_addr = file.superblock().root_group_address;
    let attrs = file.attributes_at(root_addr).expect("read attributes");
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "version");
    assert_eq!(LittleEndian::read_u64(&attrs[0].raw_data), 100);
}

#[test]
fn open_path_navigates_to_child() {
    let (cursor, _, dataset_address) = build_compact_group_file();
    let file = Hdf5File::open(cursor).expect("open");

    // Navigate to "data" (hard link to dataset).
    let addr = file.open_path("/data").expect("open /data");
    assert_eq!(addr, dataset_address);

    // Navigate to "subgroup".
    let subgroup_addr = file.open_path("subgroup").expect("open subgroup");
    // Just verify we get a non-zero address that differs from dataset.
    assert_ne!(subgroup_addr, 0);
    assert_ne!(subgroup_addr, dataset_address);
}

#[test]
fn open_path_returns_not_found_for_missing() {
    let (cursor, _, _) = build_compact_group_file();
    let file = Hdf5File::open(cursor).expect("open");
    let result = file.open_path("/nonexistent_group/dataset");
    assert!(result.is_err());
    match result.unwrap_err() {
        consus_core::Error::NotFound { .. } => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn list_group_at_matches_list_root_group() {
    let (cursor, root_address, _) = build_compact_group_file();
    let file = Hdf5File::open(cursor).expect("open");

    let via_root = file.list_root_group().expect("list root");
    let via_at = file.list_group_at(root_address).expect("list at");

    assert_eq!(via_root.len(), via_at.len());
    for (a, b) in via_root.iter().zip(via_at.iter()) {
        assert_eq!(a.0, b.0); // same names
        assert_eq!(a.1, b.1); // same addresses
    }
}

#[test]
fn attribute_decode_value_u32() {
    let raw = make_scalar_u32_attribute("temperature", 273);
    let attr = consus_hdf5::attribute::Hdf5Attribute::parse(
        &raw,
        &consus_hdf5::address::ParseContext::new(8, 8),
    )
    .expect("parse");

    let value = attr.decode_value().expect("decode");
    match value {
        consus_core::AttributeValue::Uint(v) => assert_eq!(v, 273),
        other => panic!("expected Uint(273), got {other:?}"),
    }
}

#[test]
fn dataset_with_attributes_roundtrip() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};
    use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
    use core::num::NonZeroUsize;

    let data_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let attr_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };

    let data_shape = Shape::fixed(&[2]);
    let attr_shape = Shape::scalar();
    let raw_data: Vec<u8> = [7u32, 13].iter().flat_map(|v| v.to_le_bytes()).collect();
    let attr_data = 42u32.to_le_bytes();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "obs",
            &data_dt,
            &data_shape,
            &raw_data,
            &DatasetCreationProps::default(),
            &[("scale", &attr_dt, &attr_shape, &attr_data)],
        )
        .expect("add dataset with attrs");

    let bytes = builder.finish().expect("finish");
    let cursor = MemCursor::from_bytes(bytes);
    let file = Hdf5File::open(cursor).expect("open");

    let children = file.list_root_group().expect("list root");
    assert_eq!(children.len(), 1);
    let (_, dataset_addr, _) = &children[0];

    let attrs = file.attributes_at(*dataset_addr).expect("read attrs");
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "scale");
    assert_eq!(
        LittleEndian::read_u64(&{
            let mut pad = [0u8; 8];
            pad[..4].copy_from_slice(&attrs[0].raw_data);
            pad
        }),
        42
    );
}

#[test]
fn open_path_resolves_soft_link() {
    // build_compact_group_file creates:
    //   hard link  data       -> dataset_address
    //   soft link  soft_data  -> /data  (absolute)
    // Resolving /soft_data must return the same address as /data.
    let (cursor, _, dataset_address) = build_compact_group_file();
    let file = Hdf5File::open(cursor).expect("open");

    let soft_addr = file
        .open_path("/soft_data")
        .expect("soft link must resolve");
    assert_eq!(
        soft_addr, dataset_address,
        "soft_data must resolve to the same address as data"
    );

    // Navigating through an external link must return UnsupportedFeature.
    let ext_result = file.open_path("/external_data");
    match ext_result {
        Err(consus_core::Error::UnsupportedFeature { .. }) => {}
        other => panic!(
            "expected UnsupportedFeature for external link, got {:?}",
            other
        ),
    }
}
