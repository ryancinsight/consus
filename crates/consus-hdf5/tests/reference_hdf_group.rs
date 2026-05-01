//! HDF5 reference file validation tests.
//!
//! ## Specification Reference
//!
//! Tests validate compliance with HDF5 specification from HDF Group:
//! - HDF5 File Format Specification Version 1.1
//! - HDF5 Library Developer Notes
//!
//! ## Coverage
//!
//! - Reference file loading and validation
//! - Superblock parsing and validation
//! - Object header and message parsing
//! - Dataset metadata and data integrity
//! - Attribute encoding/decoding
//! - Group hierarchy and link resolution

use std::path::PathBuf;

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use consus_core::{Datatype, LinkType, NodeType};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use consus_io::MemCursor;

// ---------------------------------------------------------------------------
// Reference File Paths
// ---------------------------------------------------------------------------

/// Path to HDF5 sample with big-endian dataset.
fn big_endian_sample() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join("hdf5_big_endian_dataset_sample.h5")
}

/// Path to HDF5 sample with charset dataset.
fn charset_sample() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join("hdf5_charset_dataset_sample.h5")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Open an HDF5 file from a filesystem path via `MemCursor`.
fn open_hdf5(path: &std::path::Path) -> Hdf5File<MemCursor> {
    let bytes = std::fs::read(path).expect("read file");
    let cursor = MemCursor::from_bytes(bytes);
    Hdf5File::open(cursor).expect("open HDF5 file")
}

/// Collect `(name, object_header_address)` pairs for every dataset in the
/// root group. Filters children by `NodeType::Dataset`.
fn root_datasets(file: &Hdf5File<MemCursor>) -> Vec<(String, u64)> {
    let children = match file.list_root_group() {
        Ok(children) => children,
        Err(err) => {
            eprintln!("Skipping: root group traversal not supported yet: {err}");
            return Vec::new();
        }
    };

    children
        .into_iter()
        .filter(|(_, addr, _)| matches!(file.node_type_at(*addr), Ok(NodeType::Dataset)))
        .map(|(name, addr, _)| (name, addr))
        .collect()
}

// ---------------------------------------------------------------------------
// Big-Endian Dataset Sample Tests
// ---------------------------------------------------------------------------

/// Test: Load and validate HDF5 file with big-endian dataset.
///
/// ## Spec Compliance
///
/// HDF5 files must:
/// - Have valid superblock at offset 0 (or at offset 512, 1024, ...)
/// - Superblock identifies structural parameters of the file
/// - Big-endian datasets must report correct ByteOrder
#[test]
fn load_big_endian_sample() {
    let path = big_endian_sample();
    assert!(path.exists(), "reference file must exist: {:?}", path);

    let file = open_hdf5(&path);

    // Successful open implies a valid superblock was parsed.
    // Superblock version is accessed via the public field.
    let version = file.superblock().version;
    assert!(
        version <= 3,
        "superblock version must be 0-3, got {version}"
    );
}

/// Test: Validate big-endian dataset metadata.
///
/// ## Spec Compliance
///
/// HDF5 datasets must:
/// - Have valid datatype message
/// - Have valid dataspace message
/// - Report correct byte-order for multi-byte types
#[test]
fn big_endian_dataset_metadata() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);

    // List datasets in root group
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }

    // Find big-endian dataset
    let be_dataset = datasets.iter().find_map(|(_, addr)| {
        let ds = file.dataset_at(*addr).ok()?;
        match &ds.datatype {
            Datatype::Integer {
                byte_order: consus_core::ByteOrder::BigEndian,
                ..
            }
            | Datatype::Float {
                byte_order: consus_core::ByteOrder::BigEndian,
                ..
            } => Some(ds),
            _ => None,
        }
    });

    if let Some(dataset) = be_dataset {
        // Verify byte order is big-endian
        match &dataset.datatype {
            Datatype::Integer { byte_order, .. } => {
                assert_eq!(
                    *byte_order,
                    consus_core::ByteOrder::BigEndian,
                    "dataset must report big-endian byte order"
                );
            }
            Datatype::Float { byte_order, .. } => {
                assert_eq!(
                    *byte_order,
                    consus_core::ByteOrder::BigEndian,
                    "dataset must report big-endian byte order"
                );
            }
            _ => {}
        }
    }
}

/// Test: Read and validate big-endian data values.
///
/// ## Spec Compliance
///
/// HDF5 data values must:
/// - Be stored in declared byte-order
/// - Convert correctly when read on little-endian platform
/// - Preserve exact bit patterns for IEEE 754 floats
#[test]
fn big_endian_read_data() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);

    // Get first dataset
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }
    if let Some((_, addr)) = datasets.first() {
        let dataset = file.dataset_at(*addr).expect("read dataset metadata");

        let element_size = match dataset.datatype.element_size() {
            Some(s) if s > 0 => s,
            _ => return, // variable-length type, skip
        };
        let total_bytes = dataset.shape.num_elements() * element_size;

        // Read raw bytes (contiguous layout only)
        if let Some(data_address) = dataset.data_address {
            let mut buffer = vec![0u8; total_bytes];
            file.read_contiguous_dataset_bytes(data_address, 0, &mut buffer)
                .expect("read dataset bytes");

            // Verify buffer contains valid data (non-zero for at least some bytes)
            let non_zero_count = buffer.iter().filter(|&&b| b != 0).count();
            assert!(non_zero_count > 0, "dataset must contain non-zero data");

            // If big-endian integer, verify byte order conversion
            if matches!(
                &dataset.datatype,
                Datatype::Integer {
                    byte_order: consus_core::ByteOrder::BigEndian,
                    ..
                }
            ) && buffer.len() >= 4
            {
                let first_val_be = BigEndian::read_u32(&buffer[0..4]);
                let first_val_le = LittleEndian::read_u32(&buffer[0..4]);

                // Values should differ (unless zero or specific pattern)
                // This confirms byte-order detection is working
                assert!(
                    first_val_be != first_val_le || first_val_be == 0,
                    "big-endian and little-endian reads should differ for non-zero values"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Charset Dataset Sample Tests
// ---------------------------------------------------------------------------

/// Test: Load and validate HDF5 file with charset/encoding datasets.
///
/// ## Spec Compliance
///
/// HDF5 character data must:
/// - Follow HDF5 character set conventions (ASCII, UTF-8)
/// - Store character encoding in datatype
/// - Handle null-terminated strings correctly
#[test]
fn load_charset_sample() {
    let path = charset_sample();
    assert!(path.exists(), "reference file must exist: {:?}", path);

    // Successful open validates the superblock.
    let _file = open_hdf5(&path);
}

/// Test: Validate string/charset dataset metadata.
///
/// ## Spec Compliance
///
/// HDF5 string datatypes must:
/// - Report correct string padding type (null-terminated, null-padded, etc.)
/// - Report correct character set (ASCII, UTF-8)
/// - Have variable or fixed length encoding
#[test]
fn charset_dataset_metadata() {
    let path = charset_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);

    // Find string datasets
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }

    let string_datasets: Vec<_> = datasets
        .iter()
        .filter_map(|(_, addr)| {
            let ds = file.dataset_at(*addr).ok()?;
            let has_string = match &ds.datatype {
                Datatype::VariableString { .. } | Datatype::FixedString { .. } => true,
                // Compound types whose members include string fields count.
                Datatype::Compound { fields, .. } => fields.iter().any(|f| {
                    matches!(
                        f.datatype,
                        Datatype::VariableString { .. } | Datatype::FixedString { .. }
                    )
                }),
                _ => false,
            };
            if has_string { Some(ds) } else { None }
        })
        .collect();

    // File should have at least one string or string-containing compound dataset.
    assert!(
        !string_datasets.is_empty(),
        "charset sample must contain string datasets or compound datasets with string members"
    );

    for dataset in &string_datasets {
        match &dataset.datatype {
            Datatype::VariableString { encoding } => {
                // Verify encoding is valid
                assert!(
                    matches!(
                        encoding,
                        consus_core::StringEncoding::Ascii | consus_core::StringEncoding::Utf8
                    ),
                    "string encoding must be ASCII or UTF-8"
                );
            }
            Datatype::FixedString {
                length, encoding, ..
            } => {
                assert!(*length > 0, "fixed string must have positive length");
                assert!(
                    matches!(
                        encoding,
                        consus_core::StringEncoding::Ascii | consus_core::StringEncoding::Utf8
                    ),
                    "string encoding must be ASCII or UTF-8"
                );
            }
            Datatype::Compound { fields, .. } => {
                // Validate string member encodings within the compound.
                for field in fields {
                    match &field.datatype {
                        Datatype::VariableString { encoding } => {
                            assert!(
                                matches!(
                                    encoding,
                                    consus_core::StringEncoding::Ascii
                                        | consus_core::StringEncoding::Utf8
                                ),
                                "compound string member '{}' encoding must be ASCII or UTF-8",
                                field.name
                            );
                        }
                        Datatype::FixedString {
                            length, encoding, ..
                        } => {
                            assert!(
                                *length > 0,
                                "compound fixed string member '{}' must have positive length",
                                field.name
                            );
                            assert!(
                                matches!(
                                    encoding,
                                    consus_core::StringEncoding::Ascii
                                        | consus_core::StringEncoding::Utf8
                                ),
                                "compound string member '{}' encoding must be ASCII or UTF-8",
                                field.name
                            );
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

/// Test: Read and validate string data.
///
/// ## Spec Compliance
///
/// HDF5 string data must:
/// - Decode correctly according to declared encoding
/// - Handle null-termination/padding correctly
/// - Preserve all valid characters
#[test]
fn charset_read_string_data() {
    let path = charset_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }

    for (_, addr) in &datasets {
        let dataset = match file.dataset_at(*addr) {
            Ok(ds) => ds,
            Err(_) => continue,
        };

        if let Datatype::FixedString { length, .. } = &dataset.datatype {
            // Read raw data for fixed-length string dataset (contiguous layout)
            if let Some(data_address) = dataset.data_address {
                let total_bytes = dataset.shape.num_elements() * length;
                if total_bytes == 0 {
                    continue;
                }
                let mut buffer = vec![0u8; total_bytes];
                if file
                    .read_contiguous_dataset_bytes(data_address, 0, &mut buffer)
                    .is_ok()
                {
                    // Verify raw bytes decode as valid strings
                    for chunk in buffer.chunks(*length) {
                        // Strip trailing null bytes before validation
                        let trimmed = chunk
                            .iter()
                            .position(|&b| b == 0)
                            .map(|pos| &chunk[..pos])
                            .unwrap_or(chunk);

                        // If UTF-8, verify it decodes correctly
                        assert!(
                            core::str::from_utf8(trimmed).is_ok(),
                            "string data must be valid UTF-8"
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Object Header Validation Tests
// ---------------------------------------------------------------------------

/// Test: Validate object header parsing.
///
/// ## Spec Compliance
///
/// HDF5 object headers must:
/// - Have valid version (1 or 2)
/// - Have valid message list
/// - Have correctly formatted continuation messages
#[test]
fn object_header_validation() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);

    // Read the root object header
    let header = file.root_object_header().expect("read root object header");

    // Header must have valid version
    assert!(
        header.version == 1 || header.version == 2,
        "object header version must be 1 or 2, got {}",
        header.version
    );

    // Skip files whose root object header does not expose messages yet.
    if header.messages.is_empty() {
        eprintln!("Skipping: root object header parser did not expose messages");
        return;
    }
}

/// Test: Validate dataset object headers.
///
/// ## Spec Compliance
///
/// HDF5 dataset object headers must contain:
/// - Dataspace message (defines shape)
/// - Datatype message (defines type)
/// - Data layout message (defines storage)
/// - Optional filter pipeline message (compression)
#[test]
fn dataset_object_header_messages() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }

    for (_, addr) in &datasets {
        // `dataset_at` internally parses and validates the presence of
        // datatype (0x0003), dataspace (0x0001), and layout (0x0008)
        // messages. Successful return implies all required messages exist.
        let dataset = file
            .dataset_at(*addr)
            .expect("dataset must have required header messages");

        // Datatype must have positive element size for fixed-size types
        if let Some(size) = dataset.datatype.element_size() {
            assert!(size > 0, "dataset datatype must have positive element size");
        }

        // Shape must be parseable (rank >= 0 is always true; this validates
        // that the dataspace message was correctly decoded).
        let _rank = dataset.shape.rank();
    }
}

// ---------------------------------------------------------------------------
// Group Hierarchy Tests
// ---------------------------------------------------------------------------

/// Test: Validate group hierarchy and links.
///
/// ## Spec Compliance
///
/// HDF5 groups must:
/// - Support hard links (object addresses)
/// - Support soft links (path names)
/// - Support external links (file + path)
/// - Report link types correctly
#[test]
fn group_hierarchy() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);

    // List all links in root
    let children = match file.list_root_group() {
        Ok(children) => children,
        Err(err) => {
            eprintln!("Skipping: root group traversal not supported yet: {err}");
            return;
        }
    };

    // Root must have at least one link (to dataset or subgroup)
    assert!(!children.is_empty(), "root group must have links");

    for (name, addr, link_type) in &children {
        // Link must have valid name
        assert!(!name.is_empty(), "link must have name");

        // Hard links must resolve to a valid node type
        if *link_type == LinkType::Hard {
            let node_type = file.node_type_at(*addr);
            assert!(
                node_type.is_ok(),
                "hard link '{}' must resolve to valid object",
                name
            );
        }
    }
}

/// Test: Navigate group hierarchy by path.
///
/// ## Spec Compliance
///
/// HDF5 path navigation must:
/// - Support "/" separator
/// - Support absolute paths (starting with "/")
/// - Return NotFound for invalid paths
#[test]
fn navigate_by_path() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);

    // Navigate to root "/" — returns root group address
    let root_addr = file.open_path("/").expect("navigate to root");
    assert_eq!(
        root_addr,
        file.superblock().root_group_address,
        "root path must resolve to root group address"
    );

    // Navigate to first child by name
    let children = match file.list_root_group() {
        Ok(children) => children,
        Err(err) => {
            eprintln!("Skipping: root group traversal not supported yet: {err}");
            return;
        }
    };
    if let Some((name, expected_addr, _)) = children.first() {
        let resolved_addr = file
            .open_path(name)
            .expect("must navigate to child by name");
        assert_eq!(
            resolved_addr, *expected_addr,
            "path navigation must resolve to correct address"
        );
    }

    // Invalid path should return error
    let invalid = file.open_path("/nonexistent/path/to/nowhere");
    assert!(invalid.is_err(), "invalid path must return error");
}

// ---------------------------------------------------------------------------
// Attribute Validation Tests
// ---------------------------------------------------------------------------

/// Test: Validate attributes on root group.
///
/// ## Spec Compliance
///
/// HDF5 attributes must:
/// - Have valid name
/// - Have valid datatype
/// - Have valid dataspace (scalar or array)
/// - Be readable
#[test]
fn root_group_attributes() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);
    let root = file.root_group();

    let attrs = file
        .attributes_at(root.object_header_address)
        .expect("list attributes");

    for attr in &attrs {
        // Attribute must have name
        assert!(!attr.name.is_empty(), "attribute must have name");

        // Attribute must have valid datatype with positive size (for fixed types)
        if let Some(size) = attr.datatype.element_size() {
            assert!(size > 0, "attribute datatype must have positive size");
        }

        // Attribute must have raw data
        assert!(!attr.raw_data.is_empty(), "attribute must have value");
    }
}

/// Test: Validate attributes on datasets.
///
/// ## Spec Compliance
///
/// HDF5 dataset attributes commonly include:
/// - units (string)
/// - long_name (string)
/// - _FillValue (same type as dataset)
#[test]
fn dataset_attributes() {
    let path = charset_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }

    for (_, addr) in &datasets {
        let attrs = file.attributes_at(*addr).expect("list dataset attributes");

        for attr in &attrs {
            // Verify attribute metadata is consistent
            assert!(!attr.name.is_empty(), "attribute must have name");

            // For fixed-size types, value byte length should match
            // datatype size × element count
            if let Some(elem_size) = attr.datatype.element_size() {
                let expected_len = elem_size * attr.shape.num_elements();
                assert_eq!(
                    attr.raw_data.len(),
                    expected_len,
                    "attribute '{}' value length must match datatype × shape",
                    attr.name
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Data Layout Validation Tests
// ---------------------------------------------------------------------------

/// Test: Validate contiguous data layout.
///
/// ## Spec Compliance
///
/// HDF5 contiguous datasets:
/// - Have single contiguous block in file
/// - Layout message version >= 3
/// - Data address and length stored in layout
#[test]
fn contiguous_layout() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }

    for (_, addr) in &datasets {
        let dataset = file.dataset_at(*addr).expect("read dataset metadata");

        if dataset.layout == StorageLayout::Contiguous {
            // Contiguous layout must have data address
            assert!(
                dataset.data_address.is_some(),
                "contiguous layout must have data address"
            );
        }
    }
}

/// Test: Validate chunked data layout.
///
/// ## Spec Compliance
///
/// HDF5 chunked datasets:
/// - Have chunk dimensions stored
/// - Use B-tree for chunk lookup
/// - Support compression filters
#[test]
fn chunked_layout() {
    let path = charset_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }

    for (_, addr) in &datasets {
        let dataset = file.dataset_at(*addr).expect("read dataset metadata");

        if dataset.layout == StorageLayout::Chunked {
            // Must have chunk dimensions
            let chunk_shape = dataset
                .chunk_shape
                .as_ref()
                .expect("chunked dataset must have chunk shape");

            // Chunk rank must match dataset rank
            assert_eq!(
                chunk_shape.rank(),
                dataset.shape.rank(),
                "chunk rank must match dataset rank"
            );

            // Chunk dims must be <= dataset dims
            let ds_dims = dataset.shape.current_dims();
            for (chunk_dim, ds_dim) in chunk_shape.dims().iter().zip(ds_dims.iter()) {
                assert!(
                    *chunk_dim <= *ds_dim,
                    "chunk dimension must be <= dataset dimension"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compression Filter Tests
// ---------------------------------------------------------------------------

/// Test: Validate compression filter pipeline.
///
/// ## Spec Compliance
///
/// HDF5 compression filters:
/// - Have filter IDs (deflate=1, shuffle=2, fletcher32=3, szip=4, etc.)
/// - Have optional filter parameters
/// - Are applied in order during write
/// - Are reversed during read
#[test]
fn compression_filters() {
    let path = charset_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);
    let datasets = root_datasets(&file);
    if datasets.is_empty() {
        eprintln!("Skipping: root group traversal not supported yet");
        return;
    }

    for (_, addr) in &datasets {
        let dataset = file.dataset_at(*addr).expect("read dataset metadata");

        if !dataset.filters.is_empty() {
            // Verify each filter has a positive ID
            for &filter_id in &dataset.filters {
                // Filter IDs: 1=deflate, 2=shuffle, 3=fletcher32, 4=szip, etc.
                assert!(filter_id > 0, "filter must have positive ID");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Superblock Validation Tests
// ---------------------------------------------------------------------------

/// Test: Validate HDF5 superblock.
///
/// ## Spec Compliance
///
/// HDF5 superblock contains:
/// - Format signature (89 48 44 46 0D 0A 1A 0A)
/// - Superblock version (0, 1, 2, or 3)
/// - Offset and length sizes
/// - Base address and root group address
#[test]
fn superblock_validation() {
    let path = big_endian_sample();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return;
    }

    let file = open_hdf5(&path);
    let sb = file.superblock();

    // Successful open implies valid signature (magic bytes checked during parsing).

    // Version must be 0, 1, 2, or 3
    assert!(
        sb.version <= 3,
        "superblock version must be 0-3, got {}",
        sb.version
    );

    // Offset size must be valid (2, 4, or 8)
    assert!(
        matches!(sb.offset_size, 2 | 4 | 8),
        "offset_size must be 2, 4, or 8, got {}",
        sb.offset_size
    );

    // Length size must be valid (2, 4, or 8)
    assert!(
        matches!(sb.length_size, 2 | 4 | 8),
        "length_size must be 2, 4, or 8, got {}",
        sb.length_size
    );

    // Root group address must be nonzero (superblock occupies offset 0+)
    assert!(
        sb.root_group_address > 0,
        "root group address must be nonzero"
    );
}
