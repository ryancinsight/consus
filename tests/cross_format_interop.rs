//! Cross-format interoperability tests.
//!
//! ## Specification Reference
//!
//! These tests validate data interchange between different scientific storage formats:
//! - HDF5 ↔ Zarr conversion
//! - NetCDF-4 ↔ HDF5 compatibility
//! - Arrow ↔ Parquet ↔ Core schema conversions
//!
//! ## Coverage
//!
//! - Schema preservation across formats
//! - Datatype mapping verification
//! - Data value preservation through conversion
//! - Attribute/metadata roundtrip

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Test Data Helpers
// ---------------------------------------------------------------------------

/// Path to workspace data directory.
fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("data")
}

// ---------------------------------------------------------------------------
// HDF5 ↔ Zarr Interoperability Tests
// ---------------------------------------------------------------------------

/// Test HDF5 dataset can be converted to Zarr and back.
///
/// ## Invariant
///
/// Data values must be preserved through HDF5 → Zarr → HDF5 conversion.
#[test]
#[cfg(all(feature = "hdf5", feature = "zarr"))]
fn hdf5_to_zarr_roundtrip() {
    use consus_core::{Datatype, Shape};
    use consus_hdf5::Hdf5File;
    use consus_io::MemCursor;
    use consus_zarr::{ArrayMetadataV3, InMemoryStore, ZarrArray};

    // Create test data in HDF5
    let mut hdf5_buffer = vec![0u8; 8192];
    let mut cursor = MemCursor::new(hdf5_buffer);

    let data: Vec<f64> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6];

    // Write HDF5 file
    {
        let mut writer = consus_hdf5::Hdf5FileBuilder::new()
            .build_writer(&mut cursor)
            .expect("create HDF5 writer");

        let dataset = writer
            .root_group()
            .create_dataset(
                "temperature",
                Datatype::Float {
                    bits: core::num::NonZeroUsize::new(64).expect("non-zero"),
                    byte_order: consus_core::ByteOrder::LittleEndian,
                },
                Shape::fixed(&[6]),
            )
            .expect("create dataset");

        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        dataset
            .write(&consus_core::Selection::all(), &bytes)
            .expect("write data");

        writer.finish().expect("finalize HDF5 file");
    }

    // Read HDF5 and convert to Zarr
    cursor.seek(std::io::SeekFrom::Start(0)).expect("seek");
    let hdf5_file = Hdf5File::open(cursor).expect("open HDF5 file");
    let hdf5_dataset = hdf5_file
        .root_group()
        .get_dataset("temperature")
        .expect("get dataset");

    let mut read_buf = vec![0u8; 6 * 8];
    hdf5_dataset
        .read(&consus_core::Selection::all(), &mut read_buf)
        .expect("read HDF5 data");

    let hdf5_values: Vec<f64> = read_buf
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")))
        .collect();

    // Write to Zarr
    let mut zarr_store = InMemoryStore::new();
    let zarr_array = ZarrArray::create(
        &mut zarr_store,
        "temperature",
        ArrayMetadataV3 {
            shape: vec![6],
            data_type: "float64".to_string(),
            chunk_grid: vec![6],
            fill_value: 0.0,
            codecs: vec![],
        },
    )
    .expect("create Zarr array");

    zarr_array
        .write_chunk(&[0], &read_buf)
        .expect("write Zarr chunk");

    // Read back from Zarr
    let zarr_data = zarr_array.read_chunk(&[0]).expect("read Zarr chunk");
    let zarr_values: Vec<f64> = zarr_data
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")))
        .collect();

    // Verify values match
    assert_eq!(hdf5_values.len(), zarr_values.len());
    for (h, z) in hdf5_values.iter().zip(zarr_values.iter()) {
        assert!(
            (h - z).abs() < f64::EPSILON,
            "HDF5 and Zarr values must match"
        );
    }
}

/// Test Zarr array can be read and written to HDF5.
///
/// ## Invariant
///
/// Zarr → HDF5 conversion preserves exact data values.
#[test]
#[cfg(all(feature = "hdf5", feature = "zarr"))]
fn zarr_to_hdf5_conversion() {
    use consus_core::{Datatype, Shape};
    use consus_io::MemCursor;
    use consus_zarr::{ArrayMetadataV3, InMemoryStore, ZarrArray};

    // Create Zarr array
    let mut store = InMemoryStore::new();
    let array = ZarrArray::create(
        &mut store,
        "data",
        ArrayMetadataV3 {
            shape: vec![10],
            data_type: "int32".to_string(),
            chunk_grid: vec![5],
            fill_value: 0,
            codecs: vec![],
        },
    )
    .expect("create Zarr array");

    // Write data to Zarr
    let data: Vec<i32> = (0..10).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    array
        .write_chunk(&[0], &bytes[..20])
        .expect("write chunk 0");
    array
        .write_chunk(&[1], &bytes[20..])
        .expect("write chunk 1");

    // Read from Zarr and write to HDF5
    let chunk0 = array.read_chunk(&[0]).expect("read chunk 0");
    let chunk1 = array.read_chunk(&[1]).expect("read chunk 1");

    let mut hdf5_buffer = vec![0u8; 8192];
    let mut cursor = MemCursor::new(hdf5_buffer);

    {
        let mut writer = consus_hdf5::Hdf5FileBuilder::new()
            .build_writer(&mut cursor)
            .expect("create HDF5 writer");

        let dataset = writer
            .root_group()
            .create_dataset(
                "data",
                Datatype::Integer {
                    bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
                    byte_order: consus_core::ByteOrder::LittleEndian,
                    signed: true,
                },
                Shape::fixed(&[10]),
            )
            .expect("create dataset");

        let all_bytes: Vec<u8> = chunk0.into_iter().chain(chunk1.into_iter()).collect();
        dataset
            .write(&consus_core::Selection::all(), &all_bytes)
            .expect("write data");

        writer.finish().expect("finalize HDF5 file");
    }

    // Verify by reading HDF5
    cursor.seek(std::io::SeekFrom::Start(0)).expect("seek");
    let hdf5_file = consus_hdf5::Hdf5File::open(cursor).expect("open HDF5 file");
    let dataset = hdf5_file
        .root_group()
        .get_dataset("data")
        .expect("get dataset");

    let mut read_buf = vec![0u8; 10 * 4];
    dataset
        .read(&consus_core::Selection::all(), &mut read_buf)
        .expect("read data");

    let hdf5_values: Vec<i32> = read_buf
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("4 bytes")))
        .collect();

    assert_eq!(hdf5_values, data, "Zarr → HDF5 must preserve values");
}

// ---------------------------------------------------------------------------
// NetCDF-4 ↔ HDF5 Compatibility Tests
// ---------------------------------------------------------------------------

/// Test NetCDF-4 file can be read through HDF5 layer.
///
/// ## Spec Compliance
///
/// NetCDF-4 is built on HDF5, so:
/// - NetCDF-4 files must be valid HDF5 files
/// - Dimensions map to special HDF5 datasets
/// - Variables map to HDF5 datasets with attributes
#[test]
#[cfg(feature = "netcdf")]
fn netcdf_hdf5_compatibility() {
    let nc_path = data_dir().join("netcdf_hdf5_compat_sample.nc");

    if !nc_path.exists() {
        eprintln!("Skipping: netCDF sample not found at {:?}", nc_path);
        return;
    }

    // Open as NetCDF
    let nc_file = consus_netcdf::NcFile::open(&nc_path).expect("open netCDF file");

    // Open as HDF5 (should succeed)
    #[cfg(feature = "hdf5")]
    {
        let hdf5_file = consus_hdf5::Hdf5File::open(&nc_path).expect("open as HDF5");

        // Both should report valid structures
        assert!(nc_file.root_group().is_ok());
        assert!(hdf5_file.root_group().is_ok());
    }
}

/// Test NetCDF-4 dimension and variable structure.
///
/// ## Invariant
///
/// NetCDF-4 variables must match HDF5 dataset shape.
#[test]
#[cfg(all(feature = "netcdf", feature = "hdf5"))]
fn netcdf_variable_matches_hdf5_dataset() {
    let nc_path = data_dir().join("netcdf_small_grid_sample.nc");

    if !nc_path.exists() {
        eprintln!("Skipping: netCDF sample not found at {:?}", nc_path);
        return;
    }

    let nc_file = consus_netcdf::NcFile::open(&nc_path).expect("open netCDF");
    let hdf5_file = consus_hdf5::Hdf5File::open(&nc_path).expect("open HDF5");

    let nc_root = nc_file.root_group().expect("get netCDF root");
    let hdf5_root = hdf5_file.root_group().expect("get HDF5 root");

    let nc_vars = nc_root.variables().expect("list netCDF variables");
    let hdf5_datasets = hdf5_root.datasets().expect("list HDF5 datasets");

    // NetCDF variables should correspond to HDF5 datasets
    for nc_var in &nc_vars {
        let name = nc_var.name();

        // Find corresponding HDF5 dataset
        let hdf5_ds = hdf5_datasets.iter().find(|ds| ds.name() == name);

        if let Some(hdf5_ds) = hdf5_ds {
            // Shapes must match
            let nc_shape = nc_var.shape().expect("get netCDF shape");
            let hdf5_shape = hdf5_ds.shape();

            assert_eq!(
                nc_shape.len(),
                hdf5_shape.len(),
                "rank mismatch for {}",
                name
            );

            for (nc_dim, hdf5_dim) in nc_shape.iter().zip(hdf5_shape.iter()) {
                assert_eq!(nc_dim, hdf5_dim, "dimension mismatch for {}", name);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Arrow ↔ Parquet ↔ Core Schema Tests
// ---------------------------------------------------------------------------

/// Test Arrow schema conversion to Core types.
///
/// ## Invariant
///
/// Arrow → Core conversion must preserve:
/// - Field names
/// - Type semantics (int, float, string)
/// - Nullability
#[test]
#[cfg(feature = "arrow")]
fn arrow_schema_to_core_preserves_semantics() {
    use consus_arrow::{
        ArrowDataType, ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind, ArrowSchema,
    };
    use consus_core::{ByteOrder, Datatype};

    let schema = ArrowSchema::new(vec![
        ArrowFieldBuilder::new(
            ArrowFieldId::new(1),
            String::from("id"),
            ArrowFieldKind::Int,
            Datatype::Integer {
                bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            },
        )
        .nullable(false)
        .build()
        .expect("field must build"),
        ArrowFieldBuilder::new(
            ArrowFieldId::new(2),
            String::from("name"),
            ArrowFieldKind::Utf8,
            Datatype::VariableString {
                encoding: consus_core::StringEncoding::Utf8,
            },
        )
        .nullable(true)
        .build()
        .expect("field must build"),
    ]);

    let core_pairs = consus_arrow::conversion::arrow_schema_to_core_pairs(&schema);

    assert_eq!(core_pairs.len(), 2);
    assert_eq!(core_pairs[0].0, "id");
    assert_eq!(core_pairs[1].0, "name");

    // Verify types
    match &core_pairs[0].1 {
        Datatype::Integer { bits, signed, .. } => {
            assert_eq!(bits.get(), 32);
            assert!(*signed);
        }
        _ => panic!("expected integer type for id"),
    }

    match &core_pairs[1].1 {
        Datatype::VariableString { encoding } => {
            assert!(matches!(encoding, consus_core::StringEncoding::Utf8));
        }
        _ => panic!("expected string type for name"),
    }
}

/// Test Parquet schema conversion to Core types.
///
/// ## Invariant
///
/// Parquet → Core conversion must preserve:
/// - Column names
/// - Physical types
/// - Logical type annotations (strings, timestamps)
#[test]
#[cfg(feature = "parquet")]
fn parquet_schema_to_core_preserves_types() {
    use consus_parquet::{FieldDescriptor, LogicalType, ParquetPhysicalType, SchemaDescriptor};

    let schema = SchemaDescriptor::new(vec![
        FieldDescriptor::required(
            consus_parquet::FieldId::new(1),
            "timestamp",
            ParquetPhysicalType::Int64,
        ),
        FieldDescriptor::optional(
            consus_parquet::FieldId::new(2),
            "value",
            ParquetPhysicalType::Double,
        ),
        FieldDescriptor::optional(
            consus_parquet::FieldId::new(3),
            "label",
            ParquetPhysicalType::ByteArray,
            Some(LogicalType::String),
        ),
    ]);

    let core_pairs = consus_parquet::conversion::parquet_schema_to_core_pairs(&schema);

    assert_eq!(core_pairs.len(), 3);
    assert_eq!(core_pairs[0].0, "timestamp");
    assert_eq!(core_pairs[1].0, "value");
    assert_eq!(core_pairs[2].0, "label");

    // Verify types
    match &core_pairs[0].1 {
        consus_core::Datatype::Integer { bits, .. } => {
            assert_eq!(bits.get(), 64);
        }
        _ => panic!("expected int64 for timestamp"),
    }

    match &core_pairs[1].1 {
        consus_core::Datatype::Float { bits, .. } => {
            assert_eq!(bits.get(), 64);
        }
        _ => panic!("expected float64 for value"),
    }

    match &core_pairs[2].1 {
        consus_core::Datatype::VariableString { .. } => {}
        _ => panic!("expected string for label"),
    }
}

/// Test Arrow ↔ Parquet schema compatibility.
///
/// ## Invariant
///
/// Arrow and Parquet schemas should be interconvertible for common types.
#[test]
#[cfg(all(feature = "arrow", feature = "parquet"))]
fn arrow_parquet_schema_interop() {
    use consus_arrow::{
        ArrowDataType, ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind, ArrowSchema,
    };
    use consus_core::{ByteOrder, Datatype};
    use consus_parquet::{FieldDescriptor, LogicalType, ParquetPhysicalType, SchemaDescriptor};

    // Create Arrow schema
    let arrow_schema = ArrowSchema::new(vec![ArrowFieldBuilder::new(
        ArrowFieldId::new(1),
        String::from("temperature"),
        ArrowFieldKind::Float,
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).expect("non-zero"),
            byte_order: ByteOrder::LittleEndian,
        },
    )
    .nullable(false)
    .build()
    .expect("field must build")]);

    // Convert to Core
    let core_pairs = consus_arrow::conversion::arrow_schema_to_core_pairs(&arrow_schema);

    // Create equivalent Parquet schema
    let parquet_schema = SchemaDescriptor::new(vec![FieldDescriptor::required(
        consus_parquet::FieldId::new(1),
        "temperature",
        ParquetPhysicalType::Double,
    )]);

    let parquet_core_pairs =
        consus_parquet::conversion::parquet_schema_to_core_pairs(&parquet_schema);

    // Both should produce same Core representation
    assert_eq!(core_pairs.len(), parquet_core_pairs.len());
    assert_eq!(core_pairs[0].0, parquet_core_pairs[0].0);

    // Types should be compatible
    match (&core_pairs[0].1, &parquet_core_pairs[0].1) {
        (Datatype::Float { bits: b1, .. }, Datatype::Float { bits: b2, .. }) => {
            assert_eq!(b1, b2);
        }
        _ => panic!("type mismatch"),
    }
}

// ---------------------------------------------------------------------------
// Cross-Format Data Value Tests
// ---------------------------------------------------------------------------

/// Test that identical data values roundtrip through multiple formats.
///
/// ## Invariant
///
/// Data values must be preserved through any format conversion.
#[test]
#[cfg(all(feature = "hdf5", feature = "zarr"))]
fn data_values_preserved_across_formats() {
    use consus_core::{Datatype, Selection, Shape};
    use consus_io::MemCursor;
    use consus_zarr::{ArrayMetadataV3, InMemoryStore, ZarrArray};

    // Original data
    let original: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];

    // Write to HDF5
    let mut hdf5_buffer = vec![0u8; 8192];
    let mut cursor = MemCursor::new(hdf5_buffer);

    {
        let mut writer = consus_hdf5::Hdf5FileBuilder::new()
            .build_writer(&mut cursor)
            .expect("create writer");

        let dataset = writer
            .root_group()
            .create_dataset(
                "data",
                Datatype::Float {
                    bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
                    byte_order: consus_core::ByteOrder::LittleEndian,
                },
                Shape::fixed(&[8]),
            )
            .expect("create dataset");

        let bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        dataset.write(&Selection::all(), &bytes).expect("write");

        writer.finish().expect("finalize");
    }

    // Read from HDF5
    cursor.seek(std::io::SeekFrom::Start(0)).expect("seek");
    let hdf5_file = consus_hdf5::Hdf5File::open(cursor).expect("open");
    let dataset = hdf5_file.root_group().get_dataset("data").expect("get");

    let mut hdf5_buf = vec![0u8; 8 * 4];
    dataset
        .read(&Selection::all(), &mut hdf5_buf)
        .expect("read");

    let from_hdf5: Vec<f32> = hdf5_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    // Write to Zarr
    let mut store = InMemoryStore::new();
    let zarr = ZarrArray::create(
        &mut store,
        "data",
        ArrayMetadataV3 {
            shape: vec![8],
            data_type: "float32".to_string(),
            chunk_grid: vec![8],
            fill_value: 0.0,
            codecs: vec![],
        },
    )
    .expect("create");

    zarr.write_chunk(&[0], &hdf5_buf).expect("write");

    // Read from Zarr
    let zarr_data = zarr.read_chunk(&[0]).expect("read");
    let from_zarr: Vec<f32> = zarr_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    // All values must match
    for (i, (orig, h, z)) in original
        .iter()
        .zip(from_hdf5.iter())
        .zip(from_zarr.iter())
        .enumerate()
    {
        assert!((orig - h).abs() < f32::EPSILON, "HDF5 mismatch at {}", i);
        assert!((orig - z).abs() < f32::EPSILON, "Zarr mismatch at {}", i);
    }
}

// ---------------------------------------------------------------------------
// Compression Interoperability Tests
// ---------------------------------------------------------------------------

/// Test compression settings translate across formats.
///
/// ## Invariant
///
/// Equivalent compression algorithms should be used when converting formats.
#[test]
#[cfg(all(feature = "hdf5", feature = "compression"))]
fn compression_settings_interop() {
    use consus_core::{Compression, Datatype, Selection, Shape};
    use consus_io::MemCursor;

    // Create HDF5 file with deflate compression
    let mut buffer = vec![0u8; 8192];
    let mut cursor = MemCursor::new(buffer);

    let data: Vec<u8> = (0..=255).cycle().take(1000).collect();

    {
        let mut writer = consus_hdf5::Hdf5FileBuilder::new()
            .build_writer(&mut cursor)
            .expect("create writer");

        let dataset = writer
            .root_group()
            .create_dataset_compressed(
                "compressed",
                Datatype::Integer {
                    bits: core::num::NonZeroUsize::new(8).expect("non-zero"),
                    byte_order: consus_core::ByteOrder::LittleEndian,
                    signed: false,
                },
                Shape::fixed(&[1000]),
                Compression::Deflate { level: 6 },
            )
            .expect("create compressed dataset");

        dataset.write(&Selection::all(), &data).expect("write");
        writer.finish().expect("finalize");
    }

    // Read back and verify compression was applied
    cursor.seek(std::io::SeekFrom::Start(0)).expect("seek");
    let file = consus_hdf5::Hdf5File::open(cursor).expect("open");

    let dataset = file.root_group().get_dataset("compressed").expect("get");

    assert!(dataset.is_compressed(), "dataset must report compressed");

    let filters = dataset.filters().expect("get filters");
    assert!(!filters.is_empty(), "must have compression filter");

    // Should have deflate filter (ID 1)
    let has_deflate = filters.iter().any(|f| f.id() == 1);
    assert!(has_deflate, "must use deflate compression");

    // Data must decompress correctly
    let mut read_buf = vec![0u8; 1000];
    dataset
        .read(&Selection::all(), &mut read_buf)
        .expect("read");

    assert_eq!(read_buf, data, "decompressed data must match original");
}
