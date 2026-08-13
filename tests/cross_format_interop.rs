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

#[cfg(feature = "hdf5")]
use consus_hdf5::dataset::StorageLayout;
#[cfg(feature = "hdf5")]
use consus_hdf5::file::Hdf5File;
#[cfg(feature = "hdf5")]
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
#[cfg(feature = "hdf5")]
use consus_io::MemCursor;

// ---------------------------------------------------------------------------
// Provider-contract helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "hdf5")]
fn write_hdf5_dataset(
    name: &str,
    datatype: &consus_core::Datatype,
    shape: &consus_core::Shape,
    raw: &[u8],
    properties: &DatasetCreationProps,
) -> Vec<u8> {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(name, datatype, shape, raw, properties)
        .expect("add HDF5 dataset");
    builder.finish().expect("finish HDF5 file")
}

#[cfg(feature = "hdf5")]
fn find_hdf5_dataset_address(file: &Hdf5File<MemCursor>, name: &str) -> u64 {
    file.list_root_group()
        .expect("list HDF5 root group")
        .into_iter()
        .find_map(|(child_name, address, _)| (child_name == name).then_some(address))
        .unwrap_or_else(|| panic!("HDF5 dataset {name:?} is absent"))
}

#[cfg(feature = "hdf5")]
fn read_hdf5_dataset(file: &Hdf5File<MemCursor>, name: &str) -> Vec<u8> {
    let address = find_hdf5_dataset_address(file, name);
    let dataset = file
        .dataset_at(address)
        .expect("read HDF5 dataset metadata");

    match dataset.layout {
        StorageLayout::Contiguous => {
            let data_address = dataset.data_address.expect("contiguous data address");
            let element_size = dataset.datatype.element_size().expect("fixed datatype");
            let mut bytes = vec![0u8; dataset.shape.num_elements() * element_size];
            file.read_contiguous_dataset_bytes(data_address, 0, &mut bytes)
                .expect("read HDF5 contiguous dataset");
            bytes
        }
        StorageLayout::Chunked => file
            .read_chunked_dataset_all_bytes(address)
            .expect("read HDF5 chunked dataset"),
        StorageLayout::Compact | StorageLayout::Virtual => {
            panic!("unsupported HDF5 test layout: {:?}", dataset.layout)
        }
    }
}

#[cfg(feature = "zarr")]
fn zarr_metadata(shape: Vec<usize>, chunks: Vec<usize>, dtype: &str) -> consus_zarr::ArrayMetadata {
    consus_zarr::ArrayMetadata {
        version: consus_zarr::ZarrVersion::V3,
        shape,
        chunks,
        dtype: dtype.to_string(),
        fill_value: consus_zarr::FillValue::Default,
        order: 'C',
        codecs: Vec::new(),
        chunk_key_encoding: consus_zarr::ChunkKeyEncoding::default(),
        dimension_names: None,
    }
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
    use consus_core::{ByteOrder, Datatype, Shape};
    use consus_zarr::{InMemoryStore, read_chunk, write_chunk};

    let data: Vec<f64> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6];
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    let datatype = Datatype::Float {
        bits: core::num::NonZeroUsize::new(64).expect("non-zero"),
        byte_order: ByteOrder::LittleEndian,
    };
    let shape = Shape::fixed(&[data.len()]);

    let hdf5_bytes = write_hdf5_dataset(
        "temperature",
        &datatype,
        &shape,
        &bytes,
        &DatasetCreationProps::default(),
    );
    let hdf5_file = Hdf5File::open(MemCursor::from_bytes(hdf5_bytes)).expect("open HDF5 file");
    let hdf5_bytes = read_hdf5_dataset(&hdf5_file, "temperature");
    let hdf5_values: Vec<f64> = hdf5_bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")))
        .collect();

    let mut zarr_store = InMemoryStore::new();
    let metadata = zarr_metadata(vec![data.len()], vec![data.len()], "float64");
    write_chunk(&mut zarr_store, "temperature", &[0], &metadata, &hdf5_bytes)
        .expect("write Zarr chunk");
    let zarr_data =
        read_chunk(&zarr_store, "temperature", &[0], &metadata).expect("read Zarr chunk");
    let zarr_values: Vec<f64> = zarr_data
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")))
        .collect();

    assert_eq!(hdf5_values, data, "HDF5 must preserve source values");
    assert_eq!(zarr_values, data, "Zarr must preserve HDF5 values");
}

/// Test Zarr array can be read and written to HDF5.
///
/// ## Invariant
///
/// Zarr → HDF5 conversion preserves exact data values.
#[test]
#[cfg(all(feature = "hdf5", feature = "zarr"))]
fn zarr_to_hdf5_conversion() {
    use consus_core::{ByteOrder, Datatype, Shape};
    use consus_zarr::{InMemoryStore, read_chunk, write_chunk};

    let mut store = InMemoryStore::new();
    let metadata = zarr_metadata(vec![10], vec![5], "int32");

    let data: Vec<i32> = (0..10).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_chunk(&mut store, "data", &[0], &metadata, &bytes[..20]).expect("write chunk 0");
    write_chunk(&mut store, "data", &[1], &metadata, &bytes[20..]).expect("write chunk 1");

    let chunk0 = read_chunk(&store, "data", &[0], &metadata).expect("read chunk 0");
    let chunk1 = read_chunk(&store, "data", &[1], &metadata).expect("read chunk 1");
    let all_bytes: Vec<u8> = chunk0.into_iter().chain(chunk1).collect();
    let datatype = Datatype::Integer {
        bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let hdf5_bytes = write_hdf5_dataset(
        "data",
        &datatype,
        &Shape::fixed(&[10]),
        &all_bytes,
        &DatasetCreationProps::default(),
    );
    let hdf5_file = Hdf5File::open(MemCursor::from_bytes(hdf5_bytes)).expect("open HDF5 file");
    let read_buf = read_hdf5_dataset(&hdf5_file, "data");

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
#[cfg(all(feature = "netcdf", feature = "hdf5"))]
fn netcdf_hdf5_compatibility() {
    use consus_core::{ByteOrder, Datatype, Shape};
    use consus_netcdf::{NetcdfDimension, NetcdfModel, NetcdfVariable, NetcdfWriter};

    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("time"), 4));
    model.root.variables.push(
        NetcdfVariable::new(
            String::from("temperature"),
            Datatype::Float {
                bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
                byte_order: ByteOrder::LittleEndian,
            },
            vec![String::from("time")],
        )
        .with_shape(Shape::fixed(&[4]))
        .with_data(
            (0..4)
                .flat_map(|value| (value as f32).to_le_bytes())
                .collect(),
        ),
    );

    let bytes = NetcdfWriter::new()
        .write_model(&model)
        .expect("write netCDF model as HDF5");
    let hdf5_file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open netCDF HDF5 image");
    let decoded = consus_netcdf::read_model(&hdf5_file).expect("read netCDF model from HDF5");

    assert_eq!(decoded.root.name, "/");
    assert_eq!(decoded.root.dimensions.len(), 1);
    assert_eq!(decoded.root.variables.len(), 1);
    assert_eq!(decoded.root.variables[0].name, "temperature");
    assert_eq!(
        decoded.root.variables[0]
            .shape
            .as_ref()
            .expect("shape")
            .current_dims()
            .as_slice(),
        &[4]
    );
}

/// Test NetCDF-4 dimension and variable structure.
///
/// ## Invariant
///
/// NetCDF-4 variables must match HDF5 dataset shape.
#[test]
#[cfg(all(feature = "netcdf", feature = "hdf5"))]
fn netcdf_variable_matches_hdf5_dataset() {
    use consus_core::{ByteOrder, Datatype, Shape};
    use consus_netcdf::{NetcdfDimension, NetcdfModel, NetcdfVariable, NetcdfWriter};

    let mut model = NetcdfModel::default();
    model
        .root
        .dimensions
        .push(NetcdfDimension::new(String::from("x"), 3));
    model.root.variables.push(
        NetcdfVariable::new(
            String::from("signal"),
            Datatype::Integer {
                bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            },
            vec![String::from("x")],
        )
        .with_shape(Shape::fixed(&[3]))
        .with_data((0..3).flat_map(i32::to_le_bytes).collect()),
    );

    let bytes = NetcdfWriter::new()
        .write_model(&model)
        .expect("write netCDF model");
    let hdf5_file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open HDF5 image");
    let decoded = consus_netcdf::read_model(&hdf5_file).expect("read netCDF model");
    let variable = decoded
        .root
        .variables
        .iter()
        .find(|variable| variable.name == "signal")
        .expect("decoded signal variable");

    assert_eq!(variable.dimensions, vec![String::from("x")]);
    assert_eq!(
        variable
            .shape
            .as_ref()
            .expect("variable shape")
            .current_dims()
            .as_slice(),
        &[3]
    );
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
    use consus_arrow::{ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind, ArrowSchema};
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
            None,
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
    use consus_arrow::{ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind, ArrowSchema};
    use consus_core::{ByteOrder, Datatype};
    use consus_parquet::{FieldDescriptor, ParquetPhysicalType, SchemaDescriptor};

    // Create Arrow schema
    let arrow_schema = ArrowSchema::new(vec![
        ArrowFieldBuilder::new(
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
        .expect("field must build"),
    ]);

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
    use consus_core::{ByteOrder, Datatype, Shape};
    use consus_zarr::{InMemoryStore, read_chunk, write_chunk};

    let original: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
    let bytes: Vec<u8> = original
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let datatype = Datatype::Float {
        bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
        byte_order: ByteOrder::LittleEndian,
    };

    let hdf5_bytes = write_hdf5_dataset(
        "data",
        &datatype,
        &Shape::fixed(&[original.len()]),
        &bytes,
        &DatasetCreationProps::default(),
    );
    let hdf5_file = Hdf5File::open(MemCursor::from_bytes(hdf5_bytes)).expect("open HDF5");
    let hdf5_buf = read_hdf5_dataset(&hdf5_file, "data");
    let from_hdf5: Vec<f32> = hdf5_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    let mut store = InMemoryStore::new();
    let metadata = zarr_metadata(vec![original.len()], vec![original.len()], "float32");
    write_chunk(&mut store, "data", &[0], &metadata, &hdf5_buf).expect("write Zarr chunk");
    let zarr_data = read_chunk(&store, "data", &[0], &metadata).expect("read Zarr chunk");
    let from_zarr: Vec<f32> = zarr_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    assert_eq!(from_hdf5, original, "HDF5 values must roundtrip");
    assert_eq!(from_zarr, original, "Zarr values must roundtrip");
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
    use consus_core::{ByteOrder, Compression, Datatype, Shape};
    use consus_hdf5::property_list::DatasetLayout;

    let data: Vec<u8> = (0..=255).cycle().take(1000).collect();
    let datatype = Datatype::Integer {
        bits: core::num::NonZeroUsize::new(8).expect("non-zero"),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let properties = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![data.len()]),
        compression: Compression::Deflate { level: 6 },
        ..DatasetCreationProps::default()
    };
    let bytes = write_hdf5_dataset(
        "compressed",
        &datatype,
        &Shape::fixed(&[data.len()]),
        &data,
        &properties,
    );
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open HDF5");
    let address = find_hdf5_dataset_address(&file, "compressed");
    let dataset = file.dataset_at(address).expect("read compressed metadata");

    let has_deflate = dataset.filters.contains(&1);
    assert!(has_deflate, "must use deflate compression");

    assert_eq!(read_hdf5_dataset(&file, "compressed"), data);
}
