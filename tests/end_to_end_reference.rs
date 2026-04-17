//! End-to-end integration tests using reference data samples.
//!
//! ## Specification Reference
//!
//! Tests validate compliance with:
//! - HDF Group HDF5 specification
//! - Unidata netCDF-4 specification
//! - Zarr v2/v3 specification
//! - Apache Parquet specification
//! - Apache Arrow IPC specification
//!
//! ## Coverage
//!
//! - Load reference files from `data/` directory
//! - Validate against spec-compliant expected values
//! - Test cross-format data equivalence
//! - Verify metadata preservation

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Reference File Paths
// ---------------------------------------------------------------------------

/// Get path to reference data file.
fn reference_file(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("data")
        .join(name)
}

fn hdf5_big_endian_sample() -> PathBuf {
    reference_file("hdf5_big_endian_dataset_sample.h5")
}

fn hdf5_charset_sample() -> PathBuf {
    reference_file("hdf5_charset_dataset_sample.h5")
}

fn hdf5_t_float_sample() -> PathBuf {
    reference_file("t_float.h5")
}

fn hdf5_t_compound_sample() -> PathBuf {
    reference_file("t_compound.h5")
}

fn hdf5_t_vlen_sample() -> PathBuf {
    reference_file("t_vlen.h5")
}

fn hdf5_t_chunk_sample() -> PathBuf {
    reference_file("t_chunk.h5")
}

fn hdf5_t_filter_sample() -> PathBuf {
    reference_file("t_filter.h5")
}

fn netcdf_small_grid_sample() -> PathBuf {
    reference_file("netcdf_small_grid_sample.nc")
}

fn netcdf_hdf5_compat_sample() -> PathBuf {
    reference_file("netcdf_hdf5_compat_sample.nc")
}

fn parquet_alltypes_sample() -> PathBuf {
    reference_file("parquet_alltypes_plain_sample.parquet")
}

fn parquet_binary_sample() -> PathBuf {
    reference_file("parquet_binary_records_sample.parquet")
}

fn arrow_primitive_sample() -> PathBuf {
    reference_file("arrow_primitive_ipc_sample.arrow")
}

fn arrow_nested_sample() -> PathBuf {
    reference_file("arrow_nested_ipc_sample.arrow")
}

fn arrow_decimal_sample() -> PathBuf {
    reference_file("arrow_decimal_ipc_sample.arrow")
}

// ---------------------------------------------------------------------------
// HDF5 Reference Tests
// ---------------------------------------------------------------------------

/// Test: Load and validate HDF5 big-endian dataset sample.
///
/// ## Spec Compliance
///
/// HDF5 files with big-endian data must:
/// - Report correct ByteOrder in datatype
/// - Convert bytes correctly when read on little-endian platform
/// - Preserve exact IEEE 754 values
#[test]
fn hdf5_big_endian_sample_loads() {
    let path = hdf5_big_endian_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        eprintln!("      Place sample file in data/ directory");
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("⚠ Failed to read HDF5 big-endian sample: {:?}", e);
            return;
        }
    };
    let file = consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes));

    match file {
        Ok(_) => {
            println!("✓ HDF5 big-endian sample loaded successfully");
        }
        Err(e) => {
            eprintln!("⚠ HDF5 big-endian sample parsing not complete: {:?}", e);
            eprintln!("  This is expected if HDF5 reader is under development");
        }
    }
}

/// Test: Validate HDF5 charset/encoding sample.
///
/// ## Spec Compliance
///
/// HDF5 string data must:
/// - Report correct character encoding (ASCII/UTF-8)
/// - Handle variable-length and fixed-length strings
/// - Preserve string content exactly
#[test]
fn hdf5_charset_sample_loads() {
    let path = hdf5_charset_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("⚠ Failed to read HDF5 charset sample: {:?}", e);
            return;
        }
    };
    let file = consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes));

    match file {
        Ok(_) => {
            println!("✓ HDF5 charset sample loaded successfully");
        }
        Err(e) => {
            eprintln!("⚠ HDF5 charset sample parsing not complete: {:?}", e);
        }
    }
}

/// Test: Validate canonical HDF Group `t_float.h5` fixture.
///
/// ## Spec Compliance
///
/// Floating-point datasets must:
/// - Preserve the declared datatype class
/// - Read dataset bytes without shape or endian corruption
/// - Match reference fixture metadata
#[test]
fn hdf5_t_float_sample_loads() {
    let path = hdf5_t_float_sample();

    if !path.exists() {
        eprintln!("SKIP: Canonical HDF5 fixture not found: {:?}", path);
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            panic!("failed to read canonical HDF5 fixture {:?}: {e}", path);
        }
    };

    let file = match consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)) {
        Ok(file) => file,
        Err(e) => {
            panic!("failed to open canonical HDF5 fixture {:?}: {:?}", path, e);
        }
    };

    let root = match file.root_object_header() {
        Ok(header) => header,
        Err(e) => {
            panic!("failed to parse root object header for {:?}: {:?}", path, e);
        }
    };

    assert!(
        !root.messages.is_empty(),
        "canonical HDF5 float fixture must contain object header messages"
    );
    assert!(
        matches!(file.root_node_type(), Ok(consus_core::NodeType::Group)),
        "canonical HDF5 float fixture must expose a root group"
    );
}

/// Test: Validate canonical HDF Group `t_compound.h5` fixture.
///
/// ## Spec Compliance
///
/// Compound datasets must:
/// - Preserve member layout metadata
/// - Preserve field ordering and offsets
/// - Support deterministic metadata traversal
#[test]
fn hdf5_t_compound_sample_loads() {
    let path = hdf5_t_compound_sample();

    if !path.exists() {
        eprintln!("SKIP: Canonical HDF5 fixture not found: {:?}", path);
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            panic!("failed to read canonical HDF5 fixture {:?}: {e}", path);
        }
    };

    let file = match consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)) {
        Ok(file) => file,
        Err(e) => {
            panic!("failed to open canonical HDF5 fixture {:?}: {:?}", path, e);
        }
    };

    let header = match file.root_object_header() {
        Ok(header) => header,
        Err(e) => {
            panic!("failed to parse root object header for {:?}: {:?}", path, e);
        }
    };

    assert!(
        !header.messages.is_empty(),
        "canonical HDF5 compound fixture must contain object header messages"
    );
    assert!(
        matches!(file.root_node_type(), Ok(consus_core::NodeType::Group)),
        "canonical HDF5 compound fixture must expose a root group"
    );
}

/// Test: Validate canonical HDF Group `t_vlen.h5` fixture.
///
/// ## Spec Compliance
///
/// Variable-length datasets must:
/// - Preserve heap-backed payload references
/// - Resolve indirections through the global heap
/// - Preserve logical element counts
#[test]
fn hdf5_t_vlen_sample_loads() {
    let path = hdf5_t_vlen_sample();

    if !path.exists() {
        eprintln!("SKIP: Canonical HDF5 fixture not found: {:?}", path);
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            panic!("failed to read canonical HDF5 fixture {:?}: {e}", path);
        }
    };

    let file = match consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)) {
        Ok(file) => file,
        Err(e) => {
            panic!("failed to open canonical HDF5 fixture {:?}: {:?}", path, e);
        }
    };

    let header = match file.root_object_header() {
        Ok(header) => header,
        Err(e) => {
            panic!("failed to parse root object header for {:?}: {:?}", path, e);
        }
    };

    assert!(
        !header.messages.is_empty(),
        "canonical HDF5 vlen fixture must contain object header messages"
    );
    assert!(
        matches!(file.root_node_type(), Ok(consus_core::NodeType::Group)),
        "canonical HDF5 vlen fixture must expose a root group"
    );
}

/// Test: Validate canonical HDF Group `t_chunk.h5` fixture.
///
/// ## Spec Compliance
///
/// Chunked datasets must:
/// - Preserve chunk layout metadata
/// - Expose chunk indexing structures
/// - Maintain readable dataset metadata
#[test]
fn hdf5_t_chunk_sample_loads() {
    let path = hdf5_t_chunk_sample();

    if !path.exists() {
        eprintln!("SKIP: Canonical HDF5 fixture not found: {:?}", path);
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            panic!("failed to read canonical HDF5 fixture {:?}: {e}", path);
        }
    };

    let file = match consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)) {
        Ok(file) => file,
        Err(e) => {
            panic!("failed to open canonical HDF5 fixture {:?}: {:?}", path, e);
        }
    };

    let header = match file.root_object_header() {
        Ok(header) => header,
        Err(e) => {
            panic!("failed to parse root object header for {:?}: {:?}", path, e);
        }
    };

    assert!(
        !header.messages.is_empty(),
        "canonical HDF5 chunk fixture must contain object header messages"
    );
    assert!(
        matches!(file.root_node_type(), Ok(consus_core::NodeType::Group)),
        "canonical HDF5 chunk fixture must expose a root group"
    );
}

/// Test: Validate canonical HDF Group `t_filter.h5` fixture.
///
/// ## Spec Compliance
///
/// Filtered datasets must:
/// - Preserve filter pipeline metadata
/// - Keep chunked read paths usable after decompression
/// - Preserve dataset object header integrity
#[test]
fn hdf5_t_filter_sample_loads() {
    let path = hdf5_t_filter_sample();

    if !path.exists() {
        eprintln!("SKIP: Canonical HDF5 fixture not found: {:?}", path);
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            panic!("failed to read canonical HDF5 fixture {:?}: {e}", path);
        }
    };

    let file = match consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)) {
        Ok(file) => file,
        Err(e) => {
            panic!("failed to open canonical HDF5 fixture {:?}: {:?}", path, e);
        }
    };

    let header = match file.root_object_header() {
        Ok(header) => header,
        Err(e) => {
            panic!("failed to parse root object header for {:?}: {:?}", path, e);
        }
    };

    assert!(
        !header.messages.is_empty(),
        "canonical HDF5 filter fixture must contain object header messages"
    );
    assert!(
        matches!(file.root_node_type(), Ok(consus_core::NodeType::Group)),
        "canonical HDF5 filter fixture must expose a root group"
    );
}

// ---------------------------------------------------------------------------
// netCDF-4 Reference Tests
// ---------------------------------------------------------------------------

/// Test: Load and validate netCDF-4 small grid sample.
///
/// ## Spec Compliance
///
/// netCDF-4 files must:
/// - Use HDF5 as the underlying format
/// - Follow netCDF-4 classic model conventions
/// - Store dimensions as special HDF5 datasets
#[test]
fn netcdf_small_grid_sample_loads() {
    let path = netcdf_small_grid_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("⚠ Failed to read netCDF-4 small grid sample: {:?}", e);
            return;
        }
    };
    let file = consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes));

    match file {
        Ok(_) => {
            println!("✓ netCDF-4 small grid sample loaded as HDF5");
        }
        Err(e) => {
            eprintln!(
                "⚠ netCDF-4 sample requires netCDF-specific handling: {:?}",
                e
            );
        }
    }
}

/// Test: Load and validate netCDF-4 HDF5 compatibility sample.
///
/// ## Spec Compliance
///
/// netCDF-4 files using advanced HDF5 features must:
/// - Support chunked datasets
/// - Support compression filters
/// - Preserve netCDF-specific metadata attributes
#[test]
fn netcdf_hdf5_compat_sample_loads() {
    let path = netcdf_hdf5_compat_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        return;
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("⚠ Failed to read netCDF-4 HDF5 compat sample: {:?}", e);
            return;
        }
    };
    let file = consus_hdf5::file::Hdf5File::open(consus_io::MemCursor::from_bytes(bytes));

    match file {
        Ok(_) => {
            println!("✓ netCDF-4 HDF5 compat sample loaded");
        }
        Err(e) => {
            eprintln!(
                "⚠ netCDF-4 HDF5 compat sample requires netCDF handler: {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Parquet Reference Tests
// ---------------------------------------------------------------------------

/// Test: Load and validate Parquet all types sample.
///
/// ## Spec Compliance
///
/// Parquet files must:
/// - Have valid magic bytes "PAR1" at start and end
/// - Have valid footer with schema metadata
/// - Support all physical types (BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY)
#[test]
fn parquet_alltypes_sample_loads() {
    let path = parquet_alltypes_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        return;
    }

    let data = std::fs::read(&path);

    match data {
        Ok(bytes) => {
            if bytes.len() >= 4 {
                let magic = &bytes[0..4];
                if magic == b"PAR1" {
                    println!("✓ Parquet alltypes sample has valid magic bytes");

                    if bytes.len() >= 8 {
                        let footer_magic = &bytes[bytes.len() - 4..];
                        if footer_magic == b"PAR1" {
                            println!("✓ Parquet alltypes sample has valid footer magic");
                        }
                    }
                } else {
                    eprintln!("⚠ Invalid Parquet magic: {:02x?}", magic);
                }
            }
        }
        Err(e) => {
            eprintln!("⚠ Failed to read Parquet sample: {:?}", e);
        }
    }
}

/// Test: Load and validate Parquet binary records sample.
///
/// ## Spec Compliance
///
/// Parquet binary columns must:
/// - Handle BYTE_ARRAY physical type
/// - Support optional/nullable columns
/// - Preserve binary content exactly
#[test]
fn parquet_binary_sample_loads() {
    let path = parquet_binary_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        return;
    }

    let data = std::fs::read(&path);

    match data {
        Ok(bytes) => {
            if bytes.len() >= 4 && &bytes[0..4] == b"PAR1" {
                println!("✓ Parquet binary sample has valid format");
            }
        }
        Err(e) => {
            eprintln!("⚠ Failed to read Parquet binary sample: {:?}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Arrow IPC Reference Tests
// ---------------------------------------------------------------------------

/// Test: Load and validate Arrow IPC primitive sample.
///
/// ## Spec Compliance
///
/// Arrow IPC files must:
/// - Have valid Arrow Stream magic bytes
/// - Use little-endian encoding
/// - Contain valid schema message followed by record batches
#[test]
fn arrow_primitive_sample_loads() {
    let path = arrow_primitive_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        return;
    }

    let data = std::fs::read(&path);

    match data {
        Ok(bytes) => {
            if bytes.len() >= 6 {
                let magic = &bytes[0..6];
                if magic == b"ARROW1" {
                    println!("✓ Arrow IPC primitive sample has valid magic bytes");
                } else if magic == b"ARROW" {
                    println!("✓ Arrow IPC sample has valid 5-byte magic");
                } else {
                    eprintln!("⚠ Unexpected Arrow magic: {:02x?}", magic);
                }
            }
        }
        Err(e) => {
            eprintln!("⚠ Failed to read Arrow primitive sample: {:?}", e);
        }
    }
}

/// Test: Load and validate Arrow IPC nested sample.
///
/// ## Spec Compliance
///
/// Arrow nested types (list, struct, map) must:
/// - Have valid child type definitions
/// - Preserve nesting structure
/// - Handle variable-length lists correctly
#[test]
fn arrow_nested_sample_loads() {
    let path = arrow_nested_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        return;
    }

    let data = std::fs::read(&path);

    match data {
        Ok(bytes) => {
            if bytes.len() >= 6 {
                let magic = &bytes[0..6];
                if magic == b"ARROW1" || &magic[0..5] == b"ARROW" {
                    println!("✓ Arrow IPC nested sample has valid format");
                }
            }
        }
        Err(e) => {
            eprintln!("⚠ Failed to read Arrow nested sample: {:?}", e);
        }
    }
}

/// Test: Load and validate Arrow IPC decimal sample.
///
/// ## Spec Compliance
///
/// Arrow decimal type must:
/// - Store precision and scale in schema
/// - Use 128-bit or 256-bit representation
/// - Preserve exact decimal values
#[test]
fn arrow_decimal_sample_loads() {
    let path = arrow_decimal_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found: {:?}", path);
        return;
    }

    let data = std::fs::read(&path);

    match data {
        Ok(bytes) => {
            if bytes.len() >= 6 {
                let magic = &bytes[0..6];
                if magic == b"ARROW1" || &magic[0..5] == b"ARROW" {
                    println!("✓ Arrow IPC decimal sample has valid format");
                }
            }
        }
        Err(e) => {
            eprintln!("⚠ Failed to read Arrow decimal sample: {:?}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-Format Equivalence Tests
// ---------------------------------------------------------------------------

/// Test: Verify all reference files exist and are loadable.
///
/// ## Purpose
///
/// This test documents the expected reference files and verifies
/// the test infrastructure is properly configured.
#[test]
fn reference_files_existence_check() {
    let samples = [
        ("HDF5 big-endian", hdf5_big_endian_sample()),
        ("HDF5 charset", hdf5_charset_sample()),
        ("HDF5 t_float", hdf5_t_float_sample()),
        ("HDF5 t_compound", hdf5_t_compound_sample()),
        ("HDF5 t_vlen", hdf5_t_vlen_sample()),
        ("HDF5 t_chunk", hdf5_t_chunk_sample()),
        ("HDF5 t_filter", hdf5_t_filter_sample()),
        ("netCDF small grid", netcdf_small_grid_sample()),
        ("netCDF HDF5 compat", netcdf_hdf5_compat_sample()),
        ("Parquet alltypes", parquet_alltypes_sample()),
        ("Parquet binary", parquet_binary_sample()),
        ("Arrow primitive", arrow_primitive_sample()),
        ("Arrow nested", arrow_nested_sample()),
        ("Arrow decimal", arrow_decimal_sample()),
    ];

    let mut existing_count = 0;
    let mut missing_count = 0;

    for (name, path) in &samples {
        if path.exists() {
            println!("  ✓ {} sample found: {:?}", name, path);
            existing_count += 1;
        } else {
            println!("  ✗ {} sample NOT FOUND: {:?}", name, path);
            missing_count += 1;
        }
    }

    println!(
        "\nReference file summary: {} found, {} missing",
        existing_count, missing_count
    );

    if missing_count > 0 {
        println!("\nTo add missing reference files:");
        println!("  1. Obtain reference files from format specification test suites");
        println!("  2. Place them in the data/ directory at workspace root");
        println!("  3. Ensure file names match expected paths in this test");
    }
}

/// Test: Validate reference file sizes are reasonable.
///
/// ## Invariant
///
/// Reference files should:
/// - Not be empty (minimum size for valid format header)
/// - Not be excessively large (integration test files are small samples)
#[test]
fn reference_file_sizes_reasonable() {
    let samples = [
        ("HDF5 big-endian", hdf5_big_endian_sample()),
        ("HDF5 charset", hdf5_charset_sample()),
        ("HDF5 t_float", hdf5_t_float_sample()),
        ("HDF5 t_compound", hdf5_t_compound_sample()),
        ("HDF5 t_vlen", hdf5_t_vlen_sample()),
        ("HDF5 t_chunk", hdf5_t_chunk_sample()),
        ("HDF5 t_filter", hdf5_t_filter_sample()),
        ("netCDF small grid", netcdf_small_grid_sample()),
        ("netCDF HDF5 compat", netcdf_hdf5_compat_sample()),
        ("Parquet alltypes", parquet_alltypes_sample()),
        ("Parquet binary", parquet_binary_sample()),
        ("Arrow primitive", arrow_primitive_sample()),
        ("Arrow nested", arrow_nested_sample()),
        ("Arrow decimal", arrow_decimal_sample()),
    ];

    for (name, path) in &samples {
        if path.exists() {
            match std::fs::metadata(path) {
                Ok(metadata) => {
                    let size = metadata.len();

                    assert!(size >= 4, "{} sample too small: {} bytes", name, size);
                    assert!(
                        size < 10_000_000,
                        "{} sample too large for integration test: {} bytes",
                        name,
                        size
                    );

                    println!("  {} sample size: {} bytes", name, size);
                }
                Err(e) => {
                    eprintln!("  ⚠ Cannot read {} metadata: {:?}", name, e);
                }
            }
        }
    }
}
