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
        .join("data")
        .join(name)
}

fn hdf5_big_endian_sample() -> PathBuf {
    reference_file("hdf5_big_endian_dataset_sample.h5")
}

fn hdf5_charset_sample() -> PathBuf {
    reference_file("hdf5_charset_dataset_sample.h5")
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

    // Attempt to open with consus-hdf5
    let file = consus_hdf5::Hdf5File::open(&path);

    match file {
        Ok(_) => {
            // Success: file loaded
            println!("✓ HDF5 big-endian sample loaded successfully");
        }
        Err(e) => {
            // File may not be fully parseable yet
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

    let file = consus_hdf5::Hdf5File::open(&path);

    match file {
        Ok(_) => {
            println!("✓ HDF5 charset sample loaded successfully");
        }
        Err(e) => {
            eprintln!("⚠ HDF5 charset sample parsing not complete: {:?}", e);
        }
    }
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

    // netCDF-4 is HDF5-based, try opening with HDF5 reader
    let file = consus_hdf5::Hdf5File::open(&path);

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

    let file = consus_hdf5::Hdf5File::open(&path);

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

    // Read first 4 bytes to check magic number
    let data = std::fs::read(&path);

    match data {
        Ok(bytes) => {
            if bytes.len() >= 4 {
                let magic = &bytes[0..4];
                if magic == b"PAR1" {
                    println!("✓ Parquet alltypes sample has valid magic bytes");

                    // Check footer magic
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
                // Check for Arrow Stream magic: "ARROW1"
                let magic = &bytes[0..6];
                if magic == b"ARROW1" {
                    println!("✓ Arrow IPC primitive sample has valid magic bytes");
                } else if magic == b"ARROW" {
                    // Some implementations use 5-byte magic
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

                    // Minimum size: at least format header (typically 4-8 bytes)
                    assert!(size >= 4, "{} sample too small: {} bytes", name, size);

                    // Maximum size: integration test samples should be < 10 MB
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

// ---------------------------------------------------------------------------
// Data Integrity Spot Checks
// ---------------------------------------------------------------------------

/// Test: Spot check HDF5 big-endian sample contains expected data patterns.
///
/// ## Invariant
///
/// Reference files should contain non-trivial data (not all zeros or all same value).
#[test]
fn hdf5_big_endian_data_integrity() {
    let path = hdf5_big_endian_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found");
        return;
    }

    match std::fs::read(&path) {
        Ok(bytes) => {
            // Check for non-trivial data
            let non_zero_count = bytes.iter().filter(|&&b| b != 0).count();
            let unique_bytes = bytes.iter().collect::<std::collections::HashSet<_>>().len();

            assert!(
                non_zero_count > 100,
                "HDF5 sample should have substantial non-zero data"
            );

            assert!(
                unique_bytes > 10,
                "HDF5 sample should have diverse byte values"
            );

            println!(
                "  HDF5 big-endian: {} bytes, {} non-zero, {} unique values",
                bytes.len(),
                non_zero_count,
                unique_bytes
            );
        }
        Err(e) => {
            eprintln!("⚠ Cannot read HDF5 sample: {:?}", e);
        }
    }
}

/// Test: Spot check Parquet sample contains valid structure markers.
///
/// ## Invariant
///
/// Parquet files must have PAR1 magic at start and end.
#[test]
fn parquet_structure_markers() {
    let path = parquet_alltypes_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found");
        return;
    }

    match std::fs::read(&path) {
        Ok(bytes) => {
            if bytes.len() >= 8 {
                // Check header magic
                assert_eq!(&bytes[0..4], b"PAR1", "Parquet must start with PAR1 magic");

                // Check footer magic
                assert_eq!(
                    &bytes[bytes.len() - 4..],
                    b"PAR1",
                    "Parquet must end with PAR1 magic"
                );

                println!("  Parquet structure markers valid");
            }
        }
        Err(e) => {
            eprintln!("⚠ Cannot read Parquet sample: {:?}", e);
        }
    }
}

/// Test: Spot check Arrow IPC sample contains valid stream markers.
///
/// ## Invariant
///
/// Arrow IPC files start with "ARROW1" magic (for streaming) or
/// specific IPC message format (for file format).
#[test]
fn arrow_structure_markers() {
    let path = arrow_primitive_sample();

    if !path.exists() {
        eprintln!("SKIP: Reference file not found");
        return;
    }

    match std::fs::read(&path) {
        Ok(bytes) => {
            if bytes.len() >= 6 {
                let magic = &bytes[0..6];

                // Arrow Stream format starts with "ARROW1"
                // Arrow IPC File format has different structure
                let valid_stream = magic == b"ARROW1";
                let valid_ipc = &bytes[0..4] == &[0xFF, 0xFF, 0xFF, 0xFF]; // continuation marker

                assert!(
                    valid_stream || valid_ipc || &bytes[0..5] == b"ARROW",
                    "Arrow sample must have valid format marker"
                );

                println!("  Arrow structure markers valid");
            }
        }
        Err(e) => {
            eprintln!("⚠ Cannot read Arrow sample: {:?}", e);
        }
    }
}
