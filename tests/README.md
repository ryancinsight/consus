# Consus Test Suite

Comprehensive test suite for the Consus scientific storage library, covering all format backends with specification-compliant validation.

## Architecture

The test suite follows a **SSOT (Single Source of Truth)** architecture:

```
consus/
├── crates/
│   ├── consus-core/tests/           # Core type and trait tests
│   ├── consus-io/tests/             # I/O abstraction tests
│   ├── consus-compression/tests/    # Compression codec tests + shared fixtures
│   ├── consus-hdf5/tests/           # HDF5 format tests
│   ├── consus-zarr/tests/           # Zarr v2/v3 format tests
│   ├── consus-netcdf/tests/         # netCDF-4 format tests
│   ├── consus-arrow/tests/          # Arrow interoperability tests
│   └── consus-parquet/tests/        # Parquet interoperability tests
├── tests/                           # Workspace-level integration tests
│   ├── cross_format_interop.rs      # Cross-format conversion tests
│   ├── end_to_end_reference.rs      # Reference file validation
│   └── property_integration.rs      # Property-based integration tests
└── data/                            # Reference data samples
    ├── hdf5_big_endian_dataset_sample.h5
    ├── hdf5_charset_dataset_sample.h5
    ├── netcdf_small_grid_sample.nc
    ├── netcdf_hdf5_compat_sample.nc
    ├── parquet_alltypes_plain_sample.parquet
    ├── parquet_binary_records_sample.parquet
    ├── arrow_primitive_ipc_sample.arrow
    ├── arrow_nested_ipc_sample.arrow
    └── arrow_decimal_ipc_sample.arrow
```

## Test Categories

### 1. Unit Tests (`tests/unit_*.rs`)

Each crate contains unit tests for individual components:

- **consus-core**: Datatype, Shape, Selection, Compression, canonical type verification
- **consus-io**: MemCursor, ReadAt/WriteAt traits, async I/O
- **consus-compression**: Checksum algorithms (CRC-32, Fletcher-32, Lookup3), codec roundtrips, filter pipelines
- **consus-hdf5**: Header parsing, attribute encoding, dataset I/O, link resolution
- **consus-zarr**: Metadata parsing (v2/v3), store operations, chunk I/O
- **consus-netcdf**: Dimensions, attributes, variables, CF conventions

### 2. Reference File Validation (`tests/reference_*.rs`)

Tests validate against official specification test files:

- **HDF Group**: HDF5 specification compliance
- **Unidata**: netCDF-4 specification compliance
- **Zarr Spec**: Zarr v2/v3 specification examples
- **Apache**: Parquet and Arrow IPC specification compliance

### 3. Round-Trip Tests (`tests/roundtrip_*.rs`)

Verify data preservation through write-read cycles:

- Write file → read back → verify byte-identical values
- Dataset creation → write → read → verify exact matches
- Group hierarchy preservation
- Attribute roundtrip
- Compression filter roundtrip

### 4. Property-Based Tests (`tests/property_*.rs`)

Using `proptest` for invariant verification:

- Arbitrary data shapes and sizes
- Random data patterns through compression
- Schema conversion properties
- Selection bounds validation
- Cross-format consistency

### 5. Cross-Format Interoperability (`tests/cross_format_interop.rs`)

Validate data interchange between formats:

- HDF5 ↔ Zarr conversion
- netCDF-4 ↔ HDF5 compatibility
- Arrow ↔ Parquet ↔ Core schema conversions
- Data value preservation through multiple format hops

## Running Tests

### Run All Unit Tests

```bash
# Run tests for all crates
cargo test --workspace

# Run tests for specific crate
cargo test -p consus-core
cargo test -p consus-hdf5
cargo test -p consus-zarr
```

### Run Integration Tests

```bash
# Run workspace-level integration tests
cd tests
cargo test

# Run with all format backends enabled
cargo test --features all-formats

# Run specific integration test
cargo test --test end_to_end_reference
cargo test --test cross_format_interop --features hdf5,zarr
```

### Run Property Tests

```bash
# Property tests run as part of regular test suite
cargo test --test property_integration

# Run with more cases for deeper coverage
PROPTEST_CASES=1000 cargo test --test property_integration
```

### Run with Specific Features

```bash
# Run HDF5 tests only
cargo test -p consus-hdf5 --features std

# Run compression tests with all codecs
cargo test -p consus-compression --features deflate,zstd,lz4

# Run async I/O tests
cargo test -p consus-io --features async-io
```

## Test Coverage Summary

| Crate | Unit Tests | Integration Tests | Property Tests | Reference Tests |
|-------|------------|-------------------|----------------|-----------------|
| consus-core | 156 | - | 24 | 25 (SSOT) |
| consus-io | 140 | - | 30 | - |
| consus-compression | 85 | - | 20 | 48 (checksum) |
| consus-hdf5 | - | 20+ | - | 799 |
| consus-zarr | - | 41 | 15 | 7 |
| consus-netcdf | 591 | 2 | - | 499 |
| consus-arrow | - | 28 | - | - |
| Integration | - | 3 | 45+ | 9 |

**Total**: 2000+ individual test cases

## Shared Test Fixtures

Located in `consus-compression/tests/fixtures/mod.rs`:

```rust
use consus_compression::tests::fixtures;

// Reproducible pseudo-random data
let data = fixtures::random_data(1024);

// Gradient pattern (highly compressible)
let gradient = fixtures::gradient_data(1024);

// All zeros / all ones
let zeros = fixtures::zeroes(1024);
let ones = fixtures::ones(1024);

// Alternating pattern
let alt = fixtures::alternating_data(1024);
```

All fixtures use deterministic generation for reproducibility.

## Test Data Requirements

Reference files in `data/` directory should be:

1. **Obtained from official sources**:
   - HDF Group test suite
   - Unidata netCDF-4 examples
   - Zarr-Python test data
   - Apache Arrow/Parquet test vectors

2. **Small but representative**:
   - < 10 MB per file (integration test samples)
   - Contains non-trivial data patterns
   - Covers edge cases (big-endian, special characters, etc.)

3. **Named according to convention**:
   - `{format}_{feature}_sample.{ext}`
   - Example: `hdf5_big_endian_dataset_sample.h5`

## Specification Compliance

### HDF5 (HDF Group)

- Superblock parsing (v0, v1, v2)
- Object header messages
- B-tree navigation
- Dataset layouts (contiguous, chunked)
- Filter pipeline (deflate, shuffle, szip, blosc)
- Attribute encoding
- Link resolution (hard, soft, external)

### netCDF-4 (Unidata)

- Dimension handling (fixed, unlimited)
- Variable definitions
- CF convention attributes (units, long_name, standard_name)
- Coordinate variables
- Group hierarchy

### Zarr v2/v3 (Zarr Specification)

- .zarray / zarr.json metadata parsing
- Chunk key encoding
- Codec pipelines (gzip, zstd, blosc, shuffle)
- Fill value handling
- Consolidated metadata

### Parquet (Apache)

- Schema descriptors
- Physical types (BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY)
- Logical types (STRING, TIMESTAMP, DECIMAL)
- Compression codecs

### Arrow IPC (Apache)

- Schema messages
- Record batches
- Nested types (list, struct)
- Dictionary encoding

## Mathematical Invariants Verified

### Data Integrity

- `compress(decompress(data)) == data` for all codecs
- `write_then_read(values) == values` for all formats
- `to_bytes(from_bytes(value)) == value` for all datatypes

### Shape and Selection

- `element_count(shape) == product(dims)`
- `hyperslab(start, count, stride)` stays within bounds
- `selection.element_count(shape)` matches expected count

### Schema Conversion

- `arrow_to_core(arrow_schema).len() == arrow_schema.fields.len()`
- `parquet_to_core(parquet_schema).len() == parquet_schema.columns.len()`
- Field names preserved through all conversions

### Byte Order

- `from_le_bytes(to_le_bytes(value)) == value`
- `from_be_bytes(to_be_bytes(value)) == value`
- Big-endian datasets preserve byte order marker

## Continuous Integration

Tests are designed for CI environments:

1. **Timeout discipline**: All long-running tests use bounded timeouts
2. **Parallel execution**: Tests are independent and thread-safe
3. **Feature gates**: Tests skip gracefully when features disabled
4. **Resource limits**: Memory allocations bounded, no unbounded growth

## Adding New Tests

When adding tests, follow these principles:

1. **Assert on VALUES**: Never use `is_ok()` without checking the contained value
2. **Derive test data**: Use analytically derived values, not magic numbers
3. **Feature gate**: Use `#[cfg(feature = "...")]` for optional dependencies
4. **Document spec reference**: Include specification section in doc comments
5. **Use shared fixtures**: Import from `consus-compression::tests::fixtures`

## Test Output Interpretation

- `✓` - Test passed, file loaded successfully
- `⚠` - Test passed with warnings (e.g., feature incomplete)
- `✗` - Test failed
- `SKIP` - Test skipped (missing reference file or disabled feature)

## Known Limitations

Current limitations tracked for future work:

1. HDF5 reader/writer incomplete (in development)
2. Zarr S3 backend tests require `s3` feature
3. netCDF-4 writer API under development
4. Compound datatype support limited to specific formats
5. Async I/O tests require tokio runtime

## Contact

For questions about test suite architecture:

- Ryan Clanton (ryanclanton@outlook.com)
- GitHub: @ryancinsight