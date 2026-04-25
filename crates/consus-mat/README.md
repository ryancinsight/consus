# consus-mat

MATLAB `.mat` file reader for the [Consus](https://github.com/ryancinsight/consus)
scientific storage workspace. Supports format levels v4, v5, and v7.3 (HDF5-backed).

## Format coverage

| Version | Encoding | Feature gate | Status      |
|---------|----------|--------------|-------------|
| v4      | Binary   | always       | Implemented |
| v5      | Binary   | always       | Implemented |
| v7.3    | HDF5     | `v73`        | Implemented |

## Feature flags

| Flag       | Default | Description |
|------------|---------|-------------|
| `std`      | on      | Enables `loadmat` and standard I/O error propagation. |
| `alloc`    | on      | Enables heap-allocated model types and `loadmat_bytes`. |
| `v73`      | on      | Enables MAT v7.3 support via `consus-hdf5`. |
| `compress` | on      | Enables MAT v5 `miCOMPRESSED` zlib decompression via `flate2`. |

## Quick start

```rust
use consus_mat::{loadmat_bytes, MatArray, MatNumericClass};

let bytes = std::fs::read("data.mat").unwrap();
let file = loadmat_bytes(&bytes).unwrap();

for (name, value) in &file.variables {
    match value {
        MatArray::Numeric(na) => {
            println!("{name}: {:?} {:?}", na.class, na.shape);
        }
        MatArray::Char(ca)    => println!("{name}: \"{}\"", ca.data),
        MatArray::Logical(la) => println!("{name}: logical {:?}", la.shape),
        MatArray::Sparse(sa)  => println!("{name}: sparse {}x{}", sa.nrows, sa.ncols),
        MatArray::Cell(ca)    => println!("{name}: cell {:?}", ca.shape),
        MatArray::Struct(sa)  => {
            let fields: Vec<&str> = sa.field_names().collect();
            println!("{name}: struct {:?} fields={:?}", sa.shape, fields);
        }
    }
}
```

## Canonical model

Every variable loaded from a `.mat` file is returned as a `MatArray` variant:

| MATLAB class              | Rust variant        |
|---------------------------|---------------------|
| double/single/intN/uintN  | `MatArray::Numeric` |
| char                      | `MatArray::Char`    |
| logical                   | `MatArray::Logical` |
| sparse                    | `MatArray::Sparse`  |
| cell                      | `MatArray::Cell`    |
| struct                    | `MatArray::Struct`  |

## MAT v5 compression behavior

`miCOMPRESSED` payloads are handled by an explicit feature policy:

- **`compress` enabled** — zlib-compressed `miMATRIX` payloads are decompressed and parsed.
- **`compress` disabled** — encountering `miCOMPRESSED` returns `MatError::UnsupportedFeature`
  with a deterministic message.

## Rejection policies

The following format features are intentionally unsupported and return
`MatError::UnsupportedFeature`:

- MAT v4 sparse matrices
- MAT v4 VAX/Cray machine types (M=2..=4)
- MAT v5 `mxOBJECT_CLASS`
- MAT v7.3 sparse datasets
- MAT v7.3 compact and virtual HDF5 dataset layouts

## Version-specific notes

### v4
Records begin at byte 0; there is no file-level header. Each record is a
20-byte header followed by the variable name and column-major data bytes.
Character matrices use f64 values encoding Unicode code points.

### v5
A 128-byte file header carries the endian indicator. Top-level elements with
unknown type codes are skipped after their declared payload is consumed.
Logical arrays are signalled by the `FLAG_LOGICAL` bit in the array flags
sub-element rather than a distinct class code.

### v7.3 (HDF5-backed)
MATLAB class is stored as the `MATLAB_class` HDF5 attribute. Root-group
children named `#refs#` or starting with `#` are treated as internal
references and excluded from the variable map. Character dataset byte order is
derived from the HDF5 dataset datatype rather than assumed to be little-endian.
Struct variables are stored as HDF5 groups; their shape is currently fixed to
`[1, 1]` (scalar struct). Cell array child element order is determined by
parsing child-group names as decimal integers.
