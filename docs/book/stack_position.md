# Position in the Stack

## What Consus Owns

Consus is the scientific storage provider for Atlas. It owns:

- **Multi-format I/O** — HDF5, Zarr, netCDF, Parquet, Arrow, FITS, NPY, MAT,
  NWB, ONNX backends behind a unified facade
- **Canonical type system** — `Datatype`, `Shape`, `Selection`, `Hyperslab`
  as the cross-format SSOT
- **Compression codec registry** — DEFLATE, Zstandard, Blosc, LZ4, BZip2
- **Abstract storage traits** — `FileRead`, `FileWrite`, `DatasetRead`,
  `DatasetWrite`, `GroupRead`, `GroupWrite`, `SelectionOps`

Consus does **not** own physics equations, numerical solvers, memory
allocation policy, or visualization.

## Where Consus Sits

```
eunomia → aequitas → ... physics providers (helios, kwavers, CFDrs, ritk) ...
                               │
                               ▼ (store results)
                            consus (scientific storage)
                               │
                               ▼
                          local filesystem, HPC storage, external adapters
```

Physics providers and domain solvers write their results through Consus.
Application code reads study data from Consus and passes it to solvers or
visualization.

## Consumers

| Consumer | How Consus is used |
|----------|-------------------|
| `helios` | Radiation-therapy dose maps and DICOM series |
| `kwavers` | Ultrasound simulation result storage |
| `ritk` | Medical image series and tractography data |
| `CFDrs` | Fluid simulation output (velocity fields, pressure) |
| `tyche` | Sensitivity analysis study schemas |

## No-`std` Support

`consus-core` is `no_std`-compatible. Enable the `alloc` feature for compound
types and `std` for full I/O integration. Format backends require `std`.

## Tyche Integration

The versioned Consus study schema (TYCHE-005) connects Tyche sensitivity
analysis runs to Consus storage: each sampling design, forward-model evaluation,
and Sobol index report is persisted as a versioned Consus dataset so studies
are reproducible from the stored parameters alone.
