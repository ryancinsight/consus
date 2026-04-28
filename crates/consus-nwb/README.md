# consus-nwb

Pure-Rust read/write crate for Neurodata Without Borders (NWB) 2.x files.

## Overview

NWB 2.x is an HDF5-based format for neurophysiology data standardized by the
NWB consortium. It encodes session metadata, time series recordings, spike-sorted
unit tables, electrode geometry, and subject information under a versioned
namespace schema.

This crate provides:

- **Read path** — open an NWB 2.x file from a byte slice; query session
  metadata, `TimeSeries` data, flat Units spike times, and subject metadata.
- **Write path** — construct NWB 2.x files via `NwbFileBuilder`; write
  `TimeSeries` (timestamp or uniform-rate), flat Units spike times, and
  subject metadata.
- **Model layer** — pure data types: `TimeSeries`, `UnitsTable`,
  `ElectrodeTable`, `NwbSessionMetadata`, `NwbSubjectMetadata`.

The storage backend is `consus-hdf5`, a pure-Rust HDF5 implementation. No C
library dependency is required.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std`   | yes     | Enables standard library (file I/O paths, `std::error::Error`). |
| `alloc` | yes     | Enables heap allocation (required for all `String`/`Vec` types). |

## Quick Start — Reading

```rust
use consus_nwb::file::NwbFile;

let bytes: &[u8] = /* NWB file bytes */;
let nwb = NwbFile::open(bytes)?;

// NWB format version
let version = nwb.nwb_version()?;          // -> NwbVersion

// Session metadata
let meta = nwb.session_metadata()?;         // -> NwbSessionMetadata
println!("{}", meta.identifier());
println!("{}", meta.session_description());
println!("{}", meta.session_start_time());

// TimeSeries by path
let ts = nwb.time_series("acquisition/my_ts")?;  // -> TimeSeries
println!("{} samples", ts.len());

// Enumerate TimeSeries in container groups
let acq_paths  = nwb.list_acquisition()?;             // -> Vec<String>
let proc_paths = nwb.list_processing("behavior")?;    // -> Vec<String>

// Units spike times (flat)
let spike_times = nwb.units_spike_times()?;           // -> Vec<f64>

// Subject metadata
let subject = nwb.subject()?;                         // -> NwbSubjectMetadata
println!("{:?}", subject.subject_id());
```

## Quick Start — Writing

```rust
use consus_nwb::file::NwbFileBuilder;
use consus_nwb::model::TimeSeries;
use consus_nwb::metadata::NwbSubjectMetadata;

let mut builder = NwbFileBuilder::new(
    "2.7.0",
    "session-001",
    "Tetrode recording from CA1",
    "2024-01-15T10:00:00",
)?;

// TimeSeries with explicit timestamps
builder.write_time_series(&TimeSeries::with_timestamps(
    "lick_times",
    vec![0.5, 1.2, 2.3],   // data
    vec![0.5, 1.2, 2.3],   // timestamps (seconds)
))?;

// TimeSeries with uniform sample rate
builder.write_time_series(&TimeSeries::with_rate(
    "lfp",
    vec![0.1, -0.1, 0.2],  // data
    0.0,                    // starting_time (seconds)
    30_000.0,               // rate (Hz)
))?;

// Units — flat spike times
builder.write_units(&[0.1, 0.2, 0.5, 1.0, 1.1])?;

// Subject metadata (all fields optional)
builder.write_subject(&NwbSubjectMetadata::from_parts(
    Some("sub-001".to_owned()),
    Some("Mus musculus".to_owned()),
    Some("M".to_owned()),
    Some("P30".to_owned()),
    Some("C57BL/6 mouse".to_owned()),
))?;

let bytes: Vec<u8> = builder.finish()?;
```

## Architecture

```
consus-nwb
├── conventions  — NWB namespace and neurodata type resolution
├── file         — NwbFile (read) + NwbFileBuilder (write)
├── group        — Group traversal helpers
├── io           — NWB-specific I/O helpers
├── metadata     — NwbSessionMetadata, NwbSubjectMetadata
├── model        — TimeSeries, UnitsTable, ElectrodeTable
├── namespace    — NWB namespace registry
├── storage      — HDF5-backed dataset/attribute read helpers
├── validation   — Schema conformance checking
└── version      — NWB version detection
```

Each module has a single responsibility. `file` is the only module that
aggregates across the others; all domain types live in `model` and `metadata`
with no upward dependencies.

## NWB Compliance

### Implemented

**Read path**

| Capability | Status |
|---|---|
| NWB version detection (`/nwb_version`, `/.specloc`) | ✓ |
| Session metadata (`/file_create_date`, `/identifier`, `/session_description`, `/session_start_time`) | ✓ |
| `TimeSeries` data and timestamps/rate from any HDF5 path | ✓ |
| `list_acquisition` / `list_processing` group enumeration | ✓ |
| Units flat spike times (`/units/spike_times`) | ✓ |
| Subject metadata (`/general/subject/*`) | ✓ |

**Write path**

| Capability | Status |
|---|---|
| NWB 2.x root structure and required attributes | ✓ |
| `TimeSeries` with explicit timestamps | ✓ |
| `TimeSeries` with uniform rate + `starting_time` | ✓ |
| Units flat spike times (`/units/spike_times`) | ✓ |
| Subject metadata | ✓ |

### Not yet implemented

- **Per-unit `UnitsTable`** — `VectorIndex`-decoded per-unit spike time
  intervals are defined in the model layer (`UnitsTable`) but are not yet wired
  into the file read/write API. Scheduled for a subsequent milestone.
- **`ElectrodeTable`** — electrode geometry and channel metadata are defined in
  the model layer but not yet exposed through `NwbFile` or `NwbFileBuilder`.
- **Namespace YAML parsing** — the `namespace` module provides a registry
  structure; runtime parsing of NWB namespace YAML bundles is not yet
  implemented.
- **`NWBFile` neurodata type coercion** — full `neurodata_type_def` /
  `neurodata_type_inc` resolution across the inheritance hierarchy is partial.

## NWB Specification References

- NWB 2.x format specification: <https://nwb-schema.readthedocs.io/en/latest/format.html>
- HDMF DynamicTable (VectorIndex, VectorData): <https://hdmf-common-schema.readthedocs.io/>

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.

`SPDX-License-Identifier: MIT OR Apache-2.0`
