//! NWBFile top-level container.
//!
//! An NWBFile is an HDF5 file conforming to the NWB 2.x specification.
//! This module provides the entry point for opening and reading NWB files
//! and the builder for writing them.
//!
//! ## Specification
//!
//! Reference: *NWB 2.x Format Specification*
//! <https://nwb-schema.readthedocs.io/en/latest/format.html>
//!
//! ## Architecture
//!
//! ```text
//! NwbFile<'a>
//!   ├── open(bytes)              — validate HDF5 + NWB root attributes
//!   ├── nwb_version()            — detect NWB spec version
//!   ├── session_metadata()       — read required session-level attributes
//!   ├── time_series(path)        — read a TimeSeries neurodata group
//!   ├── units_spike_times()      — read flat Units spike times
//!   ├── units_table()            — read Units VectorIndex table
//!   ├── electrode_table()        — read electrodes DynamicTable
//!   ├── subject()                — read subject metadata
//!   ├── list_specifications()    — list namespace names from /specifications/
//!   └── read_specification(ns, ver) — read and parse namespace spec YAML
//!
//! NwbFileBuilder
//!   ├── new(...)                 — create root NWB metadata
//!   ├── write_time_series(ts)    — emit TimeSeries group
//!   ├── write_units(spikes)      — emit flat Units spike times
//!   ├── write_units_table(...)   — emit Units VectorData + VectorIndex table
//!   ├── write_electrode_table(...)— emit electrodes DynamicTable
//!   ├── write_subject(...)       — emit general/subject
//!   ├── write_namespace_specs(specs) — emit /specifications/{ns}/{ver}/namespace datasets
//!   └── finish()                 — return HDF5 bytes
//! ```
//!
//! ## Invariants
//!
//! - [`NwbFile::open`] only succeeds when the file passes
//!   [`crate::validation::validate_root_attributes`].
//! - All read methods are pure with respect to the file image: they never
//!   mutate the underlying byte slice.
//! - The `'a` lifetime ties the `NwbFile` to the byte slice it was opened
//!   from; the file cannot outlive its source data.

mod builder;
mod reader;
#[cfg(all(test, feature = "alloc"))]
mod tests;

pub use builder::NwbFileBuilder;
pub use reader::NwbFile;
