//! netCDF-4 Python bindings: `PyNetcdfFile`, `PyNetcdfWriter`.
//!
//! `PyNetcdfFile` reads a netCDF-4/HDF5 file from raw bytes, exposes the
//! root-group structure, and decodes variable data.
//!
//! `PyNetcdfWriter` builds a `NetcdfModel` with typed data payloads and
//! serialises it to HDF5 bytes via `NetcdfWriter`.

mod codec;
mod dtype;
mod file;
mod writer;

pub use file::PyNetcdfFile;
pub use writer::PyNetcdfWriter;
