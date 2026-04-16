//! Filter pipeline for chained data transformations.
//!
//! HDF5 and Zarr apply ordered sequences of filters (shuffle, compression,
//! checksum) to chunk data. This module provides the trait-based abstraction
//! and concrete filter implementations.
//!
//! ## Hierarchy
//!
//! ```text
//! pipeline/
//! ├── traits     # Filter, FilterDirection
//! ├── shuffle    # Byte shuffle/unshuffle
//! ├── nbit       # N-bit packing/unpacking
//! └── executor   # Pipeline execution engine
//! ```
//!
//! ## Feature Gate
//!
//! All pipeline types require the `alloc` feature because filter operations
//! produce variable-length output via [`alloc::vec::Vec`].

pub mod executor;
pub mod nbit;
pub mod shuffle;
pub mod traits;

pub use executor::FilterPipeline;
pub use nbit::NbitFilter;
pub use shuffle::ShuffleFilter;
pub use traits::{Filter, FilterDirection};
