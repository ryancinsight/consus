//! Checksum algorithms for data integrity verification.
//!
//! This module is the SSOT for all checksum computations in Consus.
//! No other crate may duplicate these implementations.
//!
//! ## Hierarchy
//!
//! ```text
//! checksum/
//! ├── traits       # Checksum trait
//! ├── crc32        # CRC-32 (IEEE 802.3)
//! ├── fletcher32   # Fletcher-32 (HDF5 filter ID 3)
//! └── lookup3      # Jenkins lookup3 (HDF5 v2 metadata checksums)
//! ```

pub mod crc32;
pub mod fletcher32;
pub mod lookup3;
pub mod traits;

pub use crc32::Crc32;
pub use fletcher32::Fletcher32;
pub use lookup3::Lookup3;
pub use traits::Checksum;
