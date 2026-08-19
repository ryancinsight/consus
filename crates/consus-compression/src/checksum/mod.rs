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
//! ├── reflected    # Shared table generation for reflected CRCs
//! ├── crc32        # CRC-32 (IEEE 802.3)
//! ├── crc32c       # CRC-32C (Castagnoli; Zarr v3's checksum codec)
//! ├── fletcher32   # Fletcher-32 (HDF5 filter ID 3)
//! └── lookup3      # Jenkins lookup3 (HDF5 v2 metadata checksums)
//! ```
//!
//! ## Choosing between CRC-32 and CRC-32C
//!
//! They share an interface and nothing else: different polynomials, and no
//! agreement on any realistic input. The format dictates which is correct —
//! HDF5's filter wants IEEE, Zarr v3's checksum codec wants Castagnoli — so
//! neither is a default and neither substitutes for the other.

pub mod crc32;
pub mod crc32c;
pub mod fletcher32;
pub mod lookup3;
mod reflected;
pub mod traits;

pub use crc32::Crc32;
pub use crc32c::Crc32c;
pub use fletcher32::Fletcher32;
pub use lookup3::Lookup3;
pub use traits::Checksum;
