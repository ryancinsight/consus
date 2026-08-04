//! Zarr v3 sharded array support.
//!
//! ## Specification
//!
//! Implements the Zarr v3 sharding codec (`sharding_indexed`) per:
//! <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html>
//!
//! ## Shard File Layout
//!
//! ```text
//! [inner_chunk_0_bytes][inner_chunk_1_bytes]...[inner_chunk_N-1_bytes][shard_index]
//! ```
//!
//! The shard index is at the END of the file. Each entry is 16 bytes:
//! `(offset: u64 LE, nbytes: u64 LE)`. Uninitialized chunks use `(u64::MAX, u64::MAX)`.
//! Inner chunk offsets are absolute byte positions from the start of the shard file.

mod config;
mod error;
mod io;

#[cfg(feature = "alloc")]
pub use config::{ShardingConfig, extract_sharding_config};
pub use error::ShardError;
pub use io::inner_linear_index;
#[cfg(feature = "alloc")]
pub use io::{read_inner_chunk_from_shard, write_shard};

#[cfg(all(test, feature = "alloc"))]
mod tests;
