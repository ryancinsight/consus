//! Abstract I/O trait definitions forming the Dependency Inversion boundary.
//!
//! ## Architecture
//!
//! Format backends and I/O implementations depend on these traits,
//! never on concrete types. This enables:
//!
//! - In-memory buffers for testing
//! - Memory-mapped I/O for performance
//! - `no_std` environments with custom I/O providers
//!
//! ## Trait Hierarchy
//!
//! ```text
//! ReadAt            (positioned read, stateless, concurrent-safe)
//! WriteAt           (positioned write)
//! Length            (byte length query)
//! Truncate          (resize source/sink)
//! SeekFrom          (seek position enum, no_std)
//! Seekable          (cursor-based seeking)
//! RandomAccess      = ReadAt + WriteAt + Length + Truncate
//! ```

pub mod combined;
pub mod extent;
pub mod read;
pub mod seek;
pub mod write;

// ── Sync trait re-exports ──────────────────────────────────────────

pub use combined::RandomAccess;
pub use extent::{Length, Truncate};
pub use read::ReadAt;
pub use seek::{SeekFrom, Seekable};
pub use write::WriteAt;
