//! Local, global, and fractal heap implementations.
//!
//! ## Specification
//!
//! Heaps store variable-length data (names, VL strings, etc.).

pub mod fractal;
pub mod global;
pub mod local;

pub use fractal::FRACTAL_HEAP_SIGNATURE;
pub use fractal::FractalHeapHeader;
#[cfg(feature = "alloc")]
pub use fractal::FractalHeapId;
pub use global::{GLOBAL_HEAP_SIGNATURE, GlobalHeapObject};
pub use local::{LOCAL_HEAP_SIGNATURE, LocalHeap};
