//! MAT v7.3 reader module.
pub mod reader;
#[cfg(feature = "v73")]
pub use reader::read_mat_v73;
