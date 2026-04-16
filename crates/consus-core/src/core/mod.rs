//! Abstract traits and error hierarchy for the Consus storage model.
//!
//! ## Architecture
//!
//! - `error`: Comprehensive error types and `Result` alias.
//! - `traits`: Object-safe traits for format-agnostic storage access
//!   (Dependency Inversion Principle).

pub mod error;
pub mod traits;
