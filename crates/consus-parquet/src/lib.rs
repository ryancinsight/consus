//! # consus-parquet
//!
//! Apache Parquet interoperability layer for the Consus storage library.
//!
//! ## Specification
//!
//! Apache Parquet is a columnar storage format for big-data ecosystems.
//! Reference: <https://parquet.apache.org/documentation/latest/>
//!
//! ### Integration Strategy
//!
//! Consus does not reimplement Parquet from scratch. Instead, it provides:
//! 1. **Schema mapping**: Consus datatypes ↔ Parquet logical/physical types
//! 2. **Hybrid storage**: embed Parquet tables inside Consus hierarchical containers
//! 3. **Arrow bridge**: zero-copy conversion between Consus arrays and Arrow arrays
//!
//! This enables ML/big-data interop (Polars, Spark, DuckDB) while maintaining
//! Consus's hierarchical organization for scientific datasets.
//!
//! ## Status
//!
//! Phase 3 — structural skeleton.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod schema_map;
