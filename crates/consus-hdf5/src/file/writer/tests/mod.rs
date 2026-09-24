//! Unit tests for the writer modules, split from the former monolith by family.

#[cfg(all(test, feature = "alloc"))]
mod builder;
#[cfg(all(test, feature = "alloc"))]
mod chunk_index;
#[cfg(all(test, feature = "alloc"))]
mod dataset_encoding;
#[cfg(all(test, feature = "alloc"))]
mod group_links;
#[cfg(all(test, feature = "alloc"))]
mod state;
