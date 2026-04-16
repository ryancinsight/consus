//! Object and region reference types.
//!
//! References encode typed pointers to objects or regions within
//! the storage hierarchy. The referent's identity is format-specific;
//! this module defines the canonical reference class taxonomy.
//!
//! ## Formal Specification
//!
//! A reference `r` has a class `c ∈ {Object, Region}` and a format-specific
//! address `a`. The class determines the valid dereference operations:
//! - `Object`: dereferences to a `Node` (group or dataset).
//! - `Region`: dereferences to a `(Node, Selection)` pair.

/// Reference class taxonomy.
///
/// ## Invariant
///
/// A reference value is associated with exactly one class.
/// The class determines the format-specific encoding and the
/// set of valid dereference operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceType {
    /// Reference to an object (group or dataset) by address.
    Object,
    /// Reference to a region (hyperslab or point set) within a dataset.
    Region,
}
