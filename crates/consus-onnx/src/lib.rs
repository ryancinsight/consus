//! Bounded, zero-copy ONNX protobuf document parsing.
//!
//! The reader owns ONNX format decoding without coupling document parsing to a
//! tensor runtime. Names and initializer payloads borrow the source bytes;
//! graph collections allocate only within explicit [`ParseLimits`].

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
mod document;
#[cfg(feature = "alloc")]
mod parse;
#[cfg(feature = "alloc")]
mod wire;

#[cfg(feature = "alloc")]
pub use document::{
    ElementType, Initializer, ModelDocument, Node, OperatorSet, TensorInfo, ValueInfo,
};
#[cfg(feature = "alloc")]
pub use parse::{ParseError, ParseLimits, parse_model};
