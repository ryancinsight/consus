//! Fractal heap ID model and byte-order-aware decode.

use super::{FractalHeapHeader, read_uint_le};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use consus_core::{Error, Result};

// ---------------------------------------------------------------------------
// Heap ID
// ---------------------------------------------------------------------------

/// Decoded fractal heap object ID.
///
/// The encoding scheme is selected by bits 6-7 of the first byte of the raw
/// heap ID (see module-level documentation).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub enum FractalHeapId {
    /// Managed object: located at `offset` within the managed address space
    /// of the fractal heap, with the given `length` in bytes.
    Managed {
        /// Byte offset within the heap's managed space.
        offset: u64,
        /// Object length in bytes.
        length: u64,
    },

    /// Tiny object: the object data is stored inline in the heap ID bytes
    /// themselves (max ~14 bytes depending on heap-ID length).
    Tiny {
        /// Inline object data.
        data: Vec<u8>,
    },

    /// Huge object: stored outside the managed space and indexed by a v2
    /// B-tree.  `btree_key` is the lookup key (or a direct address when the
    /// heap header's flags bit 0 is set).
    Huge {
        /// B-tree key or direct file address.
        btree_key: u64,
    },
}

/// Decode a raw heap ID according to the fractal heap header parameters.
///
/// # Encoding
///
/// ```text
/// byte 0:  bits 6-7 = type  (0 = managed, 1 = tiny, 2 = huge)
///          bits 4-5 = version (must be 0)
///          bits 0-3 = type-specific
/// ```
///
/// **Managed (type 0)**
///
/// | Offset              | Size                              | Field  |
/// |---------------------|-----------------------------------|--------|
/// | 1                   | ⌈max_heap_size_bits / 8⌉          | Offset |
/// | 1 + offset_bytes    | heap_id_length − 1 − offset_bytes | Length |
///
/// **Tiny (type 1)**
///
/// Remaining bytes (1 .. heap_id_length) carry inline object data.
///
/// **Huge (type 2)**
///
/// Remaining bytes encode the v2 B-tree key as a little-endian integer.
///
/// # Errors
///
/// - [`Error::InvalidFormat`] on empty input, unknown type, or truncated ID.
#[cfg(feature = "alloc")]
pub fn decode_heap_id(id_bytes: &[u8], header: &FractalHeapHeader) -> Result<FractalHeapId> {
    if id_bytes.is_empty() {
        return Err(Error::InvalidFormat {
            message: String::from("empty fractal heap ID"),
        });
    }

    let first = id_bytes[0];
    let id_type = (first >> 6) & 0x03;

    match id_type {
        // -- Managed ---------------------------------------------------------
        0 => {
            let offset_bytes = (header.max_heap_size_bits as usize).div_ceil(8);
            let min_len = 1 + offset_bytes;
            if id_bytes.len() < min_len {
                return Err(Error::InvalidFormat {
                    message: String::from("managed heap ID too short"),
                });
            }
            let offset = read_uint_le(&id_bytes[1..], offset_bytes);
            let length_bytes = id_bytes.len() - 1 - offset_bytes;
            let length = if length_bytes > 0 {
                read_uint_le(&id_bytes[1 + offset_bytes..], length_bytes)
            } else {
                0
            };
            Ok(FractalHeapId::Managed { offset, length })
        }

        // -- Tiny ------------------------------------------------------------
        1 => {
            // Bits 0-3 of byte 0 encode (actual_length − 1) for normal tiny
            // objects.  The inline data occupies bytes 1..heap_id_length.
            let data = id_bytes[1..].to_vec();
            Ok(FractalHeapId::Tiny { data })
        }

        // -- Huge ------------------------------------------------------------
        2 => {
            let key_bytes = id_bytes.len() - 1;
            if key_bytes == 0 {
                return Err(Error::InvalidFormat {
                    message: String::from("huge heap ID has no key bytes"),
                });
            }
            let btree_key = read_uint_le(&id_bytes[1..], key_bytes.min(8));
            Ok(FractalHeapId::Huge { btree_key })
        }

        _ => Err(Error::InvalidFormat {
            message: alloc::format!("unknown fractal heap ID type: {id_type}"),
        }),
    }
}
