//! Chunk key encoding strategies.
//!
//! ### Key Encoding
//!
//! - **v2**: dot-separated indices (e.g., `0.0.0`)
//! - **v3 default**: slash-separated with `c/` prefix (e.g., `c/0/0/0`)
//! - **v3 v2-compat**: dot-separated (e.g., `0.0.0`)

#[cfg(feature = "alloc")]
use alloc::string::String;

/// Separator used in chunk key encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKeySeparator {
    /// Dot separator (v2 default, v3 v2-compat).
    Dot,
    /// Slash separator (v3 default).
    Slash,
}

/// Generate the chunk key for given grid coordinates.
///
/// ## Example
///
/// `chunk_key(&[3, 1, 4], ChunkKeySeparator::Dot)` -> `"3.1.4"`
/// `chunk_key(&[3, 1, 4], ChunkKeySeparator::Slash)` -> `"c/3/1/4"`
#[cfg(feature = "alloc")]
pub fn chunk_key(coords: &[usize], separator: ChunkKeySeparator) -> String {
    use core::fmt::Write;

    let mut key = String::new();
    match separator {
        ChunkKeySeparator::Dot => {
            for (i, c) in coords.iter().enumerate() {
                if i > 0 {
                    key.push('.');
                }
                write!(&mut key, "{c}").expect("write to String");
            }
        }
        ChunkKeySeparator::Slash => {
            key.push('c');
            for c in coords {
                write!(&mut key, "/{c}").expect("write to String");
            }
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_chunk_key() {
        let key = chunk_key(&[3, 1, 4], ChunkKeySeparator::Dot);
        assert_eq!(key, "3.1.4");
    }

    #[test]
    fn v3_chunk_key() {
        let key = chunk_key(&[3, 1, 4], ChunkKeySeparator::Slash);
        assert_eq!(key, "c/3/1/4");
    }

    #[test]
    fn scalar_chunk_key() {
        let key = chunk_key(&[], ChunkKeySeparator::Dot);
        assert_eq!(key, "");
    }
}
