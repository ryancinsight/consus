#[cfg(feature = "alloc")]
use alloc::string::String;

/// Compute the element size in bytes for a given dtype string.
///
/// Returns `None` for variable-length types (vlen, unicode, bytes).
///
/// ## Supported dtypes
///
/// - Boolean: 1 byte
/// - Integer: 1, 2, 4, 8 bytes (signed and unsigned)
/// - Float: 2, 4, 8, 16 bytes
/// - Complex: 8, 16 bytes (real+imag)
/// - String: fixed (size encoded in dtype) or variable (None)
/// - Bitfield: size encoded in dtype
/// - Time: 8 bytes
/// - Reference: 8 bytes
/// - Enum: base type size
/// - Array: product of dims × base size
/// - Compound: sum of field sizes
/// - Opaque: size encoded in dtype
#[cfg(feature = "alloc")]
pub fn dtype_to_element_size(dtype: &str) -> Option<usize> {
    let dtype = dtype.trim().to_lowercase();

    match dtype.as_str() {
        "bool" | "bool_" => return Some(1),
        "int8" => return Some(1),
        "uint8" | "i1" | "u1" => return Some(1),
        "int16" | "i2" => return Some(2),
        "uint16" | "u2" => return Some(2),
        "int32" | "i4" => return Some(4),
        "uint32" | "u4" => return Some(4),
        "int64" | "i8" => return Some(8),
        "uint64" | "u8" => return Some(8),
        "float16" | "f2" => return Some(2),
        "float32" | "f4" => return Some(4),
        "float64" | "f8" => return Some(8),
        "complex64" | "c8" => return Some(8),
        "complex128" | "c16" => return Some(16),
        "reference" | "object" => return Some(8),
        "string" | "utf8" | "unicode" => return None,
        "bytes" | "binary" => return None,
        value if value.starts_with("vlen<") => return None,
        _ => {}
    }

    let bytes = dtype.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let (rest, _) = match bytes[0] {
        b'<' | b'>' | b'=' | b'|' => (&bytes[1..], bytes[0]),
        _ => (bytes, b'|'),
    };

    if rest.is_empty() {
        return None;
    }

    match rest[0] {
        b'b' => Some(1),
        b'i' | b'u' | b'f' | b'c' | b's' | b'v' | b'B' | b'e' => parse_size_suffix(rest),
        b'o' => Some(8),
        b'm' | b't' => Some(8),
        _ => None,
    }
}

#[cfg(feature = "alloc")]
fn parse_size_suffix(rest: &[u8]) -> Option<usize> {
    if rest.len() < 2 {
        return if rest[0] == b'v' { Some(1) } else { None };
    }

    let size_str: String = rest[1..]
        .iter()
        .take_while(|&&byte| byte.is_ascii_digit())
        .map(|&byte| byte as char)
        .collect();
    size_str.parse().ok()
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_to_element_size_v3_named() {
        assert_eq!(dtype_to_element_size("bool"), Some(1));
        assert_eq!(dtype_to_element_size("int8"), Some(1));
        assert_eq!(dtype_to_element_size("uint16"), Some(2));
        assert_eq!(dtype_to_element_size("int32"), Some(4));
        assert_eq!(dtype_to_element_size("uint64"), Some(8));
        assert_eq!(dtype_to_element_size("float32"), Some(4));
        assert_eq!(dtype_to_element_size("float64"), Some(8));
        assert_eq!(dtype_to_element_size("complex64"), Some(8));
        assert_eq!(dtype_to_element_size("complex128"), Some(16));
        assert_eq!(dtype_to_element_size("float16"), Some(2));
    }

    #[test]
    fn test_dtype_to_element_size_v2_numpy() {
        assert_eq!(dtype_to_element_size("<f8"), Some(8));
        assert_eq!(dtype_to_element_size("<i4"), Some(4));
        assert_eq!(dtype_to_element_size(">u2"), Some(2));
        assert_eq!(dtype_to_element_size("<c8"), Some(8));
        assert_eq!(dtype_to_element_size("|S10"), Some(10));
        assert_eq!(dtype_to_element_size("<V16"), Some(16));
    }

    #[test]
    fn test_dtype_to_element_size_variable() {
        assert_eq!(dtype_to_element_size("string"), None);
        assert_eq!(dtype_to_element_size("utf8"), None);
        assert_eq!(dtype_to_element_size("vlen<unicode>"), None);
        assert_eq!(dtype_to_element_size("vlen<uint8>"), None);
        assert_eq!(dtype_to_element_size("bytes"), None);
    }
}
