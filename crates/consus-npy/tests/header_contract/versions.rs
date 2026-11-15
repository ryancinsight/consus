//! Version selection and header text decoding of the NPY reader.

use super::{framed, trace};
use consus_npy::{Error, NpyElement, read_npy};
use std::io::Cursor;

fn dictionary_with_comment<T: NpyElement>(comment: &[u8]) -> Vec<u8> {
    let dictionary = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': (2,), }}",
        T::DTYPE
    );
    [dictionary.as_bytes(), b" # ", comment, b"\n"].concat()
}

fn rejects_unsupported_versions<T: NpyElement + std::fmt::Debug + PartialEq>(values: [T; 2]) {
    // numpy.lib.format accepts exactly (1, 0), (2, 0) and (3, 0); a larger
    // minor number and a major with no layout are rejected.
    for (major, minor, accepted) in [
        (1, 0, true),
        (2, 0, true),
        (3, 0, true),
        (1, 1, false),
        (1, 7, false),
        (1, 255, false),
        (2, 1, false),
        (3, 1, false),
        (3, 9, false),
        (0, 0, false),
        (4, 0, false),
        (255, 0, false),
    ] {
        let _case = tracing::info_span!("case", major, minor, dtype = T::DTYPE).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            framed(major, minor, &dictionary_with_comment::<T>(b""), &values)
        };
        let result = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<T>(Cursor::new(bytes))
        };
        let _span = tracing::info_span!("assert").entered();
        if accepted {
            let array = result.expect("NumPy reads this version");
            assert_eq!(array.shape(), [2]);
            assert_eq!(array.values(), values);
        } else {
            match result {
                Err(Error::InvalidFormat(message)) => {
                    assert_eq!(message, format!("unsupported NPY version {major}.{minor}"));
                }
                other => panic!("NumPy rejects version {major}.{minor}; received {other:?}"),
            }
        }
    }
}

fn decodes_headers_by_version<T: NpyElement + std::fmt::Debug + PartialEq>(values: [T; 2]) {
    // A lone 0xE9 byte is U+00E9 in Latin-1 and invalid UTF-8. NumPy decodes
    // versions 1.0 and 2.0 as Latin-1 and loads the array, and decodes 3.0 as
    // UTF-8. The two-byte UTF-8 form of U+00E9 is two Latin-1 characters in
    // 1.0 and 2.0 and one character in 3.0; inside a comment all are data.
    let latin1 = &b"caf\xe9"[..];
    let utf8 = "caf\u{e9}".as_bytes();
    for (major, comment, accepted) in [
        (1, latin1, true),
        (2, latin1, true),
        (3, latin1, false),
        (1, utf8, true),
        (2, utf8, true),
        (3, utf8, true),
    ] {
        let _case = tracing::info_span!("case", major, ?comment, dtype = T::DTYPE).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            framed(major, 0, &dictionary_with_comment::<T>(comment), &values)
        };
        let result = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<T>(Cursor::new(bytes))
        };
        let _span = tracing::info_span!("assert").entered();
        if accepted {
            let array = result.expect("header text decodes in this version");
            assert_eq!(array.shape(), [2]);
            assert_eq!(array.values(), values);
            assert!(!array.is_fortran_order());
        } else {
            match result {
                Err(Error::InvalidFormat(message)) => {
                    assert!(message.starts_with("header is not UTF-8: "), "{message:?}");
                }
                other => panic!("NumPy cannot decode this header; received {other:?}"),
            }
        }
    }
}

#[test]
fn all_supported_scalars_accept_only_numpy_versions() {
    let _trace = trace();
    let _test = tracing::info_span!("only_numpy_versions").entered();
    rejects_unsupported_versions([7_i32, -3]);
    rejects_unsupported_versions([i64::MAX, i64::MIN]);
    rejects_unsupported_versions([1.5_f32, -2.25]);
    rejects_unsupported_versions([1.5_f64, -2.25]);
}

#[test]
fn all_supported_scalars_decode_headers_by_version() {
    let _trace = trace();
    let _test = tracing::info_span!("decode_by_version").entered();
    decodes_headers_by_version([7_i32, -3]);
    decodes_headers_by_version([i64::MAX, i64::MIN]);
    decodes_headers_by_version([1.5_f32, -2.25]);
    decodes_headers_by_version([1.5_f64, -2.25]);
}

#[test]
fn latin1_text_inside_a_key_is_data_not_a_decoding_error() {
    let _trace = trace();
    let _test = tracing::info_span!("latin1_key").entered();
    let header = b"{'descr': '<i4', 'fortran_order': False, 'shape': (2,), '\xe9': 1}\n";
    let bytes = framed(1, 0, header, &[7_i32, -3]);
    let _span = tracing::info_span!("assert").entered();
    match read_npy::<i32>(Cursor::new(bytes)) {
        Err(Error::InvalidFormat(message)) => {
            assert_eq!(message, "unexpected header key \"\u{e9}\"");
        }
        other => panic!("NumPy rejects the unknown key; received {other:?}"),
    }
}
