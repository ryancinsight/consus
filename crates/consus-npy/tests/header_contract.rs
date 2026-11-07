//! Value-semantic regression witnesses for the NPY dictionary boundary.
use consus_npy::{Error, read_npy};
use std::io::Cursor;
use tracing_subscriber::fmt::format::FmtSpan;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .without_time()
            .finish(),
    )
}

fn fixture(header: &str) -> Vec<u8> {
    let mut header = header.to_owned();
    let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(
        &u16::try_from(header.len())
            .expect("small fixture header")
            .to_le_bytes(),
    );
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&7_i32.to_le_bytes());
    bytes.extend_from_slice(&(-3_i32).to_le_bytes());
    bytes
}

#[test]
fn reads_double_quoted_dictionary_keys() {
    let _trace = trace();
    let _test = tracing::info_span!("reads_double_quoted_dictionary_keys").entered();
    let bytes = {
        let _span = tracing::info_span!("arrange").entered();
        fixture(r#"{"descr": "<i4", "fortran_order": False, "shape": (2,), }"#)
    };
    let result = {
        let _span = tracing::info_span!("act").entered();
        read_npy::<i32>(Cursor::new(bytes))
    };
    let _span = tracing::info_span!("assert").entered();
    let array = result.expect("NumPy accepts either dictionary-key quote style");
    assert_eq!(array.shape(), [2]);
    assert_eq!(array.values(), [7, -3]);
    assert!(!array.is_fortran_order());
}

#[test]
fn rejects_malformed_boolean_and_shape_literals() {
    let _trace = trace();
    let _test = tracing::info_span!("rejects_malformed_boolean_and_shape_literals").entered();
    for header in [
        "{'descr': '<i4', 'fortran_order': Falsehood, 'shape': (2,), }",
        "{'descr': '<i4', 'fortran_order': False, 'shape': (2,,), }",
        "{'descr': '<i4', 'fortran_order': False, 'shape': (2), }",
    ] {
        let _case = tracing::info_span!("case", header).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            fixture(header)
        };
        let result = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<i32>(Cursor::new(bytes))
        };
        let _span = tracing::info_span!("assert").entered();
        match result {
            Err(Error::InvalidFormat(message)) => assert!(!message.is_empty()),
            other => panic!("NumPy rejects this malformed header; received {other:?}"),
        }
    }
}

fn array_fixture<T: consus_npy::NpyElement>(header: &str, major: u8, values: &[T]) -> Vec<u8> {
    let prefix = if major == 1 { 10 } else { 12 };
    let mut header = header.to_owned();
    header.extend(std::iter::repeat_n(
        ' ',
        (64 - ((prefix + header.len() + 1) % 64)) % 64,
    ));
    header.push('\n');
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.extend_from_slice(&[major, 0]);
    if major == 1 {
        bytes.extend_from_slice(
            &u16::try_from(header.len())
                .expect("bounded fixture")
                .to_le_bytes(),
        );
    } else {
        bytes.extend_from_slice(
            &u32::try_from(header.len())
                .expect("bounded fixture")
                .to_le_bytes(),
        );
    }
    bytes.extend_from_slice(header.as_bytes());
    for &value in values {
        value.write_to(&mut bytes).expect("in-memory scalar write");
    }
    bytes
}

fn permutations<T: consus_npy::NpyElement + std::fmt::Debug + PartialEq>(values: [T; 2]) {
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for mask in 0..16 {
            for major in 1..=3 {
                for fortran in [false, true] {
                    let _case =
                        tracing::info_span!("case", ?order, mask, major, fortran, dtype = T::DTYPE)
                            .entered();
                    let bytes = {
                        let _span = tracing::info_span!("arrange").entered();
                        let quote = |bit| if mask & (1 << bit) == 0 { '\'' } else { '"' };
                        let keys = ["descr", "fortran_order", "shape"];
                        let literal = [
                            format!("{q}{}{q}", T::DTYPE, q = quote(3)),
                            if fortran { "True" } else { "False" }.to_owned(),
                            "(2,)".to_owned(),
                        ];
                        let fields = order
                            .iter()
                            .map(|&i| format!("{q}{}{q}: {}", keys[i], literal[i], q = quote(i)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        array_fixture(&format!("{{{fields}}}"), major, &values)
                    };
                    let decoded = {
                        let _span = tracing::info_span!("act").entered();
                        read_npy::<T>(Cursor::new(bytes))
                            .expect("valid quoted dictionary permutation")
                    };
                    let _span = tracing::info_span!("assert").entered();
                    assert_eq!(decoded.shape(), [2]);
                    assert_eq!(decoded.values(), values);
                    assert_eq!(decoded.is_fortran_order(), fortran);
                }
            }
        }
    }
}

#[test]
fn all_supported_scalars_preserve_dictionary_permutations() {
    let _trace = trace();
    let _test = tracing::info_span!("dictionary_permutations").entered();
    permutations([7_i32, -3]);
    permutations([i64::MAX, i64::MIN]);
    permutations([1.5_f32, -2.25]);
    permutations([1.5_f64, -2.25]);
}

#[test]
fn complete_dictionary_is_required() {
    let _trace = trace();
    let _test = tracing::info_span!("complete_dictionary_is_required").entered();
    let cases = [
        (
            "'descr': '<i4', 'fortran_order': False, 'shape': (2,)",
            "expected '{'",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2,), } ignored",
            "content follows",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2,), 'ignored': 1}",
            "unexpected header key",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2,,), }",
            "invalid shape axis",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False, 'shape': (,2), }",
            "invalid shape axis",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2), }",
            "trailing comma",
        ),
        (
            "{'descr': '<i4', 'fortran_order': Trueish, 'shape': (2,), }",
            "not a boolean",
        ),
        (
            "{'descr': '<i4', 'fortran_order': Falsehood, 'shape': (2,), }",
            "not a boolean",
        ),
        ("{'descr': '<i4', 'shape': (2,), }", "missing fortran_order"),
        ("{'fortran_order': False, 'shape': (2,), }", "missing descr"),
        (
            "{'descr': '<i4', 'fortran_order': False, }",
            "missing shape",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False, 'shape': (-1,), }",
            "invalid shape axis",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2,), ",
            "unterminated",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False 'shape': (2,), }",
            "expected ','",
        ),
        (
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2,), \"",
            "unescaped newline",
        ),
    ];
    for (header, reason) in cases {
        let _case = tracing::info_span!("case", header, reason).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            fixture(header)
        };
        let result = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<i32>(Cursor::new(bytes))
        };
        let _span = tracing::info_span!("assert").entered();
        match result {
            Err(Error::InvalidFormat(message)) => {
                assert!(message.contains(reason), "{message:?} must name {reason:?}")
            }
            other => panic!("expected invalid-format diagnostic {reason:?}, received {other:?}"),
        }
    }
}

#[test]
fn decoded_strings_and_repeated_keys_follow_literal_values() {
    let _trace = trace();
    let _test = tracing::info_span!("literal_values").entered();
    for header in [
        r#"{'\x64escr': '\u003ci4', 'fortran_order': False, 'shape': (2,), }"#,
        r#"{'descr': '<i4', 'fortran_order': True, 'fortran_order': False, 'shape': (2,), }"#,
        r#"{'descr': '<f8', 'descr': '<i4', 'fortran_order': False, 'shape': (2,), }"#,
        r#"{'descr': '<i4', 'fortran_order': False, 'shape': (7,), 'shape': (2,), }"#,
    ] {
        let _case = tracing::info_span!("case", header).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            fixture(header)
        };
        let decoded = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<i32>(Cursor::new(bytes)).expect("valid literal values")
        };
        let _span = tracing::info_span!("assert").entered();
        assert_eq!(decoded.shape(), [2]);
        assert_eq!(decoded.values(), [7, -3]);
        assert!(!decoded.is_fortran_order());
    }
}

#[test]
fn scalar_and_empty_shapes_keep_exact_cardinality() {
    let _trace = trace();
    let _test = tracing::info_span!("cardinality").entered();
    for (shape, expected, values) in [
        ("()", vec![], vec![7_i32]),
        ("(0,)", vec![0], vec![]),
        ("(2, 0)", vec![2, 0], vec![]),
        ("(1, 2)", vec![1, 2], vec![7, -3]),
    ] {
        let _case = tracing::info_span!("case", shape).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            array_fixture(
                &format!("{{'descr': '<i4', 'fortran_order': False, 'shape': {shape}}}"),
                1,
                &values,
            )
        };
        let decoded = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<i32>(Cursor::new(bytes)).expect("valid shape")
        };
        let _span = tracing::info_span!("assert").entered();
        assert_eq!(decoded.shape(), expected);
        assert_eq!(decoded.values(), values);
    }
}

#[test]
fn shape_integer_literals_have_python_token_boundaries() {
    let _trace = trace();
    let _test = tracing::info_span!("integer_literals").entered();
    for axis in [
        "2", "+2", "+ 2", "0x2", "0X_2", "0o2", "0b10", "2_0", "((2))",
    ] {
        let _case = tracing::info_span!("accepted", axis).entered();
        let expected = if axis == "2_0" { 20 } else { 2 };
        let values = (0..expected).collect::<Vec<i32>>();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            array_fixture(
                &format!("{{'descr': '<i4', 'fortran_order': False, 'shape': ({axis},)}}"),
                1,
                &values,
            )
        };
        let array = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<i32>(Cursor::new(bytes)).expect("valid integer literal")
        };
        let _span = tracing::info_span!("assert").entered();
        assert_eq!(
            array.shape(),
            [usize::try_from(expected).expect("small positive value")]
        );
        assert_eq!(array.values(), values);
    }
    for axis in [
        "02", "0_2", "2_", "_2", "2__0", "-2", "--2", "2.0", "2 + 0", "0x", "0b2",
    ] {
        let _case = tracing::info_span!("rejected", axis).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            fixture(&format!(
                "{{'descr': '<i4', 'fortran_order': False, 'shape': ({axis},)}}"
            ))
        };
        let result = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<i32>(Cursor::new(bytes))
        };
        let _span = tracing::info_span!("assert").entered();
        match result {
            Err(Error::InvalidFormat(message)) => assert!(message.contains("invalid shape axis")),
            other => panic!("invalid integer accepted: {other:?}"),
        }
    }
}

#[test]
fn npz_members_use_the_same_dictionary_boundary() {
    let _trace = trace();
    let _test = tracing::info_span!("npz_header_boundary").entered();
    let archive = {
        let _span = tracing::info_span!("arrange").entered();
        let mut writer = consus_zip::ZipWriter::new(Vec::new());
        writer
            .add_entry(
                "good.npy",
                &fixture(r#"{"descr": "<i4", "fortran_order": False, "shape": (2,), }"#),
                consus_zip::Method::Deflated,
            )
            .expect("archive valid member");
        writer
            .add_entry(
                "bad.npy",
                &fixture("{'descr': '<i4', 'fortran_order': Falsehood, 'shape': (2,), }"),
                consus_zip::Method::Stored,
            )
            .expect("archive malformed member");
        writer.finish().expect("finish real ZIP")
    };
    let (good, bad) = {
        let _span = tracing::info_span!("act").entered();
        let mut reader =
            consus_npy::NpzReader::new(Cursor::new(archive)).expect("valid ZIP container");
        (reader.by_name::<i32>("good"), reader.by_name::<i32>("bad"))
    };
    let _span = tracing::info_span!("assert").entered();
    let good = good.expect("valid member dictionary");
    assert_eq!(good.shape(), [2]);
    assert_eq!(good.values(), [7, -3]);
    match bad {
        Err(Error::InvalidFormat(message)) => assert!(message.contains("not a boolean")),
        other => panic!("invalid member header accepted: {other:?}"),
    }
}
