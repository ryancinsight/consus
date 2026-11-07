//! Public NPY and NPZ regressions retained by independent review.
use consus_npy::{NpzReader, read_npy};
use consus_zip::{Method, ZipWriter};
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

const FIXTURES: [&[u8]; 3] = [
    include_bytes!("data/valid-literal-1.npy"),
    include_bytes!("data/valid-literal-2.npy"),
    include_bytes!("data/valid-literal-3.npy"),
];

#[test]
fn preserves_reviewed_literal_framing_in_npy_and_npz() {
    let _trace = trace();
    let _test = tracing::info_span!("reviewed_literal_framing").entered();
    for (index, fixture) in FIXTURES.into_iter().enumerate() {
        let _case = tracing::info_span!("case", index).entered();
        let archive = {
            let _span = tracing::info_span!("arrange").entered();
            let mut writer = ZipWriter::new(Vec::new());
            writer
                .add_entry("array.npy", fixture, Method::Deflated)
                .expect("write real NPZ member");
            writer.finish().expect("finish real NPZ archive")
        };
        let (plain, archived) = {
            let _span = tracing::info_span!("act").entered();
            let mut reader = NpzReader::new(Cursor::new(archive)).expect("valid ZIP container");
            (read_npy::<i32>(fixture), reader.by_name::<i32>("array"))
        };
        let _span = tracing::info_span!("assert").entered();
        for result in [plain, archived] {
            let array = result.expect("reviewed valid Python-literal dictionary remains readable");
            assert_eq!(array.shape(), [2]);
            assert_eq!(array.values(), [7, -3]);
            assert!(!array.is_fortran_order());
        }
    }
}

fn encoded(header: &str) -> Vec<u8> {
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(
        &u16::try_from(header.len())
            .expect("small literal header")
            .to_le_bytes(),
    );
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&7_i32.to_le_bytes());
    bytes.extend_from_slice(&(-3_i32).to_le_bytes());
    bytes
}

#[test]
fn comments_are_trivia_only_outside_quoted_values() {
    let _trace = trace();
    let _test = tracing::info_span!("quoted_comment_boundaries").entered();
    let valid = "(({'de' 'scr': '<i' '4', # ignored ) , 'shape'\n'fortran_order': False, 'shape': (2 # ignored ) ,\n,),})) # trailing comment\n";
    let bytes = {
        let _span = tracing::info_span!("arrange").entered();
        encoded(valid)
    };
    let result = {
        let _span = tracing::info_span!("act").entered();
        read_npy::<i32>(Cursor::new(bytes))
    };
    {
        let _span = tracing::info_span!("assert").entered();
        let array = result.expect("comments and literal concatenation preserve the dictionary");
        assert_eq!(array.shape(), [2]);
        assert_eq!(array.values(), [7, -3]);
    }
    let bytes = {
        let _span = tracing::info_span!("arrange").entered();
        encoded("{'descr': '<i4' '#not trivia', 'fortran_order': False, 'shape': (2,),}")
    };
    let result = {
        let _span = tracing::info_span!("act").entered();
        read_npy::<i32>(Cursor::new(bytes))
    };
    let _span = tracing::info_span!("assert").entered();
    match result {
        Err(consus_npy::Error::DtypeMismatch { stored, requested }) => {
            assert_eq!(stored, "<i4#not trivia");
            assert_eq!(requested, "<i4");
        }
        other => panic!("quoted comment marker must remain dtype data: {other:?}"),
    }
}

#[test]
fn framing_does_not_relax_malformed_token_rejection() {
    let _trace = trace();
    let _test = tracing::info_span!("framed_malformed_tokens").entered();
    for header in [
        "({'descr': '<i4', # comment\n'fortran_order': Falsehood, 'shape': (2,),})",
        "({'descr': '<i4', 'fortran_order': False, 'shape': (2,,),})",
        "({'descr': '<i4' + '', 'fortran_order': False, 'shape': (2,),})",
        "({'descr': '<i4', 'fortran_order': False, 'shape': (2,),}",
        "{'descr': '<i4', 'fortran_order': False, 'shape': (2 # comment\n0,),}",
    ] {
        let _case = tracing::info_span!("case", header).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            encoded(header)
        };
        let result = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<i32>(Cursor::new(bytes))
        };
        let _span = tracing::info_span!("assert").entered();
        match result {
            Err(consus_npy::Error::InvalidFormat(message)) => assert!(!message.is_empty()),
            other => panic!("malformed expression accepted: {other:?}"),
        }
    }
}

#[test]
fn adjacent_text_literals_preserve_prefix_and_quote_forms() {
    let _trace = trace();
    let _test = tracing::info_span!("adjacent_text_literal_forms").entered();
    for header in [
        r#"{'descr': '<i4' r'' u"", 'fortran_order': False, 'shape': (2,)}"#,
        r#"{'descr': '<i4' '''''' """""", 'fortran_order': False, 'shape': (2,)}"#,
        r#"{'descr': '<i4' R'' U"", 'fortran_order': False, 'shape': (2,)}"#,
    ] {
        let _case = tracing::info_span!("case", header).entered();
        let bytes = {
            let _span = tracing::info_span!("arrange").entered();
            encoded(header)
        };
        let result = {
            let _span = tracing::info_span!("act").entered();
            read_npy::<i32>(Cursor::new(bytes))
        };
        let _span = tracing::info_span!("assert").entered();
        let array = result.expect("adjacent empty text literals preserve the dtype");
        assert_eq!(array.shape(), [2]);
        assert_eq!(array.values(), [7, -3]);
    }
}
