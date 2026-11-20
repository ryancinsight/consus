//! Ordinary, well-formed numeric files for opt-in representation admission.
#![cfg(feature = "std")]

use consus_core::{ByteOrder, Datatype, Shape, test_support::assert_rejects};
use consus_hdf5::datatype::IeeeFloatEncoding;
use consus_hdf5::file::{Hdf5File, reader, writer};
use consus_hdf5::object_header::message_types;
use consus_io::MemCursor;
use core::num::NonZeroUsize;
use tracing::{info_span, subscriber::DefaultGuard};
use tracing_subscriber::fmt::format::FmtSpan;

fn trace() -> DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(FmtSpan::FULL)
            .finish(),
    )
}

fn float(width: usize, byte_order: ByteOrder) -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(width).expect("fixture width"),
        byte_order,
    }
}

// Re-emit the entire object header with the provider writer, preserving valid
// header lengths/checksums. No local HDF5 parser or checksum implementation.
fn fixture(
    datatype: &Datatype,
    name: &str,
    edit: impl FnOnce(&mut [u8]),
) -> (Hdf5File<MemCursor>, u64) {
    let _arrange = info_span!("arrange", name, ?datatype).entered();
    let shape = if name == "geometry" {
        Shape::fixed(&[15])
    } else {
        Shape::fixed(&[2, 2, 2])
    };
    let count = if name == "geometry" { 15 } else { 8 };
    // Zero is representable in every valid numeric layout used below.
    let payload = vec![0; count * datatype.element_size().expect("fixed width")];
    let mut builder = writer::Hdf5FileBuilder::new(writer::FileCreationProps::default());
    builder
        .add_dataset(
            name,
            datatype,
            &shape,
            &payload,
            &writer::DatasetCreationProps::default(),
        )
        .expect("ordinary zero dataset");
    let bytes = builder.finish().expect("valid HDF5 file");
    let original = Hdf5File::open(MemCursor::from_bytes(bytes.clone())).expect("open fixture");
    let (_, address, _) = original
        .list_root_group()
        .expect("root")
        .into_iter()
        .find(|(n, _, _)| n == name)
        .expect("named dataset");
    let mut header =
        reader::read_object_header(original.source(), address, original.context()).expect("header");
    let message = header
        .messages
        .iter_mut()
        .find(|m| m.message_type == message_types::DATATYPE)
        .expect("datatype message");
    if let Datatype::Float { bits, byte_order } = datatype {
        // Independent literal oracle from the HDF5/IEEE field definitions, not
        // the new admission method or a second invocation of encode_datatype.
        let properties: [u8; 12] = match bits.get() {
            16 => [0, 0, 16, 0, 10, 5, 0, 10, 15, 0, 0, 0],
            32 => [0, 0, 32, 0, 23, 8, 0, 23, 127, 0, 0, 0],
            64 => [0, 0, 64, 0, 52, 11, 0, 52, 255, 3, 0, 0],
            _ => panic!("fixture supports only IEEE interchange widths"),
        };
        assert_eq!(&message.data[8..20], &properties);
        assert_eq!(
            message.data[1],
            if *byte_order == ByteOrder::LittleEndian {
                0x20
            } else {
                0x21
            }
        );
        assert_eq!(
            message.data[2],
            u8::try_from(bits.get() - 1).expect("IEEE sign bit")
        );
    }
    edit(&mut message.data);
    let messages: Vec<_> = header
        .messages
        .iter()
        .map(|m| (m.message_type, m.data.as_slice()))
        .collect();
    let mut state = writer::WriteState::new(writer::FileCreationProps::default());
    state.eof = address;
    let mut sink = MemCursor::from_bytes(bytes);
    assert_eq!(
        writer::write_object_header_v2(&mut sink, &mut state, &messages)
            .expect("re-emit valid header"),
        address
    );
    let file = Hdf5File::open(sink).expect("reconstructed standard file");
    assert_eq!(
        file.dataset_at(address)
            .expect("ordinary metadata remains supported")
            .shape,
        shape
    );
    (file, address)
}

fn rejection(edit: impl FnOnce(&mut [u8]), dataset: &str) {
    let canonical = float(64, ByteOrder::LittleEndian);
    let (file, address) = fixture(&canonical, dataset, edit);
    let result = {
        let _act = info_span!("act").entered();
        file.dataset_at_with_ieee_float_encoding(address, IeeeFloatEncoding::Binary64LittleEndian)
    };
    let _assert = info_span!("assert").entered();
    // The general API is intentionally unchanged: its projection still loses
    // this distinction. Only the explicitly checked admission rejects it.
    assert_eq!(
        file.dataset_at(address)
            .expect("general API still supports custom float")
            .datatype,
        canonical
    );
    assert_rejects(
        &result,
        "dataset datatype does not use requested Binary64LittleEndian representation",
    );
}

#[test]
fn canonical_widths_orders_and_message_versions_are_admitted() {
    let _trace = trace();
    let _test = info_span!("canonical_ieee").entered();
    for (width, order, expected) in [
        (
            16,
            ByteOrder::LittleEndian,
            IeeeFloatEncoding::Binary16LittleEndian,
        ),
        (
            16,
            ByteOrder::BigEndian,
            IeeeFloatEncoding::Binary16BigEndian,
        ),
        (
            32,
            ByteOrder::LittleEndian,
            IeeeFloatEncoding::Binary32LittleEndian,
        ),
        (
            32,
            ByteOrder::BigEndian,
            IeeeFloatEncoding::Binary32BigEndian,
        ),
        (
            64,
            ByteOrder::LittleEndian,
            IeeeFloatEncoding::Binary64LittleEndian,
        ),
        (
            64,
            ByteOrder::BigEndian,
            IeeeFloatEncoding::Binary64BigEndian,
        ),
    ] {
        for version in 1u8..=5 {
            let _case = info_span!("case", width, ?order, version).entered();
            let datatype = float(width, order);
            let (file, address) = fixture(&datatype, "volume", |d| d[0] = (version << 4) | 1);
            let dataset = {
                let _act = info_span!("act").entered();
                file.dataset_at_with_ieee_float_encoding(address, expected)
                    .expect("canonical representation")
            };
            let _assert = info_span!("assert").entered();
            assert_eq!(dataset.datatype, datatype);
            assert_eq!(dataset.object_header_address, address);
            assert_eq!(dataset.shape.current_dims().as_slice(), [2, 2, 2]);
            assert_eq!(
                file.read_dataset_raw(&dataset, None)
                    .expect("ordinary zero payload"),
                vec![0; width]
            );
        }
    }
}

#[test]
fn unused_padding_preferences_do_not_change_full_width_ieee_values() {
    let _trace = trace();
    let _test = info_span!("equivalent_padding").entered();
    for padding in 0u8..8 {
        let (file, address) = fixture(&float(64, ByteOrder::LittleEndian), "volume", |d| {
            d[1] |= padding << 1
        });
        let dataset = {
            let _act = info_span!("act", padding).entered();
            file.dataset_at_with_ieee_float_encoding(
                address,
                IeeeFloatEncoding::Binary64LittleEndian,
            )
            .expect("no unused bits")
        };
        let _assert = info_span!("assert").entered();
        assert_eq!(
            file.read_dataset_raw(&dataset, None).expect("zero payload"),
            vec![0; 64]
        );
    }
}

#[test]
fn alternate_exponent_bias_is_rejected_for_volume_and_geometry() {
    let _trace = trace();
    let _test = info_span!("alternate_bias").entered();
    for name in ["volume", "geometry"] {
        rejection(|d| d[16..20].copy_from_slice(&1022u32.to_le_bytes()), name);
    }
}

#[test]
fn reordered_nonoverlapping_fields_are_not_ieee_storage() {
    let _trace = trace();
    let _test = info_span!("reordered_fields").entered();
    // Sign at bit 0; 52 mantissa bits at 1..52; 11 exponent bits at 53..63.
    rejection(
        |d| {
            d[2] = 0;
            d[12] = 53;
            d[14] = 1;
        },
        "volume",
    );
}

#[test]
fn alternate_exponent_and_mantissa_widths_are_rejected() {
    let _trace = trace();
    let _test = info_span!("alternate_fields").entered();
    // 53 mantissa bits, 10 exponent bits and one sign bit, still 64 total.
    rejection(
        |d| {
            d[12] = 53;
            d[13] = 10;
            d[15] = 53;
            d[16..20].copy_from_slice(&511u32.to_le_bytes());
        },
        "volume",
    );
}

#[test]
fn explicit_or_absent_mantissa_normalization_is_rejected() {
    let _trace = trace();
    let _test = info_span!("normalization").entered();
    for normalization in [0, 0x10] {
        rejection(|d| d[1] = normalization, "volume");
    }
}

#[test]
fn padded_binary32_in_eight_byte_storage_is_not_binary64() {
    let _trace = trace();
    let _test = info_span!("padded_float").entered();
    rejection(
        |d| {
            d[2] = 31;
            d[10..12].copy_from_slice(&32u16.to_le_bytes());
            d[12] = 23;
            d[13] = 8;
            d[15] = 23;
            d[16..20].copy_from_slice(&127u32.to_le_bytes());
        },
        "volume",
    );
}

#[test]
fn ordinary_nonmatching_width_order_and_integer_are_rejected_only_by_opt_in() {
    let _trace = trace();
    let _test = info_span!("other_numeric_types").entered();
    for datatype in [
        float(32, ByteOrder::LittleEndian),
        float(64, ByteOrder::BigEndian),
        Datatype::Integer {
            bits: NonZeroUsize::new(64).expect("width"),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        },
    ] {
        let (file, address) = fixture(&datatype, "volume", |_| {});
        let result = {
            let _act = info_span!("act").entered();
            file.dataset_at_with_ieee_float_encoding(
                address,
                IeeeFloatEncoding::Binary64LittleEndian,
            )
        };
        let _assert = info_span!("assert").entered();
        assert_eq!(
            file.dataset_at(address).expect("general metadata").datatype,
            datatype
        );
        assert_rejects(
            &result,
            "dataset datatype does not use requested Binary64LittleEndian representation",
        );
    }
}

#[test]
fn shifted_significant_region_is_not_canonical_binary64() {
    let _trace = trace();
    let _test = info_span!("shifted_significant_region").entered();
    // A valid binary32 significant region at storage bits 8..39 in an 8-byte
    // element. Field positions are relative to that significant region.
    rejection(
        |d| {
            d[2] = 31;
            d[8..10].copy_from_slice(&8u16.to_le_bytes());
            d[10..12].copy_from_slice(&32u16.to_le_bytes());
            d[12] = 23;
            d[13] = 8;
            d[15] = 23;
            d[16..20].copy_from_slice(&127u32.to_le_bytes());
        },
        "volume",
    );
}

#[test]
fn vax_byte_order_is_not_mistaken_for_big_endian() {
    let _trace = trace();
    let _test = info_span!("vax_byte_order").entered();
    let datatype = float(64, ByteOrder::BigEndian);
    let (file, address) = fixture(&datatype, "volume", |d| {
        d[0] = 0x31;
        d[1] |= 0x40;
    });
    let result = {
        let _act = info_span!("act").entered();
        file.dataset_at_with_ieee_float_encoding(address, IeeeFloatEncoding::Binary64BigEndian)
    };
    let _assert = info_span!("assert").entered();
    assert_eq!(
        file.dataset_at(address)
            .expect("legacy projection remains unchanged")
            .datatype,
        datatype
    );
    assert_rejects(
        &result,
        "dataset datatype does not use requested Binary64BigEndian representation",
    );
}

#[test]
fn checked_lookup_does_not_read_the_object_header_twice() {
    use consus_io::ReadAt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counted<'a> {
        source: &'a MemCursor,
        reads: &'a AtomicUsize,
    }
    impl ReadAt for Counted<'_> {
        fn read_at(&self, offset: u64, bytes: &mut [u8]) -> consus_core::Result<()> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.source.read_at(offset, bytes)
        }
    }

    let _trace = trace();
    let _test = info_span!("metadata_read_cost").entered();
    let (source, address) = fixture(&float(64, ByteOrder::LittleEndian), "volume", |_| {});
    let reads = AtomicUsize::new(0);
    let file = Hdf5File::open(Counted {
        source: source.source(),
        reads: &reads,
    })
    .expect("counted source");
    let (ordinary, checked, ordinary_reads, checked_reads) = {
        let _act = info_span!("act").entered();
        reads.store(0, Ordering::Relaxed);
        let ordinary = file.dataset_at(address).expect("ordinary metadata");
        let ordinary_reads = reads.load(Ordering::Relaxed);
        reads.store(0, Ordering::Relaxed);
        let checked = file
            .dataset_at_with_ieee_float_encoding(address, IeeeFloatEncoding::Binary64LittleEndian)
            .expect("checked metadata");
        let checked_reads = reads.load(Ordering::Relaxed);
        (ordinary, checked, ordinary_reads, checked_reads)
    };
    let _assert = info_span!("assert", ordinary_reads, checked_reads).entered();
    assert!(ordinary_reads > 0);
    assert_eq!(checked_reads, ordinary_reads);
    assert_eq!(checked.datatype, ordinary.datatype);
    assert_eq!(checked.shape, ordinary.shape);
    assert_eq!(checked.data_address, ordinary.data_address);
}
