//! Adversarial-input acceptance tests for the HDF5 parser trust boundary.
//!
//! Each test drives a hostile metadata value through a parser that a fuzzer
//! reaches from `Hdf5File::open` and asserts a **typed error**. The failure
//! mode each one guards against is uncatchable in Rust — a capacity-overflow
//! panic, an allocator abort, or a stack overflow — so a passing assertion
//! here is the only evidence that the bound is present.
//!
//! ## Why these values
//!
//! - `u64::MAX` record and byte counts overflow the `Layout` computation and
//!   panic deterministically on every host. Intermediate values (`1 << 40`,
//!   say) instead reach the allocator, return null, and `abort()` the whole
//!   process — a stronger failure that cannot be observed from inside a test
//!   harness. Both come from the same missing bound; the panicking value is
//!   the one a committed test can assert on.
//! - The nesting depth exceeds `ParseBudget::max_depth` by two orders of
//!   magnitude, so the rejection is unambiguous rather than borderline.

#![cfg(feature = "alloc")]

use consus_core::{Error, ParseBudget};
use consus_hdf5::address::ParseContext;
use consus_hdf5::btree::v2::{BTreeV2Header, collect_all_records, find_huge_object_record};
use consus_hdf5::dataset::chunk::{ChunkLocation, read_chunk_raw};
use consus_hdf5::datatype::compound::parse_datatype;
use consus_hdf5::heap::fractal::{FractalHeapHeader, read_huge_object, read_managed_object};
use consus_hdf5::heap::global::GlobalHeapCollection;
use consus_io::MemCursor;

/// A resource-limit rejection, as opposed to any other typed failure.
#[track_caller]
fn assert_resource_limit(error: &Error, what: &str) {
    assert!(
        matches!(error, Error::ResourceLimit { .. }),
        "{what}: expected Error::ResourceLimit, got {error:?}"
    );
}

fn fractal_header(root_indirect_rows: u16, table_width: u16) -> FractalHeapHeader {
    FractalHeapHeader {
        heap_id_length: 7,
        io_filter_size: 0,
        flags: 0,
        max_managed_object_size: 1024,
        huge_object_btree_address: 0,
        free_managed_space: 0,
        free_space_manager_address: 0,
        managed_space: 256,
        allocated_managed_space: 256,
        managed_object_count: 1,
        huge_object_size: 0,
        huge_object_count: 0,
        tiny_object_size: 0,
        tiny_object_count: 0,
        table_width,
        starting_rows: 1,
        starting_block_size: 256,
        max_direct_block_size: 65_536,
        max_heap_size_bits: 16,
        root_block_address: 0,
        root_indirect_rows,
    }
}

// ---------------------------------------------------------------------------
// Fix 1 — length-field-driven unbounded allocation
// ---------------------------------------------------------------------------

/// `total_records = u64::MAX` must not reach `Vec::with_capacity`.
///
/// Before the bound, `collect_all_records` opened with
/// `Vec::with_capacity(header.total_records as usize)`, so an 8-byte header
/// field chose the allocation size with no reference to the file's real
/// extent. `usize::MAX * size_of::<BTreeV2Record>()` exceeds `isize::MAX`,
/// producing a capacity-overflow panic before a single record was read.
///
/// `total_records` is a *hint* the traversal then verifies, not a constraint
/// the format requires to be honest, so the fix clamps the reservation rather
/// than rejecting the file. The observable contract is therefore that the
/// parse proceeds and rejects the node on its own merits — here, a leaf whose
/// signature is not "BTLF". Reaching any `Err` at all is what distinguishes
/// the fixed code: on the parent commit this call panicked instead of
/// returning.
#[test]
fn btree_v2_total_records_u64_max_is_rejected() {
    let ctx = ParseContext::new(8, 8);
    let header = BTreeV2Header {
        record_type: consus_hdf5::btree::v2::record_type::LINK_NAME,
        node_size: 4096,
        record_size: 16,
        depth: 0,
        // A real root address, so the empty-tree short-circuit does not hide
        // the allocation.
        root_address: 0,
        root_num_records: 1,
        split_percent: 98,
        merge_percent: 40,
        total_records: u64::MAX,
    };
    // The source is far smaller than the declared record count: 4 KiB of file
    // cannot hold 2^64 records, which is exactly what the bound must notice.
    let source = MemCursor::from_bytes(vec![0u8; 4096]);

    let error = collect_all_records(&source, &header, &ctx)
        .expect_err("u64::MAX total_records must not be reserved");
    assert!(
        matches!(error, Error::InvalidFormat { .. }),
        "btree v2 total_records: expected the traversal to reject the node, got {error:?}"
    );
}

/// A node size larger than the budget must not be zero-filled.
///
/// `BTreeV2LeafNode::parse` and `BTreeV2InternalNode::parse` both open with
/// `vec![0u8; header.node_size as usize]`, a `u32` read straight from the
/// B-tree header.
#[test]
fn btree_v2_node_size_beyond_budget_is_rejected() {
    let ctx = ParseContext::new(8, 8);
    let header = BTreeV2Header {
        record_type: consus_hdf5::btree::v2::record_type::LINK_NAME,
        node_size: u32::MAX,
        record_size: 16,
        depth: 0,
        root_address: 0,
        root_num_records: 1,
        split_percent: 98,
        merge_percent: 40,
        total_records: 1,
    };
    let source = MemCursor::from_bytes(vec![0u8; 4096]);

    let error = collect_all_records(&source, &header, &ctx)
        .expect_err("u32::MAX node_size must not be zero-filled");
    assert_resource_limit(&error, "btree v2 node_size");
}

/// An oversized chunk `location.size` must not be zero-filled.
///
/// The value comes from a chunk B-tree record, so it is attacker-chosen in
/// any file that reaches `read_chunked_dataset_all_bytes`. The allocation
/// preceded the read that would have proved the bytes exist.
#[test]
fn chunk_disk_size_beyond_budget_is_rejected() {
    let source = MemCursor::from_bytes(vec![0u8; 64]);
    let location = ChunkLocation {
        address: 0,
        size: u64::MAX,
        filter_mask: 0,
    };
    let registry = consus_compression::DefaultCodecRegistry::new();

    let error = read_chunk_raw(&source, &location, 64, 1, &[], &registry, None)
        .expect_err("u64::MAX chunk size must not be zero-filled");
    assert_resource_limit(&error, "chunk location.size");
}

/// An oversized uncompressed size on the unwritten-chunk path is bounded too.
///
/// `uncompressed_size` derives from the layout message's chunk dimensions,
/// which are equally attacker-chosen; the undefined-address branch allocated
/// it before any read at all.
#[test]
fn chunk_uncompressed_size_beyond_budget_is_rejected() {
    let source = MemCursor::from_bytes(vec![0u8; 64]);
    let location = ChunkLocation {
        address: consus_hdf5::constants::UNDEFINED_ADDRESS,
        size: 0,
        filter_mask: 0,
    };
    let registry = consus_compression::DefaultCodecRegistry::new();

    let error = read_chunk_raw(&source, &location, usize::MAX, 1, &[], &registry, None)
        .expect_err("usize::MAX uncompressed size must not be zero-filled");
    assert_resource_limit(&error, "chunk uncompressed_size");
}

/// A global-heap collection size is a file-supplied byte count and must be
/// checked before the body allocation or address arithmetic.
#[test]
fn global_heap_collection_size_beyond_budget_is_rejected() {
    let mut image = vec![0u8; 16];
    image[0..4].copy_from_slice(b"GCOL");
    image[4] = 1;
    image[8..16].copy_from_slice(&u64::MAX.to_le_bytes());

    let error =
        GlobalHeapCollection::parse(&MemCursor::from_bytes(image), 0, &ParseContext::new(8, 8))
            .expect_err("u64::MAX global heap size must not be allocated");
    assert_resource_limit(&error, "global heap collection size");
}

/// A global-heap collection size *below* its own header size must be a typed
/// error, not a subtraction underflow.
///
/// This is the sibling of the case above and needs its own test: `u64::MAX`
/// is rejected by the byte-ceiling check before the subtraction is reached,
/// so that test cannot exercise this guard. A small value passes the ceiling
/// and reaches `collection_size - header_size`, which underflows to roughly
/// `usize::MAX` — a debug-build panic, and in release a colossal body length
/// fed straight into the allocation. `header_size` here is `8 + length_bytes`
/// = 16, so any declared size below 16 is the hostile case; 4 is
/// unambiguously below it.
#[test]
fn global_heap_collection_size_below_header_size_is_rejected() {
    let mut image = vec![0u8; 16];
    image[0..4].copy_from_slice(b"GCOL");
    image[4] = 1;
    image[8..16].copy_from_slice(&4u64.to_le_bytes());

    let error =
        GlobalHeapCollection::parse(&MemCursor::from_bytes(image), 0, &ParseContext::new(8, 8))
            .expect_err("a collection smaller than its header must not underflow");
    let Error::InvalidFormat { message } = &error else {
        panic!("expected Error::InvalidFormat naming the violated invariant, got {error:?}");
    };
    assert!(
        message.contains("smaller than its") && message.contains("header"),
        "the error must name the header-size violation and both operands, got {message:?}"
    );
}

/// A contiguous dataset shape must be checked before its output buffer is
/// allocated; the shape product itself must not overflow first.
#[test]
fn contiguous_dataset_shape_beyond_budget_is_rejected() {
    let mut image = vec![0u8; 4096];
    image[0..8].copy_from_slice(&consus_hdf5::constants::HDF5_MAGIC);
    image[8] = 2;
    image[9] = 8;
    image[10] = 8;
    image[28..36].copy_from_slice(&4096u64.to_le_bytes());
    image[36..44].copy_from_slice(&96u64.to_le_bytes());
    let file = consus_hdf5::file::Hdf5File::open(MemCursor::from_bytes(image))
        .expect("minimal superblock must parse");
    let dataset = consus_hdf5::dataset::Hdf5Dataset {
        path: String::from("/hostile"),
        object_header_address: 0,
        datatype: consus_core::Datatype::Boolean,
        shape: consus_core::Shape::fixed(&[ParseBudget::DEFAULT.max_elements + 1, 1]),
        layout: consus_hdf5::dataset::StorageLayout::Contiguous,
        chunk_shape: None,
        data_address: Some(0),
        filters: Vec::new(),
    };

    let error = file
        .read_dataset_raw(&dataset, None)
        .expect_err("hostile dataset shape must not be allocated");
    assert_resource_limit(&error, "dataset element count");
}

/// Managed fractal-heap object lengths are external input, not allocation
/// permissions.
#[test]
fn managed_heap_object_length_beyond_budget_is_rejected() {
    let error = read_managed_object(
        &MemCursor::from_bytes(Vec::new()),
        &fractal_header(0, 4),
        0,
        u64::MAX,
        &ParseContext::new(8, 8),
    )
    .expect_err("u64::MAX managed-object length must not be allocated");
    assert_resource_limit(&error, "fractal heap managed object");
}

/// Indirect-block dimensions must be bounded before the table buffer is
/// materialized; a hostile width/row pair must return a typed resource error.
#[test]
fn fractal_indirect_block_size_beyond_budget_is_rejected() {
    let error = read_managed_object(
        &MemCursor::from_bytes(Vec::new()),
        &fractal_header(u16::MAX, u16::MAX),
        0,
        1,
        &ParseContext::new(8, 8),
    )
    .expect_err("hostile indirect-block dimensions must not be allocated");
    assert_resource_limit(&error, "fractal heap indirect block");
}

/// A HUGE-object B-tree record can carry an arbitrary length; the final heap
/// read applies the same byte ceiling as managed objects.
#[test]
fn huge_heap_object_length_beyond_budget_is_rejected() {
    let mut image = vec![0u8; 128];
    image[0..4].copy_from_slice(b"BTHD");
    image[6..10].copy_from_slice(&64u32.to_le_bytes());
    image[10..12].copy_from_slice(&24u16.to_le_bytes());
    image[16..24].copy_from_slice(&64u64.to_le_bytes());
    image[24..26].copy_from_slice(&1u16.to_le_bytes());
    image[26..34].copy_from_slice(&1u64.to_le_bytes());
    image[64..68].copy_from_slice(b"BTLF");
    image[69] = consus_hdf5::btree::v2::record_type::HUGE_OBJECT;
    image[70..78].copy_from_slice(&7u64.to_le_bytes());
    image[78..86].copy_from_slice(&96u64.to_le_bytes());
    image[86..94].copy_from_slice(&u64::MAX.to_le_bytes());

    let error = read_huge_object(
        &MemCursor::from_bytes(image),
        &fractal_header(0, 4),
        7,
        &ParseContext::new(8, 8),
    )
    .expect_err("u64::MAX HUGE-object length must not be allocated");
    assert_resource_limit(&error, "fractal heap huge object");
}

// ---------------------------------------------------------------------------
// Fix 2 — unbounded recursion
// ---------------------------------------------------------------------------

/// Build a datatype message nesting `depth` compound wrappers around one
/// 1-byte fixed-point leaf.
///
/// Version 3 encoding keeps each wrapper at 11 bytes — 8-byte header, a
/// 2-byte `"a\0"` member name with no padding, and a 1-byte member offset
/// (the compound's declared size of 1 selects the 1-byte offset width). The
/// entire 10 000-deep message is therefore ~110 KB: cheap to synthesise and
/// far cheaper than the ~10 000 stack frames it used to cost to parse.
fn nested_compound(depth: usize) -> Vec<u8> {
    // Innermost leaf: class 0 (fixed-point), version 3, 1 byte wide.
    let mut message = vec![0x30, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
    message.extend_from_slice(&[0u8; 4]); // bit offset + bit precision

    for _ in 0..depth {
        // Class 6 (compound), version 3, one member, declared size 1.
        let mut wrapper = vec![0x36, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
        wrapper.extend_from_slice(b"a\0"); // member name, unpadded in v3
        wrapper.push(0x00); // member byte offset, 1-byte width
        wrapper.append(&mut message);
        message = wrapper;
    }
    message
}

/// A ~10 000-deep nested compound datatype must return an error, not overflow
/// the stack.
///
/// `parse_datatype_inner` → `parse_compound` → `parse_compound_member` →
/// `parse_datatype_inner` had no depth parameter anywhere in the cycle. A
/// stack overflow raises SIGSEGV / `STATUS_STACK_OVERFLOW`; it is not a
/// catchable panic, so the process dies and takes the test harness with it.
#[test]
fn deeply_nested_compound_datatype_is_rejected() {
    let message = nested_compound(10_000);
    let error = parse_datatype(&message, &ParseBudget::DEFAULT)
        .expect_err("10 000-deep nesting must not be parsed");
    assert_resource_limit(&error, "compound nesting depth");
}

/// The same bound applies through the enum, variable-length, and array arms,
/// which reach `parse_datatype_inner` by their own routes.
#[test]
fn deeply_nested_array_datatype_is_rejected() {
    // Innermost leaf, then 10 000 array wrappers. Version 3 array props are
    // rank(1) + dims(4×rank); rank 1 keeps each wrapper at 8 + 5 = 13 bytes.
    let mut message = vec![0x30, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
    message.extend_from_slice(&[0u8; 4]);
    for _ in 0..10_000 {
        let mut wrapper = vec![0x3A, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
        wrapper.push(0x01); // rank = 1
        wrapper.extend_from_slice(&1u32.to_le_bytes()); // dims[0] = 1
        wrapper.append(&mut message);
        message = wrapper;
    }

    let error = parse_datatype(&message, &ParseBudget::DEFAULT)
        .expect_err("10 000-deep array nesting must not be parsed");
    assert_resource_limit(&error, "array nesting depth");
}

/// A variable-length sequence wrapping itself reaches the same cycle.
#[test]
fn deeply_nested_variable_length_datatype_is_rejected() {
    let mut message = vec![0x30, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
    message.extend_from_slice(&[0u8; 4]);
    for _ in 0..10_000 {
        // Class 9 (variable-length), version 3, sub-type 0 (sequence).
        let mut wrapper = vec![0x39, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00];
        wrapper.append(&mut message);
        message = wrapper;
    }

    let error = parse_datatype(&message, &ParseBudget::DEFAULT)
        .expect_err("10 000-deep VL nesting must not be parsed");
    assert_resource_limit(&error, "variable-length nesting depth");
}

// ---------------------------------------------------------------------------
// The bounds must not reject well-formed input
// ---------------------------------------------------------------------------

/// Modest nesting stays parseable: the bound rejects hostile depth, not
/// legitimate structure.
#[test]
fn shallow_nested_compound_datatype_still_parses() {
    let message = nested_compound(4);
    let datatype = parse_datatype(&message, &ParseBudget::DEFAULT)
        .expect("4-deep nesting is well within budget");
    assert!(
        matches!(datatype, consus_core::Datatype::Compound { .. }),
        "expected a compound datatype, got {datatype:?}"
    );
}

/// An honest chunk read is unaffected by the byte ceiling.
#[test]
fn well_formed_chunk_read_is_unaffected() {
    let payload = b"uncompressed chunk data";
    let source = MemCursor::from_bytes(payload.to_vec());
    let location = ChunkLocation {
        address: 0,
        size: payload.len() as u64,
        filter_mask: 0,
    };
    let registry = consus_compression::DefaultCodecRegistry::new();

    let data = read_chunk_raw(&source, &location, payload.len(), 1, &[], &registry, None)
        .expect("a well-formed chunk read must succeed");
    assert_eq!(data, payload);
}

// ---------------------------------------------------------------------------
// Fix 3 — malformed HUGE-object B-tree descent
// ---------------------------------------------------------------------------

/// A self-referential HUGE-object child must hit the descent budget instead
/// of recursing until the process stack overflows.
#[test]
fn huge_object_search_rejects_self_referential_child() {
    let mut node = vec![0u8; 32];
    node[..4].copy_from_slice(b"BTIN");
    node[4] = 0;
    node[5] = consus_hdf5::btree::v2::record_type::HUGE_OBJECT;
    // With zero records, the first child pointer starts immediately after the
    // six-byte internal-node header. It points back to this same node.
    node[6..14].copy_from_slice(&0u64.to_le_bytes());
    let source = MemCursor::from_bytes(node);
    let header = BTreeV2Header {
        record_type: consus_hdf5::btree::v2::record_type::HUGE_OBJECT,
        node_size: 32,
        record_size: 16,
        depth: 1,
        root_address: 0,
        root_num_records: 0,
        split_percent: 98,
        merge_percent: 40,
        total_records: 1,
    };
    let error = find_huge_object_record(&source, 0, &header, 7, &ParseContext::new(8, 8))
        .expect_err("self-referential HUGE-object child must terminate");
    assert_resource_limit(&error, "self-referential HUGE-object descent");
}

/// A truncated HUGE-object internal node with no child pointers must return a
/// format error rather than underflowing `len() - 1`.
#[test]
fn huge_object_search_rejects_empty_child_table() {
    let mut node = vec![0u8; 10];
    node[..4].copy_from_slice(b"BTIN");
    node[4] = 0;
    node[5] = consus_hdf5::btree::v2::record_type::HUGE_OBJECT;
    let source = MemCursor::from_bytes(node);
    let header = BTreeV2Header {
        record_type: consus_hdf5::btree::v2::record_type::HUGE_OBJECT,
        node_size: 10,
        record_size: 16,
        depth: 1,
        root_address: 0,
        root_num_records: 0,
        split_percent: 98,
        merge_percent: 40,
        total_records: 1,
    };
    let error = find_huge_object_record(&source, 0, &header, 7, &ParseContext::new(8, 8))
        .expect_err("empty child table must be rejected");
    assert!(
        matches!(error, Error::InvalidFormat { .. }),
        "empty HUGE-object child table: expected invalid format, got {error:?}"
    );
}
