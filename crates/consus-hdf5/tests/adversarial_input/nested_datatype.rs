//! Adversarial nesting tests for HDF5 datatype messages.

use super::assert_resource_limit;
use consus_core::ParseBudget;
use consus_hdf5::datatype::compound::parse_datatype;

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
