//! N-bit packing filter (HDF5 filter ID 5).
//!
//! Packs integer values using only the minimum number of bits needed,
//! reducing storage for data that does not use the full range of its type.
//!
//! ## Algorithm (Forward — Pack)
//!
//! Given `bits_per_value` < `bits_per_element`:
//!
//! For each element, extract the low `bits_per_value` bits and pack them
//! consecutively into the output byte stream, MSB-first within each byte.
//!
//! The output length in bytes is `ceil(num_elements * bits_per_value / 8)`.
//!
//! ## Algorithm (Reverse — Unpack)
//!
//! Read `bits_per_value` bits per element from the packed stream (MSB-first),
//! zero-extend each to `bits_per_element` bits, and write the result as
//! little-endian element bytes.
//!
//! ## Invariant
//!
//! For any data where all element values fit in `bits_per_value` bits:
//!
//! ```text
//! unpack(pack(data)) == data
//! ```
//!
//! ## Proof sketch
//!
//! Let `v_i` be the i-th element value with `v_i < 2^bits_per_value`.
//!
//! Pack extracts the low `bits_per_value` bits of `v_i`, which equals `v_i`
//! by the precondition, and writes them into the bitstream at bit offset
//! `i * bits_per_value`.
//!
//! Unpack reads `bits_per_value` bits from the same offset, recovering `v_i`,
//! then zero-extends to `bits_per_element` bits. Since the original upper
//! bits were zero (by the precondition), the result equals the original
//! element bytes. □

use alloc::vec;
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use consus_core::{Error, Result};

use super::traits::{Filter, FilterDirection};

/// N-bit packing filter.
///
/// Packs integer values using only `bits_per_value` bits per element,
/// where the native element width is `bits_per_element` bits.
///
/// ## HDF5
///
/// Corresponds to HDF5 filter ID 5 (`H5Z_FILTER_NBIT`).
///
/// ## Constraints
///
/// - `bits_per_element` ∈ {8, 16, 32}.
/// - `bits_per_value` ∈ [1, `bits_per_element`].
/// - Input data length must be divisible by `bits_per_element / 8`.
#[derive(Debug, Clone)]
pub struct NbitFilter {
    /// Number of significant bits per value in the packed representation.
    bits_per_value: u8,
    /// Native element width in bits (8, 16, or 32).
    bits_per_element: u8,
}

impl NbitFilter {
    /// Create a new N-bit packing filter.
    ///
    /// # Panics
    ///
    /// - Panics if `bits_per_element` is not 8, 16, or 32.
    /// - Panics if `bits_per_value` is 0 or exceeds `bits_per_element`.
    #[must_use]
    pub fn new(bits_per_value: u8, bits_per_element: u8) -> Self {
        assert!(
            bits_per_element == 8 || bits_per_element == 16 || bits_per_element == 32,
            "NbitFilter: bits_per_element must be 8, 16, or 32, got {bits_per_element}"
        );
        assert!(
            bits_per_value >= 1 && bits_per_value <= bits_per_element,
            "NbitFilter: bits_per_value must be in [1, {bits_per_element}], got {bits_per_value}"
        );
        Self {
            bits_per_value,
            bits_per_element,
        }
    }

    /// Return the configured bits per value.
    #[must_use]
    pub fn bits_per_value(&self) -> u8 {
        self.bits_per_value
    }

    /// Return the configured bits per element.
    #[must_use]
    pub fn bits_per_element(&self) -> u8 {
        self.bits_per_element
    }

    /// Read one element from `data` at byte offset `offset` as a `u32`.
    ///
    /// Reads `bits_per_element / 8` bytes in little-endian order.
    fn read_element(&self, data: &[u8], offset: usize) -> u32 {
        match self.bits_per_element {
            8 => u32::from(data[offset]),
            16 => u32::from(LittleEndian::read_u16(&data[offset..offset + 2])),
            32 => LittleEndian::read_u32(&data[offset..offset + 4]),
            _ => unreachable!(),
        }
    }

    /// Write one element value to `data` at byte offset `offset` in
    /// little-endian order, using `bits_per_element / 8` bytes.
    fn write_element(&self, data: &mut [u8], offset: usize, value: u32) {
        match self.bits_per_element {
            8 => data[offset] = value as u8,
            16 => LittleEndian::write_u16(&mut data[offset..offset + 2], value as u16),
            32 => LittleEndian::write_u32(&mut data[offset..offset + 4], value),
            _ => unreachable!(),
        }
    }

    /// Pack elements into a bitstream using `bits_per_value` bits each,
    /// MSB-first within each output byte.
    ///
    /// ## Bitstream layout
    ///
    /// Element `i` occupies bits `[i * bpv, (i+1) * bpv)` in the output
    /// bitstream, where bit 0 is the MSB of output byte 0.
    fn pack(&self, data: &[u8]) -> Vec<u8> {
        let elem_bytes = (self.bits_per_element / 8) as usize;
        let num_elements = data.len() / elem_bytes;
        let bpv = self.bits_per_value as usize;
        let total_bits = num_elements * bpv;
        let out_len = total_bits.div_ceil(8);
        let mut output = vec![0u8; out_len];

        for i in 0..num_elements {
            let value = self.read_element(data, i * elem_bytes);
            // Mask to bits_per_value low bits.
            let masked = value & ((1u32 << bpv) - 1);
            // Write `bpv` bits starting at bit offset `i * bpv` (MSB-first).
            let bit_offset = i * bpv;
            for b in 0..bpv {
                // Extract bit `b` from masked, where b=0 is the most significant
                // of the `bpv` bits.
                let bit = (masked >> (bpv - 1 - b)) & 1;
                let out_bit_idx = bit_offset + b;
                let byte_idx = out_bit_idx / 8;
                let bit_in_byte = 7 - (out_bit_idx % 8); // MSB-first
                output[byte_idx] |= (bit as u8) << bit_in_byte;
            }
        }

        output
    }

    /// Unpack elements from a bitstream, reading `bits_per_value` bits each
    /// (MSB-first), zero-extending to `bits_per_element` bits.
    ///
    /// `num_elements` is the expected number of elements to unpack.
    fn unpack(&self, data: &[u8], num_elements: usize) -> Vec<u8> {
        let elem_bytes = (self.bits_per_element / 8) as usize;
        let bpv = self.bits_per_value as usize;
        let mut output = vec![0u8; num_elements * elem_bytes];

        for i in 0..num_elements {
            let bit_offset = i * bpv;
            let mut value: u32 = 0;
            for b in 0..bpv {
                let in_bit_idx = bit_offset + b;
                let byte_idx = in_bit_idx / 8;
                let bit_in_byte = 7 - (in_bit_idx % 8); // MSB-first
                let bit = (data[byte_idx] >> bit_in_byte) & 1;
                // b=0 is the most significant bit of the value.
                value |= u32::from(bit) << (bpv - 1 - b);
            }
            self.write_element(&mut output, i * elem_bytes, value);
        }

        output
    }
}

impl Filter for NbitFilter {
    fn name(&self) -> &str {
        "nbit"
    }

    /// Apply the N-bit packing filter.
    ///
    /// ## Forward (Pack)
    ///
    /// Reads elements of `bits_per_element` bits from `data` (little-endian),
    /// packs each into `bits_per_value` bits MSB-first, and prepends a 4-byte
    /// little-endian element count header so that reverse can determine how
    /// many elements to unpack.
    ///
    /// ## Reverse (Unpack)
    ///
    /// Reads the 4-byte element count header, then unpacks `bits_per_value`
    /// bits per element back to `bits_per_element`-bit little-endian values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFormat`] if `data.len()` is not divisible by
    /// the element byte width (forward), or if the packed stream is too short
    /// to contain the declared element count (reverse).
    fn apply(&self, direction: FilterDirection, data: &[u8]) -> Result<Vec<u8>> {
        let elem_bytes = (self.bits_per_element / 8) as usize;

        match direction {
            FilterDirection::Forward => {
                if data.len() % elem_bytes != 0 {
                    return Err(Error::InvalidFormat {
                        message: alloc::format!(
                            "nbit pack: data length {} is not divisible by element size {} bytes",
                            data.len(),
                            elem_bytes,
                        ),
                    });
                }

                if data.is_empty() {
                    // Header with zero element count.
                    let mut out = vec![0u8; 4];
                    LittleEndian::write_u32(&mut out, 0);
                    return Ok(out);
                }

                let num_elements = data.len() / elem_bytes;
                let packed = self.pack(data);

                // Prepend a 4-byte little-endian element count header.
                let mut output = vec![0u8; 4 + packed.len()];
                LittleEndian::write_u32(&mut output[..4], num_elements as u32);
                output[4..].copy_from_slice(&packed);
                Ok(output)
            }
            FilterDirection::Reverse => {
                if data.len() < 4 {
                    return Err(Error::InvalidFormat {
                        message: alloc::format!(
                            "nbit unpack: data length {} is too short for header (need >= 4)",
                            data.len(),
                        ),
                    });
                }

                let num_elements = LittleEndian::read_u32(&data[..4]) as usize;

                if num_elements == 0 {
                    return Ok(Vec::new());
                }

                let bpv = self.bits_per_value as usize;
                let total_bits = num_elements * bpv;
                let expected_packed_bytes = total_bits.div_ceil(8);
                let packed_data = &data[4..];

                if packed_data.len() < expected_packed_bytes {
                    return Err(Error::InvalidFormat {
                        message: alloc::format!(
                            "nbit unpack: packed stream has {} bytes but need {} \
                             for {} elements at {} bits each",
                            packed_data.len(),
                            expected_packed_bytes,
                            num_elements,
                            bpv,
                        ),
                    });
                }

                Ok(self.unpack(packed_data, num_elements))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack/unpack 8-bit values with 4 bits per value.
    ///
    /// Input: [0x03, 0x07, 0x0F, 0x01] — all fit in 4 bits.
    ///
    /// Packed bitstream (MSB-first, 4 bits each):
    ///   0x03 = 0011, 0x07 = 0111, 0x0F = 1111, 0x01 = 0001
    ///   → 0011_0111 1111_0001 = [0x37, 0xF1]
    ///
    /// With 4-byte header: [0x04, 0x00, 0x00, 0x00, 0x37, 0xF1]
    #[test]
    fn pack_unpack_8bit_4bpv() {
        let filter = NbitFilter::new(4, 8);
        let input: Vec<u8> = vec![0x03, 0x07, 0x0F, 0x01];

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("pack must succeed");

        // Verify header: 4 elements as LE u32.
        assert_eq!(packed.len(), 4 + 2, "header(4) + packed(2) = 6 bytes");
        assert_eq!(
            LittleEndian::read_u32(&packed[..4]),
            4,
            "header must encode 4 elements"
        );

        // Verify packed bytes.
        assert_eq!(packed[4], 0x37, "first packed byte: 0011_0111 = 0x37");
        assert_eq!(packed[5], 0xF1, "second packed byte: 1111_0001 = 0xF1");

        // Round-trip.
        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("unpack must succeed");
        assert_eq!(unpacked, input, "round-trip must recover original data");
    }

    /// Pack/unpack 16-bit values with 10 bits per value.
    ///
    /// Input (LE u16): [0x03FF, 0x0001, 0x0200]
    /// Byte representation: [0xFF, 0x03, 0x01, 0x00, 0x00, 0x02]
    ///
    /// Values in 10 bits (MSB-first):
    ///   0x03FF = 11_1111_1111
    ///   0x0001 = 00_0000_0001
    ///   0x0200 = 10_0000_0000
    ///
    /// Bitstream (30 bits, padded to 4 bytes):
    ///   11_1111_1111 | 00_0000_0001 | 10_0000_0000 | 00 (padding)
    ///   = 1111_1111 1100_0000_0001 1000_0000_0000
    ///   byte 0: 1111_1111 = 0xFF
    ///   byte 1: 1100_0000 = 0xC0... wait let me recalculate more carefully.
    ///
    /// Bit positions 0..29 (MSB-first in bytes):
    ///   bits  0- 9: 1111111111   (0x03FF)
    ///   bits 10-19: 0000000001   (0x0001)
    ///   bits 20-29: 1000000000   (0x0200)
    ///
    /// byte 0 (bits 0-7):   11111111 = 0xFF
    /// byte 1 (bits 8-15):  11000000 = 0xC0 (bits 8-9 from first val, bits 10-15 from second)
    ///   bit 8,9 = 1,1 (remaining of 0x03FF)
    ///   bits 10-15 = 000000 (first 6 of 0x0001)
    ///   → 1100_0000 = 0xC0
    /// byte 2 (bits 16-23): 00011000 = 0x18
    ///   bits 16-19 = 0001 (last 4 of 0x0001)
    ///   bits 20-23 = 1000 (first 4 of 0x0200)
    ///   → 0001_1000 = 0x18
    /// byte 3 (bits 24-31): 00000000 = 0x00
    ///   bits 24-29 = 000000 (last 6 of 0x0200)
    ///   bits 30-31 = 00 (padding)
    ///   → 0000_0000 = 0x00
    #[test]
    fn pack_unpack_16bit_10bpv() {
        let filter = NbitFilter::new(10, 16);
        let mut input = vec![0u8; 6];
        LittleEndian::write_u16(&mut input[0..2], 0x03FF);
        LittleEndian::write_u16(&mut input[2..4], 0x0001);
        LittleEndian::write_u16(&mut input[4..6], 0x0200);

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("pack must succeed");

        // Header: 3 elements.
        assert_eq!(LittleEndian::read_u32(&packed[..4]), 3);

        // 3 elements × 10 bits = 30 bits → ceil(30/8) = 4 packed bytes.
        assert_eq!(packed.len(), 4 + 4, "header(4) + packed(4) = 8 bytes");

        // Verify packed bytes against analytical derivation.
        assert_eq!(packed[4], 0xFF, "packed byte 0");
        assert_eq!(packed[5], 0xC0, "packed byte 1");
        assert_eq!(packed[6], 0x18, "packed byte 2");
        assert_eq!(packed[7], 0x00, "packed byte 3");

        // Round-trip.
        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("unpack must succeed");
        assert_eq!(unpacked, input, "round-trip must recover original data");
    }

    /// Round-trip with 32-bit elements packed to 12 bits.
    ///
    /// Values chosen to fit within 12 bits (max 0xFFF = 4095).
    #[test]
    fn round_trip_32bit_12bpv() {
        let filter = NbitFilter::new(12, 32);
        let values: [u32; 5] = [0, 1, 0xABC, 0xFFF, 42];
        let mut input = vec![0u8; 20];
        for (i, &v) in values.iter().enumerate() {
            LittleEndian::write_u32(&mut input[i * 4..(i + 1) * 4], v);
        }

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("pack must succeed");

        // Header: 5 elements.
        assert_eq!(LittleEndian::read_u32(&packed[..4]), 5);

        // 5 × 12 = 60 bits → ceil(60/8) = 8 packed bytes.
        assert_eq!(packed.len(), 4 + 8);

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("unpack must succeed");
        assert_eq!(unpacked, input, "round-trip must recover original data");

        // Verify each element value.
        for (i, &expected) in values.iter().enumerate() {
            let actual = LittleEndian::read_u32(&unpacked[i * 4..(i + 1) * 4]);
            assert_eq!(
                actual, expected,
                "element {i}: expected {expected:#X}, got {actual:#X}"
            );
        }
    }

    /// Identity case: bits_per_value == bits_per_element (8-bit).
    ///
    /// When packing width equals element width, the packed bitstream
    /// contains the same bits (just with a header prepended).
    #[test]
    fn identity_8bit() {
        let filter = NbitFilter::new(8, 8);
        let input: Vec<u8> = vec![0x00, 0xFF, 0xAB, 0x12];

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("pack must succeed");

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("unpack must succeed");

        assert_eq!(unpacked, input, "identity packing must round-trip exactly");
    }

    /// Error: data length not divisible by element byte width (16-bit elements).
    #[test]
    fn error_misaligned_16bit() {
        let filter = NbitFilter::new(10, 16);
        // 3 bytes is not divisible by 2 (16-bit element width).
        let input: Vec<u8> = vec![0x01, 0x02, 0x03];

        let result = filter.apply(FilterDirection::Forward, &input);
        assert!(result.is_err(), "misaligned input must produce an error");

        match result.unwrap_err() {
            Error::InvalidFormat { message } => {
                assert!(
                    message.contains("3"),
                    "error must mention data length, got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Error: packed stream too short for declared element count.
    #[test]
    fn error_truncated_packed_stream() {
        let filter = NbitFilter::new(4, 8);
        // Header says 100 elements but only 1 byte of packed data follows.
        // 100 elements × 4 bits = 400 bits → 50 bytes needed.
        let mut bad_data = vec![0u8; 5];
        LittleEndian::write_u32(&mut bad_data[..4], 100);
        bad_data[4] = 0xFF;

        let result = filter.apply(FilterDirection::Reverse, &bad_data);
        assert!(result.is_err(), "truncated stream must produce an error");

        match result.unwrap_err() {
            Error::InvalidFormat { message } => {
                assert!(
                    message.contains("50"),
                    "error must mention required bytes, got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Error: reverse with data too short for header.
    #[test]
    fn error_too_short_for_header() {
        let filter = NbitFilter::new(4, 8);
        let result = filter.apply(FilterDirection::Reverse, &[0x01, 0x02]);
        assert!(result.is_err(), "data shorter than header must error");

        match result.unwrap_err() {
            Error::InvalidFormat { message } => {
                assert!(
                    message.contains("2"),
                    "error must mention actual length, got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Constructor panics: bits_per_element not in {8, 16, 32}.
    #[test]
    #[should_panic(expected = "bits_per_element must be 8, 16, or 32")]
    fn panics_on_invalid_bits_per_element() {
        let _f = NbitFilter::new(4, 12);
    }

    /// Constructor panics: bits_per_value == 0.
    #[test]
    #[should_panic(expected = "bits_per_value must be in")]
    fn panics_on_zero_bits_per_value() {
        let _f = NbitFilter::new(0, 8);
    }

    /// Constructor panics: bits_per_value > bits_per_element.
    #[test]
    #[should_panic(expected = "bits_per_value must be in")]
    fn panics_on_bits_per_value_exceeds_element() {
        let _f = NbitFilter::new(9, 8);
    }

    /// Empty input produces header-only output, which round-trips to empty.
    #[test]
    fn empty_input_round_trip() {
        let filter = NbitFilter::new(4, 8);

        let packed = filter
            .apply(FilterDirection::Forward, &[])
            .expect("empty pack must succeed");

        assert_eq!(packed.len(), 4, "empty pack must produce header only");
        assert_eq!(
            LittleEndian::read_u32(&packed[..4]),
            0,
            "header must encode 0 elements"
        );

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("empty unpack must succeed");
        assert!(unpacked.is_empty(), "unpack of zero elements must be empty");
    }

    /// Single-bit packing: values 0 and 1 only.
    #[test]
    fn single_bit_packing() {
        let filter = NbitFilter::new(1, 8);
        let input: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0];

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("pack must succeed");

        // 8 elements × 1 bit = 8 bits = 1 byte.
        assert_eq!(packed.len(), 4 + 1);

        // Bitstream MSB-first: 1,0,1,1,0,0,1,0 → 10110010 = 0xB2
        assert_eq!(packed[4], 0xB2, "packed byte must be 0xB2");

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("unpack must succeed");
        assert_eq!(unpacked, input, "single-bit round-trip must recover data");
    }
}
