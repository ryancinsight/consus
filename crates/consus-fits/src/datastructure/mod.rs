//! FITS structural descriptors and 2880-byte blocking math.
//!
//! ## Scope
//!
//! This module is the single source of truth for FITS structural layout
//! descriptors in `consus-fits`. It defines:
//! - logical FITS record sizing
//! - 2880-byte block alignment and padding math
//! - header card count to byte-size conversion
//! - data-unit byte span descriptors
//!
//! ## FITS invariants
//!
//! FITS stores content in logical records of 2880 bytes. Header card images are
//! 80 bytes each, so one logical record contains exactly 36 cards.
//!
//! For any payload length `n`:
//! - `padded_len(n) = ceil_div(n, 2880) * 2880`
//! - `padding_len(n) = padded_len(n) - n`
//! - `padding_len(n) ∈ [0, 2879]`
//!
//! ## Architectural role
//!
//! Higher-level HDU and file logic depend on this module for authoritative
//! byte-layout computations rather than duplicating FITS blocking rules.

#![cfg_attr(not(feature = "std"), no_std)]

use consus_core::{Error, Result};

use crate::header::card::FITS_CARD_LEN;

/// FITS logical record size in bytes.
pub const FITS_LOGICAL_RECORD_LEN: usize = 2880;

/// Number of 80-byte cards per FITS logical record.
pub const FITS_CARDS_PER_RECORD: usize = FITS_LOGICAL_RECORD_LEN / FITS_CARD_LEN;

/// FITS logical record descriptor.
///
/// A logical record is the canonical 2880-byte FITS blocking unit used for both
/// header and data padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FitsLogicalRecord;

impl FitsLogicalRecord {
    /// Return the FITS logical record size in bytes.
    pub const fn byte_len() -> usize {
        FITS_LOGICAL_RECORD_LEN
    }

    /// Return the number of 80-byte cards per logical record.
    pub const fn cards_per_record() -> usize {
        FITS_CARDS_PER_RECORD
    }

    /// Return whether `len` is aligned to a FITS logical record boundary.
    pub const fn is_aligned(len: usize) -> bool {
        len % FITS_LOGICAL_RECORD_LEN == 0
    }
}

/// FITS 2880-byte block alignment math.
///
/// This type centralizes all padding and alignment computations required for
/// header-data unit round-tripping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FitsBlockAlignment;

impl FitsBlockAlignment {
    /// Return whether `len` is aligned to a 2880-byte FITS block boundary.
    pub const fn is_aligned(len: usize) -> bool {
        FitsLogicalRecord::is_aligned(len)
    }

    /// Return the smallest 2880-byte-aligned length greater than or equal to
    /// `len`.
    pub const fn padded_len(len: usize) -> usize {
        let remainder = len % FITS_LOGICAL_RECORD_LEN;
        if remainder == 0 {
            len
        } else {
            len + (FITS_LOGICAL_RECORD_LEN - remainder)
        }
    }

    /// Return the number of trailing zero bytes required to pad `len` to the
    /// next 2880-byte boundary.
    pub const fn padding_len(len: usize) -> usize {
        Self::padded_len(len) - len
    }

    /// Return the number of 2880-byte logical records required to store `len`
    /// bytes.
    pub const fn record_count(len: usize) -> usize {
        Self::padded_len(len) / FITS_LOGICAL_RECORD_LEN
    }
}

/// Header card count descriptor.
///
/// FITS headers are sequences of 80-byte cards terminated by an `END` card and
/// padded to a 2880-byte boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FitsHeaderCardCount {
    cards: usize,
}

impl FitsHeaderCardCount {
    /// Construct a header card count.
    pub const fn new(cards: usize) -> Self {
        Self { cards }
    }

    /// Return the number of cards.
    pub const fn get(self) -> usize {
        self.cards
    }

    /// Return the unpadded header byte length.
    pub const fn byte_len(self) -> usize {
        self.cards * FITS_CARD_LEN
    }

    /// Return the padded header byte length rounded to a 2880-byte boundary.
    pub const fn padded_byte_len(self) -> usize {
        FitsBlockAlignment::padded_len(self.byte_len())
    }

    /// Return the number of logical records occupied by the padded header.
    pub const fn record_count(self) -> usize {
        FitsBlockAlignment::record_count(self.byte_len())
    }
}

/// Header block descriptor.
///
/// This type captures the byte extent of a parsed FITS header, including both
/// semantic card bytes and trailing FITS block padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FitsHeaderBlock {
    card_count: FitsHeaderCardCount,
}

impl FitsHeaderBlock {
    /// Construct a header block from a card count.
    pub const fn new(card_count: FitsHeaderCardCount) -> Self {
        Self { card_count }
    }

    /// Return the header card count.
    pub const fn card_count(self) -> FitsHeaderCardCount {
        self.card_count
    }

    /// Return the unpadded header byte length.
    pub const fn byte_len(self) -> usize {
        self.card_count.byte_len()
    }

    /// Return the padded header byte length.
    pub const fn padded_byte_len(self) -> usize {
        self.card_count.padded_byte_len()
    }

    /// Return the trailing padding length in bytes.
    pub const fn padding_len(self) -> usize {
        self.padded_byte_len() - self.byte_len()
    }

    /// Return the number of logical records occupied by the header.
    pub const fn record_count(self) -> usize {
        self.card_count.record_count()
    }
}

/// Data-unit byte span descriptor.
///
/// This type describes the on-disk byte extent of an HDU data unit. The
/// `logical_len` is the semantic payload length. The `padded_len` includes the
/// trailing FITS zero padding required to reach the next 2880-byte boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FitsDataSpan {
    offset: u64,
    logical_len: usize,
    padded_len: usize,
}

impl FitsDataSpan {
    /// Construct a data span from an absolute byte offset and logical payload
    /// length.
    ///
    /// ## Errors
    ///
    /// Returns `Error::Overflow` if the padded end offset cannot be represented
    /// in `u64`.
    pub fn new(offset: u64, logical_len: usize) -> Result<Self> {
        let padded_len = FitsBlockAlignment::padded_len(logical_len);
        let padded_len_u64 = u64::try_from(padded_len).map_err(|_| Error::Overflow)?;
        offset.checked_add(padded_len_u64).ok_or(Error::Overflow)?;
        Ok(Self {
            offset,
            logical_len,
            padded_len,
        })
    }

    /// Return the absolute byte offset of the data unit.
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Return the semantic payload length in bytes.
    pub const fn logical_len(self) -> usize {
        self.logical_len
    }

    /// Return the padded on-disk length in bytes.
    pub const fn padded_len(self) -> usize {
        self.padded_len
    }

    /// Return the trailing FITS padding length in bytes.
    pub const fn padding_len(self) -> usize {
        self.padded_len - self.logical_len
    }

    /// Return whether the data unit is empty.
    pub const fn is_empty(self) -> bool {
        self.logical_len == 0
    }

    /// Return the absolute byte offset immediately after the padded data unit.
    ///
    /// ## Errors
    ///
    /// Returns `Error::Overflow` if the end offset cannot be represented in
    /// `u64`.
    pub fn end_offset(self) -> Result<u64> {
        let padded_len = u64::try_from(self.padded_len).map_err(|_| Error::Overflow)?;
        self.offset.checked_add(padded_len).ok_or(Error::Overflow)
    }

    /// Return whether `absolute_offset` lies within the semantic payload range.
    pub fn contains_logical_offset(self, absolute_offset: u64) -> bool {
        let logical_len = match u64::try_from(self.logical_len) {
            Ok(value) => value,
            Err(_) => return false,
        };
        match self.offset.checked_add(logical_len) {
            Some(end) => absolute_offset >= self.offset && absolute_offset < end,
            None => false,
        }
    }

    /// Return whether `absolute_offset` lies within the padded on-disk range.
    pub fn contains_padded_offset(self, absolute_offset: u64) -> bool {
        let padded_len = match u64::try_from(self.padded_len) {
            Ok(value) => value,
            Err(_) => return false,
        };
        match self.offset.checked_add(padded_len) {
            Some(end) => absolute_offset >= self.offset && absolute_offset < end,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_record_constants_match_fits_spec() {
        assert_eq!(FitsLogicalRecord::byte_len(), 2880);
        assert_eq!(FitsLogicalRecord::cards_per_record(), 36);
        assert_eq!(FITS_CARDS_PER_RECORD, 36);
    }

    #[test]
    fn block_alignment_reports_alignment_and_padding() {
        assert!(FitsBlockAlignment::is_aligned(0));
        assert!(FitsBlockAlignment::is_aligned(2880));
        assert!(!FitsBlockAlignment::is_aligned(1));
        assert!(!FitsBlockAlignment::is_aligned(2879));

        assert_eq!(FitsBlockAlignment::padded_len(0), 0);
        assert_eq!(FitsBlockAlignment::padded_len(1), 2880);
        assert_eq!(FitsBlockAlignment::padded_len(2879), 2880);
        assert_eq!(FitsBlockAlignment::padded_len(2880), 2880);
        assert_eq!(FitsBlockAlignment::padded_len(2881), 5760);

        assert_eq!(FitsBlockAlignment::padding_len(0), 0);
        assert_eq!(FitsBlockAlignment::padding_len(1), 2879);
        assert_eq!(FitsBlockAlignment::padding_len(2879), 1);
        assert_eq!(FitsBlockAlignment::padding_len(2880), 0);
    }

    #[test]
    fn block_alignment_reports_record_count() {
        assert_eq!(FitsBlockAlignment::record_count(0), 0);
        assert_eq!(FitsBlockAlignment::record_count(1), 1);
        assert_eq!(FitsBlockAlignment::record_count(2880), 1);
        assert_eq!(FitsBlockAlignment::record_count(2881), 2);
    }

    #[test]
    fn header_card_count_converts_to_bytes_and_records() {
        let count = FitsHeaderCardCount::new(3);
        assert_eq!(count.get(), 3);
        assert_eq!(count.byte_len(), 240);
        assert_eq!(count.padded_byte_len(), 2880);
        assert_eq!(count.record_count(), 1);

        let full_record = FitsHeaderCardCount::new(36);
        assert_eq!(full_record.byte_len(), 2880);
        assert_eq!(full_record.padded_byte_len(), 2880);
        assert_eq!(full_record.record_count(), 1);

        let spill = FitsHeaderCardCount::new(37);
        assert_eq!(spill.byte_len(), 2960);
        assert_eq!(spill.padded_byte_len(), 5760);
        assert_eq!(spill.record_count(), 2);
    }

    #[test]
    fn header_block_reports_padding() {
        let block = FitsHeaderBlock::new(FitsHeaderCardCount::new(5));
        assert_eq!(block.byte_len(), 400);
        assert_eq!(block.padded_byte_len(), 2880);
        assert_eq!(block.padding_len(), 2480);
        assert_eq!(block.record_count(), 1);
    }

    #[test]
    fn data_span_round_trips_lengths_and_offsets() {
        let span = FitsDataSpan::new(5760, 100).unwrap();
        assert_eq!(span.offset(), 5760);
        assert_eq!(span.logical_len(), 100);
        assert_eq!(span.padded_len(), 2880);
        assert_eq!(span.padding_len(), 2780);
        assert_eq!(span.end_offset().unwrap(), 8640);
        assert!(!span.is_empty());
    }

    #[test]
    fn data_span_handles_empty_payload() {
        let span = FitsDataSpan::new(2880, 0).unwrap();
        assert_eq!(span.logical_len(), 0);
        assert_eq!(span.padded_len(), 0);
        assert_eq!(span.padding_len(), 0);
        assert_eq!(span.end_offset().unwrap(), 2880);
        assert!(span.is_empty());
    }

    #[test]
    fn data_span_contains_offsets_correctly() {
        let span = FitsDataSpan::new(1000, 3000).unwrap();
        assert!(span.contains_logical_offset(1000));
        assert!(span.contains_logical_offset(3999));
        assert!(!span.contains_logical_offset(4000));

        assert!(span.contains_padded_offset(1000));
        assert!(span.contains_padded_offset(6759));
        assert!(!span.contains_padded_offset(6760));
    }
}
