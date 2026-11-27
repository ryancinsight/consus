//! Contiguous storage for encoded root-group attribute messages.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Encoded root-group attribute payloads in one contiguous byte buffer.
///
/// Each payload's end offset is stored in `ends`; payload boundaries are
/// therefore represented by one offset vector rather than one allocation per
/// attribute.
#[cfg(feature = "alloc")]
#[derive(Debug, Default)]
pub(super) struct RootAttributePayloads {
    bytes: Vec<u8>,
    ends: Vec<usize>,
}

#[cfg(feature = "alloc")]
impl RootAttributePayloads {
    /// Append one encoded attribute payload.
    pub(super) fn push(&mut self, mut payload: Vec<u8>) {
        self.bytes.append(&mut payload);
        self.ends.push(self.bytes.len());
    }

    /// Iterate over attribute payloads in insertion order.
    pub(super) fn iter(&self) -> impl Iterator<Item = &[u8]> {
        let mut start = 0_usize;
        self.ends.iter().map(move |end| {
            let payload = self
                .bytes
                .get(start..*end)
                .expect("invariant: attribute end offsets are ordered");
            start = *end;
            payload
        })
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::RootAttributePayloads;
    use alloc::vec;

    #[test]
    fn payloads_share_one_buffer_with_ordered_boundaries() {
        let mut payloads = RootAttributePayloads::default();
        payloads.push(vec![1, 2, 3]);
        payloads.push(Vec::new());
        payloads.push(vec![4, 5]);

        assert_eq!(payloads.bytes, vec![1, 2, 3, 4, 5]);
        assert_eq!(payloads.ends, vec![3, 3, 5]);
        assert_eq!(
            payloads.iter().collect::<Vec<_>>(),
            vec![&[1_u8, 2, 3][..], &[], &[4_u8, 5][..]]
        );
    }
}
