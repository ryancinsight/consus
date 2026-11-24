#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;

/// Global heap collection signature.
pub const GLOBAL_HEAP_SIGNATURE: [u8; 4] = *b"GCOL";

/// A single object within a global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapObject {
    /// Object index (1-based; 0 is the free-space marker).
    pub index: u16,
    /// Reference count.
    pub reference_count: u32,
    /// Object data.
    #[cfg(feature = "alloc")]
    pub data: Vec<u8>,
}

/// A parsed global heap collection.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GlobalHeapCollection {
    /// Total collection size in bytes (including header).
    pub collection_size: u64,
    /// Objects in the collection (index 0 / free-space marker excluded).
    pub objects: Vec<GlobalHeapObject>,
}

#[cfg(feature = "alloc")]
impl GlobalHeapCollection {
    /// Parse a global heap collection from the given file address.
    ///
    /// Reads the collection header, validates the signature and version,
    /// then iterates over heap objects until index 0 (free-space marker)
    /// is encountered or the collection size is exhausted.
    ///
    /// ## Arguments
    ///
    /// - `source`: positioned I/O source.
    /// - `address`: file offset of the collection header.
    /// - `ctx`: parsing context (offset/length sizes from superblock).
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not "GCOL".
    /// - [`Error::InvalidFormat`] if the version is not 1.
    /// - [`Error::InvalidFormat`] if the collection is truncated.
    pub fn parse<R: ReadAt>(source: &R, address: u64, ctx: &ParseContext) -> Result<Self> {
        let l = ctx.length_bytes();
        // Header: signature(4) + version(1) + reserved(3) + collection_size(L)
        let header_size = 8 + l;
        let mut header_buf = vec![0u8; header_size];
        source.read_at(address, &mut header_buf)?;

        // Validate signature
        if header_buf[0..4] != GLOBAL_HEAP_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: String::from("invalid global heap collection signature"),
            });
        }

        // Validate version
        let version = header_buf[4];
        if version != 1 {
            return Err(Error::InvalidFormat {
                message: alloc::format!("unsupported global heap version: {version}, expected 1"),
            });
        }

        let collection_size = ctx.read_length(&header_buf[8..]);
        let collection_size_usize = ctx
            .budget
            .checked_bytes(collection_size, "global heap collection size")?;

        // Read the entire collection body
        let body_size = collection_size_usize
            .checked_sub(header_size)
            .ok_or_else(|| Error::InvalidFormat {
                message: alloc::format!(
                    "global heap collection size {} is smaller than its {}-byte header",
                    collection_size,
                    header_size
                ),
            })?;
        if body_size == 0 {
            return Ok(Self {
                collection_size,
                objects: Vec::new(),
            });
        }

        let mut body = ctx.budget.zeroed(
            u64::try_from(body_size).map_err(|_| Error::Overflow)?,
            "global heap collection body",
        )?;
        let body_address = address
            .checked_add(u64::try_from(header_size).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        source.read_at(body_address, &mut body)?;

        // Parse objects
        let mut objects = Vec::new();
        let mut cursor = 0usize;

        // Object header: index(2) + ref_count(2) + reserved(4) + size(L) = 8 + L bytes
        let obj_header_size = 8 + l;

        while cursor
            .checked_add(obj_header_size)
            .is_some_and(|end| end <= body.len())
        {
            let index = u16::from_le_bytes([body[cursor], body[cursor + 1]]);

            // Index 0 marks end of objects (free-space marker)
            if index == 0 {
                break;
            }

            let reference_count = u32::from(u16::from_le_bytes([
                body[cursor + 2],
                body[cursor + 3],
            ]));
            // bytes [cursor+4..cursor+8] are reserved

            let object_size = ctx.budget.checked_bytes(
                ctx.read_length(&body[cursor + 8..]),
                "global heap object size",
            )?;
            cursor = cursor.checked_add(obj_header_size).ok_or(Error::Overflow)?;

            // Read object data
            let object_end = cursor.checked_add(object_size).ok_or(Error::Overflow)?;
            if object_end > body.len() {
                return Err(Error::InvalidFormat {
                    message: alloc::format!(
                        "global heap object {} truncated: need {} bytes at offset {}, have {}",
                        index,
                        object_size,
                        cursor,
                        body.len()
                    ),
                });
            }

            let data = Vec::from(&body[cursor..object_end]);
            objects.push(GlobalHeapObject {
                index,
                reference_count,
                data,
            });

            // Advance past object data, padded to 8-byte boundary
            let padded_size = object_size.checked_add(7).ok_or(Error::Overflow)? & !7;
            cursor = object_end
                .checked_add(padded_size - object_size)
                .ok_or(Error::Overflow)?;
        }

        Ok(Self {
            collection_size,
            objects,
        })
    }

    /// Look up an object by its 1-based index.
    ///
    /// Returns `None` if no object with the given index exists.
    pub fn get(&self, index: u16) -> Option<&GlobalHeapObject> {
        self.objects.iter().find(|o| o.index == index)
    }

    /// Look up an object's data by its 1-based index.
    ///
    /// Returns `None` if no object with the given index exists.
    pub fn get_data(&self, index: u16) -> Option<&[u8]> {
        self.get(index).map(|o| o.data.as_slice())
    }
}
