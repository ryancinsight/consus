//! Recursive record collection walks across a B-tree v2.

use super::{BTreeV2Header, BTreeV2InternalNode, BTreeV2LeafNode, BTreeV2Record};
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use consus_core::Result;
#[cfg(feature = "alloc")]
use consus_io::ReadAt;
#[cfg(all(feature = "async", feature = "alloc"))]
use moirai_async::io::AsyncReadAt;

/// Collect all records from a B-tree v2 by recursive traversal.
///
/// Traverses the tree from the root, visiting all internal and leaf
/// nodes, and returns the raw records in tree order (left to right).
///
/// ## Arguments
///
/// - `source`: I/O source.
/// - `header`: the parsed B-tree v2 header.
/// - `ctx`: parsing context for variable-width addresses.
///
/// ## Returns
///
/// All records in the tree, in order. Returns an empty vector for
/// an empty tree (root_address == `u64::MAX`).
///
/// ## Errors
///
/// Propagates I/O and format errors from node parsing.
#[cfg(feature = "alloc")]
pub fn collect_all_records<R: ReadAt>(
    source: &R,
    header: &BTreeV2Header,
    ctx: &ParseContext,
) -> Result<Vec<BTreeV2Record>> {
    if header.root_address == crate::constants::UNDEFINED_ADDRESS {
        return Ok(Vec::new());
    }

    if header.total_records == 0 {
        return Ok(Vec::new());
    }

    // `total_records` is a u64 read straight from the header; a hostile value
    // selected the allocation before a single record was parsed. It is a hint
    // whose truth the traversal establishes, so clamp rather than reject.
    let mut records = Vec::with_capacity(
        ctx.budget
            .capacity_hint(header.total_records, size_of::<BTreeV2Record>()),
    );
    collect_records_recursive(
        source,
        header,
        header.root_address,
        header.root_num_records,
        header.depth,
        0,
        ctx,
        &mut records,
    )?;
    Ok(records)
}

/// Recursive helper for [`collect_all_records`].
#[cfg(feature = "alloc")]
fn collect_records_recursive<R: ReadAt>(
    source: &R,
    header: &BTreeV2Header,
    node_address: u64,
    num_records: u16,
    depth: u16,
    descent: u16,
    ctx: &ParseContext,
    records: &mut Vec<BTreeV2Record>,
) -> Result<()> {
    // `header.depth` is a u16 read from the file, so an adversarial tree can
    // demand 65 535 stack frames. Rust performs no tail-call elimination and
    // a stack overflow aborts, so the descent is bounded here.
    let descent = ctx.budget.descend(descent, "b-tree v2 tree depth")?;

    if depth == 0 {
        // Leaf node
        let leaf = BTreeV2LeafNode::parse(source, node_address, header, num_records, ctx)?;
        records.extend(leaf.records);
    } else {
        // Internal node
        let internal = BTreeV2InternalNode::parse(source, node_address, header, num_records, ctx)?;

        // Interleave: child[0], record[0], child[1], record[1], ..., child[N]
        let n_rec = internal.records.len();
        let n_children = internal.child_addresses.len();

        for i in 0..n_children {
            // Visit child[i]
            if i < n_children {
                let child_addr = internal.child_addresses[i];
                let child_nrec = internal.child_num_records[i];
                collect_records_recursive(
                    source,
                    header,
                    child_addr,
                    child_nrec,
                    depth - 1,
                    descent,
                    ctx,
                    records,
                )?;
            }

            // Emit record[i] (interleaved between children)
            if i < n_rec {
                records.push(internal.records[i].clone());
            }
        }
    }
    Ok(())
}

/// Collect all records from a B-tree v2 by recursive traversal, asynchronously.
#[cfg(all(feature = "async", feature = "alloc"))]
pub async fn async_collect_all_records<R: AsyncReadAt>(
    source: &R,
    header: &BTreeV2Header,
    ctx: &ParseContext,
) -> Result<Vec<BTreeV2Record>> {
    if header.root_address == crate::constants::UNDEFINED_ADDRESS {
        return Ok(Vec::new());
    }

    if header.total_records == 0 {
        return Ok(Vec::new());
    }

    // `total_records` is a u64 read straight from the header; a hostile value
    // selected the allocation before a single record was parsed. It is a hint
    // whose truth the traversal establishes, so clamp rather than reject.
    let mut records = Vec::with_capacity(
        ctx.budget
            .capacity_hint(header.total_records, size_of::<BTreeV2Record>()),
    );
    Box::pin(async_collect_records_recursive(
        source,
        header,
        header.root_address,
        header.root_num_records,
        header.depth,
        0,
        ctx,
        &mut records,
    ))
    .await?;
    Ok(records)
}

#[cfg(all(feature = "async", feature = "alloc"))]
async fn async_collect_records_recursive<R: AsyncReadAt>(
    source: &R,
    header: &BTreeV2Header,
    node_address: u64,
    num_records: u16,
    depth: u16,
    descent: u16,
    ctx: &ParseContext,
    records: &mut Vec<BTreeV2Record>,
) -> Result<()> {
    let descent = ctx.budget.descend(descent, "b-tree v2 tree depth")?;

    if depth == 0 {
        let leaf =
            BTreeV2LeafNode::async_parse(source, node_address, header, num_records, ctx).await?;
        records.extend(leaf.records);
    } else {
        let internal =
            BTreeV2InternalNode::async_parse(source, node_address, header, num_records, ctx)
                .await?;

        let n_rec = internal.records.len();
        let n_children = internal.child_addresses.len();

        for i in 0..n_children {
            if i < n_children {
                let child_addr = internal.child_addresses[i];
                let child_nrec = internal.child_num_records[i];
                Box::pin(async_collect_records_recursive(
                    source,
                    header,
                    child_addr,
                    child_nrec,
                    depth - 1,
                    descent,
                    ctx,
                    records,
                ))
                .await?;
            }

            if i < n_rec {
                records.push(internal.records[i].clone());
            }
        }
    }
    Ok(())
}
