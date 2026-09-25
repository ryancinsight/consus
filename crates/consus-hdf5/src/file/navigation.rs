//! Path navigation and node classification for `Hdf5File`.

use super::{Hdf5File, reader};
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use consus_core::{Datatype, Error, NodeType, Result};
use consus_io::ReadAt;

impl<R: ReadAt + Sync> Hdf5File<R> {
    /// Navigate to an object by slash-separated path, returning its
    /// object header address.
    ///
    /// Leading  is accepted and ignored. Empty components (double
    /// slashes) are skipped. Returns  if any component
    /// is absent.
    ///
    /// ## Soft Link Resolution
    ///
    /// Soft links (type 1) are resolved recursively up to a depth of 40
    /// hops to break potential cycles. Absolute soft link targets (beginning
    /// with ) are resolved from the root group. Relative targets are
    /// resolved within the current group.
    ///
    /// External links return .
    ///
    /// ## Errors
    ///
    /// -  if any path component is missing.
    /// -  if an object header is malformed or a
    ///   soft link cycle exceeds the maximum depth of 40.
    /// -  if an external link is traversed.
    #[cfg(feature = "alloc")]
    pub fn open_path(&self, path: &str) -> Result<u64> {
        self.open_path_from(self.superblock.root_group_address, path, 0)
    }
    /// Resolve a path from a given group address with cycle-break depth tracking.
    ///
    /// Called by [] and recursively for soft link resolution.
    ///  increments on each soft link hop; exceeding
    /// returns [] to break cycles.
    #[cfg(feature = "alloc")]
    #[cfg(feature = "alloc")]
    fn open_path_from(&self, start: u64, path: &str, depth: usize) -> Result<u64> {
        const MAX_LINK_DEPTH: usize = 40;
        if depth > MAX_LINK_DEPTH {
            return Err(Error::InvalidFormat {
                message: alloc::string::String::from(
                    "soft link cycle detected: maximum link depth exceeded",
                ),
            });
        }

        let mut current = start;
        for component in path.split('/').filter(|s| !s.is_empty()) {
            let header = reader::read_object_header(&self.source, current, &self.ctx)?;
            let mut found: Option<u64> = None;

            // Try v2 link messages first (dense or compact).
            let v2 = reader::list_group_v2(&self.source, &header, &self.ctx)?;
            for (name, addr, link_type, soft_target) in &v2 {
                if name == component {
                    match link_type {
                        consus_core::LinkType::Hard => {
                            found = Some(*addr);
                        }
                        consus_core::LinkType::Soft => {
                            if let Some(target) = soft_target {
                                let resolved = if target.starts_with('/') {
                                    self.open_path_from(
                                        self.superblock.root_group_address,
                                        target,
                                        depth + 1,
                                    )?
                                } else {
                                    self.open_path_from(current, target, depth + 1)?
                                };
                                found = Some(resolved);
                            }
                        }
                        consus_core::LinkType::External => {
                            return Err(Error::UnsupportedFeature {
                                feature: alloc::string::String::from("external link resolution"),
                            });
                        }
                    }
                    break;
                }
            }

            // Fall back to v1 symbol table. v2 groups have no SYMBOL_TABLE
            // message, so list_group_v1 returns InvalidFormat in that case
            // treat failure as an empty list rather than propagating the error.
            if found.is_none()
                && let Ok(v1) = reader::list_group_v1(&self.source, &header, &self.ctx)
            {
                for (name, addr) in &v1 {
                    if name.as_str() == component {
                        found = Some(*addr);
                        break;
                    }
                }
            }

            current = found.ok_or_else(|| Error::NotFound {
                path: alloc::string::String::from(component),
            })?;
        }
        Ok(current)
    }
    /// List children of a group at the given object header address.
    ///
    /// Returns `(name, object_header_address, link_type)` triples.
    ///
    /// Tries v2 compact/dense link messages first.  Falls back to the v1
    /// symbol-table path only when a `SYMBOL_TABLE` message is present in
    /// the object header; v2 groups with no children have no such message
    /// and correctly return an empty list rather than an error.
    #[cfg(feature = "alloc")]
    pub fn list_group_at(&self, address: u64) -> Result<Vec<(String, u64, consus_core::LinkType)>> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;

        let v2 = reader::list_group_v2(&self.source, &header, &self.ctx)?;
        if !v2.is_empty() {
            return Ok(v2.into_iter().map(|(n, a, lt, _)| (n, a, lt)).collect());
        }

        // Only attempt the v1 symbol-table path when the object header
        // contains a SYMBOL_TABLE message.  v2 groups with zero children
        // produce an empty v2 list and carry no SYMBOL_TABLE message; for
        // them an empty result is correct and the v1 path must not be tried.
        use crate::object_header::message_types;
        if reader::find_message(&header, message_types::SYMBOL_TABLE).is_none() {
            return Ok(Vec::new());
        }

        let v1 = reader::list_group_v1(&self.source, &header, &self.ctx)?;
        Ok(v1
            .into_iter()
            .map(|(name, addr)| (name, addr, consus_core::LinkType::Hard))
            .collect())
    }
    /// Classify the object at the given object header address.
    #[cfg(feature = "alloc")]
    pub fn node_type_at(&self, address: u64) -> Result<NodeType> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;
        Ok(reader::classify_object(&header))
    }
    /// Read the fill value from the object header at `address`, if present.
    #[cfg(feature = "alloc")]
    pub fn fill_value_at(&self, address: u64) -> Result<Option<Vec<u8>>> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;
        Ok(reader::read_fill_value(&header))
    }
    /// Read the canonical datatype from a committed (named) datatype object at `address`.
    ///
    /// ## HDF5 specification
    ///
    /// A committed datatype object header (§IV.A.2.3) carries exactly one
    /// Datatype message (0x0003) without Dataspace or Data Layout.
    /// `classify_object` returns `NodeType::NamedDatatype` for such objects.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the object header is missing a Datatype message
    ///   or the datatype cannot be decoded.
    #[cfg(feature = "alloc")]
    pub fn named_datatype_at(&self, address: u64) -> Result<Datatype> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;
        let dt_msg = reader::find_message(&header, crate::object_header::message_types::DATATYPE)
            .ok_or_else(|| Error::InvalidFormat {
            message: String::from("committed datatype object header missing datatype message"),
        })?;
        crate::datatype::compound::parse_datatype(&dt_msg.data, &self.ctx.budget)
    }
}
