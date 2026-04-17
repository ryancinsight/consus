#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};

#[cfg(feature = "alloc")]
use core::num::NonZeroUsize;

#[cfg(feature = "alloc")]
use consus_core::{
    AttributeWrite, ByteOrder, ChunkShape, Compression, DatasetRead, DatasetWrite, Datatype, Error,
    Extent, FileRead, FileWrite, GroupRead, GroupWrite, HasAttributes, Node, NodeType, Result,
    Selection, Shape, StringEncoding,
};
#[cfg(feature = "alloc")]
use consus_io::{Length, ReadAt, WriteAt};

#[cfg(feature = "alloc")]
use crate::{
    file::{FITS_FORMAT_NAME, FitsFile},
    hdu::{FitsHdu, FitsHduIndex, FitsHduPayload},
    header::{FitsCard, FitsHeader, HeaderValue},
};

#[cfg(feature = "alloc")]
pub const PRIMARY_GROUP_PATH: &str = "/PRIMARY";

#[cfg(feature = "alloc")]
pub const HDU_GROUPS_ROOT_PATH: &str = "/HDU";

#[cfg(feature = "alloc")]
pub const ROOT_ATTRIBUTE_PREFIX: &str = "fits.";

#[cfg(feature = "alloc")]
pub const HEADER_ATTRIBUTE_PREFIX: &str = "header.";

#[cfg(feature = "alloc")]
pub const HDU_KIND_ATTRIBUTE: &str = "fits.hdu.kind";

#[cfg(feature = "alloc")]
pub const HDU_INDEX_ATTRIBUTE: &str = "fits.hdu.index";

#[cfg(feature = "alloc")]
pub const HDU_PATH_ATTRIBUTE: &str = "fits.hdu.path";

#[cfg(feature = "alloc")]
pub const HDU_DATASET_PATH_ATTRIBUTE: &str = "fits.hdu.dataset_path";

#[cfg(feature = "alloc")]
pub const HDU_HEADER_CARD_COUNT_ATTRIBUTE: &str = "fits.hdu.header_card_count";

#[cfg(feature = "alloc")]
pub const HDU_DATA_OFFSET_ATTRIBUTE: &str = "fits.hdu.data_offset";

#[cfg(feature = "alloc")]
pub const HDU_LOGICAL_DATA_LEN_ATTRIBUTE: &str = "fits.hdu.logical_data_len";

#[cfg(feature = "alloc")]
pub const HDU_PADDED_DATA_LEN_ATTRIBUTE: &str = "fits.hdu.padded_data_len";

#[cfg(feature = "alloc")]
pub const HDU_IS_PRIMARY_ATTRIBUTE: &str = "fits.hdu.is_primary";

#[cfg(feature = "alloc")]
pub const HDU_COUNT_ATTRIBUTE: &str = "fits.hdu_count";

#[cfg(feature = "alloc")]
pub const PRIMARY_DATASET_PATH: &str = "/PRIMARY/DATA";

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitsAttribute {
    name: String,
    datatype: Datatype,
    value: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl FitsAttribute {
    pub fn new(name: impl Into<String>, datatype: Datatype, value: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            datatype,
            value,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    pub fn value(&self) -> &[u8] {
        &self.value
    }
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitsGroup {
    path: String,
    name: String,
    attributes: Vec<FitsAttribute>,
    children: Vec<(String, NodeType)>,
}

#[cfg(feature = "alloc")]
impl FitsGroup {
    pub fn root<IO>(file: &FitsFile<IO>) -> Result<Self>
    where
        IO: ReadAt + Length,
    {
        let mut children = Vec::new();
        if file.hdu_count() > 0 {
            children.push((String::from("PRIMARY"), NodeType::Group));
        }
        if file.hdu_count() > 1 {
            children.push((String::from("HDU"), NodeType::Group));
        }

        Ok(Self {
            path: String::from("/"),
            name: String::from("/"),
            attributes: root_attributes(file)?,
            children,
        })
    }

    pub fn primary<IO>(file: &FitsFile<IO>) -> Result<Self>
    where
        IO: ReadAt + Length,
    {
        let hdu = file
            .primary_hdu()
            .ok_or_else(|| not_found(PRIMARY_GROUP_PATH))?;
        Ok(Self {
            path: String::from(PRIMARY_GROUP_PATH),
            name: String::from("PRIMARY"),
            attributes: hdu_group_attributes(hdu)?,
            children: vec![(String::from("DATA"), NodeType::Dataset)],
        })
    }

    pub fn hdu_root<IO>(file: &FitsFile<IO>) -> Result<Self>
    where
        IO: ReadAt + Length,
    {
        let mut children = Vec::new();
        for index in 1..file.hdu_count() {
            children.push((index.to_string(), NodeType::Group));
        }

        Ok(Self {
            path: String::from(HDU_GROUPS_ROOT_PATH),
            name: String::from("HDU"),
            attributes: hdu_root_attributes(file)?,
            children,
        })
    }

    pub fn hdu<IO>(file: &FitsFile<IO>, index: usize) -> Result<Self>
    where
        IO: ReadAt + Length,
    {
        if index == 0 {
            return Self::primary(file);
        }

        let hdu = file
            .hdu(index)
            .ok_or_else(|| not_found(hdu_group_path(index)))?;
        Ok(Self {
            path: hdu_group_path(index),
            name: index.to_string(),
            attributes: hdu_group_attributes(hdu)?,
            children: vec![(String::from("DATA"), NodeType::Dataset)],
        })
    }

    pub fn from_path<IO>(file: &FitsFile<IO>, path: &str) -> Result<Self>
    where
        IO: ReadAt + Length,
    {
        match parse_group_path(path)? {
            FitsGroupPath::Root => Self::root(file),
            FitsGroupPath::Primary => Self::primary(file),
            FitsGroupPath::HduRoot => Self::hdu_root(file),
            FitsGroupPath::Hdu(index) => Self::hdu(file, index),
        }
    }

    pub fn attributes(&self) -> &[FitsAttribute] {
        &self.attributes
    }

    pub fn children(&self) -> &[(String, NodeType)] {
        &self.children
    }
}

#[cfg(feature = "alloc")]
impl Node for FitsGroup {
    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &str {
        &self.path
    }

    fn node_type(&self) -> NodeType {
        NodeType::Group
    }
}

#[cfg(feature = "alloc")]
impl HasAttributes for FitsGroup {
    fn num_attributes(&self) -> Result<usize> {
        Ok(self.attributes.len())
    }

    fn has_attribute(&self, name: &str) -> Result<bool> {
        Ok(self
            .attributes
            .iter()
            .any(|attribute| attribute.name() == name))
    }

    fn attribute_datatype(&self, name: &str) -> Result<Datatype> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name() == name)
            .map(|attribute| attribute.datatype().clone())
            .ok_or_else(|| not_found(name))
    }

    fn read_attribute_raw(&self, name: &str, buf: &mut [u8]) -> Result<usize> {
        let attribute = self
            .attributes
            .iter()
            .find(|attribute| attribute.name() == name)
            .ok_or_else(|| not_found(name))?;

        if buf.len() < attribute.value().len() {
            return Err(Error::BufferTooSmall {
                required: attribute.value().len(),
                provided: buf.len(),
            });
        }

        buf[..attribute.value().len()].copy_from_slice(attribute.value());
        Ok(attribute.value().len())
    }

    fn for_each_attribute(&self, visitor: &mut dyn FnMut(&str) -> bool) -> Result<()> {
        for attribute in &self.attributes {
            if !visitor(attribute.name()) {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "alloc")]
impl GroupRead for FitsGroup {
    fn num_children(&self) -> Result<usize> {
        Ok(self.children.len())
    }

    fn contains(&self, name: &str) -> Result<bool> {
        Ok(self.children.iter().any(|(child, _)| child == name))
    }

    fn child_node_type(&self, name: &str) -> Result<NodeType> {
        self.children
            .iter()
            .find(|(child, _)| child == name)
            .map(|(_, node_type)| *node_type)
            .ok_or_else(|| not_found(name))
    }

    fn for_each_child(&self, visitor: &mut dyn FnMut(&str, NodeType) -> bool) -> Result<()> {
        for (name, node_type) in &self.children {
            if !visitor(name, *node_type) {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "alloc")]
impl GroupWrite for FitsGroup {
    fn create_group(&mut self, _name: &str) -> Result<()> {
        Err(Error::UnsupportedFeature {
            feature: String::from("FITS group projection is read-only"),
        })
    }

    fn remove_child(&mut self, _name: &str) -> Result<()> {
        Err(Error::UnsupportedFeature {
            feature: String::from("FITS group projection is read-only"),
        })
    }
}

#[cfg(feature = "alloc")]
impl AttributeWrite for FitsGroup {
    fn write_attribute_raw(
        &mut self,
        _name: &str,
        _datatype: &Datatype,
        _data: &[u8],
    ) -> Result<()> {
        Err(Error::UnsupportedFeature {
            feature: String::from("FITS header attribute projection is read-only"),
        })
    }

    fn delete_attribute(&mut self, _name: &str) -> Result<()> {
        Err(Error::UnsupportedFeature {
            feature: String::from("FITS header attribute projection is read-only"),
        })
    }
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitsDataset {
    path: String,
    name: String,
    datatype: Datatype,
    shape: Shape,
    attributes: Vec<FitsAttribute>,
    hdu_index: usize,
}

#[cfg(feature = "alloc")]
impl FitsDataset {
    pub fn from_path<IO>(file: &FitsFile<IO>, path: &str) -> Result<Self>
    where
        IO: ReadAt + Length,
    {
        let dataset_path = parse_dataset_projection_path(path)?;
        let hdu = file
            .hdu(dataset_path.hdu_index)
            .ok_or_else(|| not_found(path))?;

        Ok(Self {
            path: dataset_path.path,
            name: String::from("DATA"),
            datatype: datatype_for_hdu(hdu)?,
            shape: shape_for_hdu(hdu)?,
            attributes: hdu_dataset_attributes(hdu)?,
            hdu_index: dataset_path.hdu_index,
        })
    }

    pub fn hdu_index(&self) -> usize {
        self.hdu_index
    }
}

#[cfg(feature = "alloc")]
impl Node for FitsDataset {
    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &str {
        &self.path
    }

    fn node_type(&self) -> NodeType {
        NodeType::Dataset
    }
}

#[cfg(feature = "alloc")]
impl HasAttributes for FitsDataset {
    fn num_attributes(&self) -> Result<usize> {
        Ok(self.attributes.len())
    }

    fn has_attribute(&self, name: &str) -> Result<bool> {
        Ok(self
            .attributes
            .iter()
            .any(|attribute| attribute.name() == name))
    }

    fn attribute_datatype(&self, name: &str) -> Result<Datatype> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name() == name)
            .map(|attribute| attribute.datatype().clone())
            .ok_or_else(|| not_found(name))
    }

    fn read_attribute_raw(&self, name: &str, buf: &mut [u8]) -> Result<usize> {
        let attribute = self
            .attributes
            .iter()
            .find(|attribute| attribute.name() == name)
            .ok_or_else(|| not_found(name))?;

        if buf.len() < attribute.value().len() {
            return Err(Error::BufferTooSmall {
                required: attribute.value().len(),
                provided: buf.len(),
            });
        }

        buf[..attribute.value().len()].copy_from_slice(attribute.value());
        Ok(attribute.value().len())
    }

    fn for_each_attribute(&self, visitor: &mut dyn FnMut(&str) -> bool) -> Result<()> {
        for attribute in &self.attributes {
            if !visitor(attribute.name()) {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "alloc")]
impl DatasetRead for FitsDataset {
    fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn chunk_shape(&self) -> Option<&ChunkShape> {
        None
    }

    fn compression(&self) -> Compression {
        Compression::None
    }

    fn read_raw(&self, _selection: &Selection, _buf: &mut [u8]) -> Result<usize> {
        Err(Error::UnsupportedFeature {
            feature: String::from(
                "FitsDataset is metadata-only; use FitsFile::read_dataset_raw for payload I/O",
            ),
        })
    }
}

#[cfg(feature = "alloc")]
impl DatasetWrite for FitsDataset {
    fn write_raw(&mut self, _selection: &Selection, _data: &[u8]) -> Result<()> {
        Err(Error::UnsupportedFeature {
            feature: String::from(
                "FitsDataset is metadata-only; use FitsFile::write_dataset_raw for payload I/O",
            ),
        })
    }

    fn resize(&mut self, _new_dims: &[usize]) -> Result<()> {
        Err(Error::UnsupportedFeature {
            feature: String::from("FITS datasets are fixed-size projections"),
        })
    }
}

#[cfg(feature = "alloc")]
impl AttributeWrite for FitsDataset {
    fn write_attribute_raw(
        &mut self,
        _name: &str,
        _datatype: &Datatype,
        _data: &[u8],
    ) -> Result<()> {
        Err(Error::UnsupportedFeature {
            feature: String::from("FITS header attribute projection is read-only"),
        })
    }

    fn delete_attribute(&mut self, _name: &str) -> Result<()> {
        Err(Error::UnsupportedFeature {
            feature: String::from("FITS header attribute projection is read-only"),
        })
    }
}

#[cfg(feature = "alloc")]
pub trait FitsFormatExt {
    fn fits_group(&self, path: &str) -> Result<FitsGroup>;
    fn fits_dataset(&self, path: &str) -> Result<FitsDataset>;
}

#[cfg(feature = "alloc")]
impl<IO> FitsFormatExt for FitsFile<IO>
where
    IO: ReadAt + Length,
{
    fn fits_group(&self, path: &str) -> Result<FitsGroup> {
        FitsGroup::from_path(self, path)
    }

    fn fits_dataset(&self, path: &str) -> Result<FitsDataset> {
        FitsDataset::from_path(self, path)
    }
}

#[cfg(feature = "alloc")]
pub fn root_attributes<IO>(file: &FitsFile<IO>) -> Result<Vec<FitsAttribute>>
where
    IO: ReadAt + Length,
{
    Ok(vec![
        string_attribute(format!("{ROOT_ATTRIBUTE_PREFIX}format"), FITS_FORMAT_NAME),
        u64_attribute(HDU_COUNT_ATTRIBUTE, file.hdu_count() as u64),
    ])
}

#[cfg(feature = "alloc")]
pub fn hdu_root_attributes<IO>(file: &FitsFile<IO>) -> Result<Vec<FitsAttribute>>
where
    IO: ReadAt + Length,
{
    let extension_count = file.hdu_count().saturating_sub(1);
    Ok(vec![
        string_attribute(format!("{ROOT_ATTRIBUTE_PREFIX}format"), FITS_FORMAT_NAME),
        u64_attribute(HDU_COUNT_ATTRIBUTE, extension_count as u64),
    ])
}

#[cfg(feature = "alloc")]
pub fn hdu_group_attributes(hdu: &FitsHdu) -> Result<Vec<FitsAttribute>> {
    let mut attributes = hdu_common_attributes(hdu)?;
    append_header_attributes(&mut attributes, hdu.header())?;
    Ok(attributes)
}

#[cfg(feature = "alloc")]
pub fn hdu_dataset_attributes(hdu: &FitsHdu) -> Result<Vec<FitsAttribute>> {
    let mut attributes = hdu_common_attributes(hdu)?;
    attributes.push(string_attribute(
        HDU_DATASET_PATH_ATTRIBUTE,
        hdu_dataset_path(hdu.index().get()),
    ));
    append_header_attributes(&mut attributes, hdu.header())?;
    Ok(attributes)
}

#[cfg(feature = "alloc")]
fn hdu_common_attributes(hdu: &FitsHdu) -> Result<Vec<FitsAttribute>> {
    let mut attributes = Vec::new();
    attributes.push(string_attribute(HDU_KIND_ATTRIBUTE, hdu_kind_name(hdu)));
    attributes.push(u64_attribute(HDU_INDEX_ATTRIBUTE, hdu.index().get() as u64));
    attributes.push(string_attribute(
        HDU_PATH_ATTRIBUTE,
        hdu_group_path(hdu.index().get()),
    ));
    attributes.push(u64_attribute(
        HDU_HEADER_CARD_COUNT_ATTRIBUTE,
        hdu.header().len() as u64,
    ));
    attributes.push(u64_attribute(
        HDU_DATA_OFFSET_ATTRIBUTE,
        hdu.data_span().offset(),
    ));
    attributes.push(u64_attribute(
        HDU_LOGICAL_DATA_LEN_ATTRIBUTE,
        hdu.data_span().logical_len() as u64,
    ));
    attributes.push(u64_attribute(
        HDU_PADDED_DATA_LEN_ATTRIBUTE,
        hdu.data_span().padded_len() as u64,
    ));
    attributes.push(bool_attribute(HDU_IS_PRIMARY_ATTRIBUTE, hdu.is_primary()));
    Ok(attributes)
}

#[cfg(feature = "alloc")]
fn append_header_attributes(
    attributes: &mut Vec<FitsAttribute>,
    header: &FitsHeader,
) -> Result<()> {
    for (index, card) in header.cards().iter().enumerate() {
        let name = header_attribute_name(card, index);
        let attribute = attribute_from_header_card(name, card)?;
        attributes.push(attribute);
    }
    Ok(())
}

#[cfg(feature = "alloc")]
fn header_attribute_name(card: &FitsCard, index: usize) -> String {
    let keyword = card.keyword().to_string();
    format!("{HEADER_ATTRIBUTE_PREFIX}{index:04}.{keyword}")
}

#[cfg(feature = "alloc")]
fn attribute_from_header_card(name: String, card: &FitsCard) -> Result<FitsAttribute> {
    match card.value() {
        Some(HeaderValue::String(value)) => Ok(string_attribute(name, value)),
        Some(HeaderValue::Logical(value)) => Ok(bool_attribute(name, *value)),
        Some(HeaderValue::Integer(value)) => Ok(i64_attribute(name, value.to_i64()?)),
        Some(HeaderValue::Real(value)) => Ok(f64_attribute(name, value.to_f64()?)),
        Some(HeaderValue::Complex(value)) => {
            let (real, imaginary) = value.to_f64_pair()?;
            Ok(f64_pair_attribute(name, real, imaginary))
        }
        Some(HeaderValue::Undefined) | None => {
            Ok(string_attribute(name, card.comment().unwrap_or_default()))
        }
    }
}

#[cfg(feature = "alloc")]
fn datatype_for_hdu(hdu: &FitsHdu) -> Result<Datatype> {
    match hdu.payload() {
        FitsHduPayload::Image(image) => Ok(image.bitpix().to_datatype()),
        FitsHduPayload::AsciiTable(table) => opaque_row_datatype(table.row_len()),
        FitsHduPayload::BinaryTable(table) => opaque_row_datatype(table.row_len()),
    }
}

#[cfg(feature = "alloc")]
fn shape_for_hdu(hdu: &FitsHdu) -> Result<Shape> {
    match hdu.payload() {
        FitsHduPayload::Image(image) => Ok(image.shape().clone()),
        FitsHduPayload::AsciiTable(table) => Ok(table.shape()),
        FitsHduPayload::BinaryTable(table) => Ok(table.shape()),
    }
}

#[cfg(feature = "alloc")]
fn opaque_row_datatype(row_len: usize) -> Result<Datatype> {
    if row_len == 0 {
        return Err(Error::InvalidFormat {
            message: String::from("FITS table row length must be positive"),
        });
    }

    Ok(Datatype::Opaque {
        size: row_len,
        tag: Some(String::from("fits-row")),
    })
}

#[cfg(feature = "alloc")]
fn string_attribute(name: impl Into<String>, value: impl AsRef<str>) -> FitsAttribute {
    let bytes = value.as_ref().as_bytes().to_vec();
    FitsAttribute::new(
        name,
        Datatype::FixedString {
            length: bytes.len(),
            encoding: StringEncoding::Utf8,
        },
        bytes,
    )
}

#[cfg(feature = "alloc")]
fn bool_attribute(name: impl Into<String>, value: bool) -> FitsAttribute {
    FitsAttribute::new(name, Datatype::Boolean, vec![u8::from(value)])
}

#[cfg(feature = "alloc")]
fn u64_attribute(name: impl Into<String>, value: u64) -> FitsAttribute {
    FitsAttribute::new(
        name,
        Datatype::Integer {
            bits: NonZeroUsize::new(64).expect("64 is non-zero"),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        },
        value.to_le_bytes().to_vec(),
    )
}

#[cfg(feature = "alloc")]
fn i64_attribute(name: impl Into<String>, value: i64) -> FitsAttribute {
    FitsAttribute::new(
        name,
        Datatype::Integer {
            bits: NonZeroUsize::new(64).expect("64 is non-zero"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        },
        value.to_le_bytes().to_vec(),
    )
}

#[cfg(feature = "alloc")]
fn f64_attribute(name: impl Into<String>, value: f64) -> FitsAttribute {
    FitsAttribute::new(
        name,
        Datatype::Float {
            bits: NonZeroUsize::new(64).expect("64 is non-zero"),
            byte_order: ByteOrder::LittleEndian,
        },
        value.to_le_bytes().to_vec(),
    )
}

#[cfg(feature = "alloc")]
fn f64_pair_attribute(name: impl Into<String>, left: f64, right: f64) -> FitsAttribute {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&left.to_le_bytes());
    bytes.extend_from_slice(&right.to_le_bytes());

    FitsAttribute::new(
        name,
        Datatype::Array {
            base: Arc::new(Datatype::Float {
                bits: NonZeroUsize::new(64).expect("64 is non-zero"),
                byte_order: ByteOrder::LittleEndian,
            })
            .as_ref()
            .clone()
            .into(),
            dims: vec![2],
        },
        bytes,
    )
}

#[cfg(feature = "alloc")]
fn hdu_kind_name(hdu: &FitsHdu) -> &'static str {
    if hdu.is_primary() {
        "primary"
    } else if hdu.is_image() {
        "image"
    } else if hdu.is_ascii_table() {
        "ascii_table"
    } else {
        "binary_table"
    }
}

#[cfg(feature = "alloc")]
fn hdu_group_path(index: usize) -> String {
    if index == 0 {
        String::from(PRIMARY_GROUP_PATH)
    } else {
        format!("{HDU_GROUPS_ROOT_PATH}/{index}")
    }
}

#[cfg(feature = "alloc")]
fn hdu_dataset_path(index: usize) -> String {
    if index == 0 {
        String::from(PRIMARY_DATASET_PATH)
    } else {
        format!("{HDU_GROUPS_ROOT_PATH}/{index}/DATA")
    }
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
enum FitsGroupPath {
    Root,
    Primary,
    HduRoot,
    Hdu(usize),
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct FitsDatasetPath {
    path: String,
    hdu_index: usize,
}

#[cfg(feature = "alloc")]
fn parse_group_path(path: &str) -> Result<FitsGroupPath> {
    match path {
        "/" => Ok(FitsGroupPath::Root),
        PRIMARY_GROUP_PATH => Ok(FitsGroupPath::Primary),
        HDU_GROUPS_ROOT_PATH => Ok(FitsGroupPath::HduRoot),
        _ => {
            let Some(rest) = path.strip_prefix("/HDU/") else {
                return Err(not_found(path));
            };
            if rest.is_empty() || rest.contains('/') {
                return Err(not_found(path));
            }
            let index = rest.parse::<usize>().map_err(|_| not_found(path))?;
            if index == 0 {
                return Err(not_found(path));
            }
            Ok(FitsGroupPath::Hdu(index))
        }
    }
}

#[cfg(feature = "alloc")]
fn parse_dataset_projection_path(path: &str) -> Result<FitsDatasetPath> {
    if path == PRIMARY_DATASET_PATH {
        return Ok(FitsDatasetPath {
            path: String::from(path),
            hdu_index: 0,
        });
    }

    let Some(rest) = path.strip_prefix("/HDU/") else {
        return Err(not_found(path));
    };
    let Some(index_text) = rest.strip_suffix("/DATA") else {
        return Err(not_found(path));
    };
    if index_text.is_empty() || index_text.contains('/') {
        return Err(not_found(path));
    }

    let index = index_text.parse::<usize>().map_err(|_| not_found(path))?;
    if index == 0 {
        return Err(not_found(path));
    }

    Ok(FitsDatasetPath {
        path: String::from(path),
        hdu_index: index,
    })
}

#[cfg(feature = "alloc")]
fn not_found(path: impl Into<String>) -> Error {
    Error::NotFound { path: path.into() }
}
