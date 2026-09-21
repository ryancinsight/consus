use crate::{DecodeError, DecodeErrorKind};

use super::super::Orientation;
use super::{field, metadata};

const MAX_DIRECTORIES: usize = 16;

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum DirectoryKind {
    #[default]
    Root,
    Exif,
    Gps,
    Interoperability,
    Thumbnail,
}

#[derive(Default)]
pub(super) struct PixelDensity {
    pub(super) horizontal_resolution: Option<(u32, u32)>,
    pub(super) vertical_resolution: Option<(u32, u32)>,
    pub(super) resolution_unit: Option<u16>,
}

#[derive(Default)]
pub(super) struct PresentationMetadata {
    pub(super) primary_density: PixelDensity,
    pub(super) thumbnail_density: PixelDensity,
    pub(super) thumbnail_offset: Option<usize>,
    pub(super) thumbnail_length: Option<usize>,
}

struct DirectoryContext<'a> {
    tiff: &'a [u8],
    order: ByteOrder,
    offset: usize,
    kind: DirectoryKind,
    orientation: &'a mut Option<Orientation>,
    metadata: &'a mut PresentationMetadata,
    pending: &'a mut [usize; MAX_DIRECTORIES],
    pending_kinds: &'a mut [DirectoryKind; MAX_DIRECTORIES],
    pending_len: &'a mut usize,
}

struct EntryContext<'a> {
    tiff: &'a [u8],
    order: ByteOrder,
    kind: DirectoryKind,
    entry: &'a [u8],
    tag: u16,
    field_type: u16,
    count: usize,
    orientation: &'a mut Option<Orientation>,
    metadata: &'a mut PresentationMetadata,
    pending: &'a mut [usize; MAX_DIRECTORIES],
    pending_kinds: &'a mut [DirectoryKind; MAX_DIRECTORIES],
    pending_len: &'a mut usize,
}

struct PointerContext<'a> {
    entry: &'a [u8],
    order: ByteOrder,
    field_type: u16,
    count: usize,
    tag: u16,
    pending: &'a mut [usize; MAX_DIRECTORIES],
    pending_kinds: &'a mut [DirectoryKind; MAX_DIRECTORIES],
    pending_len: &'a mut usize,
}

struct DensityContext<'a> {
    tiff: &'a [u8],
    order: ByteOrder,
    entry: &'a [u8],
    field_type: u16,
    count: usize,
    tag: u16,
    kind: DirectoryKind,
    metadata: &'a mut PresentationMetadata,
}

impl ByteOrder {
    fn read_short(self, bytes: &[u8]) -> Result<u16, DecodeError> {
        let value: [u8; 2] = bytes
            .get(..2)
            .ok_or_else(malformed)?
            .try_into()
            .map_err(|_| malformed())?;
        Ok(match self {
            Self::Little => u16::from_le_bytes(value),
            Self::Big => u16::from_be_bytes(value),
        })
    }

    fn read_long(self, bytes: &[u8]) -> Result<u32, DecodeError> {
        let value: [u8; 4] = bytes
            .get(..4)
            .ok_or_else(malformed)?
            .try_into()
            .map_err(|_| malformed())?;
        Ok(match self {
            Self::Little => u32::from_le_bytes(value),
            Self::Big => u32::from_be_bytes(value),
        })
    }
}

fn push_directory(
    pending: &mut [usize; MAX_DIRECTORIES],
    pending_kinds: &mut [DirectoryKind; MAX_DIRECTORIES],
    pending_len: &mut usize,
    offset: usize,
    kind: DirectoryKind,
) -> Result<(), DecodeError> {
    let slot = pending.get_mut(*pending_len).ok_or_else(too_large)?;
    *slot = offset;
    let kind_slot = pending_kinds.get_mut(*pending_len).ok_or_else(too_large)?;
    *kind_slot = kind;
    *pending_len = pending_len.checked_add(1).ok_or_else(too_large)?;
    Ok(())
}

pub(in super::super) fn parse(tiff: &[u8]) -> Result<Orientation, DecodeError> {
    let order = match tiff.get(..2) {
        Some(b"II") => ByteOrder::Little,
        Some(b"MM") => ByteOrder::Big,
        _ => return Err(malformed()),
    };
    if order.read_short(tiff.get(2..).ok_or_else(malformed)?)? != 42 {
        return Err(malformed());
    }
    let root = usize::try_from(order.read_long(tiff.get(4..).ok_or_else(malformed)?)?)
        .map_err(|_| too_large())?;
    if root < 8 || !root.is_multiple_of(2) {
        return Err(malformed());
    }

    let mut pending = [0_usize; MAX_DIRECTORIES];
    let mut pending_kinds = [DirectoryKind::Root; MAX_DIRECTORIES];
    let mut pending_len = 1_usize;
    pending[0] = root;
    let mut visited = [0_usize; MAX_DIRECTORIES];
    let mut visited_len = 0;
    let mut orientation = None;
    let mut presentation = PresentationMetadata::default();

    while pending_len != 0 {
        pending_len = pending_len.checked_sub(1).ok_or_else(malformed)?;
        let offset = *pending.get(pending_len).ok_or_else(malformed)?;
        let kind = *pending_kinds.get(pending_len).ok_or_else(malformed)?;
        if visited
            .get(..visited_len)
            .ok_or_else(malformed)?
            .contains(&offset)
        {
            return Err(malformed());
        }
        if visited_len == MAX_DIRECTORIES {
            return Err(too_large());
        }
        *visited.get_mut(visited_len).ok_or_else(too_large)? = offset;
        visited_len = visited_len.checked_add(1).ok_or_else(too_large)?;
        parse_directory(DirectoryContext {
            tiff,
            order,
            offset,
            kind,
            orientation: &mut orientation,
            metadata: &mut presentation,
            pending: &mut pending,
            pending_kinds: &mut pending_kinds,
            pending_len: &mut pending_len,
        })?;
    }

    metadata::validate(tiff, &presentation)?;
    Ok(orientation.unwrap_or_default())
}

fn parse_directory(context: DirectoryContext<'_>) -> Result<(), DecodeError> {
    let DirectoryContext {
        tiff,
        order,
        offset,
        kind,
        orientation,
        metadata,
        pending,
        pending_kinds,
        pending_len,
    } = context;
    if offset < 8 || !offset.is_multiple_of(2) {
        return Err(malformed());
    }
    let directory = tiff.get(offset..).ok_or_else(malformed)?;
    let entries = usize::from(order.read_short(directory)?);
    if entries == 0 {
        return Err(malformed());
    }
    let entries_bytes = entries.checked_mul(12).ok_or_else(too_large)?;
    let entries_end = 2_usize.checked_add(entries_bytes).ok_or_else(too_large)?;
    let table_end = entries_end.checked_add(4).ok_or_else(too_large)?;
    if directory.len() < table_end {
        return Err(malformed());
    }

    let entries_table = directory.get(2..entries_end).ok_or_else(malformed)?;
    for entry in entries_table.chunks_exact(12) {
        let tag = order.read_short(entry)?;
        let field_type = order.read_short(entry.get(2..).ok_or_else(malformed)?)?;
        let count = usize::try_from(order.read_long(entry.get(4..).ok_or_else(malformed)?)?)
            .map_err(|_| too_large())?;
        let unit = field::encoded_width(field_type)?;
        if count == 0 {
            return Err(malformed());
        }
        let size = count.checked_mul(unit).ok_or_else(too_large)?;
        if size > 4 {
            let value_offset =
                usize::try_from(order.read_long(entry.get(8..).ok_or_else(malformed)?)?)
                    .map_err(|_| too_large())?;
            let end = value_offset.checked_add(size).ok_or_else(too_large)?;
            if value_offset < 8
                || !value_offset.is_multiple_of(2)
                || tiff.get(value_offset..end).is_none()
            {
                return Err(malformed());
            }
        }
        parse_entry(EntryContext {
            tiff,
            order,
            kind,
            entry,
            tag,
            field_type,
            count,
            orientation,
            metadata,
            pending,
            pending_kinds,
            pending_len,
        })?;
    }

    let next =
        usize::try_from(order.read_long(directory.get(entries_end..).ok_or_else(malformed)?)?)
            .map_err(|_| too_large())?;
    if next != 0 {
        let next_kind = if kind == DirectoryKind::Root {
            DirectoryKind::Thumbnail
        } else {
            kind
        };
        push_directory(pending, pending_kinds, pending_len, next, next_kind)?;
    }
    Ok(())
}

fn parse_entry(context: EntryContext<'_>) -> Result<(), DecodeError> {
    let EntryContext {
        tiff,
        order,
        kind,
        entry,
        tag,
        field_type,
        count,
        orientation,
        metadata,
        pending,
        pending_kinds,
        pending_len,
    } = context;
    match tag {
        0x0112 => parse_orientation(entry, order, field_type, count, kind, orientation)?,
        0x8769 | 0x8825 | 0xa005 => parse_pointer(PointerContext {
            entry,
            order,
            field_type,
            count,
            tag,
            pending,
            pending_kinds,
            pending_len,
        })?,
        0xa001 => parse_color_space(entry, order, field_type, count)?,
        0x011a | 0x011b => parse_density(DensityContext {
            tiff,
            order,
            entry,
            field_type,
            count,
            tag,
            kind,
            metadata,
        })?,
        0x0128 => parse_resolution_unit(entry, order, field_type, count, kind, metadata)?,
        0x0201 | 0x0202 if kind == DirectoryKind::Thumbnail => {
            parse_thumbnail(entry, order, field_type, count, tag, metadata)?;
        }
        0x0001 if kind == DirectoryKind::Interoperability => {
            parse_interoperability(entry, field_type, count)?;
        }
        0x0111 | 0x0117 if kind == DirectoryKind::Thumbnail => return Err(unsupported()),
        0x012d | 0x013e | 0x013f | 0x0211 | 0x0214 | 0x8773 | 0xa500 => {
            return Err(unsupported());
        }
        _ => {}
    }
    Ok(())
}

fn parse_orientation(
    entry: &[u8],
    order: ByteOrder,
    field_type: u16,
    count: usize,
    kind: DirectoryKind,
    orientation: &mut Option<Orientation>,
) -> Result<(), DecodeError> {
    if field_type != 3 || count != 1 {
        return Err(malformed());
    }
    let value = Orientation::try_from(order.read_short(entry.get(8..).ok_or_else(malformed)?)?)?;
    if kind == DirectoryKind::Root && orientation.replace(value).is_some() {
        return Err(malformed());
    }
    Ok(())
}

fn parse_pointer(context: PointerContext<'_>) -> Result<(), DecodeError> {
    let PointerContext {
        entry,
        order,
        field_type,
        count,
        tag,
        pending,
        pending_kinds,
        pending_len,
    } = context;
    if field_type != 4 || count != 1 {
        return Err(malformed());
    }
    let target = usize::try_from(order.read_long(entry.get(8..).ok_or_else(malformed)?)?)
        .map_err(|_| too_large())?;
    if target == 0 {
        return Ok(());
    }
    let target_kind = match tag {
        0x8769 => DirectoryKind::Exif,
        0x8825 => DirectoryKind::Gps,
        0xa005 => DirectoryKind::Interoperability,
        _ => return Err(malformed()),
    };
    push_directory(pending, pending_kinds, pending_len, target, target_kind)
}

fn parse_color_space(
    entry: &[u8],
    order: ByteOrder,
    field_type: u16,
    count: usize,
) -> Result<(), DecodeError> {
    if field_type != 3 || count != 1 {
        return Err(malformed());
    }
    match order.read_short(entry.get(8..).ok_or_else(malformed)?)? {
        1 => Ok(()),
        0xffff => Err(unsupported()),
        _ => Err(malformed()),
    }
}

fn parse_density(context: DensityContext<'_>) -> Result<(), DecodeError> {
    let DensityContext {
        tiff,
        order,
        entry,
        field_type,
        count,
        tag,
        kind,
        metadata,
    } = context;
    if field_type != 5 || count != 1 {
        return Err(malformed());
    }
    let value = read_rational(tiff, order, entry)?;
    let density = metadata::density_for(metadata, kind)?;
    let target = if tag == 0x011a {
        &mut density.horizontal_resolution
    } else {
        &mut density.vertical_resolution
    };
    if target.replace(value).is_some() {
        return Err(malformed());
    }
    Ok(())
}

fn parse_resolution_unit(
    entry: &[u8],
    order: ByteOrder,
    field_type: u16,
    count: usize,
    kind: DirectoryKind,
    metadata: &mut PresentationMetadata,
) -> Result<(), DecodeError> {
    let density = metadata::density_for(metadata, kind)?;
    if field_type != 3 || count != 1 || density.resolution_unit.is_some() {
        return Err(malformed());
    }
    let unit = order.read_short(entry.get(8..).ok_or_else(malformed)?)?;
    if !matches!(unit, 1..=3) {
        return Err(malformed());
    }
    density.resolution_unit = Some(unit);
    Ok(())
}

fn parse_thumbnail(
    entry: &[u8],
    order: ByteOrder,
    field_type: u16,
    count: usize,
    tag: u16,
    metadata: &mut PresentationMetadata,
) -> Result<(), DecodeError> {
    if field_type != 4 || count != 1 {
        return Err(malformed());
    }
    let value = usize::try_from(order.read_long(entry.get(8..).ok_or_else(malformed)?)?)
        .map_err(|_| too_large())?;
    let target = if tag == 0x0201 {
        &mut metadata.thumbnail_offset
    } else {
        &mut metadata.thumbnail_length
    };
    if target.replace(value).is_some() {
        return Err(malformed());
    }
    Ok(())
}

fn parse_interoperability(entry: &[u8], field_type: u16, count: usize) -> Result<(), DecodeError> {
    if field_type != 2 || count != 4 {
        return Err(malformed());
    }
    if entry.get(8..12) != Some(b"R98\0") {
        return Err(unsupported());
    }
    Ok(())
}

fn read_rational(tiff: &[u8], order: ByteOrder, entry: &[u8]) -> Result<(u32, u32), DecodeError> {
    let offset = usize::try_from(order.read_long(entry.get(8..).ok_or_else(malformed)?)?)
        .map_err(|_| too_large())?;
    let end = offset.checked_add(8).ok_or_else(too_large)?;
    let bytes = tiff.get(offset..end).ok_or_else(malformed)?;
    let numerator = order.read_long(bytes)?;
    let denominator = order.read_long(bytes.get(4..).ok_or_else(malformed)?)?;
    if denominator == 0 {
        return Err(malformed());
    }
    Ok((numerator, denominator))
}

fn malformed() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Malformed)
}

fn unsupported() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Unsupported)
}

fn too_large() -> DecodeError {
    DecodeError::new(DecodeErrorKind::TooLarge)
}
