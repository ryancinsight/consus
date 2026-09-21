use super::bitstream::{Tables, parse_tables};
use crate::exif::Orientation;
use crate::{DecodeError, DecodeErrorKind, DecodeLimits, DecodedImage, PixelFormat};

const SOI: [u8; 2] = [0xff, 0xd8];
const UNSEEN: u8 = u8::MAX;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Coding {
    Sequential,
    Progressive,
    Lossless,
}

struct Component {
    id: u8,
    horizontal: u8,
    vertical: u8,
    quantization: usize,
    blocks_across: usize,
    blocks_down: usize,
    stored_across: usize,
    stored_down: usize,
    coefficient_offset: usize,
    approximation: [u8; 64],
}

struct Frame {
    coding: Coding,
    width: usize,
    height: usize,
    precision: u8,
    max_horizontal: u8,
    max_vertical: u8,
    mcu_across: usize,
    mcu_down: usize,
    components: Vec<Component>,
    coefficients: Vec<i32>,
}

#[derive(Clone, Copy)]
struct ScanComponent {
    frame_index: usize,
    dc_table: usize,
    ac_table: usize,
}

struct Scan {
    components: Vec<ScanComponent>,
    start: u8,
    end: u8,
    high: u8,
    low: u8,
}

pub(super) fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<DecodedImage, DecodeError> {
    if limits.max_encoded_bytes == 0
        || limits.max_dimension == 0
        || limits.max_pixels == 0
        || limits.max_working_bytes == 0
        || bytes.len() > limits.max_encoded_bytes
    {
        return Err(too_large());
    }
    if !bytes.starts_with(&SOI) {
        return Err(malformed());
    }

    let mut cursor = 2_usize;
    let mut frame = None;
    let mut tables = Tables::new();
    let mut quantization = [None; 4];
    let mut restart_interval = 0_usize;
    let mut scans = 0_usize;
    let mut orientation = Orientation::Normal;
    let mut saw_exif = false;
    let mut saw_jfif = false;
    let mut adobe_transform = None;

    loop {
        let (marker, after_marker) = read_marker(bytes, cursor)?;
        cursor = after_marker;
        match marker {
            0xd9 => {
                if cursor != bytes.len() || scans == 0 {
                    return Err(malformed());
                }
                let frame = frame.ok_or_else(malformed)?;
                validate_color_model(&frame, adobe_transform)?;
                return output::finish(frame, &quantization, adobe_transform, orientation, limits);
            }
            0xc0..=0xc3 => {
                if frame.is_some() || scans != 0 {
                    return Err(malformed());
                }
                frame = Some(frame::parse_frame(
                    segment(bytes, &mut cursor)?,
                    marker,
                    limits,
                )?);
            }
            0xc4 => parse_tables(segment(bytes, &mut cursor)?, &mut tables)?,
            0xdb => frame::parse_quantization(segment(bytes, &mut cursor)?, &mut quantization)?,
            0xdd => {
                let payload = segment(bytes, &mut cursor)?;
                if payload.len() != 2 {
                    return Err(malformed());
                }
                restart_interval = usize::from(read_u16(payload)?);
            }
            0xda => {
                let payload = segment(bytes, &mut cursor)?;
                let current = frame.as_mut().ok_or_else(malformed)?;
                let scan = frame::parse_scan(payload, current, &tables)?;
                cursor =
                    entropy::decode_scan(bytes, cursor, current, &tables, &scan, restart_interval)?;
                output::record_progression(current, &scan)?;
                scans = scans.checked_add(1).ok_or_else(too_large)?;
            }
            0xe0 => {
                if saw_jfif {
                    return Err(malformed());
                }
                validate_jfif(segment(bytes, &mut cursor)?)?;
                saw_jfif = true;
            }
            0xe1 => {
                if saw_exif {
                    return Err(malformed());
                }
                let payload = segment(bytes, &mut cursor)?;
                let tiff = payload.strip_prefix(b"Exif\0\0").ok_or_else(unsupported)?;
                orientation = crate::exif::parse(tiff)?;
                saw_exif = true;
            }
            0xee => {
                if adobe_transform.is_some() {
                    return Err(malformed());
                }
                adobe_transform = Some(validate_adobe(segment(bytes, &mut cursor)?)?);
            }
            0xfe => {
                let _ = segment(bytes, &mut cursor)?;
            }
            0xc5..=0xcf if !matches!(marker, 0xc8 | 0xcc) => {
                let _ = segment(bytes, &mut cursor)?;
                return Err(unsupported());
            }
            0xc8 | 0xcc | 0xdc | 0xde | 0xdf | 0xe2..=0xed | 0xef => {
                let _ = segment(bytes, &mut cursor)?;
                return Err(unsupported());
            }
            0xd8 | 0x01 | 0xd0..=0xd7 => return Err(malformed()),
            _ => return Err(malformed()),
        }
    }
}

mod entropy;
mod frame;
mod lossless;
mod output;

fn validate_color_model(frame: &Frame, adobe_transform: Option<u8>) -> Result<(), DecodeError> {
    match (frame.components.len(), adobe_transform) {
        (1, None) | (3, None | Some(0 | 1)) | (4, Some(0 | 2)) => Ok(()),
        _ => Err(unsupported()),
    }
}

fn validate_jfif(payload: &[u8]) -> Result<(), DecodeError> {
    if !payload.starts_with(b"JFIF\0") || payload.len() < 14 {
        return Err(unsupported());
    }
    let horizontal = read_u16(payload.get(8..).ok_or_else(malformed)?)?;
    let vertical = read_u16(payload.get(10..).ok_or_else(malformed)?)?;
    if horizontal == 0 || vertical == 0 || horizontal != vertical {
        return Err(unsupported());
    }
    let thumbnail = usize::from(payload[12])
        .checked_mul(usize::from(payload[13]))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(too_large)?;
    if payload.len() != 14_usize.checked_add(thumbnail).ok_or_else(too_large)? {
        return Err(malformed());
    }
    Ok(())
}

fn validate_adobe(payload: &[u8]) -> Result<u8, DecodeError> {
    if payload.len() != 12 || !payload.starts_with(b"Adobe") {
        return Err(malformed());
    }
    let transform = payload[11];
    if transform > 2 {
        return Err(unsupported());
    }
    Ok(transform)
}

fn read_marker(bytes: &[u8], mut cursor: usize) -> Result<(u8, usize), DecodeError> {
    if bytes.get(cursor) != Some(&0xff) {
        return Err(malformed());
    }
    while bytes.get(cursor) == Some(&0xff) {
        cursor = cursor.checked_add(1).ok_or_else(too_large)?;
    }
    let marker = *bytes.get(cursor).ok_or_else(malformed)?;
    if marker == 0 {
        return Err(malformed());
    }
    Ok((marker, cursor.checked_add(1).ok_or_else(too_large)?))
}

fn segment<'input>(bytes: &'input [u8], cursor: &mut usize) -> Result<&'input [u8], DecodeError> {
    let length = usize::from(read_u16(bytes.get(*cursor..).ok_or_else(malformed)?)?);
    if length < 2 {
        return Err(malformed());
    }
    let end = cursor.checked_add(length).ok_or_else(too_large)?;
    let start = cursor.checked_add(2).ok_or_else(too_large)?;
    let payload = bytes.get(start..end).ok_or_else(malformed)?;
    *cursor = end;
    Ok(payload)
}

fn read_u16(bytes: &[u8]) -> Result<u16, DecodeError> {
    let value: [u8; 2] = bytes
        .get(..2)
        .ok_or_else(malformed)?
        .try_into()
        .map_err(|_| malformed())?;
    Ok(u16::from_be_bytes(value))
}

const fn malformed() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Malformed)
}

const fn unsupported() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Unsupported)
}

const fn too_large() -> DecodeError {
    DecodeError::new(DecodeErrorKind::TooLarge)
}

const fn allocation() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Allocation)
}
\n