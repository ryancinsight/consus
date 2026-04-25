//! MAT v7.3 (HDF5-backed) reader.
#[cfg(feature = "v73")]
pub use imp::read_mat_v73;
#[cfg(feature = "v73")]
mod imp {
    use crate::error::MatError;
    use crate::model::{
        MatArray, MatCellArray, MatCharArray, MatLogicalArray, MatNumericArray, MatNumericClass,
        MatStructArray,
    };
    use alloc::{format, string::String, vec, vec::Vec};
    use consus_core::{AttributeValue, ByteOrder, Datatype, NodeType};
    use consus_hdf5::dataset::StorageLayout;
    use consus_hdf5::file::Hdf5File;
    use consus_io::{ReadAt, SliceReader};
    pub fn read_mat_v73(data: &[u8]) -> Result<Vec<(String, MatArray)>, MatError> {
        let file = Hdf5File::open(SliceReader::new(data)).map_err(MatError::from)?;
        collect_variables(&file)
    }
    fn collect_variables<R: ReadAt + Sync>(
        file: &Hdf5File<R>,
    ) -> Result<Vec<(String, MatArray)>, MatError> {
        let root = file.list_root_group().map_err(MatError::from)?;
        let mut vars = Vec::new();
        for entry in root {
            let (name, addr, _) = entry;
            if name.starts_with("#") || name == "refs" {
                continue;
            }
            let array = load_variable(file, addr)?;
            vars.push((name, array));
        }
        Ok(vars)
    }
    fn matlab_class_attr(attrs: &[consus_hdf5::attribute::Hdf5Attribute]) -> Option<String> {
        attrs
            .iter()
            .find(|a| a.name == "MATLAB_class")
            .and_then(|a| a.decode_value().ok())
            .and_then(|v| match v {
                AttributeValue::String(s) => Some(s),
                _ => None,
            })
    }
    /// Parse the `MATLAB_dims` attribute from a struct group's attributes.
    ///
    /// `MATLAB_dims` is a uint32 or uint64 1-D array encoding the MATLAB array
    /// dimensions in MATLAB (column-major) order. For a scalar struct the
    /// attribute is absent; callers fall back to `[1, 1]`.
    fn matlab_dims_attr(
        attrs: &[consus_hdf5::attribute::Hdf5Attribute],
    ) -> Option<Vec<usize>> {
        attrs
            .iter()
            .find(|a| a.name == "MATLAB_dims")
            .and_then(|a| a.decode_value().ok())
            .and_then(|v| match v {
                AttributeValue::UintArray(dims) => {
                    let result: Vec<usize> =
                        dims.into_iter().map(|d| d as usize).collect();
                    if result.is_empty() { None } else { Some(result) }
                }
                AttributeValue::Uint(d) => Some(vec![d as usize]),
                _ => None,
            })
    }
    /// Split a field's multi-element numeric array into per-struct-element scalars.
    ///
    /// For a non-scalar struct array with `numel` elements, each field HDF5 dataset
    /// contains `numel` values packed contiguously. This function splits a
    /// `MatArray::Numeric` with `numel` elements into `numel` 1-element arrays,
    /// one per struct array position.
    ///
    /// Non-numeric arrays (char, logical, cell, nested struct) are not split;
    /// they are returned as a single-element `Vec` unchanged.
    fn split_field_elements(arr: MatArray, numel: usize) -> Vec<MatArray> {
        if numel <= 1 {
            return vec![arr];
        }
        match arr {
            MatArray::Numeric(na) if na.numel() == numel && !na.real_data.is_empty() => {
                let elem_bytes = na.real_data.len() / numel;
                let class = na.class;
                let real_data = na.real_data;
                let imag_data = na.imag_data;
                (0..numel)
                    .map(|i| {
                        let real =
                            real_data[i * elem_bytes..(i + 1) * elem_bytes].to_vec();
                        let imag = imag_data.as_ref().map(|id| {
                            id[i * elem_bytes..(i + 1) * elem_bytes].to_vec()
                        });
                        MatArray::Numeric(MatNumericArray {
                            class,
                            shape: vec![1],
                            real_data: real,
                            imag_data: imag,
                        })
                    })
                    .collect()
            }
            other => vec![other],
        }
    }
    fn load_variable<R: ReadAt + Sync>(
        file: &Hdf5File<R>,
        addr: u64,
    ) -> Result<MatArray, MatError> {
        let attrs = file.attributes_at(addr).map_err(MatError::from)?;
        let class = matlab_class_attr(&attrs).unwrap_or_else(|| String::from("double"));
        match file.node_type_at(addr).map_err(MatError::from)? {
            NodeType::Group => load_group(file, addr, &class, &attrs),
            NodeType::Dataset | NodeType::NamedDatatype => load_dataset(file, addr, &class),
        }
    }
    fn raw_bytes<R: ReadAt + Sync>(file: &Hdf5File<R>, addr: u64) -> Result<Vec<u8>, MatError> {
        let ds = file.dataset_at(addr).map_err(MatError::from)?;
        match ds.layout {
            StorageLayout::Contiguous => {
                let n = ds.shape.num_elements() * ds.datatype.element_size().unwrap_or(0);
                let da = ds.data_address.ok_or_else(|| {
                    MatError::InvalidFormat(String::from(
                        "v7.3: contiguous dataset has no data address",
                    ))
                })?;
                let mut buf = vec![0u8; n];
                file.read_contiguous_dataset_bytes(da, 0, &mut buf)
                    .map_err(MatError::from)?;
                Ok(buf)
            }
            StorageLayout::Chunked => file
                .read_chunked_dataset_all_bytes(addr)
                .map_err(MatError::from),
            StorageLayout::Compact => Err(MatError::UnsupportedFeature(String::from(
                "v7.3 compact layout",
            ))),
            StorageLayout::Virtual => Err(MatError::UnsupportedFeature(String::from(
                "v7.3 virtual layout",
            ))),
        }
    }
    fn to_numeric(s: &str) -> Option<MatNumericClass> {
        match s {
            "double" => Some(MatNumericClass::Double),
            "single" => Some(MatNumericClass::Single),
            "int8" => Some(MatNumericClass::Int8),
            "int16" => Some(MatNumericClass::Int16),
            "int32" => Some(MatNumericClass::Int32),
            "int64" => Some(MatNumericClass::Int64),
            "uint8" => Some(MatNumericClass::Uint8),
            "uint16" => Some(MatNumericClass::Uint16),
            "uint32" => Some(MatNumericClass::Uint32),
            "uint64" => Some(MatNumericClass::Uint64),
            _ => None,
        }
    }
    fn decode_char_data(raw: &[u8], datatype: &Datatype) -> Result<String, MatError> {
        match datatype {
            Datatype::Integer {
                bits,
                byte_order,
                signed: false,
            } if bits.get() == 16 => {
                let mut chars = String::new();
                for chunk in raw.chunks_exact(2) {
                    let code_unit = match byte_order {
                        ByteOrder::LittleEndian => u16::from_le_bytes([chunk[0], chunk[1]]),
                        ByteOrder::BigEndian => u16::from_be_bytes([chunk[0], chunk[1]]),
                    };
                    chars.push(
                        char::from_u32(code_unit as u32).unwrap_or(char::REPLACEMENT_CHARACTER),
                    );
                }
                if raw.len() % 2 != 0 {
                    return Err(MatError::InvalidFormat(String::from(
                        "v7.3 char dataset byte length is not a multiple of 2",
                    )));
                }
                Ok(chars)
            }
            Datatype::FixedString { .. } | Datatype::VariableString { .. } => {
                String::from_utf8(raw.to_vec()).map_err(|_| {
                    MatError::InvalidFormat(String::from(
                        "v7.3 char dataset contains invalid UTF-8 string data",
                    ))
                })
            }
            other => Err(MatError::UnsupportedFeature(format!(
                "v7.3 char dataset datatype is unsupported: {other:?}"
            ))),
        }
    }
    fn load_dataset<R: ReadAt + Sync>(
        file: &Hdf5File<R>,
        addr: u64,
        class: &str,
    ) -> Result<MatArray, MatError> {
        let ds = file.dataset_at(addr).map_err(MatError::from)?;
        let shape: Vec<usize> = ds.shape.current_dims().into_iter().rev().collect();
        match class {
            "logical" => {
                let raw = raw_bytes(file, addr)?;
                let data: Vec<bool> = raw.iter().map(|&b| b != 0).collect();
                Ok(MatArray::Logical(MatLogicalArray::new(shape, data)?))
            }
            "char" => {
                let raw = raw_bytes(file, addr)?;
                let s = decode_char_data(&raw, &ds.datatype)?;
                Ok(MatArray::Char(MatCharArray::new(shape, s)?))
            }
            "sparse" => Err(MatError::UnsupportedFeature(String::from(
                "v7.3 sparse datasets are not supported",
            ))),
            nc_str if to_numeric(nc_str).is_some() => {
                let nc = to_numeric(nc_str).unwrap();
                let (real_data, imag_data) = match &ds.datatype {
                    Datatype::Compound { fields, size } => {
                        let rf = fields.iter().find(|f| f.name == "real").map(|f| {
                            (
                                f.offset,
                                f.datatype.element_size().unwrap_or(nc.element_size()),
                            )
                        });
                        let imf = fields.iter().find(|f| f.name == "imag").map(|f| {
                            (
                                f.offset,
                                f.datatype.element_size().unwrap_or(nc.element_size()),
                            )
                        });
                        if let (Some((ro, rs)), Some((io, is))) = (rf, imf) {
                            let raw = raw_bytes(file, addr)?;
                            let sz = *size;
                            let mut rd = Vec::new();
                            let mut id = Vec::new();
                            for e in raw.chunks_exact(sz) {
                                rd.extend_from_slice(&e[ro..ro + rs]);
                                id.extend_from_slice(&e[io..io + is]);
                            }
                            (rd, Some(id))
                        } else {
                            (raw_bytes(file, addr)?, None)
                        }
                    }
                    _ => (raw_bytes(file, addr)?, None),
                };
                Ok(MatArray::Numeric(MatNumericArray {
                    class: nc,
                    shape,
                    real_data,
                    imag_data,
                }))
            }
            other => Err(MatError::UnsupportedFeature(format!(
                "v7.3 unsupported class: {other}"
            ))),
        }
    }
    fn load_group<R: ReadAt + Sync>(
        file: &Hdf5File<R>,
        addr: u64,
        class: &str,
        attrs: &[consus_hdf5::attribute::Hdf5Attribute],
    ) -> Result<MatArray, MatError> {
        let children = file.list_group_at(addr).map_err(MatError::from)?;
        if class == "struct" {
            // Preserve struct array shape from the MATLAB_dims attribute.
            // For a non-scalar struct array, MATLAB_dims encodes [M, N, ...].
            // Absent attribute means scalar struct; fall back to [1, 1].
            let struct_shape = matlab_dims_attr(attrs).unwrap_or_else(|| vec![1, 1]);
            let numel: usize = struct_shape.iter().product::<usize>().max(1);
            let mut data = Vec::new();
            for child in &children {
                let (nm, fa, _) = child;
                if nm.starts_with("#") {
                    continue;
                }
                let el = load_variable(file, *fa)?;
                // For non-scalar struct arrays each field dataset contains numel
                // packed values; split into one MatArray per struct element.
                let elements = split_field_elements(el, numel);
                data.push((nm.clone(), elements));
            }
            Ok(MatArray::Struct(MatStructArray::new(struct_shape, data)?))
        } else {
            // MATLAB v7.3 cell groups use decimal string names ("0", "1", ...).
            // Sort children numerically so element order is deterministic.
            let mut sorted: Vec<_> = children
                .iter()
                .filter(|child| {
                    let (nm, _, _) = *child;
                    !nm.starts_with('#')
                })
                .collect();
            sorted.sort_by_key(|child| {
                let (nm, _, _) = *child;
                nm.parse::<usize>().unwrap_or(usize::MAX)
            });
            let mut cells = Vec::with_capacity(sorted.len());
            for child in &sorted {
                let (_, ca, _) = **child;
                cells.push(load_variable(file, ca)?);
            }
            let n = cells.len();
            Ok(MatArray::Cell(MatCellArray::new(vec![1, n], cells)?))
        }
    }
}
