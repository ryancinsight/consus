use consus_hdf5::file::Hdf5File;
use consus_io::SliceReader;

#[test]
fn test_unidata_reference_netcdf4_4_0() {
    let path = "tests/data/ref_nc_test_netcdf4_4_0.nc";
    let bytes = std::fs::read(path).expect("Failed to read file");
    
    // Test that the file opens successfully using the netCDF-4 facade
    let source = SliceReader::new(&bytes);
    let file = Hdf5File::open(source).expect("Failed to parse NetCDF-4 file as HDF5");

    // Retrieve the extracted model
    let model = consus_netcdf::read_model(&file).expect("should map valid netcdf model");
    
    // println!("{:#?}", model);

    let root = &model.root;
    let dims = &root.dimensions;
    
    assert!(!dims.is_empty(), "NetCDF-4 reference file should have dimensions");

    let vars = &root.variables;
    assert!(!vars.is_empty(), "NetCDF-4 reference file should have variables");

    // The ref_nc_test_netcdf4_4_0.nc file typically has a variable named "var" or similar.
    // Let's assert that at least one variable resolves correctly.
    for var in vars {
        assert!(var.rank() == var.dimensions.len(), "Variable rank should match dimension list");
        
        if var.name == "lat" || var.name == "lon" || var.name == "time" {
            assert!(var.coordinate_variable, "Dimension variable should be correctly identified as coordinate variable");
        }
    }

    if let Some(var) = vars.first() {
        // The data mapping for HDF5 should be valid.
        let hdf5_mapping = consus_netcdf::model::bridge::map_variable(var).expect("Valid HDF5 mapping");
        assert_eq!(hdf5_mapping.name, var.name);
    }
}

