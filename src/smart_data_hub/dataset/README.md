## Description:

We recorded two types of data: 

1. Rock properties: includes density, porosity, hydraulic conductivity, intrinsic permeability, specific heat capacity, thermal conductivity, P-wave velocity, S-wave velocity, and electrical resistivity, accompanied by their uncertainty margins for crystalline, clay, and salt host rock areas. All the rock properties are stored in YAML files.

2. Geostructural data: includes interface and orientation points for individual lithological rock formations for each host rock area, and is stored in CSV files. An additional 3D structural model for each lithological rock formation is provided as a VTK file.

## Contents:

* #### [`site`](./site/README.md)
We include a summary of site information, including lithological information on its rock formations, file paths to detailed rock properties and geometric data.

* #### [`rock_property`](./rock_property/README.md)
This folder provides rock properties, including density, porosity, hydraulic conductivity, intrinsic permeability, specific heat capacity, thermal conductivity, P-wave velocity, S-wave velocity, and electrical resistivity for each lithological rock formation. It also contains default values for data integrity. The rock types in question for consideration of default values consist of the following: Conglomerate, Mudstone, Sandstone, Limestone, Dolomite, Rocksalt, and Crystalline. 

* #### [`geometry`](./geometry/README.md)
This folder holds geostructural data in CSV files. Additionally, we provide a 3D structural model for each lithological rock formation as a VTK file. 

* #### source.bib
The file source.bib contains BibTeX entries for all sources that are mentioned in the dataset. 