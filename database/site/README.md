## Design of site YAML files  
This folder summaries geological information on siting regions, with one YAML file dedicated to each region.
An example of the YAML file entries is provided below:

```
name: DE_Crystalline
description: A generic geological model for final repository system in crystalline host rock in Germany.
geometry_input:
  source: weitkamp2021
  description: The geological structural data is mainly based on the drilling profile in Figure 26.
  point_data_path: database/geometry/DE_Crystalline/input/points.csv
  orientation_data_path: database/geometry/DE_Crystalline/input/orientations.csv
Quaternary:
  source: weitkamp2021
  geological_time:
    eon: Phanerozoic
    era: Cenozoic
    period: Quaternary
    epoch: null
    age: null
  description: It consists of unconsolidated fluvial sediments, primarily sands with gravel and alluvial loam.
  simplified_lithology: [Sandstone, Conglomerate]
  property_path: database/rock_property/DE_Crystalline/Quaternary.yaml
  geometry_surface_output_path: [database/geometry/DE_Crystalline/output/Quaternary.vtk]
```
The entries in this YAML file consist of four main parts: 
1. name: site's name and/or region's name. 
2. description: additional description of the site. 
3. geometry_input: information on input structural data, including data source, description, CSV file paths for point and orientation data for creating a [GemPy](https://www.gempy.org) model.
4. stratigraphy layers: Each stratigraphic layer is described through its geological time, lithological description, simplified lithology, and the path to the rock properties and geometry. The geological time for each rock formation is based on the [International Commission on Stratigraphy](https://stratigraphy.org/chart).

## Search Approach for Default Values
The missing values are filled based on the `simplified_lithology` entries in each stratigraphical layer. For instance, in the provided example, the simplified lithology of `Quaternary` are referenced as `[Sandstone, Conglomerate]`. In this case, the missing values are retrieved from the file `../rock_property/default/Sandstone.yaml` and `../rock_property/default/Conglomerate.yaml`.


