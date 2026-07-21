[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17939847.svg)](https://doi.org/10.5281/zenodo.17939847) [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/mbd-rwth/smart_data_hub) [![github license badge](https://img.shields.io/github/license/mbd-rwth/smart_data_hub)](https://github.com/mbd-rwth/smart_data_hub) [![Documentation Status](https://readthedocs.org/projects/smart_data_hub/badge/?version=latest)](https://smart_data_hub.readthedocs.io/en/latest/?badge=latest) [![build](https://github.com/mbd-rwth/smart_data_hub/actions/workflows/build.yml/badge.svg)](https://github.com/mbd-rwth/smart_data_hub/actions/workflows/build.yml) [![cffconvert](https://github.com/mbd-rwth/smart_data_hub/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/mbd-rwth/smart_data_hub/actions/workflows/cffconvert.yml) [![link-check](https://github.com/mbd-rwth/smart_data_hub/actions/workflows/link-check.yml/badge.svg)](https://github.com/mbd-rwth/smart_data_hub/actions/workflows/link-check.yml)

# Smart Data Hub


Welcome to the smart data hub repository, developed by the [Methods for Model-based Development in Computational Engineering](https://www.mbd.rwth-aachen.de/) (MBD) and 
[Geophysical Imaging and Monitoring](https://www.gim.rwth-aachen.de/) (GIM) groups at RWTH Aachen University. The smart 
data hub is a product of the '[Smart-Monitoring](https://www.mbd.rwth-aachen.de/go/id/sxklc?lidx=1#aaaaaaaaaasxkoh)' project, 
which aims to provide innovative solutions in data-integrated simulation studies. The Smart Data Hub bridges existing research limitations to enable efficient data compilation for specific simulations while incorporating uncertainty quantification. It is capable of providing reliable, reproducible output for data-driven decision-making processes.

## Installation
To install smart_data_hub from GitHub repository, do:

```console
git clone git@github.com:CQ-QianChen/smart_data_hub.git
cd smart_data_hub
python -m pip install .
```

OR

1. Download the zip file or clone the repository:
2. Create a conda environment using ``environment.yml`` and running the following command ``conda env create -f environment.yml``, 
3. Activate the environment with ``conda activate smart_data_hub``.


## Data Hub Architecture
The data-hub consists of a dataset integrated with a Graphic User Interface (GUI).
1. **Dataset**: It provides material properties along with their uncertainty margins and sensible defaults 
in YAML files. All relevant files can be found in the [`dataset`](./src/smart_data_hub/dataset/README.md) directory. We provide the static dataset in [Zenodo](https://doi.org/10.5281/zenodo.19886769).
### Exporting Site Data

Use `export_data.py` to extract data for a specific site and model configuration. The script reads a YAML configuration file that specifies which site to extract, sampling behavior, units to merge, and geometry.

#### Site Configuration File Format 

| Field | Description                                                                                                                                                                                                                                                                                              |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `site_name` | Identifier for the candidate site (e.g., `DE_South_Claystone`).                                                                                                                                                                                                                                          |
| `tag_dict` | Dictionary of tags used to filter the source dataset. Keys are tag types (e.g., `location`, `agency`, `simplified_lithology`); values are lists of tag names to match.                                                                                                                                   |
| `sampling_functions_by_property` | Maps a property name to the probability distribution function used when sampling/merging its values. Supported functions: `generate_lognorm`, `generate_PERT`, `generate_truncnorm`, `generate_uniform`, `generate_norm`. If a property is not listed here, a default sampling function is used instead. |

##### Optional fields

| Field | Description | Default behavior if omitted |
|---|---|---|
| `merge_unit_rock_prop` | Groups multiple detailed rock unit/sequence names into a single merged rock unit. Keys are the merged unit name; values are lists of the original rock unit names to combine under that name. | If no merging is provided — apply the provided merged rock units |
| `merge_unit_depth` | Defines the top/bottom depth (in meters) for each merged rock unit boundary, used to build the site's stratigraphic geometry. | If no depth is provided - apply the provided depths for merged units |


**Note:** `merge_unit_rock_prop` and `merge_unit_depth` are typically used together — `merge_unit_rock_prop` defines *which* rock units to combine, while `merge_unit_depth` defines *where* (at what depth) each merged unit's boundaries lie.

##### Output path

In addition, users must provide output paths for different data types.

| Path | Description                                                                                                                                                                                                                                                                 |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `path_to_save_rock_yaml`     | Output directory for the merged rock property YAML files.                                                                                                                                                                                                                   | 
| `path_to_save_site_yaml`     | Output directory for the generated site YAML file.                                                                                                                                                                                                                          | 
| `path_to_save_site_geometry` | Output directory for the generated site geometry file.                                                                                                                                                                                                                      | 

#### Usage:
```
python export_data.py --config path/to/site_config.yaml \
    --path_to_save_rock_yaml output/path/to/rock_data \
    --path_to_save_site_yaml output/path/to/site_data \
    --path_to_save_site_geometry output/path/to/geometry
```

#### Example
```yaml
site_name: DE_South_Claystone

site_scenario_name: DE_South_Claystone_Germany  # used as the output folder name for saved YAML files

tag_dict:
  location:
    - Germany

sampling_functions_by_property:  # supports: generate_lognorm, generate_PERT, generate_truncnorm, generate_uniform, generate_norm
  electrical_resistivity: generate_lognorm
  intrinsic_permeability: generate_lognorm


merge_unit_rock_prop:
  Quaternary: [Quaternary]
  Tertiary: [Tertiary]
  Jurassic_Upper: [Jurassic_Upper]
  Jurassic_Middle: [Jurassic_Middle_sequence1, Jurassic_Middle_sequence2]
  Host_rock: [Opalinus_Clay]
  Jurassic_Lower: [Jurassic_Lower_sequence1, Jurassic_Lower_sequence2]
  Keuper: [Keuper_sequence1, Keuper_sequence2, Keuper_sequence3]
  Muschelkalk: [Muschelkalk_Upper, Muschelkalk_Middle, Muschelkalk_Lower]

merge_unit_depth:
  Quaternary_top: 580.0
  Quaternary_bottom: 515.0
  Tertiary_bottom: 475.0
  Jurassic_Upper_bottom: 0.0
  Jurassic_Middle_bottom: -85.0
  Opalinus_Clay_bottom: -200.0
  Jurassic_Lower_bottom: -280.0
  Keuper_bottom: -450.0
  Muschelkalk_bottom: -575.0
```

##### Running the script

```
python -m smart_data_hub.export_data \
--config ./communication_sdh/input/config_DE_South_Claystone_Germany.yaml \
--path_to_save_rock_yaml output/DE_South_Claystone_Germany/rock_data \
--path_to_save_site_yaml output/DE_South_Claystone_Germany/site_data \
--path_to_save_site_geometry output/DE_South_Claystone_Germany/geometry
```

2. **GUI**: It was developed with [Plotly Dash](https://dash.plotly.com/) —a web-based application for interactive visualization. 
Simply run the following script:

```bash
python -m smart_data_hub.SmartGUI
```

It will then start a local flask server.
The GUI displays three sections:
   * Geomodel: provides a 3D structural geomodel for each site.
   * Chronostratigraphic chart & table: The chart indicates geological formation time of each stratum. The table provides information on rock properties.
   * Processing functions: enables customized scenarios by filtering data in the table, provides uncertainty distributions for each value via the "Merge data" button, and adds sensible defaults when data is missing via the "Add Default" button.  All relevant code can be found in the [`src`](./src)directory.
  
The following screenshot shows the GUI for the Buntsandstein layer at the DE_Rocksalt site. The highlighted, non-grey layer on the left-hand side is the currently selected layer, and the data table on the right-hand side shows the data after merging and adding default values.
![](SDH_GUI.png)

## Credits
The authors of this project are [@CQ-QianChen](https://github.com/CQ-QianChen), [@ninomenzel1](https://github.com/ninomenzel1) and 
[@mboxberg](https://github.com/mboxberg).

## License
`smart_data_hub` is released under the GNU General Public License v3.0. See [LICENSE](LICENSE) file for details.
