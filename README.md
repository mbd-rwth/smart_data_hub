# Smart Data Hub

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17939847.svg)](https://doi.org/10.5281/zenodo.17939847)


Welcome to the smart data hub repository, developed by the [Methods for Model-based Development in Computational Engineering](https://www.mbd.rwth-aachen.de/) (MBD) and 
[Geophysical Imaging and Monitoring](https://www.gim.rwth-aachen.de/) (GIM) groups at RWTH Aachen University. The smart 
data hub is a product of the '[Smart-Monitoring](https://www.mbd.rwth-aachen.de/go/id/sxklc?lidx=1#aaaaaaaaaasxkoh)' project, 
which aims to provide innovative solutions in data-integrated simulation studies. The Smart Data Hub bridges existing research limitations to enable efficient data compilation for specific simulations while incorporating uncertainty quantification. It is capable of providing reliable, reproducible output for data-driven decision-making processes.

## Data Hub Architecture
The data-hub consists of a database integrated with a Graphic User Interface (GUI).
1. **Database**: It provides material properties along with their uncertainty margins and sensible defaults 
in YAML files. All relevant files can be found in the [`database`](./database/README.md) directory.
2. **GUI**: It was developed with [Plotly Dash](https://dash.plotly.com/)—a web-based application for interactive visualization. Simply run the Python program [SmartGUI.py](./SmartGUI.py), it will then start a local flask server.
The GUI displays three sections:
   * Geomodel: provides a 3D structural geomodel for each site.
   * Chronostratigraphic chart & Table: The chart indicates geological formation time of each stratum. The table provides information on rock properties.
   * Processing functions: enables customized scenarios by filtering data in the table, provides uncertainty distributions for each value via the "Merge data" button, and adds sensible defaults when data is missing via the "Add Default" button.  All relevant codes can be found in the [`src`](./src)directory.
  
The following screenshot shows the GUI for the Buntsandstein layer at the DE_Rocksalt site. The highlighted, non-grey layer on the left-hand side is the currently selected layer, and the data table on the right-hand side shows the data after merging and adding default values.
![](SDH_GUI.png)

## Installation (For GUI)
1. Download the zip file or clone the repository:
2. Create a conda environment using ``environment.yml`` and running the following command ``conda env create -f environment.yml``, 
3. Activate the environment with ``conda activate smart_data_hub``.

## Credits
The authors of this project are [@CQ-QianChen](https://github.com/CQ-QianChen), [@ninomenzel1](https://github.com/ninomenzel1) and 
[@mboxberg](https://github.com/mboxberg).

## License
`smart_data_hub` is released under the MIT License. See [LICENSE](LICENSE) file for details.
