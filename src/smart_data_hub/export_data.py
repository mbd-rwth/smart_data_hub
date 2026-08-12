"""Script to read a site configuration YAML file.

Usage:
    python read_site_config.py --config path/to/site_config.yaml \
        --path_to_save_rock_yaml output/DE_South_Claystone/rock_data \
        --path_to_save_site_yaml output/DE_South_Claystone/site_data \
        --path_to_save_site_geometry output/DE_South_Claystone/geometry
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
import uuid
import scipy
from smart_data_hub.add_default import add_default_df
from smart_data_hub.data_tagging import filter_tagged_data
from smart_data_hub.dataframe2yaml import export2yaml
from smart_data_hub.merge_method import merge_property_value
from smart_data_hub.merge_method import generate_lognorm, generate_PERT, generate_truncnorm, generate_uniform
from smart_data_hub.merge_method import format_number_adaptive
from smart_data_hub.property2dataframe import combine_rock_site_property
from smart_data_hub.generate_id import sdh_namespace
from smart_data_hub.generate_id import get_entry_str
from smart_data_hub.dataframe2yaml import convert_to_flow_sequence

def default_merge_rock_unit():
    """The default method for merging site units.

    Returns:
        dict: default merged units
    """
    return {
        "DE_South_Claystone": {
            "Quaternary": ["Quaternary"],
            "Tertiary": ["Tertiary"],
            "Jurassic_Upper": ["Jurassic_Upper"],
            "Jurassic_Middle": ["Jurassic_Middle_sequence1", "Jurassic_Middle_sequence2"],
            "Host_rock": ["Opalinus_Clay"],
            "Jurassic_Lower": ["Jurassic_Lower_sequence1", "Jurassic_Lower_sequence2"],
            "Keuper": ["Keuper_sequence1", "Keuper_sequence2", "Keuper_sequence3"],
            "Muschelkalk": ["Muschelkalk_Upper", "Muschelkalk_Middle", "Muschelkalk_Lower"],
            "basement": ["Muschelkalk_Lower"]
        },
        "DE_North_Claystone": {
            "Quaternary": ["Quaternary"],
            "Albian": ["Albian"],
            "Aptian": ["Aptian"],
            "Host_rock": ["Barremian", "Hauterivian"],
            "Valanginian_Wealden": ["Valanginian", "Wealden"],
            "Jurassic_Upper": ["Jurassic_Upper"],
            "Jurassic_Middle": ["Jurassic_Middle"],
            "Jurassic_Lower": ["Jurassic_Lower"],
            "Keuper": ["Keuper"],
            "Buntsandstein_Muschelkalk": ["Buntsandstein_Muschelkalk", "Muschelkalk"],
            "Zechstein": ["Zechstein"],
            "basement": ["Zechstein"]
        },
        "DE_Rocksalt": {
            "Quaternary": ["Quaternary"],
            "Tertiary": ["Tertiary"],
            "Buntsandstein": ["Buntsandstein"],
            "Host_rock": [
                "Aller",
                "Leine_Anhydritmittelsalz",
                "Leine_Potash",
                "Leine_Rocksalt",
                "Leine_Hauptanhydrit",
                "Strassfurt_Potash",
                "Strassfurt_Rocksalt",
                "Stassfurt_Anhydrite_Carbonate",
            ],
            "Rotliegend": ["Rotliegend"],
            "basement": ["Rotliegend"]
        },
        "DE_Crystalline": {
            "Quaternary": ["Quaternary"],
            "Muschelkalk": ["Muschelkalk"],
            "Buntsandstein": ["Buntsandstein"],
            "Zechstein": ["Zechstein_Rocksalt", "Zechstein_Potash", "Zechstein_Anhydrite"],
            "Host_rock": ["Granite"],
            "basement": ["Granite"]
        },
    }


def default_depth_for_merged_rock_units():
    """Create a dictionary for default depth of merged units.

    Returns:
        dict: default depth of merged units.
    """
    return {
        "DE_South_Claystone": {'Quaternary_top': 
                               {'value': 580, 'value_min': 560, 'value_max': 570, 'sampling_function': 'generate_uniform'}, 
                               'Quaternary_bottom': 
                               {'value': 515, 'value_min': 505, 'value_max': 525, 'sampling_function': 'generate_uniform'}, 
                               'Tertiary_bottom': 
                               {'value': 475, 'value_min': 465, 'value_max': 485, 'sampling_function': 'generate_uniform'}, 
                               'Jurassic_Upper_bottom': 
                               {'value': 0.0, 'value_min': -10.0, 'value_max': 10.0, 'sampling_function': 'generate_uniform'}, 
                               'Jurassic_Middle_bottom': 
                               {'value': -85.0, 'value_min': -60.0, 'value_max': -110.0, 'sampling_function': 'generate_uniform'}, 
                               'Host_rock_bottom': 
                               {'value': -260.0, 'value_min': -285.0, 'value_max': -235.0, 'sampling_function': 'generate_uniform'}, 
                               'Jurassic_Lower_bottom': 
                               {'value': -400.0, 'value_min': -425.0, 'value_max': -375.0, 'sampling_function': 'generate_uniform'},  
                               'Keuper_bottom': 
                               {'value': -575.0, 'value_min': -600.0, 'value_max': -550.0, 'sampling_function': 'generate_uniform'},
                               'Muschelkalk_bottom': 
                               {'value': -800.0, 'value_min': -825.0, 'value_max': -775.0, 'sampling_function': 'generate_uniform'}, 
                               'basement_bottom': 
                               {'value': -1000.0, 'value_min': -1025.0, 'value_max': -975.0, 'sampling_function': 'generate_uniform'}},
        "DE_North_Claystone": {'Quaternary_top': 
                               {'value': 0.0, 'value_min': -10.0, 'value_max': 10.0, 'sampling_function': 'generate_uniform'}, 
                               'Quaternary_bottom': 
                               {'value': -50.0, 'value_min': -60.0, 'value_max': -40.0, 'sampling_function': 'generate_uniform'}, 
                               'Albian_bottom': 
                               {'value': -100.0, 'value_min': -110.0, 'value_max': -90.0, 'sampling_function': 'generate_uniform'}, 
                               'Aptian_bottom': 
                               {'value': -400.0, 'value_min': -425.0, 'value_max': -375.0, 'sampling_function': 'generate_uniform'}, 
                               'Host_rock_bottom': 
                               {'value': -575.0, 'value_min': -600.0, 'value_max': -550.0, 'sampling_function': 'generate_uniform'}, 
                               'Valanginian_Wealden_bottom': 
                               {'value': -650.0, 'value_min': -675.0, 'value_max': -625.0, 'sampling_function': 'generate_uniform'}, 
                               'Jurassic_Upper_bottom': 
                               {'value': -800.0, 'value_min': -825.0, 'value_max': -775.0, 'sampling_function': 'generate_uniform'}, 
                               'Jurassic_Middle_bottom': 
                               {'value': -900.0, 'value_min': -925.0, 'value_max': -875.0, 'sampling_function': 'generate_uniform'}, 
                               'Jurassic_Lower_bottom': 
                               {'value': -1000.0, 'value_min': -1025.0, 'value_max': -975.0, 'sampling_function': 'generate_uniform'}, 
                               'Keuper_bottom': 
                               {'value': -1100.0, 'value_min': -1125.0, 'value_max': -1100.0, 'sampling_function': 'generate_uniform'}, 
                               'Buntsandstein_Muschelkalk_bottom': 
                               {'value': -1500.0, 'value_min': -1525.0, 'value_max': -1475.0, 'sampling_function': 'generate_uniform'}, 
                               'Zechstein_bottom': 
                               {'value': -1600.0, 'value_min': -1625.0, 'value_max': -1575.0, 'sampling_function': 'generate_uniform'}, 
                               'basement_bottom': 
                               {'value': -1800.0, 'value_min': -1900.0, 'value_max': -1700.0, 'sampling_function': 'generate_uniform'}},
        "DE_Rocksalt": {'Quaternary_top': {'value': 0.0,
                                        'value_min': -10.0,
                                        'value_max': 10.0,
                                        'sampling_function': 'generate_uniform'},
                                        'Quaternary_bottom': {'value': -50.0,
                                        'value_min': -60.0,
                                        'value_max': -40.0,
                                        'sampling_function': 'generate_uniform'},
                                        'Tertiary_bottom': {'value': -220.0,
                                        'value_min': -245.0,
                                        'value_max': -195.0,
                                        'sampling_function': 'generate_uniform'},
                                        'Buntsandstein_bottom': {'value': -1215.0,
                                        'value_min': -1240.0,
                                        'value_max': -1190.0,
                                        'sampling_function': 'generate_uniform'},
                                        'Host_rock_bottom': {'value': -1390.0,
                                        'value_min': -1415.0,
                                        'value_max': -1365.0,
                                        'sampling_function': 'generate_uniform'},
                                        'Rotliegend_bottom': {'value': -1500.0,
                                        'value_min': -1525.0,
                                        'value_max': -1475.0,
                                        'sampling_function': 'generate_uniform'},
                                        'basement_bottom': {'value': -1800.0,
                                        'value_min': -1900.0,
                                        'value_max': -1700.0,
                                        'sampling_function': 'generate_uniform'}},
        "DE_Crystalline": {'Quaternary_top': {'value': 0.0,
                            'value_min': -10.0,
                            'value_max': 10.0,
                            'sampling_function': 'generate_uniform'},
                            'Quaternary_bottom': {'value': -50.0,
                            'value_min': -60.0,
                            'value_max': -40.0,
                            'sampling_function': 'generate_uniform'},
                            'Muschelkalk_bottom': {'value': -100.0,
                            'value_min': -110.0,
                            'value_max': -90.0,
                            'sampling_function': 'generate_uniform'},
                            'Buntsandstein_bottom': {'value': -400.0,
                            'value_min': -415.0,
                            'value_max': -385.0,
                            'sampling_function': 'generate_uniform'},
                            'Zechstein_bottom': {'value': -1000.0,
                            'value_min': -1025.0,
                            'value_max': -975.0,
                            'sampling_function': 'generate_uniform'},
                            'Host_rock_bottom': {'value': -1175.0,
                            'value_min': -1200.0,
                            'value_max': -1150.0,
                            'sampling_function': 'generate_uniformm'},
                            'basement_bottom': {'value': -1800.0,
                            'value_min': -1900.0,
                            'value_max': -1700.0,
                            'sampling_function': 'generate_uniform'}}
                                }


def export_site_merged_rock_units_yaml(
    site_name,
    merge_unit_rock_prop,
    tag_dict,
    sampling_functions_by_property,
    path_to_save_rock_yaml,
) -> None:
    """Export merged rock-unit property data for a site to YAML files.

    Args:
        site_name (str): Name of the site.
        merge_unit_rock_prop (dict[str, list[str]]): Mapping of merged unit to the rock layers that should be combined.
        tag_dict (dict[str, list[str]]): Tag filters applied to the site data.
        sampling_functions_by_property (dict[str, callable]): Mapping of property
            names to desired sampling functions.
        path_to_save_yaml (str): The path to save the output YAML files.
    """
    # get all rock properties for all sites
    df1 = combine_rock_site_property()
    # filter the data for the specific site

    df2 = df1.loc[df1["site"] == site_name]
    for merge_unit, rock_layers in merge_unit_rock_prop.items():
        # merge rock layers
        df3 = df2[df2["rock_layer"].isin(rock_layers)]
        df4 = df3.copy()
        if tag_dict is not None:
            # filter the data based on tags
            df4 = filter_tagged_data(df3, tag_dict)
        # add default data based on the litholgies for the merged rock unit
        lithologies = list({lith for liths_ls in df3["layer_simplified_lithology"] for lith in liths_ls})
        df5 = add_default_df(df4, lithologies)
        # merge the rock property data for the merged unit
        df6 = merge_property_value(
            df5, source_type="merged", sampling_functions_by_property=sampling_functions_by_property,
        )

        export2yaml(df6, f"{path_to_save_rock_yaml}/{merge_unit}.yaml")


def export_site_merged_yaml(site_name, merge_unit_rock_prop, path_to_save_site_yaml) -> None:
    """Build and export a merged YAML file describing a geological site's layers.

    Args:
        site_name (str): Name of the site.
        merge_unit_rock_prop (dict[str, list[str]]): Mapping of merged unit to the rock layers that should be combined.
        path_to_save_site_yaml (str): Directory path where the resulting YAML file will be saved.
    """
    # get all rock properties for all sites
    df1 = combine_rock_site_property()

    sub = df1[df1["name"] == site_name]
    if sub.empty:
        sub = df1[df1["site"] == site_name]

    site_name = sub["name"].iloc[0] if not sub.empty else site_name
    site_description = sub["site_description"].iloc[0] if not sub.empty else ""

    yaml_dict = {"name": site_name, "description": site_description}

    for sub_key, layer_list in merge_unit_rock_prop.items():
        sources, lithologies = [], []
        for layer in layer_list:
            rows = sub[sub["rock_layer"] == layer]
            if rows.empty:
                continue
            src = rows["layer_source"].iloc[0]
            if pd.notna(src) and src not in sources:
                sources.append(src)
            for l in rows["layer_simplified_lithology"].iloc[0]:
                if l not in lithologies:
                    lithologies.append(l)

        yaml_dict[sub_key] = {
            "source": sources[0] if len(sources) == 1 else sources,
            "simplified_lithology": lithologies,
        }

    with open(f"{path_to_save_site_yaml}/{site_name}.yaml", "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False, default_flow_style=None)


def export_site_depth_yaml(merge_unit_depth, path_to_save_site_geometry) -> None:
    """Export a YAML file describing the depths of a geological site's layers.

    Args:
        merge_unit_depth (dict[str, float]): Mapping of merged unit to the depth.
        path_to_save_site_geometry (str): Directory path where the resulting YAML file will be saved.
    """
   
    all_sampled_depth_date = {}
    for rock_interface, depth_data in merge_unit_depth.items():
        sampling_function = eval(depth_data['sampling_function'])
        sampling_dist = sampling_function(as_string=True, **depth_data)
        samples_string = sampling_dist + '.rvs(size=1000000, random_state=21)'
        samples = eval(samples_string)

        sampled_depth_date = {}
        sampled_depth_date['source'] = 'merged'
        sampled_depth_date['type'] = 'scalar'
        sampled_depth_date['value'] = format_number_adaptive(np.mean(samples))
        sampled_depth_date['value_min'] = format_number_adaptive(np.min(samples))
        sampled_depth_date['value_max'] = format_number_adaptive(np.max(samples))
        sampled_depth_date['value_std'] = format_number_adaptive(np.std(samples))
        sampled_depth_date['probability_distribution'] = {}
        sampled_depth_date['probability_distribution']['sample_size'] = int(1000000)
        sampled_depth_date['probability_distribution']['sampled_data'] = samples_string
        sampled_depth_date['unit_str'] = 'm'
        sampled_depth_date['unit_base'] = convert_to_flow_sequence([0, 1, 0, 0, 0, 0, 0])
        sampled_depth_date['description'] = f"A uniform distribution fitted from the given data values."
        sampled_depth_date['tag'] = {'ID': None}
        
        SDH_NAMESPACE = sdh_namespace()
        sampled_depth_date['tag']["ID"] = str(uuid.uuid5(SDH_NAMESPACE, get_entry_str(sampled_depth_date, rock_interface)))

        all_sampled_depth_date[rock_interface] = [sampled_depth_date]

    with open(os.path.join(path_to_save_site_geometry, "rock_interface.yaml"), "w") as f:
        yaml.safe_dump(all_sampled_depth_date, f, sort_keys=False)


REQUIRED_FIELDS = [
    "site_name",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Read a site configuration YAML file.",
    )

    parser.add_argument("--config", type=str, required=True, help="Path to the site configuration YAML file.")
    parser.add_argument(
        "--path_to_save_rock_yaml",
        type=str,
        required=True,
        help="Output directory for the merged rock property YAML files.",
    )
    parser.add_argument(
        "--path_to_save_site_yaml",
        type=str,
        required=True,
        help="Output directory for the generated site YAML file (a summary of rock unit lithologies).",
    )
    parser.add_argument(
        "--path_to_save_site_geometry",
        type=str,
        required=True,
        help="Output directory for the generated site geometry file.",
    )

    return parser.parse_args()


def load_site_yaml_config(config_path):
    """Load a YAML configuration file.

    Args:
        config_path (str): Path to the input YAML configuration file

    Raises:
        FileNotFoundError: not found message.
        ValueError: empty config file message.

    Returns:
        dict: configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        msg = f"Config file is empty: {config_path}"
        raise ValueError(msg)

    return config


def validate_config(config) -> None:
    """Validate that all required fields are present in the config.

    Args:
        config (dict): configuration dictionary

    Raises:
        ValueError: missing required field message.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        msg = f"Missing required config field(s): {missing}"
        raise ValueError(msg)


def build_site_config(config_path, path_to_save_rock_yaml, path_to_save_site_yaml, path_to_save_site_geometry):
    """Load and validate, a site configuration file. Save rock, site, geometry data with output paths given via CLI.

    Args:
        config_path (str): path to the configuration file.
        path_to_save_rock_yaml (str): output directory for the merged rock property YAML files.
        path_to_save_site_yaml (str): output directory for the generated site YAML file.
        path_to_save_site_geometry (str): output directory for the generated site geometry file.

    Returns:
        dict: configuration dictionary with output paths given via CLI.
    """
    raw_config = load_site_yaml_config(config_path)
    validate_config(raw_config)

    site_name = raw_config["site_name"]

    sampling_functions_by_property_dict = raw_config.get("sampling_functions_by_property")

    if sampling_functions_by_property_dict is not None:
        sampling_functions_by_property = {prop: eval(func) for prop, func in raw_config["sampling_functions_by_property"].items()}
    else:
        sampling_functions_by_property = None

    return {
        "site_name": raw_config["site_name"],
        "tag_dict": raw_config.get("tag_dict"),
        "sampling_functions_by_property": sampling_functions_by_property,
        "merge_unit_rock_prop": raw_config.get("merge_unit_rock_prop", default_merge_rock_unit()[site_name]),
        "merge_unit_depth": raw_config.get("merge_unit_depth", default_depth_for_merged_rock_units()[site_name]),
        "path_to_save_rock_yaml": path_to_save_rock_yaml,
        "path_to_save_site_yaml": path_to_save_site_yaml,
        "path_to_save_site_geometry": path_to_save_site_geometry,
    }


def main() -> None:
    args = parse_args()
    try:
        site_config = build_site_config(
            args.config,
            args.path_to_save_rock_yaml,
            args.path_to_save_site_yaml,
            args.path_to_save_site_geometry,
        )
    except (FileNotFoundError, ValueError):
        sys.exit(1)

    for path_key in ["path_to_save_rock_yaml", "path_to_save_site_yaml", "path_to_save_site_geometry"]:
        Path(site_config[path_key]).mkdir(parents=True, exist_ok=True)

    site_name = site_config["site_name"]
    tag_dict = site_config["tag_dict"]
    sampling_functions_by_property = site_config["sampling_functions_by_property"]
    merge_unit_rock_prop = site_config["merge_unit_rock_prop"]
    merge_unit_depth = site_config["merge_unit_depth"]

    path_to_save_rock_yaml = site_config["path_to_save_rock_yaml"]
    path_to_save_site_yaml = site_config["path_to_save_site_yaml"]
    path_to_save_site_geometry = site_config["path_to_save_site_geometry"]

    export_site_merged_rock_units_yaml(
        site_name, merge_unit_rock_prop, tag_dict, sampling_functions_by_property, path_to_save_rock_yaml,
    )
    export_site_merged_yaml(site_name, merge_unit_rock_prop, path_to_save_site_yaml)
    export_site_depth_yaml(merge_unit_depth, path_to_save_site_geometry)


if __name__ == "__main__":
    main()
