from pathlib import Path
import yaml
import os
import pandas as pd

from smart_data_hub.property2dataframe import combine_rock_site_property
from smart_data_hub.data_tagging import filter_tagged_data
from smart_data_hub.add_default import add_default_df
from smart_data_hub.merge_method import merge_property_value, generate_lognorm, generate_PERT, generate_truncnorm, generate_uniform, generate_norm
from smart_data_hub.dataframe2yaml import export2yaml

def default_merge_rock_unit():
    """The default method for merging site units.

    Returns:
        dict: default merged units
    """
    return {'DE_South_Claystone': {'Quaternary': ['Quaternary'],
            'Tertiary': ['Tertiary'],
            'Jurassic_Upper': ['Jurassic_Upper'],
            'Jurassic_Middle': ['Jurassic_Middle_sequence1',
            'Jurassic_Middle_sequence2'],
            'Host_rock': ['Opalinus_Clay'],
            'Jurassic_Lower': ['Jurassic_Lower_sequence1', 'Jurassic_Lower_sequence2'],
            'Keuper': ['Keuper_sequence1', 'Keuper_sequence2', 'Keuper_sequence3'],
            'Muschelkalk': ['Muschelkalk_Upper',
            'Muschelkalk_Middle',
            'Muschelkalk_Lower']},
            'DE_North_Claystone': {'Quaternary': ['Quaternary'],
            'Albian': ['Albian'],
            'Aptian': ['Aptian'],
            'Host_rock': ['Barremian', 'Hauterivian'],
            'Valanginian_Wealden': ['Valanginian', 'Wealden'],
            'Jurassic_Upper': ['Jurassic_Upper'],
            'Jurassic_Middle': ['Jurassic_Middle'],
            'Jurassic_Lower': ['Jurassic_Lower'],
            'Keuper': ['Keuper'],
            'Buntsandstein_Muschelkalk': ['Buntsandstein_Muschelkalk', 'Muschelkalk'],
            'Zechstein': ['Zechstein']},
            'DE_Rocksalt': {'Quaternary': ['Quaternary'],
            'Tertiary': ['Tertiary'],
            'Buntsandstein': ['Buntsandstein'],
            'Host_rock': ['Aller',
            'Leine_Anhydritmittelsalz',
            'Leine_Potash',
            'Leine_Rocksalt',
            'Leine_Hauptanhydrit',
            'Strassfurt_Potash',
            'Strassfurt_Rocksalt',
            'Stassfurt_Anhydrite_Carbonate'],
            'Rotliegend': ['Rotliegend']},
            'DE_Crystalline': {'Quaternary': ['Quaternary'],
            'Muschelkalk': ['Muschelkalk'],
            'Buntsandstein': ['Buntsandstein'],
            'Zechstein': ['Zechstein_Rocksalt',
            'Zechstein_Potash',
            'Zechstein_Anhydrite'],
            'Host_rock': ['Granite']}}

def default_depth_for_merged_rock_units():
    """Create a dictionary for default depth of merged units

    Returns:
        dict: default depth of merged units.
    """
    return {'DE_South_Claystone': {'Quaternary_top': 580.0,
            'Quaternary_bottom': 515.0,
            'Tertiary_bottom': 475.0,
            'Jurassic_Upper_bottom': 0.0,
            'Jurassic_Middle_bottom': -85.0,
            'Opalinus_Clay_bottom': -200.0,
            'Jurassic_Lower_bottom': -280.0,
            'Keuper_bottom': -450.0,
            'Muschelkalk_bottom': -575.0},
            'DE_North_Claystone': {'Quaternary_top': 0.0,
            'Quaternary_bottom': -50.0,
            'Albian_bottom': -400.0,
            'Aptian_bottom': -500.0,
            'Host_rock_bottom': -1050.0,
            'Valanginian_Wealden_bottom': -1200.0,
            'Jurassic_Upper_bottom': -1400.0,
            'Jurassic_Middle_bottom': -2300.0,
            'Jurassic_Lower_bottom': -2650.0,
            'Keuper_bottom': -3085.0,
            'Buntsandstein_Muschelkalk_bottom': -3500.0,
            'Zechstein_bottom': -4200.0},
            'DE_Rocksalt': {'Quaternary_top': 0.0,
            'Quaternary_bottom': -70.0,
            'Tertiary_bottom': -220.0,
            'Buntsandstein_bottom': -745.0,
            'Rocksalt_bottom': -1390.0,
            'Rotliegend_bottom': -1490.0},
            'DE_Crystalline': {'Quaternary_top': 20.0,
            'Quaternary_bottom': -10.0,
            'Muschelkalk_bottom': -150.0,
            'Buntsandstein_bottom': -450.0,
            'Zechstein_bottom': -886.0,
            'Granite_bottom': -1200.0}}

def export_site_merged_rock_units_yaml(
    site_name,
    merge_unit_rock_prop,
    tag_dict,
    sampling_functions_by_property,
    path_to_save_rock_yaml,
):
    """
    Export merged rock-unit property data for a site to YAML files.

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

    merge_unit_rock_prop
    df2 = df1.loc[df1['site'] == site_name]
    for merge_unit, rock_layers in merge_unit_rock_prop.items():
        # merge rock layers
        df3 = df2[df2["rock_layer"].isin(rock_layers)]
        # filter the data based on tags
        df4 = filter_tagged_data(df3, tag_dict)
        # add default data based on the litholgies for the merged rock unit
        lithologies = list(set(lith for liths_ls in df3['layer_simplified_lithology'] for lith in liths_ls))
        df5 = add_default_df(df4, lithologies)
        # merge the rock property data for the merged unit
        df6 = merge_property_value(df5, source_type="merged", sampling_functions_by_property = sampling_functions_by_property)
        if not os.path.exists(path_to_save_rock_yaml):
            os.makedirs(path_to_save_rock_yaml)
        export2yaml(df6, f"{path_to_save_rock_yaml}/{merge_unit}.yaml")


def export_site_merged_yaml(site_name, merge_unit_rock_prop, path_to_save_site_yaml):
    """
    Build and export a merged YAML file describing a geological site's layers.

    Args:
        site_name (str): Name of the site.
        merge_unit_rock_prop (dict[str, list[str]]): Mapping of merged unit to the rock layers that should be combined.
        path_to_save_site_yaml (str): Directory path where the resulting YAML file will be saved.
    """
    # get all rock properties for all sites
    df1 = combine_rock_site_property()

    sub = df1[df1['name'] == site_name]
    if sub.empty:
        sub = df1[df1['site'] == site_name]

    site_name = sub['name'].iloc[0] if not sub.empty else site_name
    site_description = sub['site_description'].iloc[0] if not sub.empty else ""

    yaml_dict = {"name": site_name, "description": site_description}

    for sub_key, layer_list in merge_unit_rock_prop.items():
        sources, lithologies = [], []
        for layer in layer_list:
            rows = sub[sub['rock_layer'] == layer]
            if rows.empty:
                continue
            src = rows['layer_source'].iloc[0]
            if pd.notna(src) and src not in sources:
                sources.append(src)
            for l in rows['layer_simplified_lithology'].iloc[0]:
                if l not in lithologies:
                    lithologies.append(l)

        yaml_dict[sub_key] = {
            "source": sources[0] if len(sources) == 1 else sources,
            "simplified_lithology": lithologies
        }
    if not os.path.exists(path_to_save_site_yaml):
            os.makedirs(path_to_save_site_yaml)

    with open(f'{path_to_save_site_yaml}/{site_name}.yaml', 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False, default_flow_style=None)


def export_site_depth_yaml(merge_unit_depth, path_to_save_site_geometry):
    """
    Export a YAML file describing the depths of a geological site's layers.

    Args:
        merge_unit_depth (dict[str, float]): Mapping of merged unit to the depth.
        path_to_save_site_geometry (str): Directory path where the resulting YAML file will be saved.
    """

    if not os.path.exists(path_to_save_site_geometry):
        os.makedirs(path_to_save_site_geometry)
    with open(os.path.join(path_to_save_site_geometry, "units.yaml"), "w") as f:
        yaml.safe_dump(merge_unit_depth, f, sort_keys=False)