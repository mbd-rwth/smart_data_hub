import pandas as pd
import numpy as np

from src.property2dataframe import load_rock_property


def load_default_df(lithologies):
    """Load default rock property DataFrame for the given lithologies.

    Args:
        lithologies (list): list of lithologies.

    Returns:
        pd.DataFrame: Default rock property DataFrame.
    """
    # load default rock property from yaml files
    add_default_yaml_list = [
        f"database/rock_property/default/{lithology}.yaml" for lithology in lithologies
    ]
    default_df = load_rock_property(add_default_yaml_list)
    return default_df


def get_matching_default_df(lithologies, missing_prop_names):
    """Get the matching default DataFrame for the given lithologies and missing properties.

    Args:
        lithologies (list): list of lithologies to look for default properties
        missing_prop_names (list): list of missing properties to find defaults for

    Returns:
        pd.DataFrame: DataFrame containing the matching default properties
    """

    if missing_prop_names == []:
        return pd.DataFrame()

    # load default rock property from yaml files
    default_df = load_default_df(lithologies)
    # extract the rows with that matching with the missing properties
    matching_default_df = default_df[default_df["property"].isin(missing_prop_names)]

    return matching_default_df


def add_default_df(property_df, lithologies):
    """Add default properties to the property DataFrame.

    Args:
        property_df (pd.DataFrame): DataFrame containing the properties to check
        lithologies (list): list of lithologies to look for default properties

    Returns:
        pd.DataFrame: DataFrame with default properties added
    """

    if property_df.empty:
        return load_default_df(lithologies)
    no_id_props = list(property_df.loc[property_df["ID"].isna()]["property"])
    # drop the missing id properties from the original DataFrame
    removed_missing_id_property_df = property_df[
        ~property_df["property"].isin(no_id_props)
    ]

    # find the missing properties
    all_properties = list(removed_missing_id_property_df["property"].unique())
    if "hydraulic_conductivity" in all_properties:
        all_properties[all_properties.index("hydraulic_conductivity")] = (
            "intrinsic_permeability"
        )
    required_property_names = set(
        [
            "density",
            "porosity",
            "intrinsic_permeability",
            "p_wave_velocity",
            "s_wave_velocity",
            "specific_heat_capacity",
            "thermal_conductivity",
            "electrical_resistivity",
        ]
    )
    missing_props = list(required_property_names - set(all_properties))

    matching_default_df = get_matching_default_df(lithologies, missing_props)

    add_default_property_df = pd.concat(
        [
            df
            for df in [removed_missing_id_property_df, matching_default_df]
            if not df.empty
        ],
        ignore_index=True,
    )
    # Replace np.nan with None
    add_default_property_df = add_default_property_df.replace({np.nan: None, "": None})

    return add_default_property_df
