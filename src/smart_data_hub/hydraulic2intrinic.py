import scipy.constants as sc
import numpy as np


def convert_value(type, convert_ratio, value):
    """Convert a value based on its type and a conversion ratio.

    Args:
        type (scalar): The type of the value.
        convert_ratio (float): The conversion ratio to apply.
        value (float, str, dict, None): The input value to convert.

    Returns:
        dict, str, float, None: The converted value.
    """

    if type == "dictionary":
        return {k: float(v) * convert_ratio for k, v in value.items()}
    elif type == "expression":
        return f"{convert_ratio}*({value})"
    else:
        try:
            return value * convert_ratio
        except TypeError:  # If value is None
            return value


def hydraulic2intrinic(input_df):
    """Convert hydraulic conductivity to intrinsic permeability.

    Args:
        input_df (pd.DataFrame): The input DataFrame containing hydraulic conductivity values.

    Returns:
        pd.DataFrame: The modified DataFrame with intrinsic permeability values.
    """

    water_vis = 0.001
    water_density = 1000
    gravitational_acceleration = sc.g

    convert_ratio = water_vis / (water_density * gravitational_acceleration)

    # Condition where property == 'hydraulic_conductivity'
    mask = input_df["property"] == "hydraulic_conductivity"
    # Convert value_min, value_max, and value_std using the conversion ratio
    input_df.loc[mask, ["value_min", "value_max", "value_std"]] = (
        input_df.loc[mask, ["value_min", "value_max", "value_std"]] * convert_ratio
    )
    # Convert value based on its type using the conversion ratio
    input_df.loc[mask, ["value"]] = input_df.loc[mask, ["value", "type"]].apply(
        lambda row: convert_value(row["type"], convert_ratio, row["value"]), axis=1
    )
    input_df.loc[mask, ["sampled_data"]] = input_df.loc[mask, ["sampled_data"]].map(
        lambda sampled_data: (
            sampled_data  # if null
            if sampled_data is None
            else (
                f"{convert_ratio}*({sampled_data})"
                if isinstance(sampled_data, str)
                else convert_ratio * sampled_data
            )
        )
    )

    # Change the property name, unit_str, and unit_base to be consistent with intrinsic permeability
    input_df.loc[mask, ["property"]] = "intrinsic_permeability"
    input_df.loc[mask, ["unit_str"]] = "m^2"
    input_df.loc[mask, ["unit_base"]] = "[0, 2, 0, 0, 0, 0, 0]"
    # Replace np.nan with None
    input_df = input_df.replace({np.nan: None})

    return input_df
