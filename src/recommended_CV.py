import scipy.stats as stats
import numpy as np
import pandas as pd
from src.property2dataframe import combine_rock_site_property
from src.hydraulic2intrinic import hydraulic2intrinic


def value_best_mask(input_pd_df):
    """Filter data in the input DataFrame that contains the best estimate value.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean series indicating which data contains the best estimate value.
    """

    is_best_mask = input_pd_df["value"].notna()

    return is_best_mask


def value_range_mask(input_pd_df):
    """Filter data in the input DataFrame that is given by a range.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean series indicating which data is given by a range.
    """
    is_range_mask = (
        input_pd_df["value"].isna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].notna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
    )

    return is_range_mask


def property_CV():
    """Calculate the coefficient of variation (CV) for each property.

    Returns:
        dict: A dictionary containing the CV for each property.
    """

    merged_df = combine_rock_site_property()

    # take only the scalar type values
    merged_df = merged_df.loc[merged_df["type"] == "scalar"]

    # convert hydraulic conductivity to intrinic permeability
    merged_df = hydraulic2intrinic(merged_df)

    is_best_mask = value_best_mask(merged_df)
    merged_df_best = merged_df[is_best_mask].copy()

    is_range_mask = value_range_mask(merged_df)
    merged_df_uniform = merged_df[is_range_mask].copy()
    merged_df_uniform["value"] = (
        merged_df_uniform["value_min"] + merged_df_uniform["value_max"]
    ) / 2.0

    # merge the best estimate and half range values
    merged_df = pd.concat([merged_df_best, merged_df_uniform], ignore_index=True)

    properties = list(set(merged_df["property"]))
    dict_property_cv = {}
    for property in properties:
        property_mask = merged_df["property"] == property
        property_df = merged_df[property_mask].copy()
        # take data within one standard deviation of the mean
        property_df = property_df[
            np.abs(stats.zscore(list(property_df["value"]))) < 1
        ].copy()

        combined_values = property_df["value"]

        mean_values = combined_values.mean()
        std_values = combined_values.std()

        dict_property_cv.update({property: std_values / mean_values})
    return dict_property_cv
