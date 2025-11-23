import numpy as np
import pandas as pd
import uuid
import scipy
import scipy.stats as stats
from scipy.stats import truncnorm, lognorm, beta, uniform
from src.property2dataframe import preserve_value_type
from src.hydraulic2intrinic import hydraulic2intrinic
from src.generate_id import sdh_namespace, get_entry_str


# --- Create masks based on different inputs ---#
def value_empty_mask(input_pd_df):
    """Filter data in the input DataFrame that is empty.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean series indicating which data is empty.
    """

    is_empty_mask = (
        input_pd_df["value"].isna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["value_min"].isna()
        & input_pd_df["value_max"].isna()
        & input_pd_df["sampled_data"].isna()
    )

    return is_empty_mask


def value_invalid_mask(input_pd_df):
    """Filter some data in the input DataFrame that contains invalid entries.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean Series indicating which data is invalid.
    """

    # The value is not provided, but the standard deviation is given.
    invalid_mask_1 = (
        input_pd_df["value"].isna()
        & input_pd_df["value_std"].notna()
        & input_pd_df["sampled_data"].isna()
    )
    # Only the minimum value is provided
    invalid_mask_2 = (
        input_pd_df["value"].isna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].isna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
    )
    # Only the maximum value is provided
    invalid_mask_3 = (
        input_pd_df["value"].isna()
        & input_pd_df["value_min"].isna()
        & input_pd_df["value_max"].notna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
    )
    is_invalid_mask = invalid_mask_1 | invalid_mask_2 | invalid_mask_3
    return is_invalid_mask


def value_pdf_mask(input_pd_df):
    """Filter some data in the input DataFrame given by a probability distribution.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean series indicating which data entries have a corresponding probability distribution.
    """
    is_pdf_mask = input_pd_df["sampled_data"].notna() & (
        (input_pd_df["type"] == "scalar")
    )

    return is_pdf_mask


def value_uniform_mask(input_pd_df):
    """Filter some data in the input DataFrame given by a range.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean series indicating which data is uniform.
    """

    is_uniform_mask = (
        input_pd_df["value"].isna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].notna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
        & ((input_pd_df["type"] == "scalar"))
    )

    return is_uniform_mask


def value_truncnorm_mask(input_pd_df):
    """Filter some data in the input DataFrame that can be assumed with a truncated normal distribution.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean series indicating which data is truncated normal.
    """
    is_truncnorm_mask = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].notna()
        & input_pd_df["value_std"].notna()
        & input_pd_df["sampled_data"].isna()
        & ((input_pd_df["type"] == "scalar"))
    )

    is_truncnorm_mask_porosity = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].isna()
        & input_pd_df["value_std"].notna()
        & input_pd_df["sampled_data"].isna()
        & (input_pd_df["property"] == "porosity")
        & ((input_pd_df["type"] == "scalar"))
    )

    return is_truncnorm_mask | is_truncnorm_mask_porosity


def value_lognorm_mask(input_pd_df):
    """Filter some data in the input DataFrame that can be assumed with a log-normal distribution.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean series indicating which data is log-normal.
    """

    is_lognorm_mask1 = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].isna()
        & input_pd_df["value_max"].isna()
        & input_pd_df["value_std"].notna()
        & input_pd_df["sampled_data"].isna()
        & ((input_pd_df["type"] == "scalar"))
    )

    is_lognorm_mask2 = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].isna()
        & input_pd_df["value_max"].notna()
        & input_pd_df["value_std"].notna()
        & input_pd_df["sampled_data"].isna()
        & ((input_pd_df["type"] == "scalar"))
    )

    is_lognorm_mask3 = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].isna()
        & input_pd_df["value_std"].notna()
        & input_pd_df["sampled_data"].isna()
        & (input_pd_df["property"] != "porosity")
        & ((input_pd_df["type"] == "scalar"))
    )

    is_lognorm_mask4 = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].isna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
        & (input_pd_df["property"] != "porosity")
        & ((input_pd_df["type"] == "scalar"))
    )

    is_lognorm_mask5 = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].isna()
        & input_pd_df["value_max"].notna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
        & ((input_pd_df["type"] == "scalar"))
    )

    is_lognorm_mask_single = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].isna()
        & input_pd_df["value_max"].isna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
        & ((input_pd_df["type"] == "scalar"))
    )

    return (
        is_lognorm_mask1
        | is_lognorm_mask2
        | is_lognorm_mask3
        | is_lognorm_mask4
        | is_lognorm_mask5
        | is_lognorm_mask_single
    )


def value_PERT_mask(input_pd_df):
    """Filter some data in the input DataFrame that can be assumed with a PERT distribution.

    Args:
        input_pd_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Boolean series indicating which data is PERT.
    """

    is_PERT_mask = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].notna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
        & ((input_pd_df["type"] == "scalar"))
    )

    is_PERT_mask_porosity = (
        input_pd_df["value"].notna()
        & input_pd_df["value_min"].notna()
        & input_pd_df["value_max"].isna()
        & input_pd_df["value_std"].isna()
        & input_pd_df["sampled_data"].isna()
        & (input_pd_df["property"] == "porosity")
        & ((input_pd_df["type"] == "scalar"))
    )

    return is_PERT_mask | is_PERT_mask_porosity


# --- Generate different distributions based on the input parameters---#
def generate_truncnorm(value, value_std, value_min, value_max):
    """Generate a truncated normal distribution based on the given parameters.

    Args:
        value (float): The mean of the original dataset.
        value_std (float): The standard deviation of the original dataset.
        value_min (float): The minimum value of the original dataset.
        value_max (float): The maximum value of the original dataset.

    Returns:
        scipy.stats._distn_infrastructure.rv_continuous_frozen: A truncated normal distribution object.
    """
    a = (value_min - value) / value_std
    b = (value_max - value) / value_std
    return truncnorm(a, b, loc=value, scale=value_std)


def generate_PERT(value, value_min, value_max):
    """Generate a PERT distribution based on the given parameters.

    Args:
        value (float): The mean of the original dataset.
        value_min (float): The minimum value of the original dataset.
        value_max (float): The maximum value of the original dataset.

    Returns:
        scipy.stats._distn_infrastructure.rv_continuous_frozen: A beta distribution object.
    """
    # PERT distribution parameters calculated based on the book Risk Analysis: A Quantitative Guide by David Vose

    value_mu = 4
    value_a = 1 + value_mu * ((value - value_min) / (value_max - value_min))
    value_b = 1 + value_mu * ((value_max - value) / (value_max - value_min))

    return beta(value_a, value_b, loc=value_min, scale=value_max - value_min)


def generate_lognorm(value, value_std, value_min):
    """Generate a log-normal distribution based on the given parameters.

    Args:
        value (float): The mean of the original dataset.
        value_std (float): The standard deviation of the original dataset.
        value_min (float): Shift the distribution to start from the minimum value.

    Returns:
        scipy.stats._distn_infrastructure.rv_continuous_frozen: A log-normal distribution object.
    """
    if value_min is None:
        value_min = 0.0
    adjusted_mean = value - value_min
    # lognorm parameters calculated based on  https://doi.org/10.5281/ZENODO.4305949
    log_std = np.sqrt(np.log(1 + (value_std / adjusted_mean) ** 2))
    log_mean = np.log(adjusted_mean) - 0.5 * log_std**2

    return lognorm(s=log_std, scale=np.exp(log_mean), loc=value_min)


def generate_uniform(value_min, value_max):
    """Generate a uniform distribution based on the given parameters.

    Args:
        value_min (float): The minimum value of the original dataset.
        value_max (float): The maximum value of the original dataset.

    Returns:
        scipy.stats._distn_infrastructure.rv_continuous_frozen: A uniform distribution object.
    """
    return uniform(loc=value_min, scale=value_max - value_min)


# Get the coefficient of variation (CV) for each property. Note: generated from recommended_CV.py
dict_property_cv = {
    "intrinsic_permeability": 4.18875,
    "p_wave_velocity": 0.13319,
    "specific_heat_capacity": 0.08823,
    "electrical_resistivity": 1.61120,
    "density": 0.07223,
    "thermal_conductivity": 0.23402,
    "porosity": 0.27030,
    "s_wave_velocity": 0.13030,
}


# --- Generate samples from different distributions ---#
def generate_samples(
    input_df_group,
    df_pdf_type,
    random_state=21,
):
    """Generate samples for a group of input data according to its specified distribution type, including uniform, truncated normal, log-normal, and PERT.

    Args:
        input_df_group (pd.DataFrame): The input DataFrame containing the data for one specific property and one type of pdf.
        df_pdf_type (str): The type of probability distribution function to use. It can be "is_pdf_df", "is_uniform_df", "is_truncnorm_df", "is_lognorm_df", or "is_PERT_df".
        random_state (int, optional): The random state for reproducibility. Defaults to 21.

    Raises:
        ValueError: If the distribution type is not recognized.

    Returns:
        np.ndarray: The generated samples falls within three standard deviations of the mean.
    """
    samples_combined = []
    # Get the property name
    property_name = input_df_group["property"].iloc[0]
    for (
        sample_size,
        sampled_data,
        value,
        value_std,
        value_min,
        value_max,
    ) in zip(
        input_df_group["sample_size"],
        input_df_group["sampled_data"],
        input_df_group["value"],
        input_df_group["value_std"],
        input_df_group["value_min"],
        input_df_group["value_max"],
    ):
        if sample_size is None or np.isnan(sample_size):
            sample_size = 1000000
        if df_pdf_type == "is_pdf_df":
            # if the scipy.stats.rvs or numpy array is provided
            if isinstance(sampled_data, str):
                samples = eval(sampled_data)
            else:
                samples = np.array(sampled_data)
        elif df_pdf_type == "is_uniform_df":
            uniform_samples = generate_uniform(value_min, value_max).rvs(
                size=sample_size, random_state=random_state
            )
            samples = uniform_samples
        elif df_pdf_type == "is_truncnorm_df":

            if value_max is None:  # if the property is porosity
                value_max = 1.0

            truncnorm_samples = generate_truncnorm(
                value, value_std, value_min, value_max
            ).rvs(size=sample_size, random_state=random_state)
            samples = truncnorm_samples
        elif df_pdf_type == "is_lognorm_df":
            if value_std is None:
                value_std = value * dict_property_cv[property_name]

            lognorm_samples = generate_lognorm(value, value_std, value_min).rvs(
                size=sample_size, random_state=random_state
            )
            samples = lognorm_samples
        elif df_pdf_type == "is_PERT_df":

            if value_max is None:  # if the property is porosity
                value_max = 1.0

            PERT_samples = generate_PERT(value, value_min, value_max).rvs(
                size=sample_size, random_state=random_state
            )
            samples = PERT_samples
        else:
            raise ValueError(f"The df_pdf_type {df_pdf_type} is not recognized!")

        # Keep porosity samples with value <= 1
        if property_name == "porosity":
            samples = samples[samples <= 1]

        # Filter samples within three standard deviation of the mean
        samples_zscore = stats.zscore(samples)
        samples_filtered = samples[np.abs(samples_zscore) < 3]
        samples_combined.append(samples_filtered)
    return np.concatenate(samples_combined)


# Present numbers with adaptive precision
def format_number_adaptive(number):
    """Format the number adaptively: If the number is smaller than 0.0001, use scientific notation and keep two significant digits.
    If the number is between 0.001 and 0.1, use three significant digits. Otherwise, round to two decimal places.

    Args:
        number (float, None, np.nan): The number to format.

    Returns:
        float: The formatted number or None.
    """
    if number is None or np.isnan(number):
        return None
    # Keep two significant digits in scientific notation
    if abs(number) < 0.0001:
        return float(f"{number:.2e}")
    # Keep three significant digits
    elif abs(number) < 0.1:
        return float(f"{number:.3g}")
    elif abs(number) < 100000:
        # Round to two decimal places
        return float(f"{number:.2f}")
    else:
        return float(f"{number:.2e}")


def get_sample_statistics(input_property_group):
    """This function concatenates samples from different distributions for a group of input data and computes statistical properties (mean, std, min, max).

    Args:
        input_property_group (pd.DataFrame): DataFrame containing values for one specific property.

    Returns:
        tuple: A tuple containing the mean, standard deviation, minimum value, maximum value of the combined samples.
               The tuple is in the following order:
               (samples_mean, samples_std, samples_min, samples_max)
    """

    # Generate samples for each distribution type
    sample_combined = []

    # Get samples from the existing pdf
    is_pdf_mask = value_pdf_mask(input_property_group)
    is_pdf_df = input_property_group[is_pdf_mask].copy()
    if not is_pdf_df.empty:
        sample_combined.append(generate_samples(is_pdf_df, "is_pdf_df"))

    # Generate uniform samples
    is_uniform_mask = value_uniform_mask(input_property_group)
    is_uniform_df = input_property_group[is_uniform_mask].copy()
    if not is_uniform_df.empty:
        sample_combined.append(generate_samples(is_uniform_df, "is_uniform_df"))

    # Generate truncated normal samples
    is_truncnorm_mask = value_truncnorm_mask(input_property_group)
    is_truncnorm_df = input_property_group[is_truncnorm_mask].copy()
    if not is_truncnorm_df.empty:
        sample_combined.append(generate_samples(is_truncnorm_df, "is_truncnorm_df"))

    # Generate log-normal samples
    is_lognorm_mask = value_lognorm_mask(input_property_group)
    is_lognorm_df = input_property_group[is_lognorm_mask].copy()
    if not is_lognorm_df.empty:
        sample_combined.append(generate_samples(is_lognorm_df, "is_lognorm_df"))

    # Generate PERT samples
    is_PERT_mask = value_PERT_mask(input_property_group)
    is_PERT_df = input_property_group[is_PERT_mask].copy()
    if not is_PERT_df.empty:
        sample_combined.append(generate_samples(is_PERT_df, "is_PERT_df"))

    samples_all = np.concatenate(sample_combined)
    # compute statistical properties
    samples_mean = np.mean(samples_all)
    samples_std = np.std(samples_all)
    samples_min = np.min(samples_all)
    samples_max = np.max(samples_all)

    return (
        format_number_adaptive(samples_mean),
        format_number_adaptive(samples_std),
        format_number_adaptive(samples_min),
        format_number_adaptive(samples_max),
    )


# --- Merge property values from multiple datasets into a single DataFrame with combined statistics ---#


def generate_unit(property_name):
    """Generate a unit string and base for a given property name.

    Args:
        property_name (str): The name of the property for which to generate the unit.

    Raises:
        TypeError: If the property name is not recognized.

    Returns:
        str: The unit string for the property.
        list: The unit base for the property.
    """
    if property_name == "density":
        unit_str = "kg/m^3"
        unit_base = [1, -3, 0, 0, 0, 0, 0]
    elif property_name == "porosity":
        unit_str = None
        unit_base = [0, 0, 0, 0, 0, 0, 0]
    elif property_name == "specific_heat_capacity":
        unit_str = "J/kg/K"
        unit_base = [0, 2, -2, -1, 0, 0, 0]
    elif property_name == "thermal_conductivity":
        unit_str = "W/m/K"
        unit_base = [1, 1, -3, -1, 0, 0, 0]
    elif property_name == "p_wave_velocity" or property_name == "s_wave_velocity":
        unit_str = "m/s"
        unit_base = [0, 1, -1, 0, 0, 0, 0]
    elif property_name == "intrinsic_permeability":
        unit_str = "m^2"
        unit_base = [0, 2, 0, 0, 0, 0, 0]
    elif property_name == "electrical_resistivity":
        unit_str = "ohmm"
        unit_base = [1, 3, -3, 0, -2, 0, 0]
    else:
        raise TypeError(f"The property name {property_name} is not recognized!")

    return unit_str, unit_base


def merge_property_value(input_property, sample_size=1000000, source_type="merged"):
    """Merge property values from multiple datasets into a single DataFrame with combined statistics. The merged results include summary statistics and truncated normal distribution parameters for each property, along with metadata such as description and combined IDs.

    Args:
        input_property (pd.DataFrame): DataFrame containing property values.
        sample_size (int): The number of samples to generate for each merged dataset. Default is 1000000.
        source_type (str): "merged" or "default". Default is "merged".

    Returns:
        pd.DataFrame: DataFrame containing merged property values, summary statistics, distribution parameters, and metadata for each property.
    """

    # Replace np.nan with None
    input_property = input_property.replace({np.nan: None, "": None})
    # Preserve float for scalar and string for expression and dictionary
    input_property[["value", "value_min", "value_max", "value_std"]] = input_property[
        ["value", "value_min", "value_max", "value_std"]
    ].map(preserve_value_type)

    # Filter out empty and unreasonable values
    is_empty_mask = value_empty_mask(input_property)
    is_unreasonable_mask = value_invalid_mask(input_property)
    input_property = input_property[~(is_empty_mask | is_unreasonable_mask)].copy()

    # Convert hydraulic conductivity to intrinic permeability
    input_property = hydraulic2intrinic(input_property).copy()
    # Filter out scalar type and non-scalar type data
    input_property_scalar = input_property.loc[input_property["type"] == "scalar"]
    input_property_nonscalar = input_property.loc[input_property["type"] != "scalar"]
    # Create an empty DataFrame with selected columns
    selected_columns = [
        "property",
        "source",
        "type",
        "value",
        "value_min",
        "value_max",
        "value_std",
        "sample_size",
        "sampled_data",
        "unit_str",
        "unit_base",
        "variable_name",
        "variable_unit_str",
        "variable_unit_base",
        "description",
        "agency",
        "location",
        "simplified_lithology",
        "ID",
    ]
    merged_property = pd.DataFrame(columns=selected_columns)
    # Group the input property by 'property'
    input_property_grouped = input_property_scalar.groupby(["property"])
    # Get SDH namespace for generating unique IDs
    SDH_NAMESPACE = sdh_namespace()
    # Merge property values for each group
    for property_group_keys, input_property_group in input_property_grouped:
        is_pdf_mask = value_pdf_mask(input_property_group)
        is_pdf_df = input_property_group[is_pdf_mask].copy()
        # If only one dataset is provided by a probability distribution, then the merge process is unnecessary
        if (input_property_group.shape[0] == 1) and (not is_pdf_df.empty):

            merged_property = pd.concat(
                [
                    property_df
                    for property_df in [merged_property, input_property_group]
                    if not property_df.empty
                ],
                ignore_index=True,
            )
        else:

            # Get statistics from the combined samples
            (
                samples_mean,
                samples_std,
                samples_min,
                samples_max,
            ) = get_sample_statistics(input_property_group)
            # Assign the parameters for the truncated normal distribution
            truncnorm_a = format_number_adaptive(
                (samples_min - samples_mean) / samples_std
            )
            truncnorm_b = format_number_adaptive(
                (samples_max - samples_mean) / samples_std
            )
            truncnorm_rvs = f"scipy.stats.truncnorm({truncnorm_a}, {truncnorm_b}, loc={samples_mean}, scale={samples_std}).rvs(size={sample_size}, random_state=21)"
            # Get meta information
            ids_list = list(input_property_group["ID"])
            ids_combined = ",".join(ids_list)
            if len(ids_list) > 1:
                print_before_id = "combined datasets with ids:"
            else:
                print_before_id = "dataset with id:"

            number_of_datasets = input_property_group.shape[0]
            property_name = property_group_keys[0]
            property_dict = {
                "property": property_name,
                "type": "scalar",
                "source": source_type,
                "value": samples_mean,
                "value_min": samples_min,
                "value_max": samples_max,
                "value_std": samples_std,
                "sample_size": sample_size,
                "sampled_data": truncnorm_rvs,
                "unit_str": generate_unit(property_name)[0],
                "unit_base": generate_unit(property_name)[1],
                "variable_name": None,
                "variable_unit_str": None,
                "variable_unit_base": None,
                "description": None,
                "agency": None,
                "location": None,
                "simplified_lithology": None,
                "ID": None,
            }
            if (source_type == "default") or (source_type == "merged"):
                property_dict["ID"] = str(
                    uuid.uuid5(
                        SDH_NAMESPACE, get_entry_str(property_dict, property_name)
                    )
                )
                property_dict["description"] = (
                    f"A truncated normal distribution fitted from {number_of_datasets} {print_before_id} {ids_combined}."
                )
            else:
                raise ValueError(
                    f"The allowed source_type entries are 'default' and 'merged'. The source_type '{source_type}' is not recognized!"
                )

            # Create a new row with the merged property values
            input_property_group_merged = pd.DataFrame([property_dict])

            # Append the merged data of each property group to the merged DataFrame
            merged_property = pd.concat(
                [
                    property_df
                    for property_df in [merged_property, input_property_group_merged]
                    if not property_df.empty
                ],
                ignore_index=True,
            )

    if (not input_property_nonscalar.empty) and (source_type == "merged"):

        merged_property = pd.concat(
            [
                property_df
                for property_df in [
                    merged_property,
                    input_property_nonscalar[selected_columns],
                ]
                if not property_df.empty
            ],
            ignore_index=True,
        )
    # Replace np.nan with None
    merged_property = merged_property.replace({np.nan: None, "": None})
    # Drop columns where all values are None
    merged_property = merged_property.dropna(axis=1, how="all")

    return merged_property
