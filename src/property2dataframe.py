import os
import yaml
import pandas as pd
import numpy as np

from src.load_path import get_path_in_dir


def preserve_value_type(value):
    """Preserve the original type of the value.

    Args:
        value (float, str, None): The input value to preserve.

    Returns:
        float, str, None: The preserved value.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def load_rock_property(yaml_rock_property_paths):
    """Load rock properties from YAML files to a DataFrame.

    Args:
        yaml_rock_property_paths (list): a list of paths to YAML files containing rock properties.

    Returns:
        pd.DataFrame: a DataFrame containing the loaded rock properties.
    """
    # load property files to DataFrame
    records = []
    for filepath in yaml_rock_property_paths:
        # save site name
        site = os.path.dirname(filepath).split("/")[-1]
        # save rock layer name
        rock_layer = os.path.splitext(os.path.basename(filepath))[0]

        # Load YAML file
        with open(filepath, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        # Iterate over each top-level property
        for prop, entries in yaml_data.items():
            for entry in entries:
                property_entry = {"site": site}
                property_entry["rock_layer"] = rock_layer
                property_entry["property"] = prop

                for key, value in entry.items():
                    if key == "probability_distribution":
                        pdf_data = value or {}
                        property_entry["sample_size"] = pdf_data.get("sample_size")
                        property_entry["sampled_data"] = pdf_data.get("sampled_data")
                    elif key == "tag":
                        tag_data = value or {}
                        if site != "default":
                            property_entry["agency"] = tag_data.get("agency")

                            property_entry["location"] = tag_data.get("location")

                            property_entry["simplified_lithology"] = tag_data.get(
                                "simplified_lithology"
                            )
                        property_entry["ID"] = tag_data.get("ID")
                    else:
                        property_entry[key] = value

                records.append(property_entry)

    # Create the DataFrame for properties
    property_df = pd.DataFrame(records)
    # Preserve float for scalar and string for expression and dictionary
    property_df[["value", "value_min", "value_max", "value_std"]] = property_df[
        ["value", "value_min", "value_max", "value_std"]
    ].map(preserve_value_type)
    # Keep the sample_size as integer
    property_df["sample_size"] = pd.to_numeric(
        property_df["sample_size"], errors="coerce"
    ).astype("Int64")
    # Convert list to numpy array
    property_df["sampled_data"] = property_df["sampled_data"].map(
        lambda sampled_data: (
            np.array(sampled_data, dtype=np.float64)
            if isinstance(sampled_data, list)
            else sampled_data
        )
    )
    # Replace np.nan with None
    property_df = property_df.replace({np.nan: None})
    return property_df


def load_site_property(site_files_path):
    """load site properties from YAML files to a DataFrame.

    Args:
        site_files_path (list): a list of paths to YAML files containing site properties.

    Returns:
        pd.DataFrame: a DataFrame containing the loaded site properties.
    """

    records = []
    for filepath in site_files_path:
        site = os.path.splitext(os.path.basename(filepath))[0]

        # Load YAML file
        with open(filepath, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            name = content.pop("name")
            site_description = content.pop("description")

            geometry_input = content.pop("geometry_input")
            geometry_source = geometry_input.get("source")
            geometry_point_data_path = geometry_input.get("point_data_path")
            geometry_orientation_data_path = geometry_input.get("orientation_data_path")

        # Iterate through rock layers
        for rock_layer, rock_data in content.items():

            site_entry = {"site": site}
            site_entry["name"] = name
            site_entry["site_description"] = site_description
            site_entry["geometry_source"] = geometry_source
            site_entry["geometry_point_data_path"] = geometry_point_data_path
            site_entry["geometry_orientation_data_path"] = (
                geometry_orientation_data_path
            )

            site_entry["rock_layer"] = rock_layer
            site_entry["layer_source"] = rock_data.get("source")
            geological_time = rock_data.get("geological_time", {})
            site_entry["eon"] = geological_time.get("eon")
            site_entry["era"] = geological_time.get("era")
            site_entry["period"] = geological_time.get("period")
            site_entry["epoch"] = geological_time.get("epoch")
            site_entry["age"] = geological_time.get("age")
            site_entry["layer_description"] = rock_data.get("description")
            site_entry["layer_simplified_lithology"] = rock_data.get(
                "simplified_lithology"
            )

            site_entry["property_path"] = rock_data.get("property_path")
            site_entry["geometry_surface_output_path"] = rock_data.get(
                "geometry_surface_output_path"
            )

            records.append(site_entry)

    # Create the DataFrame for site properties
    site_df = pd.DataFrame(records)
    return site_df


def combine_rock_site_property():
    """Combine rock properties and site properties into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the combined properties.
    """
    # load YAML file paths from the rock_property directory
    property_path = os.path.join("..", "database", "rock_property")
    property_file_paths = get_path_in_dir(property_path)
    yaml_property_paths = [
        path for path in property_file_paths if path.endswith(".yaml")
    ]

    # remove the YAML file paths in default directory
    real_path = os.path.realpath(os.path.dirname(__file__))
    default_path = os.path.realpath(
        os.path.join(
            real_path,
            os.path.join(os.path.join("..", "database", "rock_property", "default")),
        )
    )
    yaml_rock_property_paths = [
        path for path in yaml_property_paths if default_path not in path
    ]
    # load all rock layer properties to a Pandas DataFrame
    property_df = load_rock_property(yaml_rock_property_paths)

    # load YAML file paths from the site directory
    site_path = os.path.join("..", "database", "site")
    site_file_paths = get_path_in_dir(site_path)
    yaml_site_paths = [path for path in site_file_paths if path.endswith(".yaml")]
    # load all site properties to a Pandas DataFrame
    site_df = load_site_property(yaml_site_paths)

    # Merge the two DataFrame based on site and rock_layer
    merged_property_df = pd.merge(
        property_df, site_df, on=["site", "rock_layer"], how="left"
    )
    return merged_property_df
