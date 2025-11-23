import uuid
from ruamel.yaml import YAML
import os
from src.load_path import get_path_in_dir

# Preserve quotes and save None to null in YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.representer.add_representer(
    type(None),
    lambda dumper, _: dumper.represent_scalar("tag:yaml.org,2002:null", "null"),
)


# Define the SDH_NAMESPACE
def sdh_namespace():
    """Generate the SDH namespace UUID.

    Returns:
        uuid.UUID: The SDH namespace UUID.
    """
    return uuid.UUID("c02c3197-8a22-421d-9b86-66c88560e57b")


def get_list_from_sequence(value):
    """helper function to convert a string or None to a list.

    Args:
        value (str, None, list): the input value to convert.

    Raises:
        ValueError: if the input value is not a string, None, or a list.

    Returns:
        list : the converted list.
    """

    if isinstance(value, str):
        return [value]
    elif value is None:
        return []
    elif isinstance(value, list):
        return value
    else:
        raise ValueError("Input value must be a string, None, or a list.")


def normalize_str(val):
    """Normalize function for splitting and sorting strings.

    Args:
        val (str): The input string to normalize.

    Returns:
        str: The normalized string.
    """
    if val is None:
        return ""
    # Split by comma or slash
    parts = [p.strip() for chunk in val.split(",") for p in chunk.split("/")]
    # Remove empty and sort
    parts = sorted(filter(None, parts))
    return ",".join(parts)


def flatten_entry_dict(property_entry, property_name):
    """Flatten the entry dictionary.

    Args:
        property_entry (dict): The property dictionary.
        property_name (str): Property name.

    Returns:
        dict: The flattened entry dictionary.
    """

    if property_entry.get("probability_distribution"):
        pdf_info = property_entry.get("probability_distribution")
    else:
        pdf_info = property_entry
    if property_entry.get("tag"):
        tag_info = property_entry.get("tag")
    else:
        tag_info = property_entry
    flatten_dict = {
        "property": property_name,
        "type": property_entry["type"],
        "source": property_entry["source"],
        "value": property_entry["value"],
        "value_min": property_entry["value_min"],
        "value_max": property_entry["value_max"],
        "value_std": property_entry["value_std"],
        "sample_size": pdf_info["sample_size"],
        "sampled_data": pdf_info["sampled_data"],
        "unit_str": property_entry["unit_str"],
        "unit_base": property_entry["unit_base"],
        "variable_name": property_entry.get("variable_name"),
        "variable_unit_str": property_entry.get("variable_unit_str"),
        "variable_unit_base": property_entry.get("variable_unit_base"),
        "description": property_entry["description"],
        "agency": (", ".join(get_list_from_sequence(tag_info.get("agency"))) or None),
        "location": (
            ", ".join(get_list_from_sequence(tag_info.get("location"))) or None
        ),
        "simplified_lithology": (
            ", ".join(get_list_from_sequence(tag_info.get("simplified_lithology")))
            or None
        ),
    }

    return flatten_dict


def get_entry_str(property_entry, property_name):
    """Function for combining all entry fields into a single string.

    Args:
        property_entry (dict): The property dictionary.
        property_name (str): Property name.

    Returns:
        str: The combined entry string.
    """

    flatten_entry = flatten_entry_dict(property_entry, property_name)
    combined_entry_str = ", ".join(
        normalize_str(str(entry_str)) for entry_str in flatten_entry.values()
    )
    return combined_entry_str


def generate_property_id(yaml_file_path):
    """Check for missing ID and generate a unique ID for each data in the YAML file.

    Args:
        yaml_file_path (str): The path to the YAML file.
    """

    SDH_NAMESPACE = sdh_namespace()

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    missing_ID = False
    # Iterate through the data and assign IDs if missing
    for property_name, property_entries in data.items():
        for property_entry in property_entries:
            # If data is given with a valid source
            if property_entry.get("source"):
                # Check if 'ID' is empty
                if not property_entry["tag"].get("ID"):
                    missing_ID = True
                    property_entry["tag"]["ID"] = str(
                        uuid.uuid5(
                            SDH_NAMESPACE, get_entry_str(property_entry, property_name)
                        )
                    )
    # Save the updated YAML back
    if missing_ID:
        with open(yaml_file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)


def generate_id_for_all():
    """Generate IDs for all YAML files in the rock_property directory"""

    # load YAML file paths from the rock_property directory
    property_path = os.path.join("..", "database", "rock_property")
    property_file_paths = get_path_in_dir(property_path)
    yaml_property_paths = [
        path for path in property_file_paths if path.endswith(".yaml")
    ]

    # Generate IDs for each data in all YAML files if IDs are missing
    for yaml_file_path in yaml_property_paths:
        # for yaml_file_path in yaml_rock_property_paths:
        generate_property_id(yaml_file_path)
