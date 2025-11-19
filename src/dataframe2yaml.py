import yaml


# keep the flow sequence style when writing as a YAML sequence
class List_flow_sequence(list):
    pass


def flow_sequence_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def convert_to_flow_sequence(unit_base):
    """Convert unit_base to a flow sequence representation.

    Args:
        unit_base (list or nonetype): The unit_base to convert.

    Returns:
        List_flow_sequence: The flow sequence representation of the unit base.
    """

    # only convert python list
    if isinstance(unit_base, list):
        return List_flow_sequence(unit_base)
    else:
        return unit_base


# Register flow style representer
yaml.add_representer(
    List_flow_sequence, flow_sequence_representer, Dumper=yaml.SafeDumper
)
# Write None as null
yaml.add_representer(
    type(None),
    lambda dumper, _: dumper.represent_scalar("tag:yaml.org,2002:null", "null"),
)


def dataframe2yaml_str(property_df):
    """Convert the property dataframe to a YAML string.

    Args:
        property_df (pd.DataFrame): The input property dataframe.

    Returns:
        str: The YAML string representation of the property dataframe.
    """

    yaml_dict = {}
    property_df_headers = list(property_df)
    property_df_headers.remove("property")

    for _, row in property_df.iterrows():
        property = row["property"]

        if property not in yaml_dict:
            yaml_dict[property] = []

        # assemble data dictionary
        data_dict = {}
        for header_key in property_df_headers:
            if (header_key == "sample_size") or (header_key == "sampled_data"):
                if "probability_distribution" not in data_dict:
                    data_dict["probability_distribution"] = {}
                data_dict["probability_distribution"][header_key] = row[header_key]
            elif (
                (header_key == "ID")
                or (header_key == "agency")
                or (header_key == "location")
                or (header_key == "simplified_lithology")
            ):
                if "tag" not in data_dict:
                    data_dict["tag"] = {}
                data_dict["tag"][header_key] = convert_to_flow_sequence(row[header_key])
            elif (
                header_key == "site" or header_key == "rock_layer"
            ):  # do not store the site and rock_layer information
                data_dict = data_dict
            else:
                data_dict[header_key] = convert_to_flow_sequence(row[header_key])

        yaml_dict[property].append(data_dict)

    # Convert dict to YAML
    yaml_str = yaml.dump(
        yaml_dict, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper
    )

    return yaml_str


def export2yaml(property_df, output_file_path):
    """export the property dataframe to a YAML file.

    Args:
        property_df (pd.DataFrame): The input property dataframe.
        output_file_path (str): Path to the output YAML file.
    """
    yaml_str = dataframe2yaml_str(property_df)

    # Save to file
    with open(output_file_path, "w") as f:
        f.write(yaml_str)
