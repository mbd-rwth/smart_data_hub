import pandas as pd


def string_in_strings(target, strings):
    """Check if 'target' exists in 'strings'.

    Args:
        target (str): string to search for.
        strings (list, str, None): list or string containing multiple entries

    Returns:
        bool: True if target is found in the strings, False otherwise.
    """

    if strings is None:
        return False
    if isinstance(strings, str):
        if target in strings:
            return True
        else:
            return False
    elif isinstance(
        strings, list
    ):  # for columns "simplified_lithology", "location", "agency", "unit_base",or "variable_unit_base"
        for strings in strings:
            # Split comma and slash
            parts = [part.strip() for part in strings.replace("/", ",").split(",")]

            if target in parts:
                return True

        return False
    else:
        return False


def get_tagged_data_mask(property_df, tag_type, tag_names):
    """Get masks for multiple tag names in a specific tag type.

    Args:
        property_df (pd.DataFrame): Input DataFrame.
        tag_type (str): Specific tag type. It can be "agency", "location", or "simplified_lithology".
        tag_names (list): List of tag names.

    Returns:
        pd.Series: A Boolean series indicating which data entries have the specified tags.
    """
    combined_tag_mask = pd.Series(False, index=property_df.index)
    for tag_name in tag_names:
        tagged_data_mask = property_df.apply(
            lambda row: string_in_strings(tag_name, row[tag_type]), axis=1
        )
        combined_tag_mask |= tagged_data_mask

    return combined_tag_mask


def filter_tagged_data(property_df, tag_dict):
    """Filter the DataFrame based on multiple tag criteria.

    Args:
        property_df (pd.DataFrame): Input DataFrame.
        tag_dict (dict): Dictionary where keys are tag types ("agency", "location", "simplified_lithology")
                         and values are lists of tag names.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows that match all specified tag criteria.
    """
    tag_type_masks = pd.Series(True, index=property_df.index)
    for key, value in tag_dict.items():
        tag_type_mask = get_tagged_data_mask(property_df, key, value)
        tag_type_masks &= tag_type_mask
    property_df = property_df[tag_type_masks]
    return property_df
