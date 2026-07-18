import os
from pathlib import Path
from importlib.resources import files
from smart_data_hub.merge_method import merge_property_value
from smart_data_hub.merge_method import generate_lognorm, generate_PERT, generate_truncnorm, generate_uniform
from smart_data_hub.dataframe2yaml import export2yaml
from smart_data_hub.property2dataframe import combine_rock_site_property



def generate_initial_default():
    """Generate the initial default data for rock properties"""

    # combine all data into a single pandas dataframe
    merged_df = combine_rock_site_property()

    # Fill None rock type in simplified_lithology with layer_simplified_lithology
    merged_df["simplified_lithology"] = merged_df["simplified_lithology"].fillna(
        merged_df["layer_simplified_lithology"]
    )

    # Get a unqiue rock type list
    layer_unqiue_lithologies = (
        merged_df["layer_simplified_lithology"].explode().unique().tolist()
    )

    sampling_functions_by_property = {"electrical_resistivity": generate_lognorm, "intrinsic_permeability": generate_lognorm}
    for layer_unqiue_lithology in layer_unqiue_lithologies:

        input_df_for_merging = merged_df[
            merged_df["simplified_lithology"].map(
                lambda lithology: layer_unqiue_lithology in lithology
            )
        ]

        input_df_for_merging = input_df_for_merging.drop_duplicates(subset=["ID"])

        merged_property_df = merge_property_value(
            input_df_for_merging, source_type="default", sampling_functions_by_property=sampling_functions_by_property
        )
        # Only maintainers can save the generated data to YAML files.
        save_to_file = True
    
        if save_to_file:

            output_file_path = files("smart_data_hub") / "dataset" / "rock_property" / "default" / f"{layer_unqiue_lithology}.yaml"
            export2yaml(
                merged_property_df,
                output_file_path=output_file_path,
            )
