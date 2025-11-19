import os
import pyvista as pv
import pandas as pd
from src.property2dataframe import combine_rock_site_property
from src.load_path import get_path_in_dir
from src.merge_method import value_empty_mask, value_invalid_mask


def validate_property_data():
    # Load the all properties into a DataFrame
    merged_df = combine_rock_site_property()

    # --- Check if geometry data are complete ---#
    geometry_surface_output_paths_list = list(merged_df["geometry_surface_output_path"])
    geometry_surface_output_paths = list(
        set(path for sublist in geometry_surface_output_paths_list for path in sublist)
    )
    surface_output_paths = []
    for paths in geometry_surface_output_paths:
        if (paths != None) and (str(paths) != "nan"):
            for path in paths.split(", "):
                surface_output_paths.append(os.path.join(*path.split(os.sep)[-3:]))
    # Get VTK paths in geometry folder
    geometry_path = os.path.join("..", "database", "geometry")
    geometry_file_paths = get_path_in_dir(geometry_path)
    geometry_vtk_paths = [
        os.path.join(*path.split(os.sep)[-3:])
        for path in geometry_file_paths
        if path.endswith(".vtk")
    ]

    diff_geometry_names = set(surface_output_paths) - set(geometry_vtk_paths)
    assert not diff_geometry_names, f"Missing {diff_geometry_names} !"

    for file_path in geometry_file_paths:
        if file_path.endswith(".vtk"):
            rock_layer_ply = pv.read(file_path)
            assert not (
                rock_layer_ply.n_points == 0 and rock_layer_ply.n_cells == 0
            ), f"{os.path.join(*file_path.split(os.sep)[-3:])} is a empty mesh!"

    # --- Check if required property names are present ---#
    required_property_names = set(
        [
            "density",
            "porosity",
            "hydraulic_conductivity",
            "intrinsic_permeability",
            "p_wave_velocity",
            "s_wave_velocity",
            "specific_heat_capacity",
            "thermal_conductivity",
            "electrical_resistivity",
        ]
    )
    merged_property_names = set(merged_df["property"])
    diff_property_names = required_property_names - merged_property_names
    assert not diff_property_names, f"Missing {diff_property_names} !"

    # drop empty data entry
    merged_df = (merged_df.dropna(subset=["source"])).copy()

    # --- Check if there are empty data with a valid source ---#
    empty_mask = value_empty_mask(merged_df)
    assert merged_df[
        empty_mask
    ].empty, f"Rows with missing data:\n{merged_df[empty_mask]}"

    # --- check if the value entry is valid ---#
    is_invalid_mask = value_invalid_mask(merged_df)
    assert merged_df[
        is_invalid_mask
    ].empty, f"Rows with invalid entries:\n{merged_df[is_invalid_mask]}"

    # --- check if the value is in the range of value_min and value_max ---#
    # Define check values function
    def check_condition(row):
        (
            v,
            v_min,
            v_max,
        ) = (
            row["value"],
            row["value_min"],
            row["value_max"],
        )

        # if all values exist
        if pd.notna(v_min) and pd.notna(v) and pd.notna(v_max):
            return v_min < v < v_max
        # if value_min and value exist
        elif pd.notna(v_min) and pd.notna(v):
            return v_min < v
        # if value and value_max exist
        elif pd.notna(v) and pd.notna(v_max):
            return v < v_max
        # if value_min and value_max exist
        elif pd.notna(v_min) and pd.notna(v_max):
            return v_min < v_max
        else:
            return True

    condition_mask = merged_df.apply(check_condition, axis=1)
    assert merged_df[
        ~condition_mask
    ].empty, f"Rows with invalid value ranges:\n{merged_df[~condition_mask]}"

    print("All the data validation tests passed successfully!")


if __name__ == "__main__":
    validate_property_data()
