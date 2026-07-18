import os
import pyvista as pv
import pandas as pd
from pathlib import Path
import pytest
from importlib.resources import files
from smart_data_hub.property2dataframe import combine_rock_site_property
from smart_data_hub.load_path import get_path_in_dir
from smart_data_hub.merge_method import value_empty_mask, value_invalid_mask

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


@pytest.fixture(scope="module")
def merged_df():
    # Load the all properties into a DataFrame
    return combine_rock_site_property()


def geometry_file_paths():
    # Get VTK paths in geometry folder

    geometry_path = files("smart_data_hub") / "dataset" / "geometry"
    return get_path_in_dir(geometry_path)


def test_referenced_geometry_files_exist(merged_df):
    # --- Check if geometry data are complete ---#
    geometry_surface_output_paths_list = list(merged_df["geometry_surface_output_path"])
    geometry_surface_output_paths = list(
        set(
            path
            for sublist in geometry_surface_output_paths_list
            if isinstance(sublist, list)  # skip NaN / non-list entries
            for path in sublist
        )
    )
    surface_output_paths = []
    for paths in geometry_surface_output_paths:
        if (paths != None) and (str(paths) != "nan"):
            for path in paths.split(", "):
                surface_output_paths.append(os.path.join(*path.split(os.sep)[-3:]))

    geometry_vtk_paths = [
        os.path.join(*path.split(os.sep)[-3:])
        for path in geometry_file_paths()
        if path.endswith(".vtk")
    ]

    diff_geometry_names = set(surface_output_paths) - set(geometry_vtk_paths)
    assert not diff_geometry_names, f"Missing {diff_geometry_names} !"


@pytest.mark.parametrize(
    "file_path",
    [
        Path(file_path)
        for file_path in geometry_file_paths()
        if file_path.endswith(".vtk")
    ],
    ids=lambda path: str(path),
)
def test_geometry_mesh_is_not_empty(file_path):
    rock_layer_ply = pv.read(file_path)

    assert not (
        rock_layer_ply.n_points == 0 and rock_layer_ply.n_cells == 0
    ), f"{os.path.join(*file_path.split(os.sep)[-3:])} is a empty mesh!"


def test_required_properties_are_present(merged_df):
    # --- Check if required property names are present ---#
    merged_property_names = set(merged_df["property"])
    diff_property_names = required_property_names - merged_property_names
    assert not diff_property_names, f"Missing {diff_property_names} !"


@pytest.fixture(scope="module")
def sourced_property_df(merged_df):
    # drop empty data entry
    return merged_df.dropna(subset=["source"]).copy()


def test_sourced_properties_do_not_have_empty_values(sourced_property_df):
    # --- Check if there are empty data with a valid source ---#
    empty_mask = value_empty_mask(sourced_property_df)
    assert sourced_property_df[
        empty_mask
    ].empty, f"Rows with missing data:\n{sourced_property_df[empty_mask]}"


def test_sourced_properties_do_not_have_invalid_values(sourced_property_df):
    # --- check if the value entry is valid ---#
    is_invalid_mask = value_invalid_mask(sourced_property_df)
    assert sourced_property_df[
        is_invalid_mask
    ].empty, f"Rows with invalid entries:\n{sourced_property_df[is_invalid_mask]}"
