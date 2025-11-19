import os
import gempy as gp
import pyvista as pv
import numpy as np
from gempy_engine.core.data.stack_relation_type import StackRelationType


def generate_geomodel(
    site_orientation_path,
    site_points_path,
    extent_geometry,
    refinement,
    resolution,
    structural_group,
    stack_relation,
):
    """Create a GemPy 3D structural model.

    Args:
        site_orientation_path (str): Path to CSV file that stores orientation data for each surface.
        site_points_path (str): Path to CSV file that stores point data for each surface.
        extent_geometry (list): model extend in a list [x_min, x_max, y_min, y_max, z_min, z_max].
        refinement (int): Model resolution of the regular grids.
        resolution (list): Model resolution in a list [x, y, z].
        structural_group (dict): GemPy series or stacks in the model.
        stack_relation (GemPy StackRelationType): GemPy StackRelationType.

    Returns:
        gempy.core.data.geo_model.GeoModel: The computed GemPy 3D structural model.
    """

    geo_data = gp.create_geomodel(
        project_name="site",
        extent=extent_geometry,
        refinement=refinement,
        resolution=resolution,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=site_orientation_path,
            path_to_surface_points=site_points_path,
        ),
    )

    gp.map_stack_to_surfaces(geo_data, mapping_object=structural_group)

    for i in range(len(geo_data.structural_frame.structural_groups)):
        geo_data.structural_frame.structural_groups[i].structural_relation = (
            stack_relation
        )

    # Compute the geological model
    gp.compute_model(geo_data)
    return geo_data


def export_gempy2vtk(geo_model, save_output_folder_path):
    """Export each fromation surface as a VTK file.

    Args:
        geo_model (gempy.core.data.geo_model.GeoModel): A GemPy 3D structural model.
        save_output_folder_path (str): Path to the output folder.
    """

    formation_list = list(geo_model.structural_frame.element_id_name_map.values())
    formation_list.remove("basement")

    # Get vertices and edges for each surface
    surface_meshes_vertices = [
        geo_model.input_transform.apply_inverse(mesh.vertices)
        for mesh in geo_model.solutions.dc_meshes
    ]
    surface_meshes_edges = [mesh.edges for mesh in geo_model.solutions.dc_meshes]

    for i in range(len(surface_meshes_vertices)):
        surface_mesh = pv.PolyData(
            surface_meshes_vertices[i],
            np.insert(surface_meshes_edges[i], 0, 3, axis=1).ravel(),
        )
        surface_mesh.save(
            os.path.join(save_output_folder_path, f"{formation_list[i]}.vtk")
        )
        print(f"Saved {formation_list[i]}.vtk !")


def export_gempy2grid(
    geo_model, save_output_folder_path, output_filename="3D_model.vtk"
):
    """Export the regular grid of the model as a VTK file.
    Args:
        geo_model (gempy.core.data.geo_model.GeoModel): A GemPy 3D structural model.
        save_output_folder_path (str): Path to the output folder.
        output_filename (str): Name of the output VTK file.
    """
    # export the regular grid of the model
    resolution = geo_model.grid.regular_grid.resolution
    grid_3d = geo_model.regular_grid_coordinates.reshape(*(resolution + 1), 3).T
    regular_mesh = pv.StructuredGrid(*grid_3d)
    regular_mesh["Lithology"] = geo_model.solutions.raw_arrays.lith_block
    regular_mesh.save(os.path.join(save_output_folder_path, output_filename))
    print(f"Saved {output_filename} !")


def generate_geomodel_for_site(site_name, refinement, resolution):
    """Create a GemPy 3D structural model for a specific site.

    Args:
        site_name (str): Name of the site. It can be "DE_Crystalline", "DE_North_Claystone", "DE_South_Claystone", or "DE_Rocksalt".
        refinement (int): Model resolution of the regular grids.
        resolution (list): Model resolution in a list [x, y, z].

    Returns:
        gempy.core.data.geo_model.GeoModel: The computed GemPy 3D structural model for the given site.
    """

    if site_name == "DE_Crystalline":
        extent_geometry = [0, 2000, 0, 2000, -1400, 10]
        structural_group = {
            "Strat_series": [
                "Quaternary",
                "Muschelkalk",
                "Buntsandstein",
                "Zechstein_Rocksalt_sequence1",
                "Zechstein_Anhydrite_sequence1",
                "Zechstein_Rocksalt_sequence2",
                "Zechstein_Potash_sequence1",
                "Zechstein_Rocksalt_sequence3",
                "Zechstein_Anhydrite_sequence2",
                "Zechstein_Potash_sequence2",
                "Zechstein_Rocksalt_sequence4",
                "Zechstein_Anhydrite_sequence3",
                "Granite",
            ]
        }
        stack_relation = StackRelationType.ONLAP
    elif site_name == "DE_North_Claystone":
        extent_geometry = [0, 2000, 0, 2000, -4300, 0]
        structural_group = {
            "Strat_series1": "Quaternary",
            "Strat_series2": ("Albian", "Aptian"),
            "Strat_series3": ("Barremian", "Hauterivian", "Valanginian", "Wealden"),
            "Strat_series4": "Jurassic_Upper",
            "Strat_series5": "Jurassic_Middle",
            "Strat_series6": ("Jurassic_Lower", "Keuper"),
            "Strat_series7": "Buntsandstein_Muschelkalk",
            "Strat_series8": ("Buntsandstein", "Zechstein"),
        }
        stack_relation = StackRelationType.ONLAP

    elif site_name == "DE_South_Claystone":
        extent_geometry = [0, 2000, 0, 2000, -800, 530]
        structural_group = {
            "Strat_series1": "Quaternary",
            "Strat_series2": "Tertiary",
            "Strat_series3": ("Tithonian", "Kimmeridgian", "Oxfordian"),
            "Strat_series4": "Jurassic_Middle_sequence1",
            "Strat_series5": "Jurassic_Middle_sequence2",
            "Strat_series6": (
                "Opalinus_Clay",
                "Jurassic_Lower_sequence1",
                "Jurassic_Lower_sequence2",
            ),
            "Strat_series7": (
                "Keuper_sequence1",
                "Keuper_sequence2",
                "Keuper_sequence3",
                "Muschelkalk_Upper",
            ),
            "Strat_series8": ("Muschelkalk_Middle", "Muschelkalk_Lower"),
        }
        stack_relation = StackRelationType.ONLAP
    elif site_name == "DE_Rocksalt":
        extent_geometry = [0, 2000, 0, 2000, -1600, 0]
        structural_group = {
            "Strat_series": [
                "Quaternary",
                "Tertiary",
                "Buntsandstein",
                "Aller",
                "Tonmittelsalz",
                "Anhydritmittelsalz",
                "Leine_Potash",
                "Leine_Rocksalt",
                "Leine_Hauptanhydrit",
                "Strassfurt_Potash",
                "Strassfurt_Rocksalt",
                "Anhydrite",
                "Carbonate",
                "Rotliegend",
            ]
        }
        stack_relation = StackRelationType.ONLAP
    else:
        raise ValueError(
            f"Please use generate_geomodel to create the GemPy model for the {site_name} site."
        )

    geo_model = generate_geomodel(
        site_orientation_path=f"database/geometry/{site_name}/input/orientations.csv",
        site_points_path=f"database/geometry/{site_name}/input/points.csv",
        extent_geometry=extent_geometry,
        refinement=refinement,
        resolution=resolution,
        structural_group=structural_group,
        stack_relation=stack_relation,
    )
    return geo_model
