import os
from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pyvista as pv

from src.load_path import get_path_in_dir
from src.property2dataframe import (
    load_rock_property,
    load_site_property,
    preserve_value_type,
)
from src.data_tagging import get_tagged_data_mask
from src.generate_id import get_list_from_sequence
from src.add_default import add_default_df, find_missing_properties
from src.merge_method import merge_property_value
from src.dataframe2yaml import dataframe2yaml_str
from src.generate_geomodel import generate_geomodel_for_site, export_gempy2grid

operators = [
    ["ge ", ">="],
    ["le ", "<="],
    ["lt ", "<"],
    ["gt ", ">"],
    ["ne ", "!="],
    ["eq ", "="],
    ["contains "],
    ["datestartswith "],
]


def split_filter_part(filter_part):
    """Parse the filter part of the filter string. Adopted from the Dash documentation [https://dash.plotly.com/datatable/filtering] accessed on 13/10/2025.

    Args:
        filter_part (str or None): A part of the filter string.

    Returns:
        str: column name
        str: operator
        str: value
    """
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find("{") + 1 : name_part.rfind("}")]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', "`"):
                    value = value_part[1:-1].replace("\\" + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


def list2str(property_df):
    """Convert list in pandas DataFrame to string for better display in Dash DataTable.

    Args:
        property_df (pd.DataFrame): A pandas DataFrame that contains the properties of a rock layer.

    Returns:
        pd.DataFrame: A pandas DataFrame with list converted to string.
    """
    table_property_df = property_df.copy()
    to_convert_headers = [
        "simplified_lithology",
        "location",
        "agency",
        "unit_base",
        "variable_unit_base",
    ]
    property_df_columns = list(table_property_df)
    for header in property_df_columns:
        if header in to_convert_headers:
            table_property_df[header] = table_property_df[header].map(
                lambda row: ", ".join(map(str, get_list_from_sequence(row)))
            )
    return table_property_df


def str2list(property_df):
    """Convert strings in pandas DataFrame to list.

    Args:
        property_df (pd.DataFrame): A pandas DataFrame that contains the properties of a rock layer.

    Returns:
        pd.DataFrame: A pandas DataFrame with string converted to list.
    """
    export_property_df = property_df.copy()
    to_convert_headers = [
        "simplified_lithology",
        "location",
        "agency",
        "unit_base",
        "variable_unit_base",
    ]
    property_df_columns = list(export_property_df)
    for header in property_df_columns:
        if header in to_convert_headers:
            if (header == "unit_base") or (header == "variable_unit_base"):
                export_property_df[header] = export_property_df[header].map(
                    lambda row: (
                        [int(i) for i in row.split(", ")]
                        if (pd.notna(row) and row != "")
                        else None
                    )
                )
            else:
                export_property_df[header] = export_property_df[header].map(
                    lambda row: (
                        row.split(", ") if (pd.notna(row) and row != "") else None
                    )
                )
    return export_property_df


def RGBtxt_to_dict(file_path, color_type):
    """_summary_

    Args:
        file_path (str): Path to the rgb text file for the Geological Time Scale.
        color_type (str): 'to_hex' or 'to_rgb'.

    Raises:
        ValueError: Valid value for color_type: ['to_hex', 'to_rgb']
        ValueError: No tuple, please make sure the dictionary has the right structure!

    Returns:
        dict: Dictionary with keys as geological time scale and values as hex color or rgb color.
    """
    result_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:  # Ensure there are at least 2 parts (key and value)
                key = parts[0].strip()
                rgb_value = parts[1].strip()
                # Convert RGB to hex
                rgb = tuple(map(int, rgb_value.split("/")))
                if color_type == "to_hex":
                    hex_value = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
                    result_dict[key] = hex_value
                elif color_type == "to_rgb":
                    result_dict[key] = f"rgb{rgb}"
                else:
                    raise ValueError("Valid value for color_type: ['to_hex', 'to_rgb']")
            elif len(parts) != 2:
                raise ValueError(
                    "No tuple, please make sure the dictionary has the right structure!"
                )

    return result_dict


def get_geological_time(row):
    """Combine geological time scale to a list.

    Args:
        row (pd.DataFrame): A row of pandas DataFrame that contains geological time scale and rock layer name.

    Returns:
        list: A list of combined geological time scale.
    """
    combined_scales = [
        scale
        for scale in [row["eon"], row["era"], row["period"], row["epoch"], row["age"]]
        if pd.notna(scale)
    ]
    rock_layer_name = row["rock_layer"]

    if combined_scales[-1] != rock_layer_name:
        combined_scales.append(rock_layer_name)
    return combined_scales


def create_stratigraphic_table(site_strata, strata_props_descriptions):
    """Create a stratigrahic table follows the style of the international chronostratigraphic chart.

    Args:
        site_strata (list): A list of string that contains the name for each lithostratigraphic layer.
        strata_props_descriptions (list): A list of string that contains the description for each lithostratigraphic layer.

    Returns:
        plotly.graph_objs._figure.Figure: A stratigraphic table in Plotly Icicle format.
        dict: A dictionary that contains the hex color for each lithostratigraphic layer.
    """
    ids = ["Stratigraphic_Chart"]
    labels = ["Stratigraphic_Chart"]
    parents = [""]
    stratigraphy_colors = ["#D3D3D3"]

    d_k = 0
    ids_descriptions = [""]

    for strata_names in site_strata:

        root = "Stratigraphic_Chart"

        for i in range(len(strata_names)):
            id_name = "-".join(
                strata_names[: i + 1]
            )  # combining the name again using '-'
            if id_name not in ids:
                ids.append(id_name)
                try:
                    stratigraphy_colors.append(hex_colors[id_name])
                except (
                    KeyError
                ):  # if the id_name is not within the international chronostratigraphic chart, then use
                    # its parents color.
                    stratigraphy_colors.append(stratigraphy_colors[-1])

                if id_name.split("-") in site_strata:
                    ids_descriptions.append(strata_props_descriptions[d_k])
                    d_k += 1
                else:
                    ids_descriptions.append("")

                # add the splitted name
                labels.append(strata_names[i])

                parents.append(root)
            root = id_name  # hierarchy is divided by '-' and a root is always the id_name before '-'

    fig = go.Figure(
        go.Icicle(
            ids=ids,
            labels=labels,
            parents=parents,
            marker_colors=stratigraphy_colors,
            hovertext=ids_descriptions,
            textinfo="label+text",
            sort=False,
        )
    )

    stratigraphy_colors_dict = dict(zip(ids, stratigraphy_colors))
    fig.update_layout(margin=dict(t=40, l=25, r=25, b=0))
    return fig, stratigraphy_colors_dict


def filter_table_df(filter, property_df):
    """Filter the properties table based on the filter query. Adopted from the Dash documentation [https://dash.plotly.com/datatable/filtering] accessed on 13/10/2025.

    Args:
        filter (str): String that contains the filter query.
        property_df (pd.DataFrame): A pandas DataFrame that contains the properties of a rock layer.

    Returns:
        pd.DataFrame: Filtered pandas DataFrame.
    """
    filtering_expressions = filter.split(" && ")
    update_property_df = property_df.copy()
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ("eq", "ne", "lt", "le", "gt", "ge"):
            # these operators match pandas series operator method names
            update_property_df = update_property_df.loc[
                getattr(update_property_df[col_name], operator)(filter_value)
            ]
        elif operator == "contains":
            filter_value = [value.strip() for value in filter_value.split(",")]
            filtered_masks = get_tagged_data_mask(property_df, col_name, filter_value)
            update_property_df = update_property_df[filtered_masks]
        elif operator == "datestartswith":
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            update_property_df = update_property_df.loc[
                update_property_df[col_name].str.startswith(filter_value)
            ]

    return update_property_df


def load_vtkdata(
    site_name,
    sites_material_props_dict,
    stratum_colors_dict,
    clicked_layer_name,
    initial_call=False,
):
    """
    Load the vtk data for the 3D structural geomodel of the selected site.
    Args:
        site_name (str): name of the selected site.
        sites_material_props_dict (dict): a dictionary that contains the material properties for each site.
        stratum_colors_dict (dict): a dictionary that contains the hex color for each lithostratigraphic layer.
        clicked_layer_name (str): name of the clicked lithostratigraphic layer.
        initial_call (bool, optional): Initialize call to avoid display issues on macOS and Linux. Defaults to False.

    Returns:
        str: HTML string of the 3D structural geomodel.
    """
    plotter_strata = pv.Plotter()
    if initial_call:
        return plotter_strata.export_html(None)
    else:
        site_strata_df = pd.DataFrame.from_records(
            sites_material_props_dict[site_name]["site_props"]
        )

        for stratum, stratum_color in stratum_colors_dict.items():

            stratum_props_file_paths = site_strata_df.loc[
                site_strata_df["rock_layer"] == stratum, "geometry_surface_output_path"
            ].iloc[0]

            if isinstance(stratum_props_file_paths, list):  # not None or nan
                for stratum_props_file_path in stratum_props_file_paths:
                    stratum_ply = pv.read(stratum_props_file_path)
                    stratum_opacity = 1
                    # highlight the clicked layer
                    if clicked_layer_name in stratum_colors_dict.keys():
                        if clicked_layer_name != stratum:
                            stratum_color = "#D3D3D3"
                            stratum_opacity = 0.4
                    plotter_strata.add_mesh(
                        stratum_ply,
                        color=stratum_color,
                        name=stratum,
                        opacity=stratum_opacity,
                    )

        plotter_strata.show_grid()
        return plotter_strata.export_html(None)


hex_colors = RGBtxt_to_dict("RGB_stratigraphy.txt", color_type="to_hex")
site_path = os.path.join("..", "database", "site")
site_file_paths = get_path_in_dir(site_path)
yaml_site_paths = [path for path in site_file_paths if path.endswith(".yaml")]

geometry_path = os.path.join("database", "geometry")
geometry_folder_list = []
for folder_file in os.listdir(geometry_path):
    folder_file_path = os.path.join(geometry_path, folder_file)
    if os.path.isdir(folder_file_path):
        geometry_folder_list.append(folder_file)

site_dropdown_list = [os.path.basename(file_path)[:-5] for file_path in yaml_site_paths]
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Smart Data Hub"

site_dropdown = dbc.Card(
    html.Div(
        [
            dcc.Dropdown(
                id="site_dropdown",
                options=site_dropdown_list,
                placeholder="Select a site name",
                clearable=False,
            ),
        ]
    ),
)
site_geomodel = dbc.Card(
    [
        dbc.CardHeader("3D Structural Geomodel"),
        html.Div(
            [
                html.Iframe(
                    id="3d_model",
                    style={
                        "height": "min(100%, calc(33.33vw - 25px))",
                        "width": "calc(33.33vw - 25px)",
                    },
                ),
            ],
            style={"height": "calc(100vh - 316px)"},
        ),
    ]
)

export_geomodel_button = dbc.Button(
    "Export Geomodel", id="btn_export_geomodel", outline=True, color="secondary"
)
export_geomodel_dropdown = dcc.Dropdown(
    options=[
        {"label": "VTK file", "value": "VTK"},
    ],
    id="export_geomodel_dropdown",
    placeholder="Select model type",
)
export_geomodel = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(export_geomodel_button, width=6),
                dbc.Col(export_geomodel_dropdown, width=6),
            ],
        )
    ]
)

default_merge_butttons = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Add Default",
                        id="btn_add_default",
                        outline=True,
                        color="secondary",
                    ),
                    width={"size": 5, "order": 1},
                ),
                dbc.Col(
                    dbc.Button(
                        "Merge Data",
                        id="btn_merge_data",
                        outline=True,
                        color="secondary",
                    ),
                    width={"size": 6, "order": "last"},
                ),
            ],
        ),
    ],
    color="white",
    outline=True,
)

export_data_button_dropdown = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Export Data",
                        id="btn_export_data",
                        outline=True,
                        color="secondary",
                    ),
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options=[
                            {"label": "CSV file", "value": "csv"},
                            {"label": "YAML file", "value": "yaml"},
                        ],
                        id="export_data_dropdown",
                        placeholder="Select file type",
                    ),
                ),
            ],
        )
    ],
    color="white",
    outline=True,
)

lithostratigraphy_button = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(default_merge_butttons, width=6),
                dbc.Col(export_data_button_dropdown, width=5),
            ],
            justify="between",
        ),
    ]
)

lithostratigraphy = dbc.Card(
    [
        dbc.CardHeader("Lithostratigraphic Table"),
        html.Div(
            [
                dcc.Graph(
                    id="icicle_props",
                    style={
                        "height": "98%",
                    },
                ),
                html.Div(
                    id="properties_table",
                    children=[
                        html.Div(
                            children=[
                                dash_table.DataTable(
                                    id="datatable-interactivity",
                                    filter_action="custom",
                                    filter_query="",
                                ),
                            ],
                            style={
                                "overflow-y": "auto",
                                "overflow-x": "auto",
                                "top": "110px",
                                "height": "calc(100vh - 280px - 110px)",
                                "width": "calc(66.67vw - 25px - 80px)",
                                "position": "absolute",
                                "margin-left": "40px",
                            },
                        )
                    ],
                    style={"display": "none"},
                ),
            ],
            style={"height": "calc(100vh - 280px)"},
        ),
    ],
)

app.layout = dbc.Container(
    [
        dbc.Alert(
            [
                html.H1(
                    children="Smart Data Hub",
                    style={"textAlign": "center", "color": "black"},
                ),
            ],
            color="secondary",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Stack(
                        [
                            site_dropdown,
                            site_geomodel,
                            export_geomodel,
                        ],
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Stack(
                        [
                            lithostratigraphy,
                            lithostratigraphy_button,
                        ],
                    ),
                    md=8,
                ),
            ],
        ),
        dcc.Store(id="sites_material_props_dict"),
        dcc.Store(id="site_name_previous"),
        dcc.Store(id="clicked_layer_name"),
        dcc.Store(id="plotter_plydata"),
        dcc.Store(id="filtered_indices"),
        dcc.Store(id="icicle_clickData"),
        dcc.Store(id="current_datatable"),
        dcc.Store(id="strata_lithologies"),
        dcc.Download(id="export_props_data"),
        dcc.Download(id="export_geomodel_data"),
        dcc.ConfirmDialog(id="default_confirm"),
        dcc.ConfirmDialog(id="merge_confirm"),
    ],
    fluid=True,
)


# change the value in state will not return output,  unless any input has changed
@app.callback(
    Output("icicle_props", "figure"),
    Output("sites_material_props_dict", "data"),
    Input("site_dropdown", "value"),
    State("sites_material_props_dict", "data"),
)
def display_stratigraphic_table(site_name, sites_material_props_dict):
    if site_name is None:  # assign initial value
        fig = go.Figure()
        fig.update_layout(margin=dict(t=40, l=25, r=25, b=5))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig, {}
    else:
        if site_name not in sites_material_props_dict:
            # convert YAML file to pandas DataFrame
            site_props_df = load_site_property(
                [yaml_site_paths[site_dropdown_list.index(site_name)]]
            )
            # store the site properties
            sites_material_props_dict.update(
                {site_name: {"site_props": site_props_df.to_dict("records")}}
            )

        site_strata_df = pd.DataFrame.from_records(
            sites_material_props_dict[site_name]["site_props"]
        )
        site_strata_names_list = list(site_strata_df["rock_layer"])
        strata_props_descriptions = list(site_strata_df["layer_description"])
        strata_geological_time = list(site_strata_df.apply(get_geological_time, axis=1))

        fig, stratigraphy_colors_dict = create_stratigraphic_table(
            strata_geological_time, strata_props_descriptions
        )

        # store rgb colors for each stratum for 3d surface model.
        if "stratum_colors_dict" not in sites_material_props_dict[site_name].keys():
            stratum_colors = [
                stratigraphy_colors_dict["-".join(id_name)]
                for id_name in strata_geological_time
            ]
            stratum_colors_dict = dict(zip(site_strata_names_list, stratum_colors))
            sites_material_props_dict[site_name][
                "stratum_colors_dict"
            ] = stratum_colors_dict

        return fig, sites_material_props_dict


@app.callback(
    Output("site_name_previous", "data"),
    Output("icicle_clickData", "data"),
    Output("datatable-interactivity", "columns"),
    Output("datatable-interactivity", "data"),
    Output("datatable-interactivity", "filter_query"),
    Output("current_datatable", "data"),
    Output("strata_lithologies", "data"),
    Input("icicle_props", "clickData"),
    Input("sites_material_props_dict", "data"),
    Input("site_dropdown", "value"),
    State("site_name_previous", "data"),
    State("datatable-interactivity", "filter_query"),
    State("icicle_clickData", "data"),
    State("strata_lithologies", "data"),
)
def load_stratum_table(
    click_data,
    sites_material_props_dict,
    site_name,
    site_name_previous,
    filter,
    icicle_clickData_previous,
    lithologies,
):

    if click_data is None:
        datatable_columns = None
        datatable_data = None

    else:  # load the properties table
        datatable_columns = None
        datatable_data = None
        clicked_layer_name = click_data["points"][0]["label"]
        site_strata_df = pd.DataFrame.from_records(
            sites_material_props_dict[site_name]["site_props"]
        )
        site_strata_names_list = list(site_strata_df["rock_layer"])

        # set the condition of site_name == site_name_previous to make sure the properties table
        # is loaded when site_name is updated.
        if (
            clicked_layer_name in site_strata_names_list
            and site_name == site_name_previous
        ):  # if properties for the layer exist
            # load the properties table
            clicked_layer_df = load_rock_property(
                [
                    site_strata_df.loc[
                        site_strata_df["rock_layer"] == clicked_layer_name,
                        "property_path",
                    ].iloc[0]
                ]
            )
            lithologies = site_strata_df.loc[
                site_strata_df["rock_layer"] == clicked_layer_name,
                "layer_simplified_lithology",
            ].iloc[0]
            # clear the fiter quary when switch to a different rock layer
            if click_data != icicle_clickData_previous:
                filter = ""
            update_property_df = clicked_layer_df.copy()
            # columns to display
            columns_n = list(update_property_df.columns)
            none_columns = update_property_df.columns[
                update_property_df.isnull().all()
            ].tolist()
            to_remove_columns = [
                "variable_name",
                "variable_unit_str",
                "variable_unit_base",
            ]
            columns_display = [
                cn
                for cn in columns_n
                if cn
                not in (
                    ["site", "rock_layer"]
                    + [col for col in none_columns if col in to_remove_columns]
                )
            ]

            datatable_columns = [{"name": i, "id": i} for i in columns_display]
            datatable_data = list2str(update_property_df).to_dict("records")

    return (
        site_name,
        click_data,
        datatable_columns,
        datatable_data,
        filter,
        datatable_data,
        lithologies,
    )


@app.callback(
    Output("datatable-interactivity", "data", allow_duplicate=True),
    State("current_datatable", "data"),
    Input("datatable-interactivity", "filter_query"),
    prevent_initial_call=True,
)
def filter_table(datatable_data, filter):
    if datatable_data is None:
        return datatable_data
    else:

        update_property_df = filter_table_df(
            filter, str2list(pd.DataFrame.from_records(datatable_data))
        )
        datatable_data = list2str(update_property_df).to_dict("records")
        return datatable_data


@app.callback(
    Output("default_confirm", "displayed"),
    Output("default_confirm", "message"),
    Input("btn_add_default", "n_clicks"),
    State("datatable-interactivity", "data"),
    prevent_initial_call=True,
)
def confirm_add_default(add_default_clicks, datatable_data):
    # convert to pandas DataFrame
    datatable_data_pd = str2list(pd.DataFrame.from_records(datatable_data)).copy()
    # --- find missing properties --- #
    added_properties = find_missing_properties(datatable_data_pd)

    if added_properties:
        confirm_message = (
            f"{', '.join(added_properties)} will be added with default values."
        )
    else:
        confirm_message = (
            "All properties already have values. No default values will be added."
        )

    return True, confirm_message


@app.callback(
    Output("datatable-interactivity", "data", allow_duplicate=True),
    Output("datatable-interactivity", "filter_query", allow_duplicate=True),
    Output("current_datatable", "data", allow_duplicate=True),
    State("datatable-interactivity", "data"),
    Input("default_confirm", "submit_n_clicks"),
    State("datatable-interactivity", "filter_query"),
    State("strata_lithologies", "data"),
    prevent_initial_call=True,
)
def add_default(datatable_data, add_default_clicks, filter, lithologies):
    # convert to pandas DataFrame and add default values
    datatable_data_pd = str2list(pd.DataFrame.from_records(datatable_data)).copy()
    update_property_df = add_default_df(datatable_data_pd, lithologies).copy()

    # convert back to datatable format
    datatable_data = list2str(update_property_df).to_dict("records")

    filter = ""

    return datatable_data, filter, datatable_data


@app.callback(
    Output("merge_confirm", "displayed"),
    Output("merge_confirm", "message"),
    Input("btn_merge_data", "n_clicks"),
    State("datatable-interactivity", "data"),
    prevent_initial_call=True,
)
def confirm_merge_add(merge_data_clicks, datatable_data):
    # convert to pandas DataFrame
    datatable_data_merge_pd = str2list(pd.DataFrame.from_records(datatable_data)).copy()
    # --- Check if there are any missing values in sampled_data --- #
    # drop the missing id properties from the original DataFrame
    datatable_data_merge_pd_id = datatable_data_merge_pd.dropna(subset=["ID"]).copy()
    missing_sampled_data = datatable_data_merge_pd_id["sampled_data"].isna().any()
    remain_properties = list(datatable_data_merge_pd_id["property"])
    len_diff = len(remain_properties) - len(set(remain_properties))
    if missing_sampled_data or (len_diff > 0):
        confirm_message = (
            "Merge data and add a truncated normal distribution for each property."
        )
    else:
        confirm_message = "Merging data is not needed, as data is already merged, or no uncertainty is missing."

    return True, confirm_message


@app.callback(
    Output("datatable-interactivity", "data", allow_duplicate=True),
    Output("datatable-interactivity", "filter_query", allow_duplicate=True),
    Output("current_datatable", "data", allow_duplicate=True),
    State("datatable-interactivity", "data"),
    Input("merge_confirm", "submit_n_clicks"),
    State("datatable-interactivity", "filter_query"),
    prevent_initial_call=True,
)
def merge_data(datatable_data, merge_data_clicks, filter):

    update_property_df = merge_property_value(
        str2list(pd.DataFrame.from_records(datatable_data)), source_type="merged"
    ).copy()

    datatable_data = list2str(update_property_df).to_dict("records")
    filter = ""

    return datatable_data, filter, datatable_data


@app.callback(
    Output("properties_table", "style"), Input("datatable-interactivity", "data")
)
def display_properties_table(datatable_data):
    if datatable_data is None:
        return {"display": "none"}
    else:
        return {"display": "block"}


@app.callback(
    Output("btn_add_default", "disabled"),
    Output("btn_merge_data", "disabled"),
    Output("btn_export_data", "disabled"),
    Output("export_data_dropdown", "disabled"),
    Output("btn_export_geomodel", "disabled"),
    Output("export_geomodel_dropdown", "disabled"),
    Input("properties_table", "style"),
)
def lithology_table_buttons(properties_table_style):
    if properties_table_style["display"] == "none":
        return True, True, True, True, False, False
    else:
        return False, False, False, False, True, True


@app.callback(
    Output("export_props_data", "data"),
    Input("btn_export_data", "n_clicks"),
    State("datatable-interactivity", "data"),
    State("export_data_dropdown", "value"),
    State("icicle_props", "clickData"),
    prevent_initial_call=True,
)
def export_props_data(btn_export_data, datatable_data, export_file_type, click_data):
    data_df = pd.DataFrame.from_records(datatable_data)
    # Preserve float for scalar and string for expression and dictionary
    data_df[["value", "value_min", "value_max", "value_std"]] = data_df[
        ["value", "value_min", "value_max", "value_std"]
    ].map(preserve_value_type)
    # Keep the sample_size as integer
    data_df["sample_size"] = pd.to_numeric(
        data_df["sample_size"], errors="coerce"
    ).astype("Int64")
    # Replace both np.nan and empty strings with None
    data_df = data_df.replace({np.nan: None, "": None})

    # convert the joined strings to a list of strings
    data_df = str2list(data_df)

    # Get the rock layer name
    clicked_layer_name = click_data["points"][0]["label"]

    if export_file_type == "yaml":
        yaml_str = dataframe2yaml_str(data_df)
        return dict(
            content=yaml_str,
            filename=f"{clicked_layer_name}.yaml",
        )
    elif export_file_type == "csv":
        return dcc.send_data_frame(data_df.to_csv, f"{clicked_layer_name}.csv")
    else:
        return None


@app.callback(
    Output("3d_model", "srcDoc"),
    Input("site_dropdown", "value"),
    Input("sites_material_props_dict", "data"),
    Input("icicle_props", "clickData"),
    State("site_name_previous", "data"),
)
def display_geomodel(
    site_name, sites_material_props_dict, click_data, site_name_previous
):
    if site_name is None:  # initial interface
        return None
    elif site_name not in geometry_folder_list:  # when no geomodel is available.
        return "No geomodel is available!"
    else:
        stratum_colors_dict = sites_material_props_dict[site_name][
            "stratum_colors_dict"
        ]

        if click_data is None:
            clicked_layer_name = None
        else:
            if site_name == site_name_previous:
                clicked_layer_name = click_data["points"][0]["label"]
            else:
                clicked_layer_name = None

        html_model = load_vtkdata(
            site_name,
            sites_material_props_dict,
            stratum_colors_dict,
            clicked_layer_name,
        )

        return html_model.getvalue()


@app.callback(
    Output("export_geomodel_data", "data"),
    Input("btn_export_geomodel", "n_clicks"),
    State("export_geomodel_dropdown", "value"),
    State("site_dropdown", "value"),
    prevent_initial_call=True,
)
def export_geomodel(btn_export_geomodel, export_model_type, site_name):

    if export_model_type == "VTK":
        geo_model = generate_geomodel_for_site(
            site_name, refinement=4, resolution=[50, 50, 50]
        )
        # export the regular grid of the model
        save_output_folder_path = "assets"
        export_gempy2grid(
            geo_model,
            save_output_folder_path,
            output_filename=f"{site_name}_3D_model.vtk",
        )
        return dcc.send_file(
            os.path.join(save_output_folder_path, f"{site_name}_3D_model.vtk")
        )


if __name__ == "__main__":
    # Initial call to avoid display issues on macOS and Linux.
    load_vtkdata(
        site_name=None,
        sites_material_props_dict=None,
        stratum_colors_dict=None,
        clicked_layer_name=None,
        initial_call=True,
    )

    app.run(debug=False)
