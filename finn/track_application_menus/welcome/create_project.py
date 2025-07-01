import os
import shutil
from pathlib import Path
from typing import Any

import dask.array as da
import funlib.persistence as fp
import networkx as nx
import numpy as np
import pandas as pd
from funtracks.features._base import Feature
from funtracks.project import Project
from qtpy.QtWidgets import (
    QMessageBox,
)
from skimage.measure import regionprops
from tqdm import tqdm


class DialogValueError(ValueError):
    def __init__(self, message, show_dialog=True):
        super().__init__(message)
        self.show_dialog = show_dialog


def create_project(project_info: dict[str:Any]) -> Project:
    """Creates a new funtracks project with the information provided in the dialog"""

    print(project_info)
    intensity_image = project_info["intensity_image"]
    segmentation_image = project_info["segmentation_image"]
    name = project_info.get("title", "Untitled Project")
    ndim = int(project_info.get("ndim", 3))
    axes = project_info.get("axes", [])
    data_type = project_info.get("data_type", "points")
    working_dir = project_info.get("directory", Path.cwd())
    params = project_info.get("project_params")
    features = project_info.get("features")

    # remove old zarr dir if present
    zarr_dir = os.path.join(project_info.get("directory"), f"{name}.zarr")
    if os.path.exists(zarr_dir):
        shutil.rmtree(zarr_dir)

    # check if segmentation image has integer data type, warn user if not.
    if segmentation_image is not None and np.issubdtype(
        segmentation_image.dtype, np.floating
    ):
        msg = QMessageBox()
        msg.setWindowTitle("Invalid segmentation file type")
        msg.setText(
            "The datatype of the provided segmentation data is float.<br><br>"
            "Click <b>Continue</b> if you are sure you selected the correct data and"
            " it will be converted to integers.<br>"
            "Click <b>Go Back</b> to return to the import menu."
        )
        msg.addButton("Continue", QMessageBox.AcceptRole)
        goback_btn = msg.addButton("Go Back", QMessageBox.RejectRole)
        msg.setDefaultButton(goback_btn)
        msg.exec_()
        if msg.clickedButton() == goback_btn:
            raise DialogValueError(
                "Invalid segmentation file type, going back to select alternative "
                "data of type integer.",
                show_dialog=False,
            )
        segmentation_image = segmentation_image.astype(np.uint64)

    # when loading tracks from csv, we need the mapping to the seg_ids before c
    # constructing the fpds.
    create_graph_from_df = False
    seg_id_map = None
    tracks_path = project_info["tracks_path"]
    if tracks_path is not None:
        df = pd.read_csv(tracks_path)
        mapping = project_info["column_mapping"]
        scaling = axes["scaling"]
        if "channel" in axes["dimensions"]:
            scaling = list(scaling).pop(0)

        # remap based on column mapping provided by the user
        for new_col, old_col in mapping.items():
            df[new_col] = df[old_col]

        # check that the ids provided in the csv are indeed unique
        if not df["id"].is_unique:
            raise DialogValueError(
                f"The object id values in column {mapping['id']} "
                "of the provided CSV file are not unique. Please provide a csv file where"
                "each object has a unique id value."
            )

        # check that the id column contains integer values, if not relabel them to
        # unique integers
        if not pd.api.types.is_integer_dtype(df["id"]):
            print(
                f"Relabeling strings in column {mapping['id']} and "
                f"{mapping['parent_id']} to unique integers"
            )
            all_labels = pd.unique(df[["id"]].values.ravel())
            label_to_int = {label: idx + 1 for idx, label in enumerate(all_labels)}
            df["id"] = df["id"].map(label_to_int)
            df["parent_id"] = df["parent_id"].map(label_to_int).astype("Int64")

        # check if the provided segmentation matches with the dataframe (dimensions
        # and seg value of the first object). Raises DialogValueError if any problems
        # are found.
        if segmentation_image is not None:
            seg_indices = [int(i) for i in axes.get("seg_indices", list(range(ndim)))]
            test_df_seg_match(
                df,
                segmentation_image,
                scaling,
                axis_order=seg_indices,
                mapping=mapping,
            )

        # extract id to seg_id dictionary if a segmentation image was provided
        if segmentation_image is not None:
            if (df["seg_id"] == df["id"]).all():
                # no relabeling is needed, id is already equal to seg_id
                seg_id_map = None
            else:
                seg_id_map = df[["t", "id", "seg_id"]]

        create_graph_from_df = True

    # create fpds for the intensity image and segmentation data (if provided)
    intensity_fpds, segmentation_fdps = create_fpds(
        intensity_image,
        segmentation_image,
        os.path.join(working_dir, f"{name}.zarr"),
        ndim,
        axes,
        data_type,
        seg_id_map,
    )

    if create_graph_from_df:
        cand_graph = graph_from_df(
            df, segmentation_image, intensity_image, scaling[1:], features
        )
    else:
        cand_graph = None

    # Create a candidate graph with only nodes when tracking based on point detections.
    if data_type == "points":
        points_data = project_info.get("points_data")
        cand_graph = graph_from_points(points_data, axes["points_columns"])

    # # TODO: include features to measure, ndim, cand_graph_params, point detections
    return Project(
        name=name,
        project_params=params,
        raw=intensity_fpds,
        segmentation=segmentation_fdps,
        cand_graph=cand_graph,
    )


def create_empty_fp_array(
    fp_array_path: str, shape: tuple, axes: dict | None = None
) -> fp.Array:
    """Creates an empty funtracks persistence array with the specified shape and axes."""

    axis_names = axes.get("axis_names", ["axis_" + str(i) for i in range(len(shape))])
    voxel_size = axes.get("scaling", [1.0] * len(shape))
    axis_units = axes.get("units", ["px"] * len(shape))

    if "channel" in axes["dimensions"]:
        # remove the channel information, segmentation fpds can only have dimensions
        # t(z)yx
        axis_names.pop(0)
        voxel_size.pop(0)
        axis_units.pop(0)

    print(
        f"creating empty fpds with shape {shape}, voxel_size {voxel_size}, axis_names"
        f" {axis_names} units {axis_units}"
    )
    fpds = fp.prepare_ds(
        fp_array_path,
        shape=shape,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=axis_units,
        dtype=np.uint32,
    )

    return fpds


def create_fpds(
    intensity_image: da.Array | None,
    segmentation_image: da.Array | None,
    fp_array_path: str | None,
    ndim: int,
    axes: dict,
    data_type: str,
    seg_id_map: pd.DataFrame | None,
) -> tuple[fp.Array, fp.Array]:
    """Creates a funtracks persistence array from an intensity image or segmentation
    data.
    Args:
        intensity_image (da.Array | None ): Dask array of intensity image
        segmentation_image (da.Array | None): Dask array of the segmentation data.
        fp_array_path (str): Path where the funtracks persistence array will be
        created.
        axes (dict): Dictionary containing axis information like indices, names,
        units, and scaling.
        data_type: (str = "points" | "segmentation")
        seg_id_map (pd.DataFrame): mapping from seg id + time to id, in case the user
            provided external tracking data.

    Returns:
        fp.Array: A funtracks persistence array containing the data."""

    # Check if at least one of the two data paths is valid.
    if intensity_image is None and segmentation_image is None:
        # this situation is invalid, if no seg is provided we need at least intensity
        #  image to track from scratch
        raise DialogValueError(
            "No valid path to intensity data and segmentation labels was provided. "
            "We need at least an intensity image to track from scratch"
        )

    # Get the axis information
    raw_indices = [int(i) for i in axes.get("raw_indices", list(range(ndim)))]
    seg_indices = [int(i) for i in axes.get("seg_indices", list(range(ndim)))]
    axis_names = axes.get("axis_names", ["axis_" + str(i) for i in range(ndim)])
    voxel_size = axes.get("scaling", [1.0] * ndim)
    axis_units = axes.get("units", ["px"] * ndim)

    # Transpose the stack, if needed
    if intensity_image is not None:
        default_order = list(range(intensity_image.ndim))
        if raw_indices != default_order:
            intensity_image = np.transpose(intensity_image, raw_indices)

    if segmentation_image is not None:
        default_order = list(range(segmentation_image.ndim))
        if seg_indices != default_order:
            segmentation_image = np.transpose(segmentation_image, seg_indices)

    # Check if the shapes of intensity and segmentation data are matching
    if intensity_image is not None and segmentation_image is not None:
        if len(intensity_image.shape) == len(segmentation_image.shape) + 1:
            valid = intensity_image.shape[-(ndim - 1) :] == segmentation_image.shape
        else:
            valid = intensity_image.shape == segmentation_image.shape

        if not valid:
            raise DialogValueError(
                f"Segmentation data shape does not match intensity image shape.\n"
                f"Segmentation shape: {segmentation_image.shape}\n"
                f"Intensity image shape: {intensity_image.shape}\n"
                "Please go back and select matching data.",
                show_dialog=True,
            )

    # Create empty fpds for segmentation when tracking from scratch with labels
    if data_type == "segmentation" and segmentation_image is None:
        if "channel" in axes["dimensions"]:
            segmentation_fpds = create_empty_fp_array(
                os.path.join(fp_array_path, "seg"),
                shape=intensity_image.shape[-(ndim - 1) :],
                axes=axes,
            )
        else:
            segmentation_fpds = create_empty_fp_array(
                os.path.join(fp_array_path, "seg"),
                shape=intensity_image.shape,
                axes=axes,
            )

    # Create fpds for intensity and/or segmentation image
    if intensity_image is not None:
        intensity_fpds = create_fp_array(
            intensity_image,
            path=os.path.join(fp_array_path, "raw"),
            shape=intensity_image.shape,
            dimensions=axes["dimensions"],
            voxel_size=voxel_size,
            axis_names=axis_names,
            axis_units=axis_units,
            dtype=intensity_image.dtype,
        )
    else:
        intensity_fpds = None

    if segmentation_image is not None:
        if "channel" in axes["dimensions"]:
            voxel_size = voxel_size[1:]
            axis_names = axis_names[1:]
            axis_units = axis_units[1:]

        segmentation_fpds = create_fp_array(
            segmentation_image,
            path=os.path.join(fp_array_path, "seg"),
            shape=segmentation_image.shape,
            dimensions=(d for d in axes["dimensions"] if d != "channel"),
            voxel_size=voxel_size,
            axis_names=axis_names,
            axis_units=axis_units,
            dtype=np.uint64,
            seg_id_map=seg_id_map,
        )
    else:
        segmentation_fpds = None

    return intensity_fpds, segmentation_fpds


def create_fp_array(
    image: da.Array,
    path: str,
    shape: tuple[int],
    dimensions: tuple[str],
    voxel_size: tuple[float],
    axis_names: tuple[str],
    axis_units: tuple[str],
    dtype: np.dtype,
    seg_id_map: pd.DataFrame | None = None,
) -> fp.Array:
    fpds = fp.prepare_ds(
        path,
        shape=shape,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=axis_units,
        dtype=dtype,
    )

    # if there is segmentation data, check if relabeling is necessary for one of the
    # following reasons:
    #   1) User provided external tracks data, and we need to map seg_id to id
    #   2) The labels in the provided data are not unique across time.

    correct_duplicate_ids = None
    relabel_from_csv = None

    if path.endswith("seg"):
        # check whether we should relabel from csv
        relabel_from_csv = seg_id_map is not None
        # if not, check whether the data has duplicate values
        if not relabel_from_csv:
            correct_duplicate_ids = has_duplicate_ids(image)
        curr_max = 0

    # load and write each time point into the dataset
    time_index = list(dimensions).index("time")
    for time in tqdm(
        range(image.shape[time_index]), desc="Converting time points to zarr"
    ):
        # keep running track of max and call ensure unique labels, map to csv
        slc = [slice(None)] * image.ndim
        slc[time_index] = time

        if path.endswith("seg"):
            seg = image[tuple(slc)].compute()
            if relabel_from_csv:
                relabeled_seg = np.zeros_like(seg).astype(np.uint64)
                df_t = seg_id_map[seg_id_map["t"] == time]
                # Create a mapping from seg_id to id for the current time point
                seg_id_to_id = dict(zip(df_t["seg_id"], df_t["id"], strict=True))
                # Apply the mapping to the segmentation image for the current time
                # point
                for seg_id, new_id in seg_id_to_id.items():
                    relabeled_seg[seg == seg_id] = new_id
                seg = relabeled_seg
            elif correct_duplicate_ids:
                mask = seg != 0
                seg[mask] += curr_max
                curr_max = int(np.max(seg))
            # seg is now relabeled, or no relabeling was necessary
            fpds[time] = seg
        else:
            fpds[time] = image[tuple(slc)].compute()

    return fpds


def has_duplicate_ids(segmentation: np.ndarray) -> bool:
    """Checks if the segmentation has duplicate label ids across time. For efficiency,
    only checks between the first and second time frames.

    Args:
        segmentation (np.ndarray): (t, [z], y, x)

    Returns:
        bool: True if there are duplicate labels between the first two frames, and
            False otherwise.
    """
    if segmentation.shape[0] >= 2:
        first_frame = segmentation[0]
        second_frame = segmentation[1]
        if isinstance(first_frame, da.Array):
            first_frame = first_frame.compute()
            second_frame = second_frame.compute()
        first_frame_ids = set(np.unique(first_frame).tolist())
        if 0 in first_frame_ids:
            first_frame_ids.remove(0)
        second_frame_ids = set(np.unique(second_frame).tolist())

        if 0 in second_frame_ids:
            second_frame_ids.remove(0)
        return not first_frame_ids.isdisjoint(second_frame_ids)
    return False


def test_df_seg_match(
    df: pd.DataFrame,
    segmentation: da.Array,
    scale: list[float] | None,
    axis_order,
    mapping: dict,
):
    """Test if the provided segmentation, dataframe, and scale values are valid together.
    Tests the following requirements:
      - The scale, if provided, has same dimensions as the segmentation
      - The location coordinates have the same dimensions as the segmentation
      - The segmentation pixel value for the coordinates of first node corresponds
    with the provided seg_id as a basic sanity check that the csv file matches with the
    segmentation file

    Args:
        df (pd.DataFrame): the pandas dataframe to turn into tracks, with standardized
            column names
        segmentation (np.ndarray): The segmentation, a 3D or 4D array of integer labels
        scale (list[float] | None): A list of floats representing the relationship between
            the point coordinates and the pixels in the segmentation
    """
    # transpose if needed
    default_order = list(range(segmentation.ndim))
    if axis_order != default_order:
        segmentation = np.transpose(segmentation, axis_order)

    if scale is not None:
        if segmentation.ndim != len(scale):
            raise DialogValueError(
                f"Dimensions of the segmentation image ({segmentation.ndim}) "
                f"do not match the number of scale values given ({len(scale)})",
                show_dialog=True,
            )
    else:
        scale = [
            1,
        ] * segmentation.ndim

    row = df.iloc[-1]
    pos = (
        [row["t"], row["z"], row["y"], row["x"]]
        if "z" in df.columns
        else [row["t"], row["y"], row["x"]]
    )

    if segmentation.ndim != len(pos):
        raise DialogValueError(
            f"Dimensions of the segmentation ({segmentation.ndim}) do not match the "
            f"number of positional dimensions ({len(pos)})",
            show_dialog=True,
        )

    seg_id = int(row["seg_id"])
    coordinates = [
        int(coord / scale_value) for coord, scale_value in zip(pos, scale, strict=True)
    ]

    try:
        value = segmentation[tuple(coordinates)].compute()
    except IndexError:
        raise DialogValueError(
            f"Could not get the segmentation value at coordinates "
            f"{coordinates}. Segmentation data has shape "
            f"{segmentation.shape}. Please check if the axis order you"
            f" provided is correct.",
            show_dialog=True,
        )

    if not value == seg_id:
        raise DialogValueError(
            f"The value {value} found at coordinates {coordinates} "
            f"does not match the expected value ({seg_id}) from "
            f"column {mapping['seg_id']}",
            show_dialog=True,
        )


def graph_from_points(
    points_data: pd.DataFrame, column_mapping: dict[str:str]
) -> nx.DiGraph:
    """Create a graph from points data, representing t(z)yx coordinates"""

    graph = nx.DiGraph()
    for id, row in points_data.iterrows():
        if "z" in column_mapping:
            pos = [
                row.get(column_mapping["z"], None),
                row.get(column_mapping["y"], None),
                row.get(column_mapping["x"], None),
            ]
        else:
            pos = [row.get(column_mapping["y"], None), row.get(column_mapping["x"], None)]

        t = row.get(column_mapping["t"], None)

        values = pos + [t]
        if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            raise DialogValueError(
                f"Non-numerical or missing value found in position columns at row {id}: {values}",
                show_dialog=True,
            )

        attrs = {
            "t": int(t),
            "pos": pos,
        }

        graph.add_node(id + 1, **attrs)

    return graph


def graph_from_df(
    df: pd.DataFrame,
    segmentation: fp.Array | None,
    intensity_image: fp.Array | None,
    scaling: tuple[float],
    features: list[dict[str : Feature | str | bool]],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for _, row in df.iterrows():
        _id = int(row["id"])
        parent_id = row["parent_id"]
        if "z" in df.columns:
            pos = [row["z"], row["y"], row["x"]]
        else:
            pos = [row["y"], row["x"]]

        attrs = {
            "t": int(row["t"]),
            "pos": pos,
        }

        # add additional features that should be recomputed
        features_to_recompute = [
            f["feature"]
            for f in features
            if f["include"]
            and f["feature"].computed
            and f["from_column"] is None
            and f["feature"].regionprops_name is not None
        ]

        if len(features_to_recompute) > 0:
            t = int(row["t"])
            if intensity_image is not None:
                intensity = intensity_image[t].compute()
            else:
                intensity = None
            # compute the feature
            props = regionprops(
                segmentation[t].compute(), intensity_image=intensity, spacing=scaling
            )
            for regionprop in props:
                for feature in features_to_recompute:
                    # to list gives floats/ints in the case of single items
                    value = getattr(regionprop, feature.regionprops_name)
                    if isinstance(value, tuple):
                        value = [i.tolist() for i in value]
                    else:
                        value = value.tolist()
                    attrs[feature.attr_name] = value

        # optionally import extra features directly from the table, without recomputing
        features_to_import_from_df = [
            f for f in features if f["include"] and f["from_column"] is not None
        ]

        for feature in features_to_import_from_df:
            attrs[feature["feature"].attr_name] = row.get(feature["from_column"])

        # add the node to the graph
        graph.add_node(_id, **attrs)

        # add the edge to the graph, if the node has a parent
        # note: this loading format does not support edge attributes
        if not pd.isna(parent_id) and parent_id != -1:
            assert parent_id in graph.nodes, (
                f"Parent id {parent_id} of node {_id} not in graph yet"
            )
            graph.add_edge(parent_id, _id)

    return graph
