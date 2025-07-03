import os
import shutil
from pathlib import Path
from typing import Any

import dask.array as da
import funlib.persistence as fp
import networkx as nx
import numpy as np
import pandas as pd
from funtracks.cand_graph import CandGraph
from funtracks.features._base import Feature
from funtracks.features.feature_set import FeatureSet
from funtracks.features.measurement_features import Intensity  # adjust import as needed
from funtracks.nx_graph import NxGraph
from funtracks.project import Project
from funtracks.tracking_graph import TrackingGraph
from qtpy.QtWidgets import (
    QMessageBox,
)
from skimage.measure import regionprops
from tqdm import tqdm


class DialogValueError(ValueError):
    """Dialog message to display when invalid information is found"""

    def __init__(self, message, show_dialog=True):
        super().__init__(message)
        self.show_dialog = show_dialog


def create_project(project_info: dict[str:Any]) -> Project:
    """Creates a new Funtracks Project based on the information provided as dictionary.
    args:
        project_info (dict[str: Any]): dictionary to build a project from.
        - data_type [str]: either 'segmentation' or 'points'
        - intensity_image [da.Array | None] : intensity data
        - segmentation_image [da.Array | None] : segmentation data
        - points_data [pd.DataFrame]: point detection data
        - ndim [int]: the number of dimensions (incl time) of the data (3, 4, or
        - axes [dict]:
            dimensions [tuple[str]]: dimension names (e.g. 'time', 'z')
            raw_indices [tuple[int]]: index of each dimension in the raw data
            seg_indices [tuple[int]]: index of each dimension in the seg data
            axis_names [tuple(str)]: dimension names assigned by the user
            units (tuple[str]): units for each dimension, e.g. 'µm'
            scaling [tuple(float)]: spatial calibration in the same order as the
                dimensions
        - tracks_path [str | None]: path to where the tracking data csv file is
            stored (if provided)
        - column_mapping [dict[str: str] | None] : mapping of the csv column
            headers to the required tracking information (dimensions, ids)
        - convert_pixel_units (bool): whether the coordinates in the csv are still in
            pixel units and need to be remapped.
        - project_params [ProjectParams]: parameters for the project
        - cand_graph_params [CandGraphParams]: parameters for the candidate graph
        - features (list[dict[str: str|bool]]): list of features to measure,
                each with 'feature_name', 'include' (bool), and 'from_column'
                (str or None)
        - title [str]: name of the project,
        - directory [str]: path to directory where the project should be saved

    returns:
        funtracks.Project
    """

    try:
        data_type = project_info["data_type"]
        intensity_image = project_info["intensity_image"]
        segmentation_image = project_info["segmentation_image"]
        points_data = project_info["points_data"]
        ndim = project_info["ndim"]
        axes = project_info["axes"]
        tracks_path = project_info["tracks_path"]
        column_mapping = project_info["column_mapping"]
        convert_pixel_units = project_info["convert_pixel_units"]
        project_params = project_info["project_params"]
        cand_graph_params = project_info["cand_graph_params"]
        features = project_info["features"]
        name = project_info.get("title", "Untitled Project")
        working_dir = project_info.get("directory", Path.cwd())
        n_channels = 1
    except KeyError as err:
        missing_key = err.args[0]
        raise DialogValueError(
            f"The following key is missing: {missing_key}",
            show_dialog=True,
        ) from err

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

    # Sort dimensions according to default_order
    default_order = ("channel", "time", "z", "y", "x")
    default_seg_order = ("time", "z", "y", "x")
    filtered_int_order = [ax for ax in default_order if ax in axes["dimensions"]]
    filtered_seg_order = [ax for ax in default_seg_order if ax in axes["dimensions"]]
    int_order_indices = [axes["dimensions"].index(ax) for ax in filtered_int_order]
    seg_order_indices = [axes["dimensions"].index(ax) for ax in filtered_seg_order]

    # Reorder all axis-related lists and transpose data if needed
    axes["dimensions"] = [axes["dimensions"][i] for i in int_order_indices]
    axes["axis_names"] = [axes["axis_names"][i] for i in int_order_indices]
    axes["units"] = [axes["units"][i] for i in int_order_indices]
    axes["scaling"] = [axes["scaling"][i] for i in int_order_indices]

    if intensity_image is not None:
        axes["raw_indices"] = [int(axes["raw_indices"][i]) for i in int_order_indices]
        raw_indices = axes["raw_indices"]
        if raw_indices != sorted(raw_indices):
            intensity_image = np.transpose(intensity_image, raw_indices)
        if "channel" in axes["dimensions"]:
            n_channels = intensity_image.shape[0]
    if segmentation_image is not None:
        axes["seg_indices"] = [int(axes["seg_indices"][i]) for i in seg_order_indices]
        seg_indices = axes["seg_indices"]
        if seg_indices != sorted(seg_indices):
            segmentation_image = np.transpose(segmentation_image, seg_indices)

    # when loading tracks from csv, we need the mapping to the seg_ids before
    # constructing the fpds.
    create_graph_from_df = False
    seg_id_map = None

    if tracks_path is not None:
        # Read the dataframe, and remap the column headers
        scaling_dict = dict(zip(axes["dimensions"], axes["scaling"], strict=False))
        df = read_tracks_df(
            tracks_path, column_mapping, convert_pixel_units, scaling_dict
        )

        # check if the provided segmentation matches with the dataframe (dimensions
        # and seg value of the first object). Raises DialogValueError if any problems are
        # found.
        if segmentation_image is not None:
            test_df_seg_match(
                df,
                segmentation_image,
                scaling_dict,
                mapping=column_mapping,
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
        axes,
        data_type,
        seg_id_map,
    )

    # construct a graph from the csv data, if provided.
    if create_graph_from_df:
        nxgraph = graph_from_df(
            df, segmentation_image, intensity_image, scaling_dict, features
        )
        seg = segmentation_image is not None
        feature_set = FeatureSet(ndim=ndim, seg=seg, pos_attr="pos", time_attr="t")
        for feature in features:
            if isinstance(feature["feature"], Intensity):
                feature["feature"].value_names = (
                    "Intensity"
                    if n_channels == 1
                    else [f"Intensity_chan{chan}" for chan in range(n_channels)]
                )
            feature_set.add_feature(feature["feature"])  # add the Feature instance from
            # the feature dict to the feature_set

        tracking_graph = TrackingGraph(NxGraph, nxgraph, feature_set)
        cand_graph = CandGraph.from_tracking_graph(tracking_graph, cand_graph_params)
    # Create a candidate graph with only nodes when tracking based on point detections
    # without csv.
    elif data_type == "points":
        points_data = project_info.get("points_data")
        nxgraph = graph_from_points(points_data, axes["points_columns"])
        feature_set = FeatureSet(ndim=ndim, seg=False, pos_attr="pos", time_attr="t")
        tracking_graph = TrackingGraph(NxGraph, nxgraph, feature_set)
        cand_graph = CandGraph.from_tracking_graph(tracking_graph, cand_graph_params)
    else:
        cand_graph = None

    # # TODO: include features to measure, ndim, cand_graph_params, point detections
    return Project(
        name=name,
        project_params=project_params,
        raw=intensity_fpds,
        segmentation=segmentation_fdps,
        cand_graph=cand_graph,
    )


def read_tracks_df(
    tracks_path: str,
    column_mapping: dict[str:str],
    convert_pixel_units: bool,
    scaling: dict[str:float],
) -> pd.DataFrame:
    """Reads and verifies the provided csv file. Checks if the 'id' column is unique,
    raises a DialogValueError if not, and and ensures it consists of integers.
      args:
       tracks_path (str): path to the csv file.
       column_mapping (dict[str: str]): mapping of required column names to the column
        selected by the user.
     Returns:
        df (pd.DataFrame): dataframe holding the tracks data.
    """

    df = pd.read_csv(tracks_path)

    # remap based on column mapping provided by the user
    for new_col, old_col in column_mapping.items():
        df[new_col] = df[old_col]

    # check whether the coordinates are in pixel units and should be rescaled to world
    # coordinates
    if convert_pixel_units:
        for axis in ["z", "y", "x"]:
            if axis in df.columns:
                df[axis] = pd.to_numeric(df[axis], errors="coerce")  # convert to float
                df[axis] = df[axis] * scaling[axis]

    # check that the ids provided in the csv are indeed unique
    if not df["id"].is_unique:
        raise DialogValueError(
            f"The object id values in column {column_mapping['id']} "
            "of the provided CSV file are not unique. Please provide a csv file where"
            "each object has a unique id value."
        )

    # check that the id column contains integer values, if not relabel them.
    if not pd.api.types.is_integer_dtype(df["id"]):
        all_labels = pd.unique(df[["id"]].values.ravel())
        label_to_int = {label: idx + 1 for idx, label in enumerate(all_labels)}
        df["id"] = df["id"].map(label_to_int)
        df["parent_id"] = df["parent_id"].map(label_to_int).astype("Int64")

    return df


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

    dimensions = axes.get("dimensions")
    axis_names = axes.get("axis_names")
    voxel_size = axes.get("scaling")
    axis_units = axes.get("units")
    ndim = (
        len(intensity_image.shape)
        if intensity_image is not None
        else len(segmentation_image.shape)
    )

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
            dimensions=dimensions,
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
            dimensions=(d for d in dimensions if d != "channel"),
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
            fpds[tuple(slc)] = image[tuple(slc)].compute()

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
    scaling_dict: dict[str:float],
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

    scale = [scaling_dict[dim] for dim in scaling_dict if dim != "channel"]

    if segmentation.ndim != len(scale):
        raise DialogValueError(
            f"Dimensions of the segmentation image ({segmentation.ndim}) "
            f"do not match the number of scale values given ({len(scale)})",
            show_dialog=True,
        )

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
    except IndexError as err:
        raise DialogValueError(
            f"Could not get the segmentation value at coordinates "
            f"{coordinates}. Segmentation data has shape "
            f"{segmentation.shape}. Please check if the axis order you"
            f" provided is correct.",
            show_dialog=True,
        ) from err

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
    for _id, row in points_data.iterrows():
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
        if not all(isinstance(v, int | float | np.integer | np.floating) for v in values):
            raise DialogValueError(
                f"Non-numerical or missing value found in position columns at row {_id}: "
                f"{values}",
                show_dialog=True,
            )

        attrs = {
            "t": int(t),
            "pos": pos,
        }

        graph.add_node(_id + 1, **attrs)

    return graph


def graph_from_df(
    df: pd.DataFrame,
    segmentation: fp.Array | None,
    intensity_image: fp.Array | None,
    scaling_dict: dict[str:float],
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

        scaling = [scaling_dict[dim] for dim in scaling_dict if dim in ("z", "y", "x")]

        if len(features_to_recompute) > 0:
            t = int(row["t"])
            if intensity_image is not None:
                if len(intensity_image.shape) > len(segmentation.shape):
                    # intensity image has channels (should always be at index 0)
                    slc = [slice(None)] * intensity_image.ndim
                    slc[1] = t
                    intensity = intensity_image[tuple(slc)].compute()
                    # skimage.measure.regionprops wants the channel axis to be last,
                    # so we need to transpose again
                    indices = list(range(len(intensity.shape)))
                    indices.append(indices.pop(0))  # move the channel from the first to
                    # the last position
                    intensity = np.transpose(intensity, indices)
                else:
                    intensity = intensity_image[t].compute()
            else:
                intensity = None
            # compute the feature
            props = regionprops(
                (segmentation[t].compute() == _id).astype(np.uint8),
                intensity_image=intensity,
                spacing=scaling,
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
