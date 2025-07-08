import os
import shutil
from pathlib import Path
from typing import Any

import dask.array as da
import funlib.persistence as fp
import numpy as np
import pandas as pd
import zarr
from funtracks.cand_graph_utils import (
    graph_from_df,
    graph_from_points,
    graph_from_segmentation,
)
from funtracks.features.feature_set import FeatureSet
from funtracks.project import Project
from qtpy.QtWidgets import (
    QMessageBox,
)
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
        - axes [dict[str: dict[str: str|int|float]]]:
            with each dimension ((channel), t, (z), y, x) as key:
                - axis_name (str), e.g. 'time', 'z'
                - unit (str), e.g. 'µm'
                - step_size (float), spatial calibration
                - raw_index (int), index of dimension in raw data
                - seg_index (int), index of dimension in seg data
                - column (str), column in csv data that maps to this dimension
                - size (int), size of dimension in the dataset
        - tracks_path [str | None]: path to where the tracking data csv file is
            stored (if provided)
        - column_mapping [dict[str: str] | None] : mapping of the csv column
            headers to the required tracking information (dimensions, ids)
        - convert_pixel_units (bool): whether the coordinates in the csv are still in
            pixel units and need to be remapped.
        - project_params [ProjectParams]: parameters for the project
        - cand_graph_params [CandGraphParams]: parameters for the candidate graph
        - features (list[dict[str: str|bool]]): list of features to measure,
                each with 'feature' (Feature), 'include' (bool), and 'from_column'
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
    except KeyError as err:
        missing_key = err.args[0]
        raise DialogValueError(
            f"The following key is missing: {missing_key}",
            show_dialog=True,
        ) from err

    # -------------- Checks -------------#

    # Check if at least intensity image data or segmentation data is provided
    if intensity_image is None and segmentation_image is None:
        # this situation is invalid, if no seg is provided we need at least intensity
        #  image to track from scratch
        raise DialogValueError(
            "No valid path to intensity data and segmentation labels was provided. "
            "We need at least an intensity image to track from scratch"
        )

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

    # Check whether the data needs to be transposed to match the default_order
    if intensity_image is not None:
        default_order = ("channel", "time", "z", "y", "x")
        indices = [axes[dim]["raw_index"] for dim in default_order if dim in axes]
        if indices != sorted(indices):
            # Transpose the image
            intensity_image = np.transpose(intensity_image, indices)
            # Update the indices in the axes dictionary
            filtered_dims = [dim for dim in default_order if dim in axes]
            for dim in filtered_dims:
                axes[dim]["raw_index"] = filtered_dims.index(dim)

    if segmentation_image is not None:
        default_order = ("time", "z", "y", "x")
        indices = [axes[dim]["seg_index"] for dim in default_order if dim in axes]
        if indices != sorted(indices):
            # Transpose the image
            segmentation_image = np.transpose(segmentation_image, indices)
            # Update the indices in the axes dictionary
            filtered_dims = [dim for dim in default_order if dim in axes]
            for dim in filtered_dims:
                axes[dim]["seg_index"] = filtered_dims.index(dim)

    # Verify that the shapes of the intensity and segmentation data match
    if intensity_image is not None and segmentation_image is not None:
        if len(intensity_image.shape) == len(segmentation_image.shape) + 1:
            valid = intensity_image.shape[1:] == segmentation_image.shape
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

    # ------------- Create fpds ------------ #

    # when loading tracks from csv, we need the mapping to the seg_ids before
    # constructing the fpds.
    create_graph_from_df = False
    seg_id_map = None

    if tracks_path is not None:
        # Read the dataframe, check if it is valid, and remap the column headers
        df = read_tracks_df(tracks_path, column_mapping, convert_pixel_units, axes)

        # check if the provided segmentation matches with the dataframe (dimensions
        # and seg value of the first object). Raises DialogValueError if any problems are
        # found.
        if segmentation_image is not None:
            check_df_seg_match(
                df,
                segmentation_image,
                axes,
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

    # Create (empty) fpds arrays for the intensity and segmentation data (if provided).
    # Intensity fdps should be in a list, one per channel.
    intensity_fpds = None
    segmentation_fpds = None
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    if intensity_image is not None:
        intensity_fpds = []

        # initialize 'raw' first, to ensure it has a .zgroup file
        zroot = zarr.open(working_dir, mode="a")
        if "raw" not in zroot:
            zroot.create_group("raw")

        if "channel" in axes:
            n_channels = axes["channel"]["size"]
            for chan in range(n_channels):
                fpds = create_fp_array(
                    os.path.join(working_dir, f"raw/chan_{chan}"),
                    intensity_image[chan],
                    axes=axes,
                    dtype=intensity_image.dtype,
                )
                intensity_fpds.append(fpds)
        else:
            fpds = create_fp_array(
                os.path.join(working_dir, "raw/chan_0"),
                intensity_image,
                axes=axes,
                dtype=intensity_image.dtype,
            )
            intensity_fpds.append(fpds)

    if data_type == "segmentation":
        segmentation_fpds = create_fp_array(
            os.path.join(working_dir, "seg"),
            segmentation_image,
            axes=axes,
            dtype=segmentation_image.dtype
            if segmentation_image is not None
            else np.uint32,
            seg_id_map=seg_id_map,
        )

    # ----------- Construct Candidate graph ---------- #

    # construct a graph from the csv data, if provided.
    scaling = [axes[dim]["step_size"] for dim in axes if dim in ("z", "y", "x")]
    n_channels = axes["channel"]["size"] if "channel" in axes else 1
    if create_graph_from_df:
        cand_graph = graph_from_df(
            df,
            segmentation_fpds,
            intensity_fpds,
            ndim,
            n_channels,
            scaling,
            features,
            cand_graph_params,
        )

    # Create a candidate graph with only nodes when tracking based on point detections
    # without csv.
    elif data_type == "points":
        points_data = project_info.get("points_data")
        column_mapping = {}
        for dim in ("time", "z", "y", "x"):
            if dim in axes:
                column_mapping[dim] = axes[dim]["column"]

        # check that all values are numerical
        for col in column_mapping.values():
            # Try to convert to numeric, NaNs will be introduced for non-numeric values
            if pd.to_numeric(df[col], errors="coerce").isnull().any():
                raise DialogValueError(
                    f"Non-numerical or missing values found in column: {col}",
                    show_dialog=True,
                )

        feature_set = FeatureSet(ndim=ndim, seg=False, pos_attr="pos", time_attr="t")
        cand_graph = graph_from_points(
            points_data, column_mapping, feature_set, cand_graph_params
        )
        cand_graph.initialize_cand_edges()

    else:
        cand_graph = graph_from_segmentation(
            segmentation_fpds,
            intensity_fpds,
            ndim,
            n_channels,
            scaling,
            features,
            cand_graph_params,
        )
        cand_graph.initialize_cand_edges()

    # ----------- Return Project ---------#

    return Project(
        name=name,
        project_params=project_params,
        raw=intensity_fpds,
        segmentation=segmentation_fpds,
        cand_graph=cand_graph,
        zarr_path=Path(working_dir),
    )


def read_tracks_df(
    tracks_path: str,
    column_mapping: dict[str:str],
    convert_pixel_units: bool,
    axes: dict[dict[str : int | float | str]],
) -> pd.DataFrame:
    """Reads and verifies the provided csv file. Relabels the column headers according to
    the mapping provided by the user. Checks if the 'id' column is unique, raises a
    DialogValueError if not, and and ensures it consists of integers. Checks that the time
     dimension consists of integers incrementing with step size of 1.
    Args:
        tracks_path (str): path to the csv file.
        column_mapping (dict[str: str]): mapping of required column names to the column
            selected by the user.
        convert_pixel_units (bool): whether the coordinates in csv file are in pixel units
            and therefore need to be scaled according to the scaling information in the
            axes dict.
        axes (dict[dict[str: str|int|float]]): axes dictionary holding all dimensional
            information in the dataset.
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
                df[axis] = df[axis] * axes[axis]["step_size"]

    # check that the ids provided in the csv are unique
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

    # Rescale the time points to ensure integers incrementing at steps of 1.
    if np.max(df["t"]) != axes["time"]["size"] - 1:
        unique_times = np.sort(df["t"].unique())
        time_mapping = {orig: new for new, orig in enumerate(unique_times)}
        df["t"] = df["t"].map(time_mapping).astype(int)

    return df


def create_fp_array(
    path: str,
    image: da.Array | None,
    axes: dict[dict[str : str | float | int]],
    dtype: np.dtype,
    seg_id_map: pd.DataFrame | None = None,
) -> fp.Array:
    """Creates a funlib persistence array from a given image.
    Args:
        path (str): Path where the data will be stored
        image (da.Array | None ): Dask array of data, or None if an empty one should be
        created
        axes (dict[dict[str: str|int|float]]): axes dictionary holding all dimensional
            information in the dataset.
        dtype: dtype of the data
        seg_id_map (pd.DataFrame, optional): mapping from 'seg_id + time' to 'id', in case
          the user provided external tracking data.

    Returns:
        fp.Array: A funlib persistence array containing the data."""

    default_dimensions = ("time", "z", "y", "x")

    dimensions = [dim for dim in default_dimensions if dim in axes]
    axis_names = [axes[dim]["axis_name"] for dim in dimensions]
    voxel_size = [
        1 if dim in ("channel", "t") else axes[dim]["step_size"] for dim in dimensions
    ]
    axis_units = [axes[dim]["unit"] for dim in dimensions]
    shape = [axes[dim]["size"] for dim in dimensions]

    fpds = fp.prepare_ds(
        path,
        shape=shape,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=axis_units,
        dtype=dtype,
    )

    # if there is segmentation data, check the following:
    #  1) was segmentation data provided, or do we need an empty array?
    #  2) is relabeling is necessary for one of the following reasons:
    #   - User provided external tracks data, and we need to map seg_id to id
    #   - The labels in the provided data are not unique across time.

    correct_duplicate_ids = None
    relabel_from_csv = None

    if path.endswith("seg"):
        if image is None:
            return fpds
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


def has_duplicate_ids(segmentation: da.Array | np.ndarray) -> bool:
    """Checks if the segmentation has duplicate label ids across time. For efficiency,
    only checks between the first and second time frames.

    Args:
        segmentation (da.Array | np.ndarray): (t, [z], y, x)

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


def check_df_seg_match(
    df: pd.DataFrame,
    segmentation: da.Array,
    axes: dict[dict[str : str | int | float]],
    mapping: dict,
) -> None:
    """Test if the provided segmentation, dataframe, and axes values are valid together.
      - The dataframe and the segmentation have the same number of dimensions (3 or 4: (t,
       (z), y, x).
      - The number of unique time points matches the length of the time dimension of the
      segmentation data.
      - The segmentation pixel value for the coordinates of first node corresponds with
        the provided seg_id as a basic check that the csv file matches with the
        segmentation file

    Args:
        df (pd.DataFrame): the pandas dataframe to turn into tracks, with standardized
            column names
        segmentation (da.Array): The segmentation, a 3D or 4D array of integer labels
        axes (dict[dict[str: str|int|float]]): axes dictionary holding all dimensional
            information in the dataset.
    """

    row = df.iloc[-1]

    # extract position and scale, time step should be 1.
    if "z" in df.columns:
        pos = [row["t"], row["z"], row["y"], row["x"]]
        scale = [
            1,
            axes["z"]["step_size"],
            axes["y"]["step_size"],
            axes["x"]["step_size"],
        ]
    else:
        pos = [row["t"], row["y"], row["x"]]
        scale = [1, axes["y"]["step_size"], axes["x"]["step_size"]]

    if segmentation.ndim != len(pos):
        raise DialogValueError(
            f"Dimensions of the segmentation ({segmentation.ndim}) do not match the "
            f"number of positional dimensions ({len(pos)})",
            show_dialog=True,
        )

    if segmentation.shape[0] != len(df["t"].unique()):
        raise DialogValueError(
            f"The time dimension of the provided data frame does not match the"
            f"segmentation data. The segmentation data has {segmentation.shape[0]} time"
            f"points, but the dataframe has {len(df['t'].unique())} time points with a "
            f"maximum value of {np.max(df['t'])}. Please make sure that you provide a "
            f"dataframe with integer time points incrementing with step size 1.",
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
