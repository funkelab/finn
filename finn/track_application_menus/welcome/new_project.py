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
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import regionprops
from tqdm import tqdm

from finn.track_application_menus.welcome.new_project_pages import (
    Page1,
    Page2,
    Page3,
    Page4,
    Page5,
    Page6,
    Page7,
    Page8,
)


class DialogValueError(ValueError):
    def __init__(self, message, show_dialog=True):
        super().__init__(message)
        self.show_dialog = show_dialog


class NewProjectDialog(QDialog):
    """Dialog to create a new project"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.resize(600, 500)
        self.stacked = QStackedWidget(self)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stacked)

        # --- Page 1: Project goals --- #
        self.page1 = Page1()
        layout1 = QVBoxLayout()
        layout1.addWidget(self.page1)

        # Page 1: Add Next/Cancel button
        btn_layout1 = QHBoxLayout()
        btn_layout1.addStretch()
        self.cancel_btn1 = QPushButton("Cancel")
        self.next_btn1 = QPushButton("Next")
        self.next_btn1.setEnabled(True)
        btn_layout1.addWidget(self.cancel_btn1)
        btn_layout1.addWidget(self.next_btn1)
        layout1.addLayout(btn_layout1)

        page1_widget = QWidget()
        page1_widget.setLayout(layout1)
        self.stacked.addWidget(page1_widget)

        # --- Page 2: Intensity Data Selection --- #
        self.page2 = Page2(page1=self.page1)
        self.page2.validity_changed.connect(self._validate_page2)
        layout2 = QVBoxLayout()
        layout2.addWidget(self.page2)

        # Add Prev/Next/Cancel buttons
        btn_layout2 = QHBoxLayout()
        self.prev_btn1 = QPushButton("Previous")
        self.cancel_btn2 = QPushButton("Cancel")
        self.next_btn2 = QPushButton("Next")
        btn_layout2.addStretch()
        btn_layout2.addWidget(self.cancel_btn2)
        btn_layout2.addWidget(self.prev_btn1)
        btn_layout2.addWidget(self.next_btn2)
        layout2.addLayout(btn_layout2)

        page2_widget = QWidget()
        page2_widget.setLayout(layout2)
        self.stacked.addWidget(page2_widget)

        # --- Page 3: Detection data selection --- #
        self.page3 = Page3(page1=self.page1, page2=self.page2)
        self.page3.validity_changed.connect(self._validate_page3)

        layout3 = QVBoxLayout()
        layout3.addWidget(self.page3)

        # Add Prev/Next/Cancel buttons
        btn_layout3 = QHBoxLayout()
        self.prev_btn2 = QPushButton("Previous")
        self.cancel_btn3 = QPushButton("Cancel")
        self.next_btn3 = QPushButton("Next")
        btn_layout3.addStretch()
        btn_layout3.addWidget(self.cancel_btn3)
        btn_layout3.addWidget(self.prev_btn2)
        btn_layout3.addWidget(self.next_btn3)
        layout3.addLayout(btn_layout3)

        page3_widget = QWidget()
        page3_widget.setLayout(layout3)
        self.stacked.addWidget(page3_widget)

        # --- Page 4: Image data dimensions --- #
        self.page4 = Page4(self.page2, self.page3)
        self.page4.validity_changed.connect(self._validate_page4)

        layout4 = QVBoxLayout()
        layout4.addWidget(self.page4)

        # Add Prev/Next/Cancel buttons
        btn_layout4 = QHBoxLayout()
        self.prev_btn3 = QPushButton("Previous")
        self.next_btn4 = QPushButton("Next")
        self.cancel_btn4 = QPushButton("Cancel")

        btn_layout4.addStretch()
        btn_layout4.addWidget(self.cancel_btn4)
        btn_layout4.addWidget(self.prev_btn3)
        btn_layout4.addWidget(self.next_btn4)
        layout4.addLayout(btn_layout4)

        page4_widget = QWidget()
        page4_widget.setLayout(layout4)
        self.stacked.addWidget(page4_widget)

        # --- Page 5: Importing external tracks from csv --- #
        self.page5 = Page5(self.page3, self.page4)
        self.page5.validity_changed.connect(self._validate_page5)

        layout5 = QVBoxLayout()
        layout5.addWidget(self.page5)

        # Add Prev/Next/Cancel buttons
        btn_layout5 = QHBoxLayout()
        self.prev_btn4 = QPushButton("Previous")
        self.next_btn5 = QPushButton("Next")
        self.cancel_btn5 = QPushButton("Cancel")

        btn_layout5.addStretch()
        btn_layout5.addWidget(self.cancel_btn5)
        btn_layout5.addWidget(self.prev_btn4)
        btn_layout5.addWidget(self.next_btn5)
        layout5.addLayout(btn_layout5)

        page5_widget = QWidget()
        page5_widget.setLayout(layout5)
        self.stacked.addWidget(page5_widget)

        # --- Page 6: Project and candidate graph parameters --- #
        self.page6 = Page6()

        layout6 = QVBoxLayout()
        layout6.addWidget(self.page6)

        # Add Prev/Next/Cancel buttons
        btn_layout6 = QHBoxLayout()
        self.prev_btn5 = QPushButton("Previous")
        self.next_btn6 = QPushButton("Next")
        self.cancel_btn6 = QPushButton("Cancel")

        btn_layout6.addStretch()
        btn_layout6.addWidget(self.cancel_btn6)
        btn_layout6.addWidget(self.prev_btn5)
        btn_layout6.addWidget(self.next_btn6)
        layout6.addLayout(btn_layout6)

        page6_widget = QWidget()
        page6_widget.setLayout(layout6)
        self.stacked.addWidget(page6_widget)

        # --- Page 7: Features to measure --- #
        self.page7 = Page7(
            page1=self.page1,
            page2=self.page2,
            page3=self.page3,
            page4=self.page4,
            page5=self.page5,
        )
        self.page7.validity_changed.connect(self._validate_page7)
        layout7 = QVBoxLayout()
        layout7.addWidget(self.page7)

        # Add Prev/Next/Cancel buttons
        btn_layout7 = QHBoxLayout()
        self.prev_btn6 = QPushButton("Previous")
        self.next_btn7 = QPushButton("Next")
        self.cancel_btn7 = QPushButton("Cancel")

        btn_layout7.addStretch()
        btn_layout7.addWidget(self.cancel_btn7)
        btn_layout7.addWidget(self.prev_btn6)
        btn_layout7.addWidget(self.next_btn7)
        layout7.addLayout(btn_layout7)

        page7_widget = QWidget()
        page7_widget.setLayout(layout7)

        self.stacked.addWidget(page7_widget)

        # --- Page 8: Save Project --- #
        self.page8 = Page8()
        layout8 = QVBoxLayout()
        layout8.addWidget(self.page8)
        self.page8.validity_changed.connect(self._validate_page8)

        # Add Prev/Ok/Cancel buttons
        btn_layout8 = QHBoxLayout()
        self.prev_btn7 = QPushButton("Previous")
        self.ok_btn = QPushButton("OK")
        self.cancel_btn8 = QPushButton("Cancel")

        btn_layout8.addStretch()
        btn_layout8.addWidget(self.cancel_btn8)
        btn_layout8.addWidget(self.prev_btn7)
        btn_layout8.addWidget(self.ok_btn)
        layout8.addLayout(btn_layout8)

        page8_widget = QWidget()
        page8_widget.setLayout(layout8)
        self.stacked.addWidget(page8_widget)

        # --- Navigation --- #
        self.next_btn1.clicked.connect(self._go_to_page2)
        self.next_btn2.clicked.connect(self._go_to_page3)
        self.next_btn3.clicked.connect(self._go_to_page4)
        self.next_btn4.clicked.connect(self._go_to_page5_or_6)
        self.next_btn5.clicked.connect(lambda: self.stacked.setCurrentIndex(5))
        self.next_btn6.clicked.connect(self._go_to_page7)
        self.next_btn7.clicked.connect(self._go_to_page8)

        self.prev_btn1.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.prev_btn2.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        self.prev_btn3.clicked.connect(lambda: self.stacked.setCurrentIndex(2))
        self.prev_btn4.clicked.connect(lambda: self.stacked.setCurrentIndex(3))
        self.prev_btn5.clicked.connect(self._go_to_page4_or_5)
        self.prev_btn6.clicked.connect(lambda: self.stacked.setCurrentIndex(5))
        self.prev_btn7.clicked.connect(lambda: self.stacked.setCurrentIndex(6))

        self.ok_btn.clicked.connect(self._on_ok_clicked)

        # Connect cancel buttons to close the dialog
        self.cancel_btn1.clicked.connect(self._cancel)
        self.cancel_btn2.clicked.connect(self._cancel)
        self.cancel_btn3.clicked.connect(self._cancel)
        self.cancel_btn4.clicked.connect(self._cancel)
        self.cancel_btn5.clicked.connect(self._cancel)
        self.cancel_btn6.clicked.connect(self._cancel)
        self.cancel_btn7.clicked.connect(self._cancel)
        self.cancel_btn8.clicked.connect(self._cancel)

        self.stacked.setCurrentIndex(0)

    def _validate_page2(self):
        """Validate inputs on page 2 and enable/disable the NEXT button to page3."""
        self.next_btn2.setEnabled(self.page2.is_valid)

    def _validate_page3(self):
        """Validate inputs on page 3 and enable/disable the NEXT button to page4."""
        self.next_btn3.setEnabled(self.page3.is_valid)

    def _validate_page4(self):
        """Validate inputs on page 4 and enable/disable the NEXT button to page5."""
        self.next_btn4.setEnabled(self.page4.is_valid)

    def _validate_page5(self):
        """Validate inputs on page 5 and enable/disable the NEXT button to page6."""
        self.next_btn5.setEnabled(self.page5.is_valid)

    def _validate_page7(self):
        """Validate inputs on page 7 and enable/disable the OK button."""
        self.next_btn7.setEnabled(self.page7.is_valid)

    def _validate_page8(self):
        """Validate inputs on page 8 and enable/disable the OK button."""
        self.ok_btn.setEnabled(self.page8.is_valid)

    def _go_to_page2(self):
        """Go to page 2 and validate it to enable/disable the NEXT button."""
        self.stacked.setCurrentIndex(1)
        self.page2.validate()

    def _go_to_page3(self):
        """Go to page 3 and validate it to enable/disable the NEXT button."""
        self.stacked.setCurrentIndex(2)
        self.page3.validate()

    def _go_to_page4(self):
        """Go to page 4 and validate it to enable/disable the NEXT button."""
        self.stacked.setCurrentIndex(3)
        self.page4.update_table()

    def _go_to_page4_or_5(self):
        """Go to page 5 to track from scratch, otherwise go to page 4."""
        if self.page1.get_choice() == "curate_tracks":
            self.stacked.setCurrentIndex(4)
        else:
            self.stacked.setCurrentIndex(3)

    def _go_to_page5_or_6(self):
        """Go to page 5 to use external tracks, otherwise go to page 6."""
        if self.page1.get_choice() == "curate_tracks":
            self.stacked.setCurrentIndex(4)
            self.page5.validate()
        else:
            self.stacked.setCurrentIndex(5)

    def _go_to_page7(self):
        """Go to page 7 and make sure the mapping on page5 is updated first."""
        self.page5.mapping_updated.emit()  # ensure the mapping is updated
        self.stacked.setCurrentIndex(6)

    def _go_to_page8(self):
        """Go to page 8 and validate it to update the button"""
        self.page8.validate()
        self.stacked.setCurrentIndex(7)

    def _on_ok_clicked(self):
        """Collect all information entered by the user and try to build a project. Throws
        an error if invalid/incompatible information was entered."""
        try:
            self.project = self.create_project()
        except DialogValueError as e:
            if e.show_dialog:
                QMessageBox.warning(self, "Error", str(e))
            return
        self.accept()

    def _get_project_info(self) -> dict[str:Any]:
        """Collect the information from the different dialog pages.
        Returns:
            dict[str: Any] with information from the different pages as follows:
            Page 3:
                - data_type [str]: either 'segmentation' or 'points'
            Page 4:
                - intensity_image [da.Array | None] : intensity data
                - segmentation_image [da.Array | None] : segmentation data
                - points_data [pd.DataFrame]: point detection data
                - ndim [int]: the number of dimensions (incl time) of the data (3, 4, or
                    5)
                - axes [dict]:
                    dimensions [tuple[str]]: dimension names (e.g. 'time', 'z')
                    raw_indices [tuple[int]]: index of each dimension in the raw data
                    seg_indices [tuple[int]]: index of each dimension in the seg data
                    axis_names [tuple(str)]: dimension names assigned by the user
                    units (tuple[str]): units for each dimension, e.g. 'µm'
                    scaling [tuple(float)]: spatial calibration in the same order as the
                        dimensions
            Page 5:
                - tracks_path [str | None]: path to where the tracking data csv file is
                    stored (if provided)
                - tracks_mapping [dict[str: str] | None] : mapping of the csv column headers to
                    the required tracking information (dimensions, ids)
            Page 6:
                - project_params [ProjectParams]: parameters for the project
                - cand_graph_params [CandGraphParams]: parameters for the candidate graph
            Page 7:
                - features (list[dict[str: str|bool]]): list of features to measure,
                    each with 'feature_name', 'include' (bool), and 'from_column'
                    (str or None)
            Page 8:
                - title [str]: name of the project,
                - directory [str]: path to directory where the project should be saved
        """

        choice = self.page1.get_choice()

        project_info = {
            "tracks_path": None,
            "column_mapping": None,
        }

        project_info = project_info | self.page3.get_settings()
        project_info = project_info | self.page4.get_settings()
        if choice == "curate_tracks":
            project_info = project_info | self.page5.get_settings()
        project_info = project_info | self.page6.get_settings()
        project_info = project_info | self.page7.get_settings()
        project_info = project_info | self.page8.get_settings()

        return project_info

    def _cancel(self):
        """Reject the dialog, go back to welcome menu"""
        self.reject()

    def create_empty_fp_array(
        self, fp_array_path: str, shape: tuple, axes: dict | None = None
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
        self,
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

        # check if segmentation image has integer data type, warn user if not.
        if segmentation_image is not None and np.issubdtype(
            segmentation_image.dtype, np.floating
        ):
            msg = QMessageBox(self)
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
                segmentation_fpds = self.create_empty_fp_array(
                    os.path.join(fp_array_path, "seg"),
                    shape=intensity_image.shape[-(ndim - 1) :],
                    axes=axes,
                )
            else:
                segmentation_fpds = self.create_empty_fp_array(
                    os.path.join(fp_array_path, "seg"),
                    shape=intensity_image.shape,
                    axes=axes,
                )

        # Create fpds for intensity and/or segmentation image
        if intensity_image is not None:
            intensity_fpds = self.create_fp_array(
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

            segmentation_fpds = self.create_fp_array(
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
        self,
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
                correct_duplicate_ids = self._has_duplicate_ids(image)
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

    def create_project(self) -> Project:
        """Creates a new funtracks project with the information provided in the dialog"""

        project_info = self._get_project_info()

        print(project_info)
        intensity_image = project_info["intensity_image"]
        segmentation_image = project_info["segmentation_image"]
        name = project_info.get("title", "Untitled Project")
        ndim = int(project_info.get("ndim", 3))
        axes = project_info.get("axes", [])
        data_type = project_info.get("data_type", "points")
        working_dir = project_info.get("directory", Path.cwd())
        params = project_info.get("project_params", None)
        features = project_info.get("features")

        # remove old zarr dir if present
        zarr_dir = os.path.join(project_info.get("directory"), f"{name}.zarr")
        if os.path.exists(zarr_dir):
            shutil.rmtree(zarr_dir)

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
        intensity_fpds, segmentation_fdps = self.create_fpds(
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
            points_data = project_info.get("points_data", None)
            cand_graph = graph_from_points(points_data, axes["points_columns"])

        # # TODO: include features to measure, ndim, cand_graph_params, point detections
        return Project(
            name=name,
            project_params=params,
            raw=intensity_fpds,
            segmentation=segmentation_fdps,
            cand_graph=cand_graph,
        )

    @staticmethod
    def _has_duplicate_ids(segmentation: np.ndarray) -> bool:
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
    required_columns = ["id", "t", "y", "x", "parent_id"]

    graph = nx.DiGraph()
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        _id = int(row["id"])
        parent_id = row["parent_id"]
        if "z" in df.columns:
            pos = [row["z"], row["y"], row["x"]]
            required_columns.append("z")
        else:
            pos = [row["y"], row["x"]]

        attrs = {
            "t": int(row["t"]),
            "pos": pos,
        }

        # add all other columns into the attributes
        for attr in required_columns:
            del row_dict[attr]
        attrs.update(row_dict)

        if "track_id" in df.columns:
            attrs["track_id"] = int(row["track_id"])

        print("these are the features", features)
        for f in features:
            print("from column", f["from_column"])

        # add additional features from the table, if requested, or recompute them
        features_to_recompute = [
            f["feature"]
            for f in features
            if f["include"]
            and f["feature"].computed
            and f["from_column"] is None
            and f["feature"].regionprops_name is not None
        ]

        print("recompute these features", features_to_recompute)

        # import features directly from the table, without recomputing
        features_to_import_from_df = [
            f for f in features if f["include"] and f["from_column"] is not None
        ]

        print("import these features", features_to_import_from_df)

        for feature in features_to_import_from_df:
            print(feature["from_column"])
            print(feature["feature"].attr_name)
            attrs[feature["feature"].attr_name] = row.get(feature["from_column"])
            print(attrs)
            print(row.get(feature["from_column"]))
            print(row.get("surface_area"))

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

        # add the node to the graph
        print(attrs)
        return None
        graph.add_node(_id, **attrs)

        # add the edge to the graph, if the node has a parent
        # note: this loading format does not support edge attributes
        if not pd.isna(parent_id) and parent_id != -1:
            assert parent_id in graph.nodes, (
                f"Parent id {parent_id} of node {_id} not in graph yet"
            )
            graph.add_edge(parent_id, _id)

    return graph
