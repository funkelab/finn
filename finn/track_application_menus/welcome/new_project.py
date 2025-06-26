import os
import shutil
from pathlib import Path
from typing import Any

import dask.array as da
import funlib.persistence as fp
import numpy as np
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.resize(600, 500)
        self.stacked = QStackedWidget(self)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stacked)

        # --- Page 1: Project Info ---
        self.page1 = Page1()
        layout1 = QVBoxLayout()
        layout1.addWidget(self.page1)

        # Page 1: Add Next/Cancel button
        btn_layout1 = QHBoxLayout()
        btn_layout1.addStretch()
        self.cancel_btn1 = QPushButton("Cancel")
        self.next_btn = QPushButton("Next")
        self.next_btn.setEnabled(True)
        btn_layout1.addWidget(self.cancel_btn1)
        btn_layout1.addWidget(self.next_btn)
        layout1.addLayout(btn_layout1)

        page1_widget = QWidget()
        page1_widget.setLayout(layout1)
        self.stacked.addWidget(page1_widget)

        # --- Page 2: Intensity Data Selection ---
        self.page2 = Page2(page1=self.page1)
        self.page2.validity_changed.connect(self._validate_page2)
        layout2 = QVBoxLayout()
        layout2.addWidget(self.page2)

        # Add Prev/Next/Cancel buttons
        btn_layout2 = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.cancel_btn2 = QPushButton("Cancel")
        self.next_btn2 = QPushButton("Next")
        btn_layout2.addStretch()
        btn_layout2.addWidget(self.cancel_btn2)
        btn_layout2.addWidget(self.prev_btn)
        btn_layout2.addWidget(self.next_btn2)
        layout2.addLayout(btn_layout2)

        page2_widget = QWidget()
        page2_widget.setLayout(layout2)
        self.stacked.addWidget(page2_widget)

        # # --- Page 3: Detection data selection---
        self.page3 = Page3(page1=self.page1)
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

        # --- Page 4: Project info and dimensions ---
        self.page4 = Page4(self.page2, self.page3)
        self.page4.validity_changed.connect(self._validate_page4)

        layout4 = QVBoxLayout()
        layout4.addWidget(self.page4)

        # Add Prev/ONextk/Cancel buttons
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

        # --- Page 5: Importing external tracks from csv ---
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

        # --- Page 6: Project and candidate graph parameters ---
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

        # --- Page 7: Features to measure ---
        self.page7 = Page7(
            page1=self.page1,
            page2=self.page2,
            page3=self.page3,
            page4=self.page4,
            page5=self.page5,
        )
        layout7 = QVBoxLayout()
        layout7.addWidget(self.page7)

        # Add Prev/Ok/Cancel buttons
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

        # --- Page 8: Save Project ---
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

        # Connections for navigation
        self.next_btn.clicked.connect(self._go_to_page2)
        self.next_btn2.clicked.connect(self._go_to_page3)
        self.next_btn3.clicked.connect(self._go_to_page4)
        self.next_btn4.clicked.connect(self._go_to_page5_or_6)
        self.next_btn5.clicked.connect(lambda: self.stacked.setCurrentIndex(5))
        self.next_btn6.clicked.connect(self._go_to_page7)
        self.next_btn7.clicked.connect(self._go_to_page8)

        self.prev_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.prev_btn2.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        self.prev_btn3.clicked.connect(lambda: self.stacked.setCurrentIndex(2))
        self.prev_btn4.clicked.connect(self._go_to_page4)
        self.prev_btn5.clicked.connect(self._go_to_page4_or_5)
        self.prev_btn6.clicked.connect(lambda: self.stacked.setCurrentIndex(5))
        self.prev_btn7.clicked.connect(lambda: self.stacked.setCurrentIndex(6))

        self.ok_btn.clicked.connect(self.on_ok_clicked)

        self.stacked.setCurrentIndex(0)

        # Connect cancel buttons to close the dialog
        self.cancel_btn1.clicked.connect(self._cancel)
        self.cancel_btn2.clicked.connect(self._cancel)
        self.cancel_btn3.clicked.connect(self._cancel)
        self.cancel_btn4.clicked.connect(self._cancel)
        self.cancel_btn5.clicked.connect(self._cancel)
        self.cancel_btn6.clicked.connect(self._cancel)
        self.cancel_btn7.clicked.connect(self._cancel)
        self.cancel_btn8.clicked.connect(self._cancel)

    def _validate_page2(self):
        """Validate inputs on page 2 and enable/disable the NEXT button to page3."""
        self.next_btn2.setEnabled(self.page2.is_valid)

    def _validate_page3(self):
        """Validate inputs on page 3 and enable/disable the NEXT button to page4."""
        self.next_btn3.setEnabled(self.page3.is_valid)

    def _validate_page4(self):
        """Validate inputs on page 4 and enable/disable the NEXT button to page4."""
        self.next_btn4.setEnabled(self.page4.is_valid)

    def _validate_page5(self):
        """Validate inputs on page 5 and enable/disable the NEXT button to page5."""
        self.next_btn5.setEnabled(self.page5.is_valid)

    def _validate_page8(self):
        """Validate inputs on page 8 and enable/disable the OK button to page8."""
        print("validating page 8, button should be enabled", self.page8.is_valid)
        self.ok_btn.setEnabled(self.page8.is_valid)

    def _go_to_page2(self):
        self.stacked.setCurrentIndex(1)
        self.page2.validate()

    def _go_to_page3(self):
        self.stacked.setCurrentIndex(2)
        self.page3.validate()

    def _go_to_page4(self):
        self.stacked.setCurrentIndex(3)
        self.page4.validate()

    def _go_to_page4_or_5(self):
        """Go to page 5 if the user chose to track from scratch, otherwise go to page 5."""
        if self.page1.get_choice() == "curate_tracks":
            self.stacked.setCurrentIndex(4)
            self.page5.validate()
        else:
            self.stacked.setCurrentIndex(3)
            self.page4.validate()

    def _go_to_page5_or_6(self):
        """Go to page 5 if the user chose to use external tracks, otherwise go to page 6."""
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

    def on_ok_clicked(self):
        # Called when OK/Finish is clicked
        try:
            self.project = self.create_project()
        except DialogValueError as e:
            if e.show_dialog:
                QMessageBox.warning(self, "Error", str(e))
            return
        self.accept()

    def get_project_info(self) -> dict[str:Any]:
        """Create a dictionary with the information from the different dialog pages.
        Returns:
            dict[str: Any] with the following information:
            - title [str]: name of the project,
            - directory [str]: path to directory where the project should be saved
            - ndim [int]: the number of dimensions (incl time) of the data (3 or 4)
            - axes [dict]:
                dimensions [tuple[str]]: dimension names (e.g. 'time', 'z')
                indices [tuple[int]]: index of each dimension (e.g (0,1,2,3))
                axis_names [tuple(str)]: dimension names assigned by the user
                units (tuple[str]): units for each dimension, e.g. 'µm'
                scaling [tuple(float)]: spatial calibration in the same order as the dimensions
            - intensity_image [da.Array] | None : intensity data
            - segmentation_image [da.Array] | None: segmentation data
            - tracks_path [str | None]: path to where the tracking data csv file is stored (if provided)
            - tracks_mapping [dict[str: str]]: mapping of the csv column headers to the required tracking information (dimensions, ids)
            - data_type [str]: either 'segmentation' or 'points'
            - points_path [str | None]: path to the segmentation or poitns detection data, if provided.
            - features [list[dict[str: str|bool]]]: list of features to measure, each with 'feature_name', 'include' (bool), and 'from_column' (str or None)
            - project_params [ProjectParams]: parameters for the project
            - cand_graph_params [CandGraphParams]: parameters for the candidate graph
        """

        choice = self.page1.get_choice()
        project_info = {}
        data_type = self.page3.data_type
        project_info["data_type"] = data_type
        project_info["points_path"] = (
            self.page3.get_path() if data_type == "points" else None
        )

        project_info = (
            project_info | self.page4.get_settings()
        )  # intensity image, segmentation image, dimensions, axes

        project_info["tracks_path"] = (
            self.page5.get_tracks_path() if choice == "curate_tracks" else None
        )
        project_info["column_mapping"] = (
            self.page5.get_mapping() if choice == "curate_tracks" else None
        )
        project_info = project_info | self.page6.get_settings()
        project_info["features"] = self.page7.get_selected_features()
        project_info = project_info | self.page8.get_settings()  # name, directory

        return project_info

    def _cancel(self):
        self.reject()

    def create_empty_fp_array(
        self, fp_array_path: str, shape: tuple, axes: dict | None = None
    ) -> fp.Array:
        """Creates an empty funtracks persistence array with the specified shape and axes."""

        axis_names = axes.get("axis_names", ["axis_" + str(i) for i in range(len(shape))])
        voxel_size = axes.get("scaling", [1.0] * len(shape))
        axis_units = axes.get("units", ["px"] * len(shape))

        if "channel" in axes["dimensions"]:
            # remove the channel information, segmentation fpds can only have dimensions t(z)yx
            axis_names.pop(0)
            voxel_size.pop(0)
            axis_units.pop(0)

        print(
            f"creating empty fpds with shape {shape}, voxel_size {voxel_size}, axis_names {axis_names} units {axis_units}"
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
        data_type,
    ) -> fp.Array:
        """Creates a funtracks persistence array from an intensity image or segmentation data.
        Args:
            intensity_image (da.Array | None ): Dask array of intensity image
            segmentation_image (da.Array | None): Dask array of the segmentation data.
            fp_array_path (str): Path where the funtracks persistence array will be created.
            axes (dict): Dictionary containing axis information like indices, names, units, and scaling.
        Returns:
            fp.Array: A funtracks persistence array containing the data."""

        print("path to save to", fp_array_path)
        # Check if at least one of the two data paths is valid.
        if intensity_image is None and segmentation_image is None:
            # this situation is invalid, if no seg is provided we need at least intensity image to track from scratch
            raise DialogValueError(
                "No valid path to intensity data and segmentation labels was provided. We need at least an intensity image to track from scratch"
            )

        # check if segmentation image has integer data type, warn user if not.
        if segmentation_image is not None and np.issubdtype(
            segmentation_image.dtype, np.floating
        ):
            msg = QMessageBox(self)
            msg.setWindowTitle("Invalid segmentation file type")
            msg.setText(
                "The datatype of the provided segmentation data is float.<br><br>"
                "Click <b>Continue</b> if you are sure you selected the correct data and it will be converted to integers.<br>"
                "Click <b>Go Back</b> to return to the import menu."
            )
            msg.addButton("Continue", QMessageBox.AcceptRole)
            goback_btn = msg.addButton("Go Back", QMessageBox.RejectRole)
            msg.setDefaultButton(goback_btn)
            msg.exec_()
            if msg.clickedButton() == goback_btn:
                raise DialogValueError(
                    "Invalid segmentation file type, going back to select alternative data of type integer.",
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
            print(intensity_image.shape, segmentation_image.shape)
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

        print("save path:", os.path.join(fp_array_path, "raw"))

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
    ) -> fp.Array:
        print(path)
        print(shape)
        print(voxel_size)
        print(axis_names)
        print(axis_units)
        print(dtype)

        fpds = fp.prepare_ds(
            path,
            shape=shape,
            voxel_size=voxel_size,
            axis_names=axis_names,
            units=axis_units,
            dtype=dtype,
        )

        # if segmentation, do the relabeling like we do in the sample data
        if path.endswith("seg"):
            if self._has_duplicate_ids(image):
                image = ensure_unique_labels(image)

        time_index = list(dimensions).index("time")

        # load and write each time point into the dataset
        for time in tqdm(
            range(image.shape[time_index]), desc="Converting time points to zarr"
        ):
            slc = [slice(None)] * image.ndim
            slc[time_index] = time
            fpds[time] = image[tuple(slc)].compute()

        return fpds

    def create_project(self) -> Project:
        """Creates a new funtracks project with the information provided in the dialog"""

        project_info = self.get_project_info()
        intensity_image = project_info["intensity_image"]
        segmentation_image = project_info["segmentation_image"]
        name = project_info.get("title", "Untitled Project")
        ndim = int(project_info.get("ndim", 3))
        axes = project_info.get("axes", [])
        data_type = project_info.get("data_type", "points")
        working_dir = project_info.get("directory", Path.cwd())
        params = project_info.get("project_params", None)

        # remove old zarr dir if present
        zarr_dir = os.path.join(project_info.get("directory"), f"{name}.zarr")
        if os.path.exists(zarr_dir):
            shutil.rmtree(zarr_dir)

        # create fpds for the intensity image and segmentation data (if provided)
        print(project_info)

        intensity_fpds, segmentation_fdps = self.create_fpds(
            intensity_image,
            segmentation_image,
            os.path.join(working_dir, f"{name}.zarr"),
            ndim,
            axes,
            data_type,
        )

        # # TODO implement points logic
        if data_type == "points":
            points_path = project_info.get("detection_path", None)
            if points_path is None:
                raise ValueError(
                    "Points detection type selected, but no points file provided."
                )

        # # TODO: include features to measure, ndim, cand_graph_params, point detections
        return Project(
            name=name,
            project_params=params,
            raw=intensity_fpds,
            segmentation=segmentation_fdps,
            cand_graph=None,
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


def ensure_unique_labels(
    segmentation: np.ndarray | da.Array,
    multiseg: bool = False,
) -> np.ndarray:
    """Relabels the segmentation in place to ensure that label ids are unique across
    time. This means that every detection will have a unique label id.
    Useful for combining predictions made in each frame independently, or multiple
    segmentation outputs that repeat label IDs.

    Args:
        segmentation (np.ndarray | da.Array): Segmentation with dimensions ([h], t, [z], y, x).
        multiseg (bool, optional): Flag indicating if the segmentation contains
            multiple hypotheses in the first dimension. Defaults to False.
    """
    is_dask = isinstance(segmentation, da.Array)
    segmentation = segmentation.astype(np.uint64)
    orig_shape = segmentation.shape
    if multiseg:
        new_shape = (-1, *orig_shape[2:])
        segmentation = segmentation.reshape(new_shape)
    curr_max = 0
    frames = []
    for idx in range(segmentation.shape[0]):
        frame = segmentation[idx]
        if is_dask:
            mask = frame != 0
            frame = da.where(mask, frame + curr_max, frame)
            curr_max = int(da.max(frame).compute())
        else:
            mask = frame != 0
            frame[mask] += curr_max
            curr_max = int(np.max(frame))
        frames.append(frame)
    if is_dask:
        segmentation = da.stack(frames, axis=0)
        if multiseg:
            segmentation = segmentation.reshape(orig_shape)
    else:
        segmentation = np.stack(frames, axis=0)
        if multiseg:
            segmentation = segmentation.reshape(orig_shape)
    return segmentation
