import os
from pathlib import Path

import funlib.persistence as fp
import numpy as np
import tifffile
import zarr
from funtracks.project import Project
from motile_toolbox.utils.relabel_segmentation import ensure_unique_labels
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from finn.track_application_menus.browse_data import (
    CsvFileWidget,
    FileFolderDialog,
    SegmentationWidget,
)


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.resize(600, 400)
        self.stacked = QStackedWidget(self)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stacked)

        # --- Page 1: Project Info ---
        page1 = QWidget()
        layout1 = QVBoxLayout(page1)

        # Project title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Project title:"))
        self.title_edit = QLineEdit()
        self.title_edit.setText("MotileTrackerProject")
        title_layout.addWidget(self.title_edit)
        layout1.addLayout(title_layout)

        # Project directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Project directory:"))
        self.dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(browse_btn)
        layout1.addLayout(dir_layout)

        # Dimension radio buttons
        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("Dimensions:"))
        self.radio_2d = QRadioButton("2D + time")
        self.radio_3d = QRadioButton("3D + time")
        self.radio_2d.setChecked(True)
        self.dim_group = QButtonGroup(self)
        self.dim_group.addButton(self.radio_2d)
        self.dim_group.addButton(self.radio_3d)
        dim_layout.addWidget(self.radio_2d)
        dim_layout.addWidget(self.radio_3d)
        layout1.addLayout(dim_layout)

        # Table for axes
        self.table = QTableWidget(4, 5)
        self.table.setHorizontalHeaderLabels(
            ["Dimension", "Index", "Name", "Unit", "Step size"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout1.addWidget(self.table)

        # Fill table with default values
        self._update_table()

        # Connect radio buttons to update table
        self.radio_2d.toggled.connect(self._update_table)
        self.radio_3d.toggled.connect(self._update_table)

        # Page 1: Add Cancel button
        btn_layout1 = QHBoxLayout()
        btn_layout1.addStretch()
        self.cancel_btn1 = QPushButton("Cancel")
        self.next_btn = QPushButton("Next")
        btn_layout1.addWidget(self.cancel_btn1)
        btn_layout1.addWidget(self.next_btn)
        layout1.addLayout(btn_layout1)

        self.next_btn.setEnabled(False)
        self.title_edit.textChanged.connect(self._validate_inputs)
        self.dir_edit.textChanged.connect(self._validate_inputs)

        self.stacked.addWidget(page1)

        # --- Page 2: Data Selection ---
        page2 = QWidget()
        layout2 = QVBoxLayout(page2)

        # Intensity image
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity image:"))
        self.intensity_path = QLineEdit()
        intensity_browse = QPushButton("Browse")
        intensity_browse.clicked.connect(self._browse_intensity)
        intensity_layout.addWidget(self.intensity_path)
        intensity_layout.addWidget(intensity_browse)
        layout2.addLayout(intensity_layout)

        # Radiobuttons for data type
        data_type_group = QGroupBox("Detection type")
        data_type_layout = QHBoxLayout(data_type_group)
        self.radio_seg = QRadioButton("Segmentation labels")
        self.radio_points = QRadioButton("Point detections")
        self.radio_seg.setChecked(True)
        data_type_layout.addWidget(self.radio_seg)
        data_type_layout.addWidget(self.radio_points)
        layout2.addWidget(data_type_group)

        # Stacked widget for segmentation/points
        self.data_stacked = QStackedWidget()
        self.seg_widget = SegmentationWidget()
        self.csv_widget = CsvFileWidget()
        self.data_stacked.addWidget(self.seg_widget)
        self.data_stacked.addWidget(self.csv_widget)
        layout2.addWidget(self.data_stacked)

        self.radio_seg.toggled.connect(self._update_data_stack)
        self.radio_points.toggled.connect(self._update_data_stack)

        # Page 2: Add Cancel button
        btn_layout2 = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.cancel_btn2 = QPushButton("Cancel")
        self.ok_btn = QPushButton("OK")
        btn_layout2.addStretch()
        btn_layout2.addWidget(self.prev_btn)
        btn_layout2.addWidget(self.cancel_btn2)
        btn_layout2.addWidget(self.ok_btn)
        layout2.addLayout(btn_layout2)

        self.stacked.addWidget(page2)

        # After self.intensity_path, self.seg_widget, self.csv_widget, etc. are created:
        self.intensity_path.textChanged.connect(self._validate_page2)
        self.radio_seg.toggled.connect(self._validate_page2)
        self.radio_points.toggled.connect(self._validate_page2)

        # For SegmentationWidget
        self.seg_widget.image_path_line.textChanged.connect(self._validate_page2)
        self.seg_widget.radio_file.toggled.connect(self._validate_page2)
        self.seg_widget.radio_manual.toggled.connect(self._validate_page2)

        # For CsvFileWidget
        self.csv_widget.csv_path_line.textChanged.connect(self._validate_page2)
        self.csv_widget.radio_file.toggled.connect(self._validate_page2)
        self.csv_widget.radio_manual.toggled.connect(self._validate_page2)

        # Connections for navigation
        self.next_btn.clicked.connect(self._go_to_page2)
        self.prev_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.ok_btn.clicked.connect(self.accept)
        self.stacked.setCurrentIndex(0)

        # Connect cancel buttons to close the dialog
        self.cancel_btn1.clicked.connect(self._cancel)
        self.cancel_btn2.clicked.connect(self._cancel)

    def _go_to_page2(self):
        self.stacked.setCurrentIndex(1)
        self._validate_page2()

    def _browse_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.dir_edit.setText(directory)

    def _browse_intensity(self) -> None:
        """Open custom dialog to select either a file or a folder"""
        dialog = FileFolderDialog(self)
        if dialog.exec_():
            selected_path = dialog.get_selected_path()
            if selected_path:
                self.intensity_path.setText(selected_path)

    def _validate_inputs(self):
        title_ok = bool(self.title_edit.text().strip())
        dir_path = self.dir_edit.text().strip()
        dir_ok = bool(dir_path) and os.path.isdir(dir_path)

        # Check that axis indices are unique
        indices = []
        for row in range(self.table.rowCount()):
            index_widget = self.table.cellWidget(row, 1)
            if index_widget is not None:
                indices.append(index_widget.currentText())
        indices_unique = len(indices) == len(set(indices))

        self.next_btn.setEnabled(title_ok and dir_ok and indices_unique)

    def _validate_page2(self):
        # Validate intensity image
        intensity_path = self.intensity_path.text().strip()
        intensity_ok = bool(intensity_path) and os.path.exists(intensity_path)
        # Validate detection (segmentation/csv/manual)
        if self.radio_seg.isChecked():
            is_manual = self.seg_widget.is_manual()
            seg_path = self.seg_widget.get_path()
            seg_ok = is_manual or (bool(seg_path) and os.path.exists(seg_path))
            valid = intensity_ok and seg_ok
        else:
            is_manual = self.csv_widget.is_manual()
            csv_path = self.csv_widget.get_path()
            csv_ok = is_manual or (bool(csv_path) and os.path.exists(csv_path))
            valid = intensity_ok and csv_ok
        self.ok_btn.setEnabled(valid)

    def _update_table(self):
        is_3d = self.radio_3d.isChecked()
        axes = ["time", "z", "y", "x"] if is_3d else ["time", "y", "x"]
        axes_indices = [0, 1, 2, 3] if is_3d else [0, 1, 2]
        units = {
            "time": ["time point", "sec", "min", "hour", "day"],
            "z": ["nm", "µm", "mm", "cm", "m"],
            "y": ["nm", "µm", "mm", "cm", "m"],
            "x": ["nm", "µm", "mm", "cm", "m"],
        }
        default_units = {
            "time": "time point",
            "z": "µm",
            "y": "µm",
            "x": "µm",
        }
        stepsize = {"time": 1.0, "z": 1.0, "y": 1.0, "x": 1.0}
        self.table.setRowCount(len(axes))
        for row, axis in enumerate(axes):
            # Axis label (not editable)
            item = QTableWidgetItem(axis)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)

            # Axis indices (dropdown)
            axis_indices = QComboBox()
            axis_indices.addItems([str(i) for i in axes_indices])
            axis_indices.setCurrentText(str(axes_indices[row]))
            axis_indices.currentIndexChanged.connect(self._validate_inputs)
            self.table.setCellWidget(row, 1, axis_indices)
            # Axis name (editable)
            axis_name = QLineEdit(axis)
            self.table.setCellWidget(row, 2, axis_name)
            # Unit (dropdown)
            unit_combo = QComboBox()
            unit_combo.addItems(units[axis])
            unit_combo.setCurrentText(default_units[axis])
            self.table.setCellWidget(row, 3, unit_combo)
            # Step size (QDoubleSpinBox)
            step_spin = QDoubleSpinBox()
            step_spin.setDecimals(3)
            step_spin.setSingleStep(0.1)
            step_spin.setValue(stepsize[axis])
            step_spin.setMinimum(0.0)
            self.table.setCellWidget(row, 4, step_spin)

    def _update_data_stack(self):
        if self.radio_seg.isChecked():
            self.data_stacked.setCurrentIndex(0)
        else:
            self.data_stacked.setCurrentIndex(1)

    def create_empty_fp_array(
        self, fp_array_path: str, shape: tuple, axes: dict | None = None
    ) -> fp.Array:
        """Creates an empty funtracks persistence array with the specified shape and axes."""

        axis_names = axes.get("axis_names", ["axis_" + str(i) for i in range(len(shape))])
        voxel_size = axes.get("scaling", [1.0] * len(shape))
        axis_units = axes.get("units", ["px"] * len(shape))

        fpds = fp.prepare_ds(
            fp_array_path,
            shape=shape,
            voxel_size=voxel_size,
            axis_names=axis_names,
            units=axis_units,
            dtype=np.uint32,
        )

        return fpds

    def create_fp_array(
        self,
        image_path: str | None = None,
        fp_array_path: str | None = None,
        axes: dict | None = {},
    ) -> fp.Array:
        """Creates a funtracks persistence array from an intensity image or segmentation data.
        Args:
            image_path (str): Path to the intensity image or segmentation data.
            fp_array_path (str): Path where the funtracks persistence array will be created.
            axes (dict): Dictionary containing axis information like indices, names, units, and scaling.
        Returns:
            fp.Array: A funtracks persistence array containing the data."""

        # extract array size from the intensity image
        if image_path.endswith(".tif"):
            image = np.squeeze(tifffile.imread(image_path))
        elif ".zarr" in image_path:
            image = np.squeeze(zarr.open(image_path))
        else:
            raise ValueError("Intensity image must be a tiff or zarr file.")
        # Reorder the dimensions of the data to make sure the order is t(z)yx:
        shape = image.shape
        axis_indices = axes.get("axis_indices", list(range(len(shape))))
        axis_names = axes.get("axis_names", ["axis_" + str(i) for i in range(len(shape))])
        voxel_size = axes.get("scaling", [1.0] * len(shape))
        axis_units = axes.get("units", ["px"] * len(shape))

        default_order = list(range(image.ndim))  # e.g. [0,1,2,3] for tzyx
        if axis_indices != default_order:
            print("transposing intensity image to match axis indices")
            image = np.transpose(image, np.argsort(axis_indices))

        fpds = fp.prepare_ds(
            fp_array_path,
            shape=shape,
            voxel_size=voxel_size,
            axis_names=axis_names,
            units=axis_units,
            dtype=np.uint32,
        )

        # if segmentation, do the relabeling like we do in the sample data
        if fp_array_path.endswith("seg"):
            if self._has_duplicate_ids(image):
                image = ensure_unique_labels(image)

        # load and write each time point into the dataset
        for time in tqdm(range(axis_indices[0]), desc="Converting time points to zarr"):
            fpds[time] = image[time]

        return fpds

    def create_project(self) -> Project:
        intensity_image = None
        segmentation = None
        points = None

        # creates a new funtracks project with the information provided in the dialog
        project_info = self.get_project_info()
        name = project_info.get("name", "Untitled Project")
        axes = project_info.get("axes", [])
        detection_type = project_info.get("detection_type", "points")
        working_dir = project_info.get("directory", Path.cwd())
        params = project_info.get("project_params", None)

        # create fpds for the intensity image and segmentation data (if provided)
        intensity_image_path = project_info.get("intensity_image", None)
        if intensity_image_path is not None:
            int_fps_path = os.path.join(working_dir, "motile_tracker.zarr/int")
            intensity_image = self.create_fp_array(
                intensity_image_path, int_fps_path, axes
            )

        if detection_type == "segmentation":
            seg_path = project_info.get("detection_path", None)
            seg_fps_path = os.path.join(working_dir, "motile_tracker.zarr/int")

            if seg_path is None:
                segmentation = self.create_empty_fp_array(
                    seg_fps_path, intensity_image.shape, axes
                )

            else:
                segmentation = self.create_fp_array(seg_path, seg_fps_path, axes)
                if segmentation.shape != intensity_image.shape:
                    raise ValueError(
                        "Segmentation data shape does not match intensity image shape. "
                        f"Segmentation shape: {segmentation.shape}, Intensity image shape: {intensity_image.shape}"
                    )

        elif detection_type == "points":
            points_path = project_info.get("detection_path", None)
            if points_path is None:
                raise ValueError(
                    "Points detection type selected, but no points file provided."
                )

        return Project(
            name=name,
            project_params=params,
            raw=intensity_image,
            segmentation=segmentation,
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
            first_frame_ids = set(np.unique(segmentation[0]).tolist())
            first_frame_ids.remove(0)
            second_frame_ids = set(np.unique(segmentation[1]).tolist())
            second_frame_ids.remove(0)
            return not first_frame_ids.isdisjoint(second_frame_ids)
        return False

    def get_project_info(self):
        # creates a new funtracks project with the information provided in the dialog

        info = {
            "title": self.title_edit.text(),
            "directory": self.dir_edit.text(),
            "dimensions": "4" if self.radio_3d.isChecked() else "3",
            "axes": {
                "dimensions": [],
                "indices": [],
                "axis_names": [],
                "units": [],
                "scaling": [],
            },
            "intensity_image": self.intensity_path.text(),
            "detection_type": "segmentation" if self.radio_seg.isChecked() else "points",
            "detection_path": None,
        }
        for row in range(self.table.rowCount()):
            axis = self.table.item(row, 0).text()
            index = self.table.cellWidget(row, 1).currentText()
            axis_name = self.table.cellWidget(row, 2).text()
            unit = self.table.cellWidget(row, 3).currentText()
            step_size = self.table.cellWidget(row, 4).value()
            info["axes"]["dimensions"].append(axis)
            info["axes"]["indices"].append(index)
            info["axes"]["axis_names"].append(axis_name)
            info["axes"]["units"].append(unit)
            info["axes"]["scaling"].append(step_size)

        if self.radio_seg.isChecked():
            info["detection_path"] = self.seg_widget.image_path_line.text()
        else:
            info["detection_path"] = self.csv_widget.get_path()

        return info

    def _cancel(self):
        self.reject()
