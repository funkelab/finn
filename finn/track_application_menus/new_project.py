import os
import shutil
from pathlib import Path
from typing import Any

import dask.array as da
import funlib.persistence as fp
import numpy as np
from funtracks.features.measurement_features import featureset
from funtracks.params._base import Params
from funtracks.params.cand_graph_params import CandGraphParams
from funtracks.params.project_params import ProjectParams
from funtracks.project import Project
from psygnal import Signal
from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from finn.track_application_menus.browse_data import DataWidget
from finn.track_application_menus.csv_widget import CSVWidget


class DialogValueError(ValueError):
    def __init__(self, message, show_dialog=True):
        super().__init__(message)
        self.show_dialog = show_dialog


class Page1(QWidget):
    """Page 1 of the Project dialog, to enter project information"""

    choice_updated = Signal()

    def __init__(self):
        super().__init__()

        # Ask the user about their project goal
        goal_layout = QVBoxLayout()

        label = QLabel("What would you like to do?")
        label.setAlignment(Qt.AlignHCenter)
        goal_layout.addWidget(label)

        self.track_from_scratch = QRadioButton(
            "I have intensity data and want to track objects from scratch by manually placing points or segmentation labels."
        )
        self.track_from_detections = QRadioButton(
            "I have (label or point) detections and want to track objects from these detections."
        )
        self.curate_tracks = QRadioButton(
            "I have tracking data from external software that I want to view and/or curate."
        )
        self.track_from_scratch.toggled.connect(self.choice_updated)
        self.track_from_detections.toggled.connect(self.choice_updated)
        self.curate_tracks.toggled.connect(self.choice_updated)
        self.track_from_detections.setChecked(True)
        self.goal_group = QButtonGroup(self)
        self.goal_group.addButton(self.track_from_scratch)
        self.goal_group.addButton(self.track_from_detections)
        self.goal_group.addButton(self.curate_tracks)
        goal_layout.addWidget(self.track_from_scratch)
        goal_layout.addWidget(self.track_from_detections)
        goal_layout.addWidget(self.curate_tracks)

        # Add goal_layout to a widget to set is size
        goal_widget = QWidget()
        goal_widget.setLayout(goal_layout)
        goal_widget.setMaximumHeight(200)

        # Wrap everything in a group box
        box = QGroupBox("Project Goal")
        layout = QVBoxLayout()
        layout.addWidget(goal_widget)
        box.setLayout(layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def get_choice(self) -> str | None:
        """Return the choice made by the user regarding what they want to do."""

        if self.track_from_scratch.isChecked():
            return "track_from_scratch"
        if self.track_from_detections.isChecked():
            return "track_from_detections"
        if self.curate_tracks.isChecked():
            return "curate_tracks"
        return None


class Page2(QWidget):
    """Page 2 of the Project dialog, to select the intensity data source"""

    validity_changed = Signal()

    def __init__(self, page1: Page1):
        super().__init__()

        self.page1 = page1
        self.page1.choice_updated.connect(self._toggle_question)

        self.is_valid = False
        layout = QVBoxLayout()
        self.show_intensity_widget = False

        # Ask the user if they have intensity data (mandatory when tracking from scratch)
        intensity_layout = QVBoxLayout()
        intensity_layout.setSpacing(15)

        title_layout = QHBoxLayout()
        label = QLabel("Do you have intensity data?")
        label.setAlignment(Qt.AlignHCenter)
        title_layout.addWidget(label)
        intensity_layout.addLayout(title_layout)

        button_layout = QHBoxLayout()
        self.yes = QRadioButton("Yes")
        self.no = QRadioButton("No")
        self.yes.setChecked(True)
        self.intensity_group = QButtonGroup(self)
        self.intensity_group.addButton(self.yes)
        self.intensity_group.addButton(self.no)
        self.yes.toggled.connect(self._toggle_intensity_widget)
        self.no.toggled.connect(self._toggle_intensity_widget)
        button_layout.setAlignment(Qt.AlignHCenter)
        button_layout.addWidget(self.yes)
        button_layout.addWidget(self.no)
        intensity_layout.addLayout(button_layout)

        self.intensity_button_widget = QWidget()
        self.intensity_button_widget.setLayout(intensity_layout)
        self.intensity_button_widget.setMaximumHeight(100)
        if self.page1.get_choice() == "track_from_scratch":
            self.intensity_button_widget.setVisible(False)
        else:
            self.intensity_button_widget.setVisible(True)

        layout.addWidget(self.intensity_button_widget)

        # Provide a widget to enter the path to the intensity data
        self.intensity_data_widget = DataWidget()
        self.intensity_data_widget.validity_changed.connect(self.validate)
        self.intensity_data_widget.setToolTip(
            "<qt><i>"
            "Image data can either be a single tif (3D+time or 2D+time) stack, a "
            "folder containing a time series of 2D or 3D tif images, or a zarr folder."
            "</i></qt>"
        )

        layout.addWidget(self.intensity_data_widget)

        # Wrap everything in a group box
        box = QGroupBox("Intensity image data")
        box.setLayout(layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def _toggle_question(self):
        """Toggle the visibility of the intensity question based on the user's choice."""

        if self.page1.get_choice() == "track_from_scratch":
            self.intensity_button_widget.setVisible(False)
            self.intensity_data_widget.setVisible(True)
        else:
            self.intensity_button_widget.setVisible(True)
            if self.yes.isChecked():
                self.intensity_data_widget.setVisible(True)
            else:
                self.intensity_data_widget.setVisible(False)

        self.validate()

    def _toggle_intensity_widget(self):
        """Toggle the visibility of the intensity widget based on the user's choice."""

        if self.intensity_button_widget.isVisible() and self.yes.isChecked():
            self.intensity_data_widget.setVisible(True)
        else:
            self.intensity_data_widget.setVisible(False)

        self.validate()

    def get_path(self) -> str | None:
        """Return the path to the intensity data, if provided."""
        if self.page1.get_choice() == "track_from_scratch" or self.yes.isChecked():
            return self.intensity_data_widget.get_path()
        return None

    def validate(self) -> None:
        """Check whether the intensity data widget is valid and emit a signal."""

        if self.page1.get_choice() == "track_from_scratch" or self.yes.isChecked():
            path = self.get_path()
            if path is None or not os.path.exists(path):
                self.is_valid = False
            else:
                self.is_valid = True
        else:  # user selected no intensity data and was not required to do so
            self.is_valid = True

        self.validity_changed.emit()


class Page3(QWidget):
    """Page 3 of the Project dialog, to select the detection data source"""

    validity_changed = Signal()
    type_updated = Signal()

    def __init__(self, page1: Page1):
        super().__init__()

        self.is_valid = False
        self.data_type = "segmentation"
        self.page1 = page1
        self.page1.choice_updated.connect(self._toggle_data_widget_visibility)

        # Ask the user if they have intensity data (mandatory when tracking from scratch)
        data_layout = QVBoxLayout()
        data_layout.setSpacing(15)

        title_layout = QHBoxLayout()
        self.label = QLabel("Do you have point or label detection data?")
        self.label.setAlignment(Qt.AlignHCenter)
        title_layout.addWidget(self.label)
        data_layout.addLayout(title_layout)

        button_layout = QHBoxLayout()
        self.points = QRadioButton("Points")
        self.labels = QRadioButton("Labels")
        self.labels.setChecked(True)
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.points)
        self.button_group.addButton(self.labels)
        self.points.toggled.connect(self._toggle_data_type)
        self.labels.toggled.connect(self._toggle_data_type)
        button_layout.setAlignment(Qt.AlignHCenter)
        button_layout.addWidget(self.points)
        button_layout.addWidget(self.labels)
        data_layout.addLayout(button_layout)

        layout = QVBoxLayout()
        layout.addLayout(data_layout)

        # Provide a widget to enter the path to the detection data
        self.data_widget = DataWidget()
        self.data_widget.validity_changed.connect(self.validate)
        layout.addWidget(self.data_widget)

        # Wrap everything in a group box
        box = QGroupBox("Detection data")
        box.setLayout(layout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

        self._toggle_data_type()

    def _toggle_data_widget_visibility(self):
        if self.page1.get_choice() == "track_from_scratch":
            self.data_widget.setVisible(False)
            self.label.setText(
                "Do you want to track by placing points or segmentation labels?"
            )
            self.is_valid = True
        else:
            self.data_widget.setVisible(True)
            self.label.setText("Do you have point or label detection data?")
            self.validate()

    def _toggle_data_type(self):
        """Toggle the visibility of the intensity widget based on the user's choice."""

        self.data_type = "segmentation" if self.labels.isChecked() else "points"
        self.data_widget.update_type(self.data_type)
        self.type_updated.emit()

    def get_path(self) -> str | None:
        """Return the path to the data, if provided."""
        if (
            self.page1.get_choice() == "track_from_scratch"
            or self.data_widget.get_path() is None
        ):
            return None
        return self.data_widget.get_path()

    def validate(self) -> None:
        """Check whether the intensity data widget is valid and emit a signal."""

        if self.page1.get_choice() == "track_from_scratch":
            self.is_valid = True
        else:
            path = self.get_path()
            if (
                path is None
                or not os.path.exists(path)
                or (self.points.isChecked() and not path.endswith(".csv"))
            ):
                self.is_valid = False
            else:
                self.is_valid = True
        self.validity_changed.emit()


class Page4(QWidget):
    validity_changed = Signal()
    dim_updated = Signal()

    def __init__(self, page2: Page2 | None = None, page3: Page3 | None = None):
        super().__init__()

        self.page2 = page2
        self.page2.validity_changed.connect(self._guess_data_dimensions)
        self.page3 = page3
        self.page3.validity_changed.connect(self._guess_data_dimensions)
        self.page3.type_updated.connect(self._guess_data_dimensions)

        self.ndim = 3
        self.incl_z = False
        self.has_channels = False
        self.allow_channels = False
        self.intensity_image_shape = None
        self.segmentation_image_shape = None
        self.is_valid = False

        self.units = {
            "channel": ["channel"],
            "time": ["time point", "sec", "min", "hour", "day"],
            "z": ["nm", "µm", "mm", "cm", "m"],
            "y": ["nm", "µm", "mm", "cm", "m"],
            "x": ["nm", "µm", "mm", "cm", "m"],
        }
        self.default_units = {
            "channel": "channel",
            "time": "time point",
            "z": "µm",
            "y": "µm",
            "x": "µm",
        }
        self.step_size = {"channel": 1, "time": 1.0, "z": 1.0, "y": 1.0, "x": 1.0}

        layout = QVBoxLayout()

        # Display the shape of the intensity and/or segmentation data
        self.channel_label = QLabel(
            "<i>Please rearrange your dimension order such that the channel axis is the first dimension</i>"
        )
        layout.addWidget(self.channel_label)

        self.intensity_label = QLabel("Intensity data shape:")
        layout.addWidget(self.intensity_label)

        self.seg_label = QLabel("Segmentation data shape:")
        layout.addWidget(self.seg_label)

        self.incompatible_label = QLabel(
            "❌ The shapes of the intensity data and the segmentation data do not match.\n\n Please check if your intensity data has channels or go back to select different data."
        )
        layout.addWidget(self.incompatible_label)

        # Table for axes
        self.table = QTableWidget(5, 6)
        self.table.setHorizontalHeaderLabels(
            ["Dimension", "Index (intensity)", "Index (seg)", "Name", "Unit", "Step size"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Guess dimensions from the data and fill table with default values
        self._guess_data_dimensions()

        # wrap everything in a group box
        box = QGroupBox("Project Information")
        box.setLayout(layout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def _guess_data_dimensions(self):
        self.intensity_label.setVisible(False)
        self.seg_label.setVisible(False)
        self.intensity_image_shape = None
        self.segmentation_image_shape = None

        # Load the data as dask arrays, and extract the shape.
        if self.page2.get_path() is not None:
            self.intensity_image = self.page2.intensity_data_widget.load_images()
            self.intensity_image_shape = self.intensity_image.shape
            self.intensity_label.setText(
                f"Intensity data shape: {self.intensity_image_shape} -> {self.intensity_image_shape}"
            )
            self.intensity_label.setVisible(True)
        else:
            self.intensity_image = None

        if self.page3.get_path() is not None and self.page3.data_type == "segmentation":
            self.segmentation_image = self.page3.data_widget.load_images()
            self.segmentation_image_shape = self.segmentation_image.shape
            self.seg_label.setText(
                f"Segmentation data shape: {self.segmentation_image_shape} -> {self.segmentation_image_shape}"
            )
            self.seg_label.setVisible(True)
        else:
            self.segmentation_image = None

        # check whether we should display the checkbox to ask about the channel dimension
        self._set_allow_channels()

        # Check if the shapes for intensity data and segmentation data are compatible
        self._update_table()

    def _set_allow_channels(self) -> None:
        """Channels label should be visible if the intensity data has one dimension more than the segmentation data"""

        if self.intensity_image_shape is None and self.segmentation_image_shape is None:
            self.allow_channels = False
            self.has_channels = True
            self.ndim = 3
        elif (
            self.intensity_image_shape is not None
            and self.segmentation_image_shape is None
        ):
            self.allow_channels = len(self.intensity_image_shape) == 4
            self.ndim = len(self.intensity_image_shape)
        elif (
            self.intensity_image_shape is not None
            and self.segmentation_image_shape is not None
        ):
            self.ndim = np.max(
                (len(self.intensity_image_shape), len(self.segmentation_image_shape))
            )
            self.allow_channels = (
                len(self.intensity_image_shape) == len(self.segmentation_image_shape) + 1
            )
        else:
            self.allow_channels = False
            self.ndim = len(self.segmentation_image_shape)

        if not self.allow_channels:
            self.has_channels = False

    def _update_channel_label_visibility(self, visible: bool) -> None:
        self.channel_label.setVisible(visible)

    def _update_axis(self, axis_name: str):
        """Updates the top row of the table with the dimension chosen by the user, either 'channel' or 'z'"""

        # Axis name (editable)
        self.table.setCellWidget(0, 3, QLineEdit(axis_name))
        # Unit (dropdown)
        unit_combo = QComboBox()
        unit_combo.addItems(self.units[axis_name])
        unit_combo.setCurrentText(self.default_units[axis_name])
        self.table.setCellWidget(0, 4, unit_combo)
        # Step size (QDoubleSpinBox)
        if axis_name == "channel":
            step_spin = QLabel("1")
            self.channel_label.setVisible(True)
            self.has_channels = True
        else:
            step_spin = QDoubleSpinBox()
            step_spin.setDecimals(3)
            step_spin.setSingleStep(0.1)
            step_spin.setValue(self.step_size[axis_name])
            step_spin.setMinimum(0.0)
            self.channel_label.setVisible(False)
            self.has_channels = False
        self.table.setCellWidget(0, 5, step_spin)

        self.dim_updated.emit()
        self.validate()

    def _update_table(self):
        self.table.setVisible(True)

        self._set_allow_channels()

        print(self.ndim)

        if self.ndim == 5:
            axes = ["channel", "time", "z", "y", "x"]
            self.channel_label.setVisible(True)
            self.has_channels = True
        elif self.ndim == 4:
            if self.allow_channels:
                axes = [["channel", "z"], "time", "y", "x"]
            else:
                axes = ["time", "z", "y", "x"]
        else:
            axes = ["time", "y", "x"]

        self.incl_z = "z" in axes

        self.table.setRowCount(len(axes))
        for row, axis in enumerate(axes):
            # Axis label (not editable, except when the user has to choose between 'channel' and 'z')
            if isinstance(axis, list):
                print("updating table with combobox to chooise between channel and z")
                item = QComboBox()
                item.addItems(axis)
                item.currentTextChanged.connect(self._update_axis)
                item.setCurrentText("channel")
                axis_name = item.currentText()
                self.table.setCellWidget(row, 0, item)
            else:
                print("no combobox needed")
                existing_widget = self.table.cellWidget(row, 0)
                if existing_widget is not None:
                    self.table.removeCellWidget(row, 0)
                item = QTableWidgetItem(axis)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                axis_name = axis
                self.table.setItem(row, 0, item)

            # Axis indices (dropdown)
            self.raw_axis_indices = QComboBox()
            if self.intensity_image_shape is None:
                self.raw_axis_indices.setEnabled(False)
            else:
                self.raw_axis_indices.setEnabled(True)
                indices = range(len(self.intensity_image_shape))
                self.raw_axis_indices.addItems([str(i) for i in indices])
                self.raw_axis_indices.setCurrentText(str(indices[row]))
                self.raw_axis_indices.currentTextChanged.connect(self.validate)
            self.table.setCellWidget(row, 1, self.raw_axis_indices)

            self.seg_axis_indices = QComboBox()
            if self.segmentation_image_shape is None:
                self.seg_axis_indices.setEnabled(False)
            else:
                self.seg_axis_indices.setEnabled(True)
                indices = range(len(self.segmentation_image_shape))
                self.seg_axis_indices.addItems([str(i) for i in indices])
                if self.has_channels:
                    self.seg_axis_indices.setCurrentText(str(indices[row - 1]))
                    if row == 0:
                        self.seg_axis_indices.setEnabled(False)
                else:
                    self.seg_axis_indices.setCurrentText(str(indices[row]))
                self.seg_axis_indices.currentTextChanged.connect(self.validate)

            self.table.setCellWidget(row, 2, self.seg_axis_indices)

            # Axis name (editable)
            self.table.setCellWidget(row, 3, QLineEdit(axis_name))
            # Unit (dropdown)
            unit_combo = QComboBox()
            unit_combo.addItems(self.units[axis_name])
            unit_combo.setCurrentText(self.default_units[axis_name])
            self.table.setCellWidget(row, 4, unit_combo)
            # Step size (QDoubleSpinBox)
            if row == 0 and self.has_channels:
                # just use a label, no step size for channels
                step_spin = QLabel("1")
            else:
                step_spin = QDoubleSpinBox()
                step_spin.setDecimals(3)
                step_spin.setSingleStep(0.1)
                step_spin.setValue(self.step_size[axis_name])
                step_spin.setMinimum(0.0)
            self.table.setCellWidget(row, 5, step_spin)

        self.dim_updated.emit()
        self.validate()

    def validate(self):
        """Validate inputs on this page"""

        # Check that axis indices are unique
        raw_indices = []
        for row in range(self.table.rowCount()):
            index_widget = self.table.cellWidget(row, 1)
            if index_widget is not None:
                raw_indices.append(index_widget.currentText())
        indices_raw_unique = len(raw_indices) == len(set(raw_indices))
        if self.raw_axis_indices.isEnabled() and not indices_raw_unique:
            raw_axis_ok = False
            self.table.horizontalHeaderItem(1).setForeground(QtGui.QColor("red"))
            self.incompatible_label.setText(
                "❌ Please ensure that you choose unique intensity image axis indices."
            )
        else:
            raw_axis_ok = True
            self.table.horizontalHeaderItem(1).setForeground(QtGui.QColor("white"))

        seg_indices = []
        for row in range(self.table.rowCount()):
            index_widget = self.table.cellWidget(row, 2)
            if index_widget is not None and index_widget.isEnabled():
                seg_indices.append(index_widget.currentText())
        indices_seg_unique = len(seg_indices) == len(set(seg_indices))
        if self.seg_axis_indices.isEnabled() and not indices_seg_unique:
            seg_axis_ok = False
            self.table.horizontalHeaderItem(2).setForeground(QtGui.QColor("red"))
            self.incompatible_label.setText(
                "❌ Please ensure that you choose unique segmentation image axis indices."
            )
        else:
            seg_axis_ok = True
            self.table.horizontalHeaderItem(2).setForeground(QtGui.QColor("white"))

        # check that the selected axis order results in matching shapes
        self.intensity_label.setText(
            f"Intensity data shape: {self.intensity_image_shape} -> {self.intensity_image_shape} "
        )
        self.seg_label.setText(
            f"Segmentation data shape: {self.segmentation_image_shape} -> {self.segmentation_image_shape}"
        )

        if self.intensity_image_shape is not None:
            reordered_raw_dims = tuple(
                self.intensity_image_shape[int(i)] for i in raw_indices
            )
            color = "green" if raw_axis_ok else "red"
            self.intensity_label.setText(
                f"Intensity data shape: {self.intensity_image_shape} -> "
                f"<span style='color:{color};'>{reordered_raw_dims}</span>"
            )

        if self.segmentation_image_shape is not None:
            reordered_seg_dims = tuple(
                self.segmentation_image_shape[int(i)] for i in seg_indices
            )
            color = "green" if seg_axis_ok else "red"
            self.seg_label.setText(
                f"Segmentation data shape: {self.segmentation_image_shape} -> "
                f"<span style='color:{color};'>{reordered_seg_dims}</span>"
            )

        if (
            self.intensity_image_shape is not None
            and self.segmentation_image_shape is not None
        ):
            if not (raw_axis_ok and seg_axis_ok):
                shape_match = False
            elif self.has_channels:
                shape_match = reordered_raw_dims[-(self.ndim - 1) :] == reordered_seg_dims
            else:
                shape_match = reordered_raw_dims[-self.ndim :] == reordered_seg_dims
        else:
            shape_match = True

        if not shape_match:
            self.incompatible_label.setText(
                "❌ The shapes of the intensity data and the segmentation data do not match."
            )

        if shape_match and seg_axis_ok and raw_axis_ok:
            self.incompatible_label.setText("✅")

        self.is_valid = raw_axis_ok and seg_axis_ok and shape_match
        self.validity_changed.emit()

    def get_settings(self) -> dict[str:Any]:
        """Get the settings on page4"""

        info = {
            "intensity_image": self.intensity_image,
            "segmentation_image": self.segmentation_image,
            "ndim": self.ndim,
            "axes": {
                "dimensions": [],
                "raw_indices": [],
                "seg_indices": [],
                "axis_names": [],
                "units": [],
                "scaling": [],
            },
        }

        for row in range(self.table.rowCount()):
            axis = (
                self.table.item(row, 0).text()
                if isinstance(self.table.item(row, 0), QTableWidgetItem)
                else self.table.item(row, 0).currentText()
            )
            raw_index = self.table.cellWidget(row, 1).currentText()
            seg_index = self.table.cellWidget(row, 2).currentText()
            axis_name = self.table.cellWidget(row, 3).text()
            unit = self.table.cellWidget(row, 4).currentText()
            step_size = self.table.cellWidget(row, 5).value()
            info["axes"]["dimensions"].append(axis)
            if self.intensity_image_shape is not None:
                info["axes"]["raw_indices"].append(raw_index)
            if self.segmentation_image_shape is not None:
                info["axes"]["seg_indices"].append(seg_index)
            info["axes"]["axis_names"].append(axis_name)
            info["axes"]["units"].append(unit)
            info["axes"]["scaling"].append(step_size)

        return info


class Page5(QWidget):
    """Page5, to import external tracks from CSV"""

    validity_changed = Signal()
    mapping_updated = Signal()

    def __init__(self, page3: Page3, page4: Page4):
        super().__init__()
        self.is_valid = False
        self.page3 = page3
        self.page4 = page4
        self.page3.type_updated.connect(self._update_has_segmentation)
        self.page4.dim_updated.connect(self._update_incl_z)

        self.incl_z = self.page4.incl_z
        self.has_segmentation = self.page3.data_type == "segmentation"

        # CSVWidget for tracks
        self.csv_widget = CSVWidget(
            add_segmentation=self.has_segmentation, incl_z=self.incl_z
        )
        self.csv_widget.validity_changed.connect(self.validate)
        self.csv_widget.mapping_updated.connect(self._emit_mapping_updated)

        # wrap everything in a group
        import_data_group = QGroupBox("Import external tracking data")
        self.csv_layout = QVBoxLayout()
        self.csv_layout.addWidget(self.csv_widget)
        import_data_group.setLayout(self.csv_layout)
        import_data_layout = QVBoxLayout()
        import_data_layout.addWidget(import_data_group)
        self.setLayout(import_data_layout)

    def _emit_mapping_updated(self):
        """Emit a signal when the mapping is updated in the CSVWidget."""

        self.mapping_updated.emit()

    def _update_has_segmentation(self):
        """Update whether the user has segmentation data based on the selection in page3."""

        if self.page3.data_type == "segmentation":
            self.has_segmentation = True
        else:
            self.has_segmentation = False

        self.csv_widget.update_field_map(seg=self.has_segmentation, incl_z=self.incl_z)

    def _update_incl_z(self):
        """Update whether the user wants to include z-dimension based on the selection in page4."""

        self.incl_z = self.page4.incl_z
        self.csv_widget.update_field_map(seg=self.has_segmentation, incl_z=self.incl_z)

    def validate(self) -> None:
        """Check whether all required information was filled out and then emit a signal"""

        self.is_valid = self.csv_widget.is_valid
        self.validity_changed.emit()

    def get_tracks_path(self) -> str:
        """Get the path to the CSV file containing the tracks"""
        return self.csv_widget.get_path()

    def get_mapping(self) -> str:
        """Get the mapping from feature name to csv field name"""
        if self.csv_widget.csv_field_widget is not None:
            # Return the mapping from feature name to csv field name
            return self.csv_widget.csv_field_widget.get_name_map()
        return {}

    def get_unmapped_columns(self) -> list[str]:
        """Get the columns that were not mapped to any feature"""

        if self.csv_widget.csv_field_widget is not None:
            return self.csv_widget.csv_field_widget.columns_left
        return []


class FeatureWidget(QWidget):
    """Widget allowing the user to choose which features to include"""

    def __init__(
        self,
        ndim: int,
        mappable_columns: list[str] | None = [],
        include_intensity: bool = False,
        data_type: str = "segmentation",
    ):
        super().__init__()

        self.feature_instances = [feature_cls(ndim=ndim) for feature_cls in featureset]

        self.features_layout = QVBoxLayout()

        self.show_mapping = bool(mappable_columns)
        self.choose_column_label = QLabel("Choose from column")
        self.features_layout.addWidget(self.choose_column_label)
        self.choose_column_label.setVisible(self.show_mapping)

        self.measurement_checkboxes = {}
        self.measurement_comboboxes = {}

        # For each feature:
        for feature in self.feature_instances:
            row_layout = QHBoxLayout()
            checkbox = QCheckBox(feature.value_names)
            checkbox.setChecked(False)
            row_layout.addWidget(checkbox)
            self.measurement_checkboxes[feature.attr_name] = checkbox
            checkbox.setEnabled(data_type == "segmentation")

            combobox = QComboBox()
            combobox.addItem("None")
            combobox.setVisible(self.show_mapping)
            combobox.setEnabled(data_type == "segmentation")

            row_layout.addWidget(combobox)
            self.measurement_comboboxes[feature.attr_name] = combobox

            self.features_layout.addLayout(row_layout)

        self.setLayout(self.features_layout)

    def _update_features(
        self,
        ndim: int,
        mappable_columns: list[str] | None = [],
        include_intensity: bool = False,
        data_type: str = "segmentation",
    ):
        self.show_mapping = bool(mappable_columns)
        self.choose_column_label.setVisible(self.show_mapping)
        self.feature_instances = [feature_cls(ndim=ndim) for feature_cls in featureset]

        for feature in self.feature_instances:
            attr_name = feature.attr_name
            # Update checkbox label if needed
            if attr_name in self.measurement_checkboxes:
                checkbox = self.measurement_checkboxes[attr_name]
                checkbox.setText(feature.value_names)
                # Enable/disable intensity checkbox if needed
                if attr_name == "intensity":
                    checkbox.setEnabled(include_intensity)
                checkbox.setEnabled(data_type == "segmentation")

            # Update combobox visibility and options
            if attr_name in self.measurement_comboboxes:
                combobox = self.measurement_comboboxes[attr_name]
                combobox.setVisible(self.show_mapping)
                if self.show_mapping:
                    # Save current selection
                    current = combobox.currentText()
                    combobox.blockSignals(True)
                    combobox.clear()
                    combobox.addItem("None")
                    combobox.addItems(mappable_columns)
                    # Restore selection if still available, else set to "None"
                    if current in mappable_columns:
                        combobox.setCurrentText(current)
                    else:
                        combobox.setCurrentText("None")
                    combobox.blockSignals(False)
                else:
                    combobox.setCurrentText("None")
                # Enable/disable intensity combobox if needed
                if attr_name == "intensity":
                    combobox.setEnabled(include_intensity)
                combobox.setEnabled(data_type == "segmentation")

    def get_selected_features(self) -> list[dict[str : str | bool]]:
        selected = []
        for feature in self.feature_instances:
            attr_name = feature.attr_name
            checkbox = self.measurement_checkboxes[attr_name]
            include = checkbox.isChecked() and checkbox.isEnabled()
            from_column = None  # Default to None if not selected/visible
            if (
                hasattr(self, "measurement_comboboxes")
                and attr_name in self.measurement_comboboxes
            ):
                combobox = self.measurement_comboboxes[attr_name]
                if combobox.isVisible() and combobox.isEnabled():
                    from_column = combobox.currentText()
                    if from_column == "None":
                        from_column = None
            selected.append(
                {
                    "feature_name": attr_name,
                    "include": include,
                    "from_column": from_column,
                }
            )
        return selected


class ParamsWidget(QWidget):
    """Widget for entering solver parameters"""

    def __init__(self, params: Params, group_title: str):
        super().__init__()

        layout = QVBoxLayout(self)

        # Solver parameters
        self.solver_params_group = QGroupBox(group_title)
        solver_layout = QVBoxLayout(self.solver_params_group)

        self.params = params()
        for name, field in self.params.model_fields.items():
            value = getattr(self.params, name)
            if isinstance(value, bool):
                checkbox = QCheckBox(field.title)
                checkbox.setChecked(value)
                checkbox.param_name = name
                checkbox.toggled.connect(self._on_param_changed)
                checkbox.setToolTip(field.description)
                solver_layout.addWidget(checkbox)
            elif isinstance(value, int):
                spinbox = QSpinBox()
                spinbox.setValue(value)
                spinbox.setMinimum(0)
                spinbox.setToolTip(field.description)
                spinbox.param_name = name
                spinbox.valueChanged.connect(self._on_param_changed)
                solver_layout.addWidget(QLabel(field.title + ":"))
                solver_layout.addWidget(spinbox)
            elif isinstance(value, float):
                double_spinbox = QDoubleSpinBox()
                double_spinbox.setValue(value)
                double_spinbox.setSingleStep(0.01)
                double_spinbox.setMinimum(0.0)
                double_spinbox.setToolTip(field.description)
                double_spinbox.param_name = name
                double_spinbox.valueChanged.connect(self._on_param_changed)
                solver_layout.addWidget(QLabel(field.title + ":"))
                solver_layout.addWidget(double_spinbox)

        self.solver_params_group.setLayout(solver_layout)
        layout.addWidget(self.solver_params_group)

    def _on_param_changed(self):
        """Update the SolverParams object when a parameter is changed"""

        sender = self.sender()
        if hasattr(sender, "param_name"):
            value = None
            if isinstance(sender, QCheckBox):
                value = sender.isChecked()
            elif isinstance(sender, QSpinBox) or isinstance(sender, QDoubleSpinBox):
                value = sender.value()
            if value is not None:
                setattr(self.params, sender.param_name, value)

    def get_param_values(self) -> Params:
        """Return a Params object with the parameter values entered by the user"""

        return self.params


class Page6(QWidget):
    """Page 6 of the Project dialog, to enter project parameters"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Create a content widget for the scroll area
        content_widget = QWidget()
        params_layout = QVBoxLayout(content_widget)

        self.project_params_widget = ParamsWidget(ProjectParams, "Project Parameters")
        self.cand_graph_params_widget = ParamsWidget(
            CandGraphParams, "Candidate Graph Parameters"
        )

        params_layout.addWidget(self.project_params_widget)
        params_layout.addWidget(self.cand_graph_params_widget)

        # Wrap in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        # Wrap in a group box
        box = QGroupBox("Project and Candidate Graph Parameters")
        box.setLayout(layout)
        layout = QVBoxLayout(self)
        layout.addWidget(box)

        self.setLayout(layout)

    def get_settings(self) -> dict[str:Any]:
        """Get the settings entered by the user on page 3"""

        settings = {
            "project_params": self.project_params_widget.get_param_values(),
            "cand_graph_params": self.cand_graph_params_widget.get_param_values(),
        }
        return settings


class Page7(QWidget):
    """Page 7 of the project dialog, to enter features to measure or import"""

    def __init__(
        self, page1: Page1, page2: Page2, page3: Page3, page4: Page4, page5: Page5
    ):
        super().__init__()

        self.page1 = page1
        self.page1.choice_updated.connect(self.update_features)
        mappable_columns = (
            self.page5.get_unmapped_columns()
            if self.page1.get_choice() == "curate_tracks"
            else []
        )
        self.page2 = page2
        self.page2.validity_changed.connect(self.update_features)
        self.include_intensity = True if self.page2.get_path() is not None else False
        self.page3 = page3
        data_type = self.page3.data_type
        self.page3.type_updated.connect(self.update_features)
        self.page4 = page4
        self.page4.dim_updated.connect(self.update_features)
        self.page5 = page5
        self.page5.mapping_updated.connect(self.update_features)

        if self.page4.incl_z:
            ndim = 4
        else:
            ndim = 3

        self.feature_widget = FeatureWidget(
            ndim=ndim,
            mappable_columns=mappable_columns,
            include_intensity=self.include_intensity,
            data_type=data_type,
        )

        # wrap in a group box
        layout = QVBoxLayout(self)
        feature_group_box = QGroupBox("Node Features")
        feature_group_layout = QVBoxLayout(feature_group_box)
        feature_group_layout.addWidget(QLabel("Choose which features to include"))
        feature_group_layout.addWidget(self.feature_widget)
        feature_group_box.setLayout(feature_group_layout)
        layout.addWidget(feature_group_box)
        self.setLayout(layout)

    def update_features(self):
        """Update the features based on the selected dimensions"""

        if self.page1.get_choice() != "curate_tracks":
            mappable_columns = []
        else:
            mappable_columns = self.page5.get_unmapped_columns()
        if self.page2.get_path() is None or not os.path.exists(self.page2.get_path()):
            self.include_intensity = False
        else:
            self.include_intensity = True

        if self.page4.incl_z:
            ndim = 4
        else:
            ndim = 3

        self.feature_widget._update_features(
            ndim,
            mappable_columns=mappable_columns,
            include_intensity=self.include_intensity,
        )

    def get_selected_features(self) -> dict[str:Any]:
        """Get the features selected by the user in the FeatureWidget"""

        return self.feature_widget.get_selected_features()


class Page8(QWidget):
    """Page 8 of the project dialog, to enter information on where to store the project"""

    validity_changed = Signal()

    def __init__(self):
        super().__init__()

        self.is_valid = False
        layout = QVBoxLayout()

        # Project title
        title_layout = QVBoxLayout()
        title_layout.addWidget(QLabel("How would you like to name your project?"))
        self.title_edit = QLineEdit()
        self.title_edit.setText("MotileTrackerProject")
        self.title_edit.textChanged.connect(self.validate)
        title_layout.addWidget(self.title_edit)
        layout.addLayout(title_layout)

        # Project directory
        dir_layout = QVBoxLayout()
        dir_layout.addWidget(QLabel("Where would you like to save your project?"))
        line_edit_browse_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.textChanged.connect(self.validate)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dir)
        line_edit_browse_layout.addWidget(self.dir_edit)
        line_edit_browse_layout.addWidget(browse_btn)
        dir_layout.addLayout(line_edit_browse_layout)
        layout.addLayout(dir_layout)

        # wrap everything in a group box
        box = QGroupBox("Project Information")
        box.setLayout(layout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def _browse_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.dir_edit.setText(directory)
        self.validate()

    def validate(self):
        title_ok = bool(self.title_edit.text().strip())
        dir_path = self.dir_edit.text().strip()
        dir_ok = bool(dir_path) and os.path.isdir(dir_path)

        self.is_valid = title_ok and dir_ok

        print("page 8 valid", self.is_valid)
        self.validity_changed.emit()

    def get_settings(self) -> dict[str:str]:
        return {
            "title": self.title_edit.text(),
            "directory": self.dir_edit.text(),
        }


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
            self.create_project()
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
