from typing import Any

import numpy as np
from psygnal import Signal
from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from finn.track_application_menus.welcome.new_project_pages.page2_raw_data import Page2
from finn.track_application_menus.welcome.new_project_pages.page3_seg_data import Page3


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
            "<i>Please rearrange your dimension order such that the channel axis is the"
            " first dimension</i>"
        )
        layout.addWidget(self.channel_label)

        self.intensity_label = QLabel("Intensity data shape:")
        layout.addWidget(self.intensity_label)

        self.seg_label = QLabel("Segmentation data shape:")
        layout.addWidget(self.seg_label)

        self.incompatible_label = QLabel(
            "❌ The shapes of the intensity data and the segmentation data do not match."
            "\n\n Please check if your intensity data has channels or go back to select"
            " different data."
        )
        layout.addWidget(self.incompatible_label)

        # Table for axes
        self.table = QTableWidget(5, 6)
        self.table.setHorizontalHeaderLabels(
            [
                "Dimension",
                "Index (intensity)",
                "Index (seg)",
                "Name",
                "Unit",
                "Step size",
            ]
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
                f"Intensity data shape: {self.intensity_image_shape} ->"
                f"{self.intensity_image_shape}"
            )
            self.intensity_label.setVisible(True)
        else:
            self.intensity_image = None

        if self.page3.get_path() is not None and self.page3.data_type == "segmentation":
            self.segmentation_image = self.page3.data_widget.load_images()
            self.segmentation_image_shape = self.segmentation_image.shape
            self.seg_label.setText(
                f"Segmentation data shape: {self.segmentation_image_shape} -> "
                f"{self.segmentation_image_shape}"
            )
            self.seg_label.setVisible(True)
        else:
            self.segmentation_image = None

        # check whether we should display the checkbox to ask about the channel dimension
        self._set_allow_channels()

        # Check if the shapes for intensity data and segmentation data are compatible
        self._update_table()

    def _set_allow_channels(self) -> None:
        """Channels label should be visible if the intensity data has one dimension more
        than the segmentation data"""

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
        """Updates the top row of the table with the dimension chosen by the user, either
        'channel' or 'z'"""

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
            self.incl_z = False
        else:
            step_spin = QDoubleSpinBox()
            step_spin.setDecimals(3)
            step_spin.setSingleStep(0.1)
            step_spin.setValue(self.step_size[axis_name])
            step_spin.setMinimum(0.0)
            self.channel_label.setVisible(False)
            self.has_channels = False
            self.incl_z = True
        self.table.setCellWidget(0, 5, step_spin)

        self.dim_updated.emit()
        self.validate()

    def _update_table(self):
        self.table.setVisible(True)

        self._set_allow_channels()

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
            # Axis label (not editable, except when the user has to choose between
            # 'channel' and 'z')
            if isinstance(axis, list):
                item = QComboBox()
                item.addItems(axis)
                item.currentTextChanged.connect(self._update_axis)
                item.setCurrentText("channel")
                axis_name = item.currentText()
                self.table.setCellWidget(row, 0, item)
            else:
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
            f"Intensity data shape: {self.intensity_image_shape} -> "
            f"{self.intensity_image_shape} "
        )
        self.seg_label.setText(
            f"Segmentation data shape: {self.segmentation_image_shape} -> "
            f"{self.segmentation_image_shape}"
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
                "❌ The shapes of the intensity data and the segmentation data do not "
                "match."
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
