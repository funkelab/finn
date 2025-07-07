from typing import Any

import pandas as pd
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
from superqt import QCollapsible

from finn.track_application_menus.welcome.new_project_pages.page2_raw_data import Page2
from finn.track_application_menus.welcome.new_project_pages.page3_seg_data import Page3


class Page4(QWidget):
    """Page 4 of the Project dialog, to set the data dimensions and scaling. Provides an
    interactive table widget to set the order of the dimensions, axis names, units, and
    scaling. Checks that the dimensions of the intensity data and detection data, if both
    provided, are compatible. Intensity data is allowed to have an additional axis for
    channels, which is not used for detection data."""

    validity_changed = Signal()
    dim_updated = Signal()

    def __init__(self, page2: Page2, page3: Page3):
        super().__init__()

        self.page2 = page2
        self.page3 = page3
        self.page3.validity_changed.connect(self._guess_data_dimensions)

        self.is_valid = False
        self.ndim = 3
        self.incl_z = False
        self.has_channels = False
        self.allow_channels = False
        self.raw = None
        self.seg = None
        self.points = None

        self.default_order = ("channel", "time", "z", "y", "x")
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
        self.detection_column_header = "Index (seg)"

        # Collapsible help widget
        instructions = QLabel(
            "<qt><i>Please check specify the order of dimensions in "
            "your data. If you intensity data has channels, please make sure to put it as"
            " the first dimensions. All other dimension sizes much match with the "
            "segmentation data, if provided. If you are importing point detections, "
            "please specify the columns that point to each of the dimension. You may also"
            " provide spatial scaling information and units.</i></qt>"
        )
        instructions.setWordWrap(True)
        collapsible_widget = QCollapsible("Explanation")
        collapsible_widget.layout().setContentsMargins(0, 0, 0, 0)
        collapsible_widget.layout().setSpacing(0)
        collapsible_widget.addWidget(instructions)
        collapsible_widget.collapse(animate=False)

        # Labels to show dynamically to provide additional information when needed.
        self.channel_label = QLabel(
            "<i>Please rearrange your dimension order such that the channel axis is the"
            " first dimension</i>"
        )
        self.incompatible_label = QLabel(
            "❌ The shapes of the intensity data and the segmentation data do not match."
            "\n\n Please check if your intensity data has channels or go back to select"
            " different data."
        )

        # Display the shape of the intensity and/or segmentation data
        self.intensity_label = QLabel("Intensity data shape:")
        self.detection_label = QLabel("Segmentation data shape:")

        # Table for axes
        self.table = QTableWidget(5, 6)
        self.table.setHorizontalHeaderLabels(
            [
                "Dimension",
                "Index (intensity)",
                self.detection_column_header,
                "Name",
                "Unit",
                "Step size",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # wrap everything in a group box
        box = QGroupBox("Data dimensions")
        box_layout = QVBoxLayout()
        box_layout.addWidget(collapsible_widget)
        box_layout.addWidget(self.channel_label)
        box_layout.addWidget(self.detection_label)
        box_layout.addWidget(self.intensity_label)
        box_layout.addWidget(self.incompatible_label)
        box_layout.addWidget(self.table)
        box.setLayout(box_layout)

        # Set the box to the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

        # Guess dimensions from the data and fill table with default values
        self._guess_data_dimensions()

    def _guess_data_dimensions(self):
        """Load the data provided by the user as dask arrays (or csv) and guess the
        dimensions. Display the shape of the data. Update the table according to the data
        type and dimensoins."""

        # return early if the previous pages are not yet valid.
        if not (self.page2.is_valid and self.page3.is_valid):
            self.is_valid = False
            return

        # Reset everything first.
        self.intensity_label.setVisible(False)
        self.detection_label.setVisible(False)
        self.raw = None
        self.seg = None
        self.points = None

        # Load the image data, if provided, as dask arrays, and extract the shape.
        if self.page2.get_path() is not None:
            self.raw = self.page2.intensity_data_widget.load_images()
            self.intensity_label.setText(
                f"Intensity data shape: {self.raw.shape} ->{self.raw.shape}"
            )
            self.intensity_label.setVisible(True)
        else:
            self.raw = None

        if self.page3.get_path() is not None and self.page3.data_type == "segmentation":
            self.seg = self.page3.data_widget.load_images()
            self.detection_label.setText(
                f"Segmentation data shape: {self.seg.shape} -> {self.seg.shape}"
            )
            self.detection_label.setVisible(True)

        # Load point data, if provided, as csv.
        elif self.page3.get_path() is not None and self.page3.data_type == "points":
            self.seg = None
            self.points = pd.read_csv(self.page3.get_path())
            self.detection_label.setText("Using columns from points data")

        # Update the column header according to the data type
        self.detection_column_header = (
            "Index (seg)" if self.page3.data_type == "segmentation" else "CSV Column"
        )

        # Populate the table according to the current data type and dimensions.
        self.update_table()

    def _set_allow_channels(self) -> None:
        """Sets the value for the number of dimensions, for 'allow_channels' (whether the
        channel dimension is optional), and, if known, for 'has_channels' (whether the
        data contains channels for sure). Depending on the number of dimensions and the
        type of data, we should give the user the option to choose between 'channel' and
        'z', or just provide fixed labels for the dimensions. 'allow_channels' refers to
        whether the user has to option to choose between 'channel' and 'z',
        'has_channels' refers to whether the data is set to have channels, depending on
        the number of dimensions."""

        if self.raw is None and self.seg is None:
            self.allow_channels = False
            self.has_channels = True
            self.ndim = 3
        elif self.raw is not None and self.seg is None:
            self.allow_channels = len(self.raw.shape) == 4  # when raw has 4 dims, it is
            # ambiguous if the 4th axis is 'z' or 'channel'. We therefore need both
            # options, which will be enabled by setting self.allow_channels = True. In the
            # case of 5 dims, 'has_channels' is set automatically set to True, as there is
            # no ambiguity.
            self.ndim = len(self.raw.shape)
        elif self.raw is not None and self.seg is not None:
            self.ndim = len(self.raw.shape)
            self.allow_channels = len(self.raw.shape) == len(self.seg.shape) + 1
        elif self.raw is None and self.seg is not None:
            self.allow_channels = False
            self.ndim = len(self.seg.shape)

        if not self.allow_channels:
            self.has_channels = False

    def _get_initial_mapping(self, csv_columns: list[str], incl_z: bool) -> list[str]:
        """Make an initial guess for mapping of csv columns to fields"""

        mapping = {}
        columns_left: list = csv_columns.copy()

        standard_fields = ["t", "z", "y", "x"]
        if not incl_z:
            standard_fields.remove("z")

        # find exact matches for standard fields
        for attribute in standard_fields:
            if attribute in columns_left:
                mapping[attribute] = attribute
                columns_left.remove(attribute)

        # assign first remaining column as best guess for remaining standard fields
        for attribute in standard_fields:
            if attribute in mapping:
                continue
            if len(columns_left) > 0:
                mapping[attribute] = columns_left.pop(0)
            else:
                # no good guesses left - just put something
                mapping[attribute] = csv_columns[-1]

        return list(mapping.values())

    def _update_axis(self, axis_name: str):
        """Updates the top row of the table with the dimension chosen by the user, either
        'channel' or 'z'"""

        # User chosen axis name (editable)
        self.table.setCellWidget(0, 3, QLineEdit(axis_name))

        # Unit (dropdown)
        unit_combo = QComboBox()
        unit_combo.addItems(self.units[axis_name])
        unit_combo.setCurrentText(self.default_units[axis_name])
        self.table.setCellWidget(0, 4, unit_combo)

        # Update widgets and labels. Step size is a fixed label with value 1 for channels,
        # but a QDoubleSpinBox for z. The seg combobox should be disabled for channels,
        # but not for z.
        if axis_name == "channel":
            step_spin = QLabel("1")

            # disable the combobox of the seg data, because it is not allowed to have
            # channels.
            widget = self.table.cellWidget(0, 2)
            if isinstance(widget, QComboBox):
                widget.setEnabled(False)

            self.channel_label.setVisible(True)
            self.incl_z = False
            self.has_channels = True

        else:
            step_spin = QDoubleSpinBox()
            step_spin.setDecimals(3)
            step_spin.setSingleStep(0.1)
            step_spin.setValue(self.step_size[axis_name])
            step_spin.setMinimum(0.0)

            widget = self.table.cellWidget(0, 2)
            if isinstance(widget, QComboBox):
                widget.setEnabled(True)

            self.channel_label.setVisible(False)
            self.incl_z = True
            self.has_channels = False

        self.table.setCellWidget(0, 5, step_spin)

        # Automatic column mapping from CSV
        if self.points is not None and axis_name == "z":
            axis_labels = list(self.points.select_dtypes(include="number").columns)
            guess_map_values = self._get_initial_mapping(list(axis_labels), self.incl_z)
            widget = self.table.cellWidget(0, 2)
            widget.setCurrentText(str(guess_map_values[1]))

        # Emit a signal that the dimensions have been updated.
        self.dim_updated.emit()

        # check whether the current settings are valid.
        self.validate()

    def update_table(self):
        """Populate the table depending on the current dimensions and datatype."""

        # check data dimensions
        self._set_allow_channels()

        # determine the axes from the number of dimensions.
        if self.ndim == 5:
            axes = ["channel", "time", "z", "y", "x"]
            self.channel_label.setVisible(True)
        elif self.ndim == 4:
            if self.allow_channels:
                axes = [["z", "channel"], "time", "y", "x"]
            else:
                axes = ["time", "z", "y", "x"]
        else:
            axes = ["time", "y", "x"]
        self.incl_z = "z" in axes
        self.has_channels = "channel" in axes

        # populate the columns row by row for the given axes.
        self.table.setRowCount(len(axes))
        for row, axis in enumerate(axes):
            # Axis label (not editable, except when the user has to choose between
            # 'channel' and 'z')
            if isinstance(axis, list):
                item = QComboBox()
                item.addItems(axis)
                item.setCurrentText("channel")
                self.has_channels = True
                item.currentTextChanged.connect(self._update_axis)  # connect to update
                # function, allowing the user to switch between 'z' and 'channel'.
                axis_name = item.currentText()
                self.table.setCellWidget(row, 0, item)
            else:
                existing_widget = self.table.cellWidget(row, 0)
                if existing_widget is not None:
                    self.table.removeCellWidget(row, 0)
                item = QTableWidgetItem(axis)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # readonly
                axis_name = axis
                self.table.setItem(row, 0, item)

            # Axis indices (dropdown)
            self.raw_axis_indices = QComboBox()
            if self.raw is None:
                self.raw_axis_indices.setEnabled(False)  # disable when not used
            else:
                self.raw_axis_indices.setEnabled(True)
            indices = range(self.ndim)
            self.raw_axis_indices.addItems([str(i) for i in indices])
            self.raw_axis_indices.setCurrentText(str(indices[row]))
            self.raw_axis_indices.currentTextChanged.connect(self.validate)
            self.table.setCellWidget(row, 1, self.raw_axis_indices)

            self.seg_axis_indices = QComboBox()
            if self.seg is None and self.points is None:
                self.seg_axis_indices.setEnabled(False)
                self.seg_axis_indices.addItems([str(i) for i in indices])
            elif self.seg is not None:
                self.seg_axis_indices.setEnabled(True)
                indices = range(
                    self.seg.ndim
                )  # get indices from seg data directly, as it
                # might have n-1 dimensions compared to raw.
                self.seg_axis_indices.addItems([str(i) for i in indices])
                if self.has_channels:
                    self.seg_axis_indices.setCurrentText(str(indices[row - 1]))
                    if row == 0:
                        self.seg_axis_indices.setEnabled(False)  # disable the first
                        # combobox since seg cannot have channels.
                else:
                    self.seg_axis_indices.setCurrentText(str(indices[row]))
                self.seg_axis_indices.currentTextChanged.connect(self.validate)
            elif self.points is not None:
                self.seg_axis_indices.setEnabled(True)
                # get all numerical columns from the data frame and ask user to map them
                # to the axis labels.
                axis_labels = list(self.points.select_dtypes(include="number").columns)
                guess_map_values = self._get_initial_mapping(
                    list(axis_labels), self.incl_z
                )
                self.seg_axis_indices.addItems([str(i) for i in axis_labels])
                if self.has_channels:
                    if row == 0:
                        self.seg_axis_indices.setEnabled(False)
                    else:
                        self.seg_axis_indices.setCurrentText(
                            str(guess_map_values[row - 1])
                        )
                else:
                    self.seg_axis_indices.setCurrentText(guess_map_values[row])
                self.seg_axis_indices.currentTextChanged.connect(self.validate)

            self.table.setCellWidget(row, 2, self.seg_axis_indices)

            # Axis name (editable)
            self.table.setCellWidget(row, 3, QLineEdit(axis_name))

            # Units (dropdown)
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

        self.table.setHorizontalHeaderItem(
            2, QTableWidgetItem(self.detection_column_header)
        )

        # Send update signal and validate
        self.dim_updated.emit()
        self.validate()

    def validate(self) -> None:
        """Validate inputs on this page and then emits a signal"""

        # extract the display order in the table (this may vary when the user gets to
        # choose between 'channel' and 'z' in the first row).
        display_order = []
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if isinstance(widget, QComboBox):
                display_order.append(widget.currentText())
            else:
                item = self.table.item(row, 0)
                if isinstance(item, QTableWidgetItem):
                    display_order.append(item.text())
        expected_order = [ax for ax in self.default_order if ax in display_order]
        reorder_indices = [display_order.index(ax) for ax in expected_order]

        # Check that axis indices are unique
        raw_indices = []
        for row in range(self.table.rowCount()):
            index_widget = self.table.cellWidget(row, 1)
            if index_widget is not None:
                raw_indices.append(index_widget.currentText())
        raw_indices = [raw_indices[i] for i in reorder_indices]
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

        # update label color according to whether the axes order is valid, and also check
        # that the selected axis order results in matching shapes between raw and seg
        if self.raw is not None:
            reordered_raw_dims = tuple(self.raw.shape[int(i)] for i in raw_indices)
            color = "green" if raw_axis_ok else "red"
            self.intensity_label.setText(
                f"Intensity data shape: {self.raw.shape} -> "
                f"<span style='color:{color};'>{reordered_raw_dims}</span>"
            )

        if self.seg is not None:
            reordered_seg_dims = tuple(self.seg.shape[int(i)] for i in seg_indices)
            color = "green" if seg_axis_ok else "red"
            self.detection_label.setText(
                f"Segmentation data shape: {self.seg.shape} -> "
                f"<span style='color:{color};'>{reordered_seg_dims}</span>"
            )

        if self.raw is not None and self.seg is not None:
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

        # Set a green checkmark if everything is fine
        if shape_match and seg_axis_ok and raw_axis_ok:
            self.incompatible_label.setText("✅")

        # Emit a signal
        self.is_valid = raw_axis_ok and seg_axis_ok and shape_match
        self.validity_changed.emit()

    def get_settings(self) -> dict[str:Any]:
        """Get the settings on this page, which should contain the selected data, the
        number of dimensions, and a dictionary representing the information in the table.
        """

        info = {
            "intensity_image": self.raw,
            "segmentation_image": self.seg,
            "points_data": self.points,
            "ndim": self.ndim,
            "axes": {},
        }

        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if isinstance(widget, QComboBox):
                axis = widget.currentText()
            else:
                item = self.table.item(row, 0)
                axis = item.text()

            axis_dict = {}
            axis_dict["axis_name"] = self.table.cellWidget(row, 3).text()
            axis_dict["unit"] = self.table.cellWidget(row, 4).currentText()
            axis_dict["step_size"] = (
                1
                if isinstance(self.table.cellWidget(row, 5), QLabel)
                else self.table.cellWidget(row, 5).value()
            )
            raw_index = (
                int(self.table.cellWidget(row, 1).currentText())
                if self.raw is not None
                else None
            )
            axis_dict["raw_index"] = raw_index
            seg_index = (
                int(self.table.cellWidget(row, 2).currentText())
                if self.seg is not None and axis != "channel"
                else None
            )
            axis_dict["seg_index"] = seg_index
            axis_dict["column"] = (
                self.table.cellWidget(row, 2).currentText()
                if self.points is not None and axis != "channel"
                else None
            )
            if self.raw is not None:
                axis_dict["size"] = self.raw.shape[raw_index]
            elif self.seg is not None:
                axis_dict["size"] = self.seg.shape[seg_index]
            info["axes"][axis] = axis_dict

        return info
