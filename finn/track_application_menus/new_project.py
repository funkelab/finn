import os
from typing import Any

import funlib.persistence as fp
import numpy as np
import tifffile
import zarr
from funtracks.features.measurement_features import featureset
from funtracks.params._base import Params
from funtracks.params.cand_graph_params import CandGraphParams
from funtracks.params.project_params import ProjectParams
from funtracks.project import Project
from motile_toolbox.utils.relabel_segmentation import ensure_unique_labels
from psygnal import Signal
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
        return ""

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
        if self.page1.get_choice() == "track_from_scratch":
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

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.is_valid = False
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

        # Dimension radio buttons
        dim_layout = QVBoxLayout()
        dim_layout.addWidget(QLabel("What dimensions does your data have?"))
        button_layout = QHBoxLayout()
        self.ndim = 3
        self.radio_2d = QRadioButton("2D + time")
        self.radio_3d = QRadioButton("3D + time")
        self.radio_2d.setChecked(True)
        self.dim_group = QButtonGroup(self)
        self.dim_group.addButton(self.radio_2d)
        self.dim_group.addButton(self.radio_3d)
        button_layout.addWidget(self.radio_2d)
        button_layout.addWidget(self.radio_3d)
        dim_layout.addLayout(button_layout)
        layout.addLayout(dim_layout)

        # Table for axes
        self.table = QTableWidget(4, 5)
        self.table.setHorizontalHeaderLabels(
            ["Dimension", "Index", "Name", "Unit", "Step size"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Fill table with default values
        self._update_table()

        # Connect radio buttons to update table
        self.radio_2d.toggled.connect(self._update_table)
        self.radio_3d.toggled.connect(self._update_table)

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

    def _update_table(self):
        self.ndim = 3 if self.radio_2d.isChecked() else 4
        is_3d = self.ndim == 4
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
            self.axis_indices = QComboBox()
            self.axis_indices.addItems([str(i) for i in axes_indices])
            self.axis_indices.setCurrentText(str(axes_indices[row]))
            self.table.setCellWidget(row, 1, self.axis_indices)
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

        self.dim_updated.emit()

    def validate(self):
        """Validate inputs on this page"""

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

        self.is_valid = title_ok and dir_ok and indices_unique
        self.validity_changed.emit()

    def get_settings(self) -> dict[str:Any]:
        """Get the settings on page1"""

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

        self.incl_z = True if self.page4.radio_3d.isChecked() else False
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

        if self.page4.radio_3d.isChecked():
            self.incl_z = True
        else:
            self.incl_z = False

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

        if self.page4.radio_3d.isChecked():
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

        if self.page4.radio_3d.isChecked():
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
        self.page4 = Page4()
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
        self.ok_btn = QPushButton("OK")
        self.cancel_btn7 = QPushButton("Cancel")

        btn_layout7.addStretch()
        btn_layout7.addWidget(self.cancel_btn7)
        btn_layout7.addWidget(self.prev_btn6)
        btn_layout7.addWidget(self.ok_btn)
        layout7.addLayout(btn_layout7)

        page7_widget = QWidget()
        page7_widget.setLayout(layout7)
        self.stacked.addWidget(page7_widget)

        # Connections for navigation
        self.next_btn.clicked.connect(self._go_to_page2)
        self.next_btn2.clicked.connect(self._go_to_page3)
        self.next_btn3.clicked.connect(self._go_to_page4)
        self.next_btn4.clicked.connect(self._go_to_page5_or_6)
        self.next_btn5.clicked.connect(lambda: self.stacked.setCurrentIndex(5))
        self.next_btn6.clicked.connect(self._go_to_page7)

        self.prev_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.prev_btn2.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        self.prev_btn3.clicked.connect(lambda: self.stacked.setCurrentIndex(2))
        self.prev_btn4.clicked.connect(self._go_to_page4)
        self.prev_btn5.clicked.connect(self._go_to_page4_or_5)
        self.prev_btn6.clicked.connect(lambda: self.stacked.setCurrentIndex(5))

        self.ok_btn.clicked.connect(self.accept)

        self.stacked.setCurrentIndex(0)

        # Connect cancel buttons to close the dialog
        self.cancel_btn1.clicked.connect(self._cancel)
        self.cancel_btn2.clicked.connect(self._cancel)
        self.cancel_btn3.clicked.connect(self._cancel)
        self.cancel_btn4.clicked.connect(self._cancel)
        self.cancel_btn5.clicked.connect(self._cancel)
        self.cancel_btn6.clicked.connect(self._cancel)
        self.cancel_btn7.clicked.connect(self._cancel)

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
            self._validate_page5()
        else:
            self.stacked.setCurrentIndex(5)

    def _go_to_page7(self):
        """Go to page 7 and make sure the mapping on page5 is updated first."""
        self.page5.mapping_updated.emit()  # ensure the mapping is updated
        self.stacked.setCurrentIndex(6)

    def get_project_info(self) -> dict[str:Any]:
        """Create a dictionary with the information from the different dialog pages.
        Returns:
            dict[str: Any] with the following information:
            - title [str]: name of the project,
            - directory [str]: path to directory where the project should be saved
            - dimensions [int]: the number of dimensions (incl time) of the data (3 or 4)
            - axes [dict]:
                dimensions [tuple[str]]: dimension names (e.g. 'time', 'z')
                indices [tuple[int]]: index of each dimension (e.g (0,1,2,3))
                axis_names [tuple(str)]: dimension names assigned by the user
                units (tuple[str]): units for each dimension, e.g. 'µm'
                scaling [tuple(float)]: spatial calibration in the same order as the dimensions
            - intensity_image [str]: path to the intensity data
            - tracks_path [str | None]: path to where the tracking data csv file is stored (if provided)
            - tracks_mapping [dict[str: str]]: mapping of the csv column headers to the required tracking information (dimensions, ids)
            - data_type [str]: either 'segmentation' or 'points'
            - data_path [str | None]: path to the segmentation data, if provided.
            - features [list[dict[str: str|bool]]]: list of features to measure, each with 'feature_name', 'include' (bool), and 'from_column' (str or None)
            - project_params [ProjectParams]: parameters for the project
            - cand_graph_params [CandGraphParams]: parameters for the candidate graph
        """

        choice = self.page1.get_choice()
        project_info = {}
        project_info["intensity_image"] = self.page2.get_path()
        project_info["data_path"] = self.page3.get_path()
        project_info["data_type"] = self.page3.data_type
        project_info = (
            project_info | self.page4.get_settings()
        )  # name, directory, dimensions, axes
        project_info["tracks_path"] = (
            self.page5.get_tracks_path() if choice == "curate_tracks" else None
        )
        project_info["column_mapping"] = (
            self.page5.get_mapping() if choice == "curate_tracks" else None
        )
        project_info = project_info | self.page6.get_settings()
        project_info["features"] = self.page7.get_selected_features()

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
        print(project_info)
        # name = project_info.get("name", "Untitled Project")
        # axes = project_info.get("axes", [])
        # detection_type = project_info.get("detection_type", "points")
        # working_dir = project_info.get("directory", Path.cwd())
        # params = project_info.get("project_params", None)

        # # create fpds for the intensity image and segmentation data (if provided)
        # intensity_image_path = project_info.get("intensity_image", None)
        # if intensity_image_path is not None:
        #     int_fps_path = os.path.join(working_dir, "motile_tracker.zarr/int")
        #     intensity_image = self.create_fp_array(
        #         intensity_image_path, int_fps_path, axes
        #     )

        # if detection_type == "segmentation":
        #     seg_path = project_info.get("detection_path", None)
        #     seg_fps_path = os.path.join(working_dir, "motile_tracker.zarr/int")

        #     if seg_path is None:
        #         segmentation = self.create_empty_fp_array(
        #             seg_fps_path, intensity_image.shape, axes
        #         )

        #     else:
        #         segmentation = self.create_fp_array(seg_path, seg_fps_path, axes)
        #         if segmentation.shape != intensity_image.shape:
        #             raise ValueError(
        #                 "Segmentation data shape does not match intensity image shape. "
        #                 f"Segmentation shape: {segmentation.shape}, Intensity image shape: {intensity_image.shape}"
        #             )

        # elif detection_type == "points":
        #     points_path = project_info.get("detection_path", None)
        #     if points_path is None:
        #         raise ValueError(
        #             "Points detection type selected, but no points file provided."
        #         )

        # return Project(
        #     name=name,
        #     project_params=params,
        #     raw=intensity_image,
        #     segmentation=segmentation,
        #     cand_graph=None,
        # )
        return project_info

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
