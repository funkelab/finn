import os
from typing import Any

import funlib.persistence as fp
import numpy as np
import tifffile
import zarr
from funtracks.features.node_features import Area
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
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from finn.track_application_menus.browse_data import DataSourceWidget, DataWidget
from finn.track_application_menus.csv_widget import CSVWidget


class Page1(QWidget):
    def __init__(self):
        super().__init__()

        layout1 = QVBoxLayout(self)
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
        self.ndim = 3
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

    def _browse_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.dir_edit.setText(directory)

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

    def validate_page1(self) -> bool:
        """Validate inputs on page 1 and enable/disable the Next button."""

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

        valid = title_ok and dir_ok and indices_unique
        return valid

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


class CreateNewWidget(QWidget):
    """Widget for 'Create new' mode: detection data (seg/points/manual)"""

    validity_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_valid = False
        layout = QVBoxLayout(self)

        self.segmentation_tooltip_text = (
            '<qt><div style="width: 300px;"><i>'
            "Image data can either be a single tif 3D+time or 2D+time stack,<br>"
            "a folder containing a time series of 2D or 3D tif images, or a zarr folder."
            "</i></div></qt>"
        )
        self.points_tooltip_text = (
            "Point detections should be provided as a CSV file with columns t, (z), y, x."
        )
        self.tooltip_text = self.segmentation_tooltip_text

        # Detection type
        data_type_group = QGroupBox("Detection type")
        data_type_layout = QHBoxLayout(data_type_group)
        self.seg_radio = QRadioButton("Segmentation labels")
        self.points_radio = QRadioButton("Point detections")
        self.seg_radio.setChecked(True)
        data_type_layout.addWidget(self.seg_radio)
        data_type_layout.addWidget(self.points_radio)

        # Data stacked (segmentation or points)
        self.data_source_group = QGroupBox("Data source")
        self.data_source_group.setToolTip(self.tooltip_text)
        data_source_layout = QHBoxLayout(self.data_source_group)
        self.data_source_group.setToolTip(self.tooltip_text)

        self.data_source_widget = DataSourceWidget()
        self.data_source_widget.validity_changed.connect(self._validate)
        self.seg_radio.toggled.connect(self._update_data_stack)
        self.points_radio.toggled.connect(self._update_data_stack)
        data_source_layout.addWidget(self.data_source_widget)

        # connect to updates to verify whether the data entered is valid
        self.seg_radio.toggled.connect(self._validate)
        self.points_radio.toggled.connect(self._validate)

        layout.addWidget(data_type_group)
        layout.addWidget(self.data_source_group)

        self.setLayout(layout)

    def get_data_type(self) -> str:
        """Return the type of data (segmentation or points) that was selected by the user"""
        return "segmentation" if self.seg_radio.isChecked() else "points"

    def get_path(self) -> str:
        """Return the path to the data that was selected, which may the user chose to track from scratch"""
        return self.data_source_widget.get_path()

    def set_tooltip_text(self) -> None:
        """Change the tooltip text depending on the choice between 'segmentation labels'
        and 'point detections'."""

        self.data_source_group.setToolTip(self.tooltip_text)

    def _update_data_stack(self):
        if self.seg_radio.isChecked():
            self.data_source_widget.update_type("segmentation")
            self.tooltip_text = self.segmentation_tooltip_text
            self.set_tooltip_text()
        else:
            self.data_source_widget.update_type("points")
            self.tooltip_text = self.points_tooltip_text
            self.set_tooltip_text()

    def _validate(self) -> None:
        # Validate detection (segmentation/csv/manual)
        is_manual = self.data_source_widget.is_manual()
        seg_path = self.data_source_widget.get_path()
        self.is_valid = is_manual or (seg_path is not None and os.path.exists(seg_path))
        self.validity_changed.emit()


class ImportExternalWidget(QWidget):
    """Widget for 'Import external tracks from CSV': add segmentation, CSVWidget"""

    validity_changed = Signal()

    def __init__(self, parent=None, incl_z=False):
        super().__init__(parent)
        self.is_valid = False
        layout = QVBoxLayout(self)

        # import group
        import_data_group = QGroupBox("Import external tracking data")
        self.import_data_layout = QVBoxLayout(import_data_group)

        # Add segmentation checkbox
        self.add_segmentation_checkbox = QCheckBox("Add segmentation data")
        self.import_data_layout.addWidget(self.add_segmentation_checkbox)

        # Segmentation widget (only image selection part)
        self.data_widget = DataWidget()
        self.data_widget.validity_changed.connect(self.validate)
        self.data_widget.is_valid = True  # True by default, as it is not mandatory
        self.import_data_layout.addWidget(self.data_widget)
        self.data_widget.hide()

        # CSVWidget for tracks
        self.csv_widget = CSVWidget(
            add_segmentation=self.add_segmentation_checkbox.isChecked(), incl_z=incl_z
        )
        self.csv_widget.validity_changed.connect(self.validate)
        self.import_data_layout.addWidget(self.csv_widget)

        layout.addWidget(import_data_group)
        # Connect checkbox to show/hide segmentation and update CSVWidget
        self.add_segmentation_checkbox.toggled.connect(self._update_segmentation)
        self.setLayout(layout)

    def _update_segmentation(self):
        show = self.add_segmentation_checkbox.isChecked()
        self.data_widget.setVisible(show)
        parent_layout = self.import_data_layout
        self.csv_widget.setParent(None)
        self.csv_widget.deleteLater()
        self.csv_widget = CSVWidget(add_segmentation=show, incl_z=self.csv_widget.incl_z)
        self.csv_widget.validity_changed.connect(self.validate)
        parent_layout.addWidget(self.csv_widget)

    def get_path(self) -> str | None:
        """Return the path to the data, if provided."""
        path = self.data_widget.get_path()
        return path if len(path) > 0 else None

    def validate(self) -> None:
        """Check whether all required information was filled out and then emit a True/False signal"""

        data_widget_ok = self.data_widget.is_valid
        csv_widget_ok = self.csv_widget.is_valid
        valid = data_widget_ok and csv_widget_ok
        self.is_valid = valid
        self.validity_changed.emit()

    def get_tracks_path(self) -> str:
        return self.csv_widget.get_path()

    def get_mapping(self) -> str:
        return self.csv_widget.get_name_map()


class Page2(QWidget):
    validity_changed = Signal()

    def __init__(self, parent=None, page1=None):
        super().__init__(parent)
        self.page1 = page1
        self.page1.radio_3d.toggled.connect(self._update_incl_z)
        self.is_valid = False
        layout2 = QVBoxLayout(self)

        # Data import choice
        import_data_group = QGroupBox("Data import")
        import_data_layout = QHBoxLayout(import_data_group)
        self.create_new_radio = QRadioButton("Create new")
        self.import_external_radio = QRadioButton("Import external tracks from CSV")
        self.create_new_radio.setChecked(True)
        import_data_layout.addWidget(self.create_new_radio)
        import_data_layout.addWidget(self.import_external_radio)
        layout2.addWidget(import_data_group)

        # Intensity data
        intensity_data_group = QGroupBox("Intensity image data")
        intensity_data_group.setToolTip(
            "<qt><i>"
            "Image data can either be a single tif (3D+time or 2D+time) stack, a "
            "folder containing a time series of 2D or 3D tif images, or a zarr folder."
            "</i></qt>"
        )

        intensity_layout = QHBoxLayout(intensity_data_group)
        self.intensity_widget = DataWidget()
        intensity_layout.addWidget(self.intensity_widget)
        layout2.addWidget(intensity_data_group)
        self.intensity_widget.validity_changed.connect(self.validate)

        # Stacked widget for the two modes 'new' and 'import'
        self.mode_stacked = QStackedWidget()
        self.create_new_widget = CreateNewWidget(self)
        self.create_new_widget.validity_changed.connect(self.validate)
        incl_z = self.page1.radio_3d.isChecked() if self.page1 else False
        self.import_external_widget = ImportExternalWidget(incl_z=incl_z)
        self.import_external_widget.validity_changed.connect(self.validate)
        self.mode_stacked.addWidget(self.create_new_widget)
        self.mode_stacked.addWidget(self.import_external_widget)
        layout2.addWidget(self.mode_stacked)

        self.create_new_radio.toggled.connect(self._update_mode)
        self.import_external_radio.toggled.connect(self._update_mode)
        self._update_mode()

        self.setLayout(layout2)

    def _update_incl_z(self):
        """Create a new ImportExternalWidget with incl_z value based on the 3D/2D selection."""
        incl_z = self.page1.ndim == 4
        old_widget = self.import_external_widget
        index = self.mode_stacked.indexOf(old_widget)
        if index != -1:
            self.mode_stacked.removeWidget(old_widget)
            old_widget.deleteLater()

        self.import_external_widget = ImportExternalWidget(incl_z=incl_z)
        self.import_external_widget.validity_changed.connect(self.validate)
        self.mode_stacked.insertWidget(index, self.import_external_widget)
        if not self.create_new_radio.isChecked():
            self.mode_stacked.setCurrentIndex(index)

    def _update_mode(self) -> None:
        """Change the index of the stacked widget to switch between the 'create new widget'
        and the 'import external tracks widget'. Also update the path to the intensity data
        and the data_widget depending on which widget was chosen."""

        if self.create_new_radio.isChecked():
            self.mode_stacked.setCurrentIndex(0)
        else:
            self.mode_stacked.setCurrentIndex(1)
        self.validate()

    def validate(self):
        if not os.path.exists(self.intensity_widget.get_path()):
            self.is_valid = False
        if self.create_new_radio.isChecked():
            self.is_valid = self.create_new_widget.is_valid
        else:
            self.is_valid = self.import_external_widget.is_valid

        self.validity_changed.emit()

    def get_settings(self) -> dict[str:str]:
        """Get the settings entered by the user"""

        settings = {
            "intensity_image": self.intensity_widget.get_path(),
            "tracks_path": None,
            "tracks_mapping": None,
        }
        if self.create_new_radio.isChecked():
            settings["data_type"] = (
                "segmentation"
                if self.create_new_widget.seg_radio.isChecked()
                else "points"
            )
            settings["data_path"] = (
                self.create_new_widget.get_path()
            )  # if data_path is None, track from scratch
        else:
            settings["data_type"] = (
                "segmentation"
                if self.import_external_widget.add_segmentation_checkbox.isChecked()
                else "points"
            )
            settings["data_path"] = (
                self.import_external_widget.get_path()
            )  # if data_path is None, track with points
            settings["tracks_path"] = self.import_external_widget.get_tracks_path()
            settings["tracks_mapping"] = self.import_external_widget.get_mapping()

        return settings


class FeatureWidget(QWidget):
    """Widget allowing the user to choose which features to include"""

    def __init__(self, ndim: int):
        super().__init__()

        feature_group_box = QGroupBox("Node Features")
        feature_group_layout = QVBoxLayout()

        layout = QVBoxLayout()
        feature_group_layout.addWidget(QLabel("Choose which features to include"))

        self.features_layout = QVBoxLayout()
        self._update_features(ndim)

        feature_group_layout.addLayout(self.features_layout)
        feature_group_box.setLayout(feature_group_layout)

        layout.addWidget(feature_group_box)
        self.setLayout(layout)

    def _update_features(self, ndim: int):
        self.measurement_checkboxes = {}
        # Remove all widgets from the layout
        while self.features_layout.count():
            item = self.features_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        # Now repopulate with new checkboxes
        self.features = [Area(ndim=ndim)]  # TODO import list of available features
        for feature in self.features:
            checkbox = QCheckBox(feature.value_names)
            checkbox.setChecked(False)
            self.measurement_checkboxes[feature.attr_name] = checkbox
            self.features_layout.addWidget(checkbox)

    def get_selected_features(self) -> list[str]:
        """Return the list of selected features"""

        selected_features = []
        for attr_name, checkbox in self.measurement_checkboxes:
            if checkbox.isChecked():
                selected_features.append(attr_name)

        return selected_features


class Page3(QWidget):
    """Page 3 of the Project dialog, to enter tracking related information"""

    validity_changed = Signal()

    def __init__(self, page1=None, page2=None):
        super().__init__()
        self.page1 = page1
        self.page2 = page2
        self.is_valid = True
        layout = QVBoxLayout()

        self.feature_widget = FeatureWidget(ndim=self.page1.ndim)
        self.page1.radio_3d.toggled.connect(self.feature_widget._update_features)

        layout.addWidget(self.feature_widget)
        self.setLayout(layout)

    def validate(self):
        self.is_valid = True
        self.validity_changed.emit()


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.resize(600, 400)
        self.stacked = QStackedWidget(self)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stacked)

        # --- Page 1: Project Info ---
        self.page1 = Page1()
        layout1 = QVBoxLayout()
        layout1.addWidget(self.page1)

        # Page 1: Add Cancel button
        btn_layout1 = QHBoxLayout()
        btn_layout1.addStretch()
        self.cancel_btn1 = QPushButton("Cancel")
        self.next_btn = QPushButton("Next")
        btn_layout1.addWidget(self.cancel_btn1)
        btn_layout1.addWidget(self.next_btn)
        layout1.addLayout(btn_layout1)

        self.next_btn.setEnabled(False)
        self.page1.axis_indices.currentIndexChanged.connect(self._validate_page1)
        self.page1.title_edit.textChanged.connect(self._validate_page1)
        self.page1.dir_edit.textChanged.connect(self._validate_page1)

        page1_widget = QWidget()
        page1_widget.setLayout(layout1)
        self.stacked.addWidget(page1_widget)

        # --- Page 2: Data Selection ---
        self.page2 = Page2(page1=self.page1)
        self.page2.validity_changed.connect(self._validate_page2)
        layout2 = QVBoxLayout()
        layout2.addWidget(self.page2)

        # Add Prev/Ok/Cancel buttons
        btn_layout2 = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.cancel_btn2 = QPushButton("Cancel")
        self.next_btn2 = QPushButton("Next")
        btn_layout2.addStretch()
        btn_layout2.addWidget(self.prev_btn)
        btn_layout2.addWidget(self.cancel_btn2)
        btn_layout2.addWidget(self.next_btn2)
        layout2.addLayout(btn_layout2)

        page2_widget = QWidget()
        page2_widget.setLayout(layout2)
        self.stacked.addWidget(page2_widget)

        # --- Page 3: Tracking settings ---
        self.page3 = Page3(page1=self.page1, page2=self.page2)
        self.page3.validity_changed.connect(self._validate_page3)
        layout3 = QVBoxLayout()
        layout3.addWidget(self.page3)

        # Add Prev/Ok/Cancel buttons
        btn_layout3 = QHBoxLayout()
        self.prev_btn2 = QPushButton("Previous")
        self.cancel_btn3 = QPushButton("Cancel")
        self.ok_btn = QPushButton("OK")
        btn_layout3.addStretch()
        btn_layout3.addWidget(self.prev_btn2)
        btn_layout3.addWidget(self.cancel_btn3)
        btn_layout3.addWidget(self.ok_btn)
        layout3.addLayout(btn_layout3)

        page3_widget = QWidget()
        page3_widget.setLayout(layout3)
        self.stacked.addWidget(page3_widget)

        # Connections for navigation
        self.next_btn.clicked.connect(self._go_to_page2)
        self.next_btn2.clicked.connect(self._go_to_page3)
        self.prev_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.prev_btn2.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        self.ok_btn.clicked.connect(self.accept)
        self.stacked.setCurrentIndex(0)

        # Connect cancel buttons to close the dialog
        self.cancel_btn1.clicked.connect(self._cancel)
        self.cancel_btn2.clicked.connect(self._cancel)
        self.cancel_btn3.clicked.connect(self._cancel)

    def _validate_page1(self):
        valid = self.page1.validate_page1()
        self.next_btn.setEnabled(valid)

    def _validate_page2(self):
        """Validate inputs on page 2 and enable/disable the NEXT button to page3."""
        self.next_btn2.setEnabled(self.page2.is_valid)

    def _validate_page3(self):
        """Validate inputs on page3 and enable/disable the OK button"""
        self.ok_btn.setEnabled(self.page3.is_valid)

    def _go_to_page2(self):
        self.stacked.setCurrentIndex(1)
        self.page2.validate()

    def _go_to_page3(self):
        self.stacked.setCurrentIndex(2)
        self.page3.validate()

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
        """

        page1_info = self.page1.get_settings()
        page2_info = self.page2.get_settings()

        info = page1_info | page2_info

        return info

    def _cancel(self):
        self.reject()
