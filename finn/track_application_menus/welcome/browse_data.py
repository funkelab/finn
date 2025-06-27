import os

import dask.array as da
import numpy as np
from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from finn_builtins.io._read import magic_imread


class DataWidget(QWidget):
    """QWidget for choosing image data on a file system"""

    validity_changed = Signal()

    def __init__(self):
        super().__init__()

        self.is_valid = False
        self.type = "segmentation"
        self.browse_function = self._browse_image

        self.setToolTip(
            "<qt><i>"
            "Image data can either be a single tif (3D+time or 2D+time) stack, a "
            "folder containing a time series of 2D or 3D tif images, or a zarr folder."
            "</i></qt>"
        )

        layout = QVBoxLayout(self)

        # File selection UI
        self.path_line = QLineEdit(self)
        self.image_browse_button = QPushButton("Browse", self)
        self.image_browse_button.setAutoDefault(0)
        self.image_browse_button.clicked.connect(self._browse)
        self.path_line.editingFinished.connect(self._validate)

        image_widget = QWidget()
        image_layout = QVBoxLayout()
        image_sublayout = QHBoxLayout()
        self.label = QLabel("Image data path")
        image_sublayout.addWidget(self.label)
        image_sublayout.addWidget(self.path_line)
        image_sublayout.addWidget(self.image_browse_button)

        image_layout.addLayout(image_sublayout)
        image_widget.setLayout(image_layout)
        image_widget.setMaximumHeight(100)

        layout.addWidget(image_widget)
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def update_type(self, type: str):
        self.type = type
        if type == "segmentation":
            self.label.setText("Image data path")
            self.path_line.setText("")
            self.browse_function = self._browse_image
            self.setToolTip(
                "<qt><i>"
                "Image data can either be a single tif (3D+time or 2D+time) stack, a "
                "folder containing a time series of 2D or 3D tif images, or a zarr "
                "folder."
                "</i></qt>"
            )
        else:
            self.label.setText("CSV file path")
            self.path_line.setText("")
            self.browse_function = self._browse_csv
            self.setToolTip(
                "<qt><i>"
                "Point data should be a CSV file with columns for the t, (z), y, x "
                "coordinates."
                "</i></qt>"
            )

    def _browse(self) -> None:
        self.browse_function()

    def _browse_image(self) -> None:
        """Open custom dialog to select either a file or a folder"""
        dialog = FileFolderDialog(self)
        if dialog.exec_():
            selected_path = dialog.get_selected_path()
            if selected_path:
                self.path_line.setText(selected_path)
        self._validate()

    def _browse_csv(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if file:
            self.path_line.setText(file)
        self._validate()

    def load_images(self) -> np.ndarray | da.Array:
        """Load the image data file(s)"""

        path = self.path_line.text()
        if path is not None and os.path.exists(path):
            try:
                data = magic_imread(path, use_dask=True)
            except:
                QMessageBox.warning(
                    self,
                    "No valid files were found!",
                    "Please provide a tif stack, tif series, or zarr file for the image "
                    "stack",
                )
                self.data = None
                return None
        else:
            data = None

        return data

    def get_path(self):
        path = self.path_line.text()
        return path if os.path.exists(path) else None

    def _validate(self):
        path = self.get_path()
        self.is_valid = path is not None and os.path.exists(path)
        self.validity_changed.emit()


class FileFolderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose an image file or a folder containing a time series")

        self.layout = QVBoxLayout(self)

        self.path_line_edit = QLineEdit(self)
        self.layout.addWidget(self.path_line_edit)

        button_layout = QHBoxLayout()

        self.file_button = QPushButton("Select file", self)
        self.file_button.clicked.connect(self.select_file)
        self.file_button.setAutoDefault(False)
        self.file_button.setDefault(False)

        button_layout.addWidget(self.file_button)

        self.folder_button = QPushButton("Select folder", self)
        self.folder_button.clicked.connect(self.select_folder)
        self.folder_button.setAutoDefault(False)
        self.folder_button.setDefault(False)
        button_layout.addWidget(self.folder_button)

        self.layout.addLayout(button_layout)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def select_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Segmentation File",
            "",
            "Segmentation Files (*.tiff *.zarr *.tif)",
        )
        if file:
            self.path_line_edit.setText(file)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if folder:
            self.path_line_edit.setText(folder)

    def get_selected_path(self):
        return self.path_line_edit.text()
