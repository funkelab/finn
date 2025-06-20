import os

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
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from finn_builtins.io._read import magic_imread


class DataSourceWidget(QWidget):
    """Widget to combine SourceWidget and DataWidget"""

    validity_changed = Signal()

    def __init__(self):
        super().__init__()
        self.is_valid = False
        self.type = "segmentation"

        self.source_selection_widget = SourceWidget()
        self.source_selection_widget.radio_file.toggled.connect(self._update_mode)
        self.source_selection_widget.radio_manual.toggled.connect(self._update_mode)

        self.data_widget = DataWidget()
        self.data_widget.validity_changed.connect(self._validate)

        layout = QVBoxLayout()
        layout.addWidget(self.source_selection_widget)
        layout.addWidget(self.data_widget)
        self.setLayout(layout)

    def update_type(self, type: str):
        self.type = type
        self.data_widget.update_type(self.type)

    def _update_mode(self):
        enabled = self.source_selection_widget.radio_file.isChecked()
        self.data_widget.setEnabled(enabled)
        self._validate()

    def _validate(self):
        """Check whether all required information is filled out"""

        if self.source_selection_widget.is_manual():
            valid = True
        path = self.data_widget.get_path()
        if path is None or not os.path.exists(path):
            valid = False
        else:
            valid = True

        self.is_valid = valid
        self.validity_changed.emit()

    def is_manual(self):
        return self.source_selection_widget.is_manual()

    def get_path(self) -> str | None:
        if self.source_selection_widget.radio_file.isChecked():
            return self.data_widget.get_path()
        return None


class SourceWidget(QWidget):
    """Widget for choosing between loading data from file or from scratch"""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        # Radio buttons for mode selection
        self.radio_file = QRadioButton("Choose from file")
        self.radio_manual = QRadioButton("Manual tracking from scratch")
        self.radio_file.setChecked(True)
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_file)
        radio_layout.addWidget(self.radio_manual)
        layout.addLayout(radio_layout)

    def is_manual(self):
        return self.radio_manual.isChecked()


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
        self.path_line.textChanged.connect(self._validate)
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
                "folder containing a time series of 2D or 3D tif images, or a zarr folder."
                "</i></qt>"
            )
        else:
            self.label.setText("CSV file path")
            self.path_line.setText("")
            self.browse_function = self._browse_csv
            self.setToolTip(
                "<qt><i>"
                "Point data should be a CSV file with columns for the t, (z), y, x coordinates."
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

    def _browse_csv(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if file:
            self.path_line.setText(file)

    def _load_images(self) -> None:
        """Load the image data file(s)"""

        path = self.path_line.text()
        if os.path.exists(path):
            try:
                data = magic_imread(path)
            except:
                QMessageBox.warning(
                    self,
                    "No valid files were found!",
                    "Please provide a tif stack, tif series, or zarr file for the image stack",
                )
                self.data = None
                return
        else:
            data = None
        self.data = data

    def get_path(self):
        return self.path_line.text()

    def _validate(self):
        print("call to validate the data widget")
        self.is_valid = os.path.exists(self.get_path())
        self.validity_changed.emit()
        print("data widget is valid:", self.is_valid)


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
