import os

import tifffile
import zarr
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


class CsvFileWidget(QWidget):
    """QWidget for specifying the path to a CSV file containing point detections"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Radio buttons for mode selection
        self.radio_file = QRadioButton("Choose from file")
        self.radio_manual = QRadioButton("Manual tracking from scratch")
        self.radio_file.setChecked(True)
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_file)
        radio_layout.addWidget(self.radio_manual)
        layout.addLayout(radio_layout)

        # File selection UI
        self.csv_path_line = QLineEdit(self)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_csv)

        label = QLabel(
            "Point detections should be provided as a "
            "CSV file with columns 't', ('z'), 'y', "
            "'x'."
        )
        font = label.font()
        font.setItalic(True)
        label.setFont(font)
        label.setWordWrap(True)
        layout.addWidget(label)

        self.csv_sublayout = QHBoxLayout()
        self.csv_sublayout.addWidget(QLabel("CSV File:"))
        self.csv_sublayout.addWidget(self.csv_path_line)
        self.csv_sublayout.addWidget(self.browse_btn)
        layout.addLayout(self.csv_sublayout)
        layout.setAlignment(Qt.AlignTop)

        # Connect radio buttons to toggle file selection UI
        self.radio_file.toggled.connect(self._update_mode)
        self._update_mode()

    def _update_mode(self):
        enabled = self.radio_file.isChecked()
        for i in range(self.csv_sublayout.count()):
            widget = self.csv_sublayout.itemAt(i).widget()
            if widget:
                widget.setEnabled(enabled)

    def _browse_csv(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if file:
            self.csv_path_line.setText(file)

    def get_path(self):
        if self.radio_manual.isChecked():
            return None  # Indicates manual tracking
        return self.csv_path_line.text()

    def is_manual(self):
        return self.radio_manual.isChecked()


class SegmentationWidget(QWidget):
    """QWidget for specifying pixel calibration"""

    update_buttons = Signal()

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

        # File selection UI
        self.image_path_line = QLineEdit(self)
        self.image_path_line.editingFinished.connect(self.update_buttons.emit)
        self.image_browse_button = QPushButton("Browse", self)
        self.image_browse_button.setAutoDefault(0)
        self.image_browse_button.clicked.connect(self._browse_segmentation)

        image_widget = QWidget()
        image_layout = QVBoxLayout()
        image_sublayout = QHBoxLayout()
        image_sublayout.addWidget(QLabel("Segmentation File Path:"))
        image_sublayout.addWidget(self.image_path_line)
        image_sublayout.addWidget(self.image_browse_button)

        label = QLabel(
            "Segmentation files can either be a single tiff stack, or a directory inside"
            " a zarr folder."
        )
        font = label.font()
        font.setItalic(True)
        label.setFont(font)
        label.setWordWrap(True)
        image_layout.addWidget(label)
        image_layout.addLayout(image_sublayout)
        image_widget.setLayout(image_layout)
        image_widget.setMaximumHeight(100)

        layout.addWidget(image_widget)
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        self.image_widget = image_widget  # For toggling
        # Connect radio buttons to toggle file selection UI
        self.radio_file.toggled.connect(self._update_mode)
        self._update_mode()

    def _update_mode(self):
        enabled = self.radio_file.isChecked()
        self.image_widget.setEnabled(enabled)

    def _browse_segmentation(self) -> None:
        """Open custom dialog to select either a file or a folder"""
        dialog = FileFolderDialog(self)
        if dialog.exec_():
            selected_path = dialog.get_selected_path()
            if selected_path:
                self.image_path_line.setText(selected_path)

    def _load_segmentation(self) -> None:
        """Load the segmentation image file"""
        if os.path.exists(self.image_path_line.text()):
            if self.image_path_line.text().endswith(".tif"):
                segmentation = tifffile.imread(self.image_path_line.text())
            elif ".zarr" in self.image_path_line.text():
                segmentation = zarr.open(self.image_path_line.text())
            else:
                QMessageBox.warning(
                    self,
                    "Invalid file type",
                    "Please provide a tiff or zarr file for the segmentation image stack",
                )
                return
        else:
            segmentation = None
        self.segmentation = segmentation

    def get_path(self):
        if self.radio_manual.isChecked():
            return None  # Indicates manual tracking
        return self.image_path_line.text()

    def is_manual(self):
        return self.radio_manual.isChecked()


class FileFolderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Tif File or Zarr Folder")

        self.layout = QVBoxLayout(self)

        self.path_line_edit = QLineEdit(self)
        self.layout.addWidget(self.path_line_edit)

        button_layout = QHBoxLayout()

        self.file_button = QPushButton("Select tiff file", self)
        self.file_button.clicked.connect(self.select_file)
        self.file_button.setAutoDefault(False)
        self.file_button.setDefault(False)

        button_layout.addWidget(self.file_button)

        self.folder_button = QPushButton("Select zarr folder", self)
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
