
import os

from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from finn.track_application_menus.welcome.browse_data import DataWidget
from finn.track_application_menus.welcome.new_project_pages.page1_goals import Page1


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
