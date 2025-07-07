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
from superqt import QCollapsible

from finn.track_application_menus.welcome.browse_data import DataWidget
from finn.track_application_menus.welcome.new_project_pages.page1_goals import Page1


class Page2(QWidget):
    """Page 2 of the Project dialog, to select the intensity data source"""

    validity_changed = Signal()

    def __init__(self, page1: Page1):
        super().__init__()

        self.page1 = page1
        self.page1.choice_updated.connect(self._toggle_question_data_widget)

        self.is_valid = False
        self.show_intensity_widget = False

        # Collapsible help widget
        instructions = QLabel(
            "<qt><i>"
            "Intensity data is required when you are tracking from scratch or if you "
            "have only point detections, but optional if you are loading "
            "segmentations. You can provide intensity image data as a single tif "
            "(3D+time or 2D+time), a folder containing a time series of 2D or 3D tif "
            "images, or a zarr folder.</i></qt>"
        )
        instructions.setWordWrap(True)
        collapsible_widget = QCollapsible("Explanation")
        collapsible_widget.layout().setContentsMargins(0, 0, 0, 0)
        collapsible_widget.layout().setSpacing(0)
        collapsible_widget.addWidget(instructions)
        collapsible_widget.collapse(animate=False)

        # Buttonwidget to ask the user if they have intensity data
        intensity_layout = QVBoxLayout()
        intensity_layout.setSpacing(15)

        title_layout = QVBoxLayout()
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
        self.yes.toggled.connect(self._toggle_question_data_widget)
        button_layout.setAlignment(Qt.AlignHCenter)
        button_layout.addWidget(self.yes)
        button_layout.addWidget(self.no)
        intensity_layout.addLayout(button_layout)

        self.intensity_button_widget = QWidget()
        self.intensity_button_widget.setLayout(intensity_layout)
        if self.page1.get_choice() == "track_from_scratch":
            self.intensity_button_widget.setVisible(False)
        else:
            self.intensity_button_widget.setVisible(True)

        # Provide a widget to enter the path to the intensity data
        self.intensity_data_widget = DataWidget()
        self.intensity_data_widget.validity_changed.connect(self.validate)

        # Wrap everything in a group box
        box = QGroupBox("Intensity image data")
        box_layout = QVBoxLayout()
        box_layout.addWidget(collapsible_widget)
        box_layout.addWidget(self.intensity_button_widget)
        box_layout.addWidget(self.intensity_data_widget)
        box.setLayout(box_layout)

        # Set the box to the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def _toggle_question_data_widget(self):
        """Toggle the visibility of the intensity question based on the user's choice."""

        if self.page1.get_choice() == "track_from_scratch":
            # providing intensity data is mandatory, so no need to ask the question and
            # show the buttons
            self.intensity_button_widget.setVisible(False)
            self.intensity_data_widget.setVisible(True)
        else:
            self.intensity_button_widget.setVisible(True)
            if self.yes.isChecked():
                self.intensity_data_widget.setVisible(True)
            else:
                # The user choose 'no', so no need to show the data widget.
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
