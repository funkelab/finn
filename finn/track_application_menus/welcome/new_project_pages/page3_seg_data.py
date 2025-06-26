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
