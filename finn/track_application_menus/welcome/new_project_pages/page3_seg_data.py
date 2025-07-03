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
from finn.track_application_menus.welcome.new_project_pages.page2_raw_data import Page2


class Page3(QWidget):
    """Page 3 of the Project dialog, to select the detection data source"""

    validity_changed = Signal()

    def __init__(self, page1: Page1, page2: Page2):
        super().__init__()

        self.is_valid = False
        self.data_type = "segmentation"
        self.page1 = page1
        self.page1.choice_updated.connect(self._toggle_data_widget_visibility)
        self.page2 = page2
        self.page2.validity_changed.connect(self.validate)

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
        button_layout.setAlignment(Qt.AlignHCenter)
        button_layout.addWidget(self.points)
        button_layout.addWidget(self.labels)
        data_layout.addLayout(button_layout)

        layout = QVBoxLayout()
        layout.addLayout(data_layout)

        self.intensity_label = QLabel(
            "<i>Intensity data is required when tracking with "
            "points. <br><br>Please make sure that you have provided a valid path to the"
            " intensity data on the previous page. </i>"
        )
        self.intensity_label.setVisible(False)
        layout.addWidget(self.intensity_label)

        self.points_label = QLabel(
            "<i>Point detections will automatically be derived from the CSV file that you"
            " will be asked to provide in the second step after this one.</i>"
        )
        self.points_label.setVisible(False)
        layout.addWidget(self.points_label)

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
        choice = self.page1.get_choice()
        if choice == "track_from_scratch":
            self.data_widget.setVisible(False)
            self.label.setText(
                "Do you want to track by placing points or segmentation labels?"
            )
            self.points_label.setVisible(False)
            self.is_valid = True
        elif choice == "curate_tracks" and self.data_type == "points":
            self.points_label.setVisible(True)
            self.data_widget.setVisible(False)
        else:
            self.data_widget.setVisible(True)
            self.label.setText("Do you have point or label detection data?")
            self.points_label.setVisible(False)
        self.validate()

    def _toggle_data_type(self):
        """Toggle the visibility of the data widget based on the user's choice."""

        self.data_type = "segmentation" if self.labels.isChecked() else "points"
        self.data_widget.update_type(self.data_type)
        self._toggle_data_widget_visibility()
        self.validate()

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
            if self.points.isChecked():
                path_valid = path is not None and (
                    os.path.exists(path) and path.endswith("csv")
                )
                # In the case of points, we do need intensity data as well.
                intensity_path = self.page2.get_path()
                intensity_path_valid = intensity_path is not None and os.path.exists(
                    intensity_path
                )
                if not intensity_path_valid:
                    self.intensity_label.setVisible(True)

                if self.page1.get_choice() == "curate_tracks":
                    self.is_valid = intensity_path_valid  # no need to provide a path to
                    # the points data here, as the points will be derived from the tracks
                    # CSV path later.
                else:
                    self.is_valid = path_valid and intensity_path_valid

            else:
                # no need to check whether we have intensity data, just a valid path to
                # labels data is sufficient.
                self.is_valid = path is not None and (
                    os.path.exists(path) and not path.endswith("csv")
                )
                self.intensity_label.setVisible(False)

        if self.is_valid:
            self.intensity_label.setVisible(False)
        self.validity_changed.emit()

    def get_settings(self) -> dict[str]:
        """Returns the data_type chosen by the user"""

        return {"data_type": self.data_type}
