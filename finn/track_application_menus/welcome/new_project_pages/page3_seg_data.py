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
from finn.track_application_menus.welcome.new_project_pages.page2_raw_data import Page2


class Page3(QWidget):
    """Page 3 of the Project dialog, to select the detection data source"""

    validity_changed = Signal()

    def __init__(self, page1: Page1, page2: Page2):
        super().__init__()

        self.page1 = page1
        self.page1.choice_updated.connect(self._update_widgets)
        self.page2 = page2
        self.page2.validity_changed.connect(self.validate)
        self.is_valid = False
        self.data_type = "segmentation"

        # Collapsible help widget
        instructions = QLabel(
            "<qt><i>"
            "If you are tracking from scratch, please indicate if you want to do so by "
            "placing points, or by drawing segmentation labels. If you are loading "
            "existing detection data, please either provide points data as a csv file "
            "with one column with the coordinates for each dimension, or a label image "
            "with the same t(z)yx dimension sizes as the intensity data (if provided). "
            "You can provide label image data as a single tif (3D+time or 2D+time) a "
            "folder containing a time series of 2D or 3D tif images, or a zarr folder. "
            "If you are curating existing point based tracking data, you will be asked "
            "to provide the csv file in a later step.</i></qt>"
        )
        instructions.setWordWrap(True)
        collapsible_widget = QCollapsible("Explanation")
        collapsible_widget.layout().setContentsMargins(0, 0, 0, 0)
        collapsible_widget.layout().setSpacing(0)
        collapsible_widget.addWidget(instructions)
        collapsible_widget.collapse(animate=False)

        # Button widget to ask the user if they have segmentation or point data
        data_layout = QVBoxLayout()
        data_layout.setSpacing(15)

        title_layout = QVBoxLayout()
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
        self.points.toggled.connect(self._update_widgets)
        button_layout.setAlignment(Qt.AlignHCenter)
        button_layout.addWidget(self.points)
        button_layout.addWidget(self.labels)
        data_layout.addLayout(button_layout)

        # Label widgets that should show up dynamically to give extra information when
        # needed
        self.intensity_label = QLabel(
            "<i>Intensity data is required when tracking with "
            "points. <br><br>Please make sure that you have provided a valid path to the"
            " intensity data on the previous page. </i>"
        )
        self.intensity_label.setVisible(False)

        self.points_label = QLabel(
            "<i>Point detections will automatically be derived from the CSV file that you"
            " will be asked to provide in the second step after this one.</i>"
        )
        self.points_label.setWordWrap(True)
        self.points_label.setVisible(False)

        # Provide a widget to enter the path to the detection data
        self.data_widget = DataWidget()
        self.data_widget.validity_changed.connect(self.validate)

        # Wrap everything in a group box
        box = QGroupBox("Detection data")
        box_layout = QVBoxLayout()
        box_layout.addWidget(collapsible_widget)
        box_layout.addWidget(self.intensity_label)
        box_layout.addWidget(self.points_label)
        box_layout.addLayout(data_layout)
        box_layout.addWidget(self.data_widget)
        box.setLayout(box_layout)

        # Set the box to the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def _update_widgets(self):
        """Change the visibility of the data widget and labels depending on the data type
        and user choice"""

        self.data_type = "segmentation" if self.labels.isChecked() else "points"
        self.data_widget.update_type(self.data_type)

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

                # no need to provide a path to the points data when curating tracks, as
                # the points will be derived from the tracks CSV path later.
                if self.page1.get_choice() == "curate_tracks":
                    self.is_valid = intensity_path_valid
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
        """Returns the data type chosen by the user"""

        return {"data_type": self.data_type}
