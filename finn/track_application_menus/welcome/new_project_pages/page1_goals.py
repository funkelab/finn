from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


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
            "I have intensity data and want to track objects from scratch by manually "
            "placing points or segmentation labels."
        )
        self.track_from_detections = QRadioButton(
            "I have (label or point) detections and want to track objects from these "
            "detections."
        )
        self.curate_tracks = QRadioButton(
            "I have tracking data from external software that I want to view and/or "
            "curate."
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
