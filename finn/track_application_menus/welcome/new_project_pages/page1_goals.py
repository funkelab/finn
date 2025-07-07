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
from superqt import QCollapsible


class Page1(QWidget):
    """Page 1 of the Project dialog, to choose the goal of the project."""

    choice_updated = Signal()

    def __init__(self):
        super().__init__()

        # Collapsible help widget
        instructions = QLabel(
            "<i>Please choose between any of the following project goals:<br> "
            "- Manually track objects by placing points or labels on your intensity "
            "data.<br> - Load existing segmentations or point detections, track "
            "objects with Motile, and view and curate the result.<br> - Load existing"
            " tracking data from a csv file and view or curate it.</i>"
        )
        instructions.setWordWrap(True)
        collapsible_widget = QCollapsible("Explanation")
        collapsible_widget.layout().setContentsMargins(0, 0, 0, 0)
        collapsible_widget.layout().setSpacing(0)
        collapsible_widget.addWidget(instructions)
        collapsible_widget.collapse(animate=False)

        # Ask the user about their project goal
        goal_layout = QVBoxLayout()
        label = QLabel("What would you like to do?")
        label.setAlignment(Qt.AlignHCenter)
        goal_layout.addWidget(label)

        self.track_from_scratch = QRadioButton(
            "Manually track objects by placing points or segmentatation labels."
        )
        self.track_from_detections = QRadioButton(
            "Track objects from in existing (label or point) detections."
        )
        self.curate_tracks = QRadioButton("Curate existing tracking data.")
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

        # Wrap everything in a group box
        box = QGroupBox("Project Goal")
        box_layout = QVBoxLayout()
        box_layout.addWidget(collapsible_widget)
        box_layout.addLayout(goal_layout)
        box.setLayout(box_layout)

        # Set the box to the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def get_choice(self) -> str | None:
        """Return user choice."""

        if self.track_from_scratch.isChecked():
            return "track_from_scratch"
        if self.track_from_detections.isChecked():
            return "track_from_detections"
        if self.curate_tracks.isChecked():
            return "curate_tracks"
        return None
