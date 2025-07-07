import os

from psygnal import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible


class Page8(QWidget):
    """Page 8 of the project dialog, to enter information on where to store the project"""

    validity_changed = Signal()

    def __init__(self):
        super().__init__()

        self.is_valid = False

        # Collapsible help widget
        instructions = QLabel(
            "<qt><i>Please specify a name and a location to save this"
            " project. Note that existing projects at this location will be overwritten."
            "</i></qt>"
        )
        instructions.setWordWrap(True)
        collapsible_widget = QCollapsible("Explanation")
        collapsible_widget.layout().setContentsMargins(0, 0, 0, 0)
        collapsible_widget.layout().setSpacing(0)
        collapsible_widget.addWidget(instructions)
        collapsible_widget.collapse(animate=False)

        # Project title
        title_layout = QVBoxLayout()
        title_layout.addWidget(QLabel("How would you like to name your project?"))
        self.title_edit = QLineEdit()
        self.title_edit.setText("MotileTrackerProject")
        self.title_edit.textChanged.connect(self.validate)
        title_layout.addWidget(self.title_edit)

        # Project directory
        dir_layout = QVBoxLayout()
        dir_layout.addWidget(QLabel("Where would you like to save your project?"))
        line_edit_browse_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.textChanged.connect(self.validate)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dir)
        line_edit_browse_layout.addWidget(self.dir_edit)
        line_edit_browse_layout.addWidget(browse_btn)
        dir_layout.addLayout(line_edit_browse_layout)

        # wrap everything in a group box
        box = QGroupBox("Project Information")
        content_layout = QVBoxLayout()
        content_layout.addWidget(collapsible_widget)
        content_layout.addLayout(title_layout)
        content_layout.addLayout(dir_layout)
        box.setLayout(content_layout)

        # Set the box to the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def _browse_dir(self) -> None:
        """Open a file browser menu to choose output directory"""

        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.dir_edit.setText(directory)
        self.validate()

    def validate(self):
        """Check if a title and valid directory have been chosen, and emit a signal."""

        title_ok = bool(self.title_edit.text().strip())
        dir_path = self.dir_edit.text().strip()
        dir_ok = bool(dir_path) and os.path.isdir(dir_path)

        self.is_valid = title_ok and dir_ok
        self.validity_changed.emit()

    def get_settings(self) -> dict[str:str]:
        """Return the project title and output directory"""

        return {
            "title": self.title_edit.text(),
            "directory": self.dir_edit.text(),
        }
