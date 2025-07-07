from pathlib import Path

from funtracks.project import Project
from qtpy.QtWidgets import (
    QFileDialog,
)


def load_project() -> Project | None:
    directory = QFileDialog.getExistingDirectory(
        parent=None,  # or use your actual parent widget
        caption="Select Project Directory",
    )
    if directory:
        return Project.load(Path(directory))
    return None
