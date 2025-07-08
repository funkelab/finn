import importlib.resources

from qtpy.QtCore import Qt
from qtpy.QtSvg import QSvgRenderer, QSvgWidget
from qtpy.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


class WelcomeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Motile Tracker")
        layout = QVBoxLayout(self)

        # Logo
        with importlib.resources.path("finn.resources", "logo.svg") as svg_path:
            renderer = QSvgRenderer(str(svg_path))
            default_size = renderer.defaultSize()

            target_width = 200
            aspect_ratio = default_size.height() / default_size.width()
            target_height = int(target_width * aspect_ratio)

            svg_widget = QSvgWidget(str(svg_path))
            svg_widget.setFixedSize(target_width, target_height)
            layout.addWidget(svg_widget, alignment=Qt.AlignCenter)

        # Title
        title_label2 = QLabel(
            """<span style="font-size: 40pt; font-weight: bold; font-family: Arial;">
            Motile Tracker</span>"""
        )
        title_label2.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label2)
        layout.addWidget(title_label2)

        # Info label
        info_label = QLabel(
            "Read the documentation "
            '<a href="https://funkelab.github.io/motile_tracker/">here</a>.<br><br>'
            "For questions or problems, please "
            '<a href="https://github.com/funkelab/motile_tracker/issues">'
            "create an issue</a>."
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # buttons
        self.choice = None
        new_project_btn = QPushButton("Start new project")
        continue_project_btn = QPushButton("Open existing project")
        example_project_btn = QPushButton("Show me an example!")
        new_project_btn.clicked.connect(self._new_project)
        continue_project_btn.clicked.connect(self._continue_project)
        example_project_btn.clicked.connect(self._example_project)
        layout.addWidget(new_project_btn)
        layout.addWidget(continue_project_btn)
        layout.addWidget(example_project_btn)

        self.setLayout(layout)
        self.setModal(True)
        self.setFixedWidth(400)

    def _new_project(self):
        self.choice = "new"
        self.accept()

    def _continue_project(self):
        self.choice = "continue"
        self.accept()

    def _example_project(self):
        self.choice = "example"
        self.accept()
