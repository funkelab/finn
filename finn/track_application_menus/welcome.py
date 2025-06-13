from qtpy.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


class WelcomeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Motile Tracker")
        layout = QVBoxLayout(self)
        label = QLabel(
            "Welcome to Motile Tracker!<br><br>"
            "Read the documentation "
            '<a href="https://funkelab.github.io/motile_tracker/">here</a>.<br><br>'
            "For questions or problems, please "
            '<a href="https://github.com/funkelab/motile_tracker/issues">'
            "create an issue</a>.<br><br>"
            "Please start a new project or open an existing one to continue."
        )
        label.setOpenExternalLinks(True)
        layout.addWidget(label)
        self.choice = None  # Will be set to "new" or "continue"
        new_project_btn = QPushButton("Start new project")
        continue_project_btn = QPushButton("Open existing project")
        import_data_btn = QPushButton("Import external data")
        new_project_btn.clicked.connect(self._new_project)
        continue_project_btn.clicked.connect(self._continue_project)
        import_data_btn.clicked.connect(self._import_data)
        layout.addWidget(new_project_btn)
        layout.addWidget(continue_project_btn)
        layout.addWidget(import_data_btn)
        self.setLayout(layout)
        self.setModal(True)

    def _new_project(self):
        self.choice = "new"
        self.accept()

    def _continue_project(self):
        self.choice = "continue"
        self.accept()

    def _import_data(self):
        self.choice = "import"
        self.accept()
