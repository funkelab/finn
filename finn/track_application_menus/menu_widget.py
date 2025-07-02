from qtpy.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout

import finn
from finn.track_application_menus.editing_menu import EditingMenu
from finn.track_data_views.views_coordinator.project_viewer import ProjectViewer

# from motile_tracker.motile.menus.motile_widget import MotileWidget


class MenuWidget(QScrollArea):
    """Combines the different tracker menus into tabs for cleaner UI"""

    def __init__(self, viewer: finn.Viewer):
        super().__init__()

        tracks_viewer = ProjectViewer.get_instance(viewer)

        # motile_widget = MotileWidget(viewer)
        editing_widget = EditingMenu(viewer)

        self.tabwidget = QTabWidget()

        # tabwidget.addTab(motile_widget, "Track with Motile")
        self.tabwidget.addTab(tracks_viewer.tracks_list, "Tracks List")
        self.tabwidget.addTab(editing_widget, "Edit Tracks")

        layout = QVBoxLayout()
        layout.addWidget(self.tabwidget)

        self.setWidget(self.tabwidget)
        self.setWidgetResizable(True)

        self.setLayout(layout)
