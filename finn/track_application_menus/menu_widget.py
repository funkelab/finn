from qtpy.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout

import finn

# from motile_tracker.motile.menus.motile_widget import MotileWidget
from finn.track_data_views.views.view_3d import View3D
from finn.track_data_views.views_coordinator.tracks_viewer import TracksViewer

from .editing_menu import EditingMenu


class MenuWidget(QScrollArea):
    """Combines the different tracker menus into tabs for cleaner UI"""

    def __init__(self, viewer: finn.Viewer):
        super().__init__()

        tracks_viewer = TracksViewer.get_instance(viewer)

        # motile_widget = MotileWidget(viewer)
        editing_widget = EditingMenu(viewer)
        view3D_widget = View3D(viewer)
        view3D_widget.update_tab.connect(self.update_3D_tab)

        self.tabwidget = QTabWidget()

        # tabwidget.addTab(motile_widget, "Track with Motile")
        self.tabwidget.addTab(view3D_widget, '3D viewing')
        self.tabwidget.addTab(tracks_viewer.tracks_list, 'Tracks List')
        self.tabwidget.addTab(editing_widget, 'Edit Tracks')

        layout = QVBoxLayout()
        layout.addWidget(self.tabwidget)

        self.setWidget(self.tabwidget)
        self.setWidgetResizable(True)

        self.setLayout(layout)

    def update_3D_tab(self):
        if self.tabwidget.currentIndex() == 0:
            self.tabwidget.setCurrentIndex(1)
            self.tabwidget.setCurrentIndex(0)
