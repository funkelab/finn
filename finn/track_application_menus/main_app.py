from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import finn
from finn.track_application_menus.menu_widget import MenuWidget
from finn.track_data_views.views.tree_view.tree_widget import TreeWidget
from finn.track_data_views.views.view_3d.orthogonal_views import OrthogonalViews


class MainApp(QWidget):
    """Combines the different tracker widgets for faster dock arrangement"""

    def __init__(self, viewer: finn.Viewer):
        super().__init__()
        self.viewer = viewer

        self.viewer.dims.events.ndim.connect(self.dims_changed)

        self.menu_widget = MenuWidget(viewer)
        self.orth_views = OrthogonalViews(viewer)
        tree_widget = TreeWidget(viewer)

        self.splitter = QSplitter(Qt.Horizontal)  # Set orientation to horizontal
        self.splitter.addWidget(self.orth_views)  # Add the orthogonal views
        self.splitter.addWidget(self.menu_widget)  # Add the menu widget
        self.splitter.setSizes([0, 200])
        viewer.window.add_dock_widget(tree_widget, area="bottom", name="Tree View")

        layout = QVBoxLayout()

        self.collapse_btn = QPushButton("Show/hide orthogonal views")
        self.collapse_btn.clicked.connect(self.toggle_orth_views)
        if self.viewer.dims.ndim < 3:
            self.collapse_btn.hide()
        layout.addWidget(self.collapse_btn)
        layout.addWidget(self.splitter)

        self.setLayout(layout)

    def toggle_orth_views(self):
        """Show/Hide the orthogonal views depending on their current state"""
        sizes = self.splitter.sizes()
        if sizes[0] > 0:
            self.splitter.setSizes([0, 200])
        else:
            self.splitter.setSizes([300, 200])

    def dims_changed(self):
        """Show/Hide the orthogonal views depending on the amount of dimensions of the viewer"""

        if self.viewer.dims.ndim > 2:
            self.collapse_btn.show()
            self.splitter.setSizes([300, 200])
        else:
            self.collapse_btn.hide()
            self.splitter.setSizes([0, 200])
