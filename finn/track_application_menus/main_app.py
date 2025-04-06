from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

import finn
from finn.track_application_menus.menu_widget import MenuWidget
from finn.track_data_views.views.tree_view.tree_widget import TreeWidget


class MainApp(QWidget):
    """Combines the different tracker widgets for faster dock arrangement"""

    def __init__(self, viewer: finn.Viewer):
        super().__init__()

        self.menu_widget = MenuWidget(viewer)
        tree_widget = TreeWidget(viewer)

        viewer.window.add_dock_widget(tree_widget, area="bottom", name="Tree View")

        layout = QVBoxLayout()
        layout.addWidget(self.menu_widget)

        self.setLayout(layout)
