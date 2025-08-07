from qtpy.QtWidgets import QVBoxLayout, QWidget

import finn
from finn.track_data_views.views.view_3d.cross_widget import CrossWidget
from finn.track_data_views.views.view_3d.multiple_view_widget import MultipleViewerWidget


class OrthogonalViews(QWidget):
    """A widget that combines the multiple viewer widget and cross widget into single widget"""

    def __init__(
        self,
        viewer: finn.Viewer,
    ):
        super().__init__()

        viewer = viewer
        multiple_viewer_widget = MultipleViewerWidget(viewer)
        cross_widget = CrossWidget(viewer)

        layout = QVBoxLayout()
        layout.addWidget(multiple_viewer_widget)
        layout.addWidget(cross_widget)

        self.setLayout(layout)
