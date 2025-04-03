from qtpy.QtWidgets import QVBoxLayout, QWidget

import finn
from finn.track_data_views.views.view_3d.cross_widget import CrossWidget
from finn.track_data_views.views.view_3d.multiple_view_widget import MultipleViewerWidget


class OrthogonalViews(QWidget):
    def __init__(
        self,
        viewer: finn.Viewer,
    ):
        super().__init__()

        self.viewer = viewer

        self.multiple_viewer_widget = MultipleViewerWidget(self.viewer)
        self.cross_widget = CrossWidget(self.viewer)

        layout = QVBoxLayout()
        layout.addWidget(self.multiple_viewer_widget)
        layout.addWidget(self.cross_widget)

        self.setLayout(layout)
