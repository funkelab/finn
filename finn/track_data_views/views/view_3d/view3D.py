from psygnal import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

import finn
from finn.track_data_views.views.view_3d.clipping_plane_sliders import PlaneSliderWidget


class View3D(QWidget):
    """Widget to combine multiple views and cross widget together."""

    update_tab = Signal()

    def __init__(
        self,
        viewer: finn.Viewer,
    ):
        super().__init__()

        self.viewer = viewer
        self.plane_widget = PlaneSliderWidget(self.viewer)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.layout().addWidget(self.plane_widget)
