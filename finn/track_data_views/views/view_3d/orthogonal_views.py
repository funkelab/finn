from psygnal import Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

import finn
from finn.track_data_views.views.view_3d.multiple_view_widget import (
    CrossWidget,
    MultipleViewerWidget,
)


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

    def update(self):
        self.multiple_viewer_widget.update()


class View3D(QWidget):
    """Widget to combine multiple views and cross widget together."""

    update_tab = Signal()

    def __init__(
        self,
        viewer: finn.Viewer,
    ):
        super().__init__()

        self.viewer = viewer
        self.viewer.dims.events.ndisplay.connect(self.display_changed)
        self.viewer.dims.events.ndim.connect(self.display_changed)

        self.orth_views = OrthogonalViews(viewer)
        widget_2d_layout = QVBoxLayout()
        widget_2d_layout.addWidget(QLabel("Viewer has less than 3 dimensions"))
        self.widget_2d = QWidget()
        self.widget_2d.setLayout(widget_2d_layout)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Add both widgets to the layout but hide them initially
        self.layout().addWidget(self.orth_views)
        self.layout().addWidget(self.widget_2d)

        self.orth_views.hide()
        self.widget_2d.hide()

        # Show the correct widget
        self.display_changed()

    def display_changed(self, event=None):
        """Switch between OrthogonalViews 2D widget based on how many dimensions the viewer has."""
        # Hide both widgets
        self.orth_views.hide()
        self.widget_2d.hide()

        # Show the appropriate widget based on ndisplay
        if self.viewer.dims.ndim >= 3:
            self.orth_views.show()
            self.orth_views.update()
            self.update_tab.emit()
        else:
            self.widget_2d.show()
