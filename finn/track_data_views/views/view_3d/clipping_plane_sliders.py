import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledRangeSlider

import finn.layers
from finn.layers.utils.plane import ClippingPlane
from finn.track_data_views.views.layers.track_labels import TrackLabels
from finn.track_data_views.views_coordinator.tracks_viewer import TracksViewer


class PlaneSliderWidget(QWidget):
    """Widget implementing sliders for 3D plane and 3D clipping plane visualization"""

    def __init__(
        self,
        viewer: finn.Viewer,
    ):
        super().__init__()

        self.viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(self.viewer)
        self.current_layer = None

        # Add a dropdown menu to select a layer to add a plane view for
        self.viewer.layers.selection.events.active.connect(self._update_layer)

        # Add buttons to switch between plane and volume mode
        button_layout = QVBoxLayout()
        plane_volume_layout = QHBoxLayout()
        self.slice_view_btn = QPushButton('Slice view')
        self.slice_view_btn.clicked.connect(self._set_slice_view)
        self.clipping_plane_btn = QPushButton('Clipping Plane')
        self.clipping_plane_btn.clicked.connect(self._set_clipping_plane_mode)
        self.volume_btn = QPushButton('Volume')
        self.volume_btn.clicked.connect(self._set_volume_mode)
        plane_volume_layout.addWidget(self.slice_view_btn)
        plane_volume_layout.addWidget(self.clipping_plane_btn)
        plane_volume_layout.addWidget(self.volume_btn)
        button_layout.addLayout(plane_volume_layout)

        # Add buttons for the different viewing directions
        plane_labels = QHBoxLayout()
        plane_set_x_btn = QPushButton('X')
        plane_set_x_btn.clicked.connect(self._set_x_orientation)
        plane_set_y_btn = QPushButton('Y')
        plane_set_y_btn.clicked.connect(self._set_y_orientation)
        plane_set_z_btn = QPushButton('Z')
        plane_set_z_btn.clicked.connect(self._set_z_orientation)
        plane_set_oblique_btn = QPushButton('Oblique')
        plane_set_oblique_btn.clicked.connect(self._set_oblique_orientation)
        plane_labels.addWidget(plane_set_x_btn)
        plane_labels.addWidget(plane_set_y_btn)
        plane_labels.addWidget(plane_set_z_btn)
        plane_labels.addWidget(plane_set_oblique_btn)

        self.clipping_plane_slider = QLabeledRangeSlider(QtCore.Qt.Horizontal)
        self.clipping_plane_slider.setValue((0, 1))
        self.clipping_plane_slider.valueChanged.connect(self._set_clipping_plane)
        self.clipping_plane_slider.setSingleStep(1)
        self.clipping_plane_slider.setTickInterval(1)
        self.clipping_plane_slider.setEnabled(False)

        # Combine buttons and sliders
        plane_layout = QVBoxLayout()
        plane_layout.addWidget(QLabel('Plane Normal'))
        plane_layout.addLayout(plane_labels)
        plane_layout.addWidget(QLabel('Clipping Plane'))
        plane_layout.addWidget(self.clipping_plane_slider)

        # Assemble main layout
        view_mode_widget_layout = QVBoxLayout()
        view_mode_widget_layout.addLayout(button_layout)
        view_mode_widget_layout.addLayout(plane_layout)

        self.setLayout(view_mode_widget_layout)
        self.setMaximumHeight(400)

    def compute_plane_range(self):
        """Compute the range of the plane and clipping plane sliders"""

        normal = np.array(self.current_layer.plane.normal)
        Lx, Ly, Lz = self.current_layer.data.shape[-3:]

        # Define the corners of the 3D image bounding box
        corners = np.array(
            [
                [0, 0, 0],
                [Lx, 0, 0],
                [0, Ly, 0],
                [0, 0, Lz],
                [Lx, Ly, 0],
                [Lx, 0, Lz],
                [0, Ly, Lz],
                [Lx, Ly, Lz],
            ]
        )

        # Project the corners onto the normal vector
        projections = np.dot(corners, normal)

        # The range of possible positions is given by the min and max projections
        min_position = np.min(projections)
        max_position = np.max(projections)

        return min_position, max_position

    def _set_x_orientation(self):
        """Align the plane to slide in the yz plane"""

        if self.current_layer is not None:
            self.current_layer.plane.normal = (0, 0, 1)
            self.current_layer.experimental_clipping_planes[0].normal = (
                0,
                0,
                1,
            )
            self.current_layer.experimental_clipping_planes[1].normal = (
                0,
                0,
                -1,
            )

            self.clipping_plane_slider.setMinimum(0)
            self.clipping_plane_slider.setMaximum(self.current_layer.data.shape[-1])
            self.clipping_plane_slider.setValue(
                (
                    int(self.current_layer.data.shape[-1] / 3),
                    int(self.current_layer.data.shape[-1] / 1.5),
                )
            )

    def _set_y_orientation(self):
        """Align the plane to slide in the xz plane"""

        if self.current_layer is not None:
            self.current_layer.plane.normal = (0, 1, 0)

            self.current_layer.experimental_clipping_planes[0].normal = (
                0,
                1,
                0,
            )
            self.current_layer.experimental_clipping_planes[1].normal = (
                0,
                -1,
                0,
            )

            self.clipping_plane_slider.setMinimum(0)
            self.clipping_plane_slider.setMaximum(self.current_layer.data.shape[-2])
            self.clipping_plane_slider.setValue(
                (
                    int(self.current_layer.data.shape[-2] / 3),
                    int(self.current_layer.data.shape[-2] / 1.5),
                )
            )

    def _set_z_orientation(self):
        """Align the plane to slide in the yx plane"""

        if self.current_layer is not None:
            self.current_layer.plane.normal = (1, 0, 0)

            self.current_layer.experimental_clipping_planes[0].normal = (
                1,
                0,
                0,
            )
            self.current_layer.experimental_clipping_planes[1].normal = (
                -1,
                0,
                0,
            )

            self.clipping_plane_slider.setMinimum(0)
            self.clipping_plane_slider.setMaximum(self.current_layer.data.shape[-3])
            self.clipping_plane_slider.setValue(
                (
                    int(self.current_layer.data.shape[-3] / 3),
                    int(self.current_layer.data.shape[-3] / 1.5),
                )
            )

    def _set_oblique_orientation(self) -> None:
        """Orient plane normal along the viewing direction"""

        if self.current_layer is not None:
            self.current_layer.plane.normal = (
                self.current_layer._world_to_displayed_data_ray(
                    self.viewer.camera.view_direction, [-3, -2, -1]
                )
            )
            min_range, max_range = self.compute_plane_range()

            self.current_layer.experimental_clipping_planes[
                0
            ].normal = self.current_layer.plane.normal
            self.current_layer.experimental_clipping_planes[1].normal = (
                -self.current_layer.plane.normal[-3],
                -self.current_layer.plane.normal[-2],
                -self.current_layer.plane.normal[-1],
            )

            self.clipping_plane_slider.setMinimum(min_range)
            self.clipping_plane_slider.setMaximum(max_range)
            self.clipping_plane_slider.setValue(
                (int(max_range / 3), int(max_range / 1.5))
            )

    def _update_layer(self, event) -> None:
        """Update the layer to which the plane viewing is applied"""

        if event.value is None or not isinstance(
            event.value, (finn.layers.Image, finn.layers.Labels, TrackLabels)
        ):
            self.slice_view_btn.setEnabled(False)
            self.volume_btn.setEnabled(False)
            self.clipping_plane_btn.setEnabled(False)
            self.clipping_plane_slider.setEnabled(False)
            self.current_layer = None
            return
        self.slice_view_btn.setEnabled(True)
        self.volume_btn.setEnabled(True)
        self.clipping_plane_btn.setEnabled(True)
        self.clipping_plane_slider.setEnabled(True)
        self.current_layer = event.value
        if len(self.current_layer.experimental_clipping_planes) == 0:
            plane = self.current_layer.plane
            self.current_layer.experimental_clipping_planes.append(
                ClippingPlane(
                    normal=plane.normal,
                    position=plane.position,
                    enabled=False,
                )
            )
            self.current_layer.experimental_clipping_planes.append(
                ClippingPlane(
                    normal=[-n for n in plane.normal],
                    position=plane.position,
                    enabled=False,
                )
            )

        if self.viewer.dims.ndisplay == 3:
            if (
                self.current_layer.depiction == 'volume'
                and self.current_layer.experimental_clipping_planes[0].enabled
            ):
                self._set_clipping_plane_mode()
                self._update_clipping_plane_slider()
            else:
                self._set_volume_mode()

    def _update_clipping_plane_slider(self):
        """Updates the values of the clipping plane slider when switching between different layers"""

        new_position = np.array(
            self.current_layer.experimental_clipping_planes[0].position
        )
        plane_normal = np.array(self.current_layer.experimental_clipping_planes[0].normal)
        slider_value1 = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )

        new_position = np.array(
            self.current_layer.experimental_clipping_planes[1].position
        )
        plane_normal = np.array(self.current_layer.experimental_clipping_planes[0].normal)
        slider_value2 = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )

        self.clipping_plane_slider.valueChanged.disconnect(self._set_clipping_plane)
        self.clipping_plane_slider.setValue((int(slider_value1), int(slider_value2)))
        self.clipping_plane_slider.valueChanged.connect(self._set_clipping_plane)

    def _set_clipping_plane_mode(self) -> None:
        """Activate the clipping plane mode on the current layer"""

        self.viewer.dims.ndisplay = 3
        self.current_layer.depiction = 'volume'

        self.clipping_plane_slider.setEnabled(True)

        for clip_plane in self.current_layer.experimental_clipping_planes:
            clip_plane.enabled = True

        max_range = self.compute_plane_range()[1]
        self.clipping_plane_slider.setMaximum(max_range)
        if self.clipping_plane_slider.value()[0] == 0:
            self.clipping_plane_slider.setValue(
                (int(max_range / 3), int(max_range / 1.5))
            )

    def _set_volume_mode(self) -> None:
        """Deactive plane viewing and go back to default volume viewing"""

        self.viewer.dims.ndisplay = 3
        self.clipping_plane_slider.setEnabled(False)
        self.current_layer.depiction = 'volume'
        for clip_plane in self.current_layer.experimental_clipping_planes:
            clip_plane.enabled = False

        if isinstance(self.current_layer, TrackLabels):
            visible_nodes = self.tracks_viewer.filter_visible_nodes()
            self.tracks_viewer.tracking_layers.update_visible(
                visible_nodes=visible_nodes, plane_nodes='all'
            )

    def _set_slice_view(self):
        """Set ndisplay to 2, triggering replacement of this widget by the orthogonal views widget"""

        self.viewer.dims.ndisplay = 2

    def on_ndisplay_changed(self) -> None:
        """Update the buttons depending on the display mode of the viewer. Buttons and sliders should only be active in 3D mode"""

        if self.current_layer is not None:
            if (
                self.current_layer.depiction == 'volume'
                and self.current_layer.experimental_clipping_planes[0].enabled
            ):
                self._set_clipping_plane_mode()
                self._update_clipping_plane_slider()
            else:
                self._set_volume_mode()

    def _set_clipping_plane(self) -> None:
        """Adjust the range of the clipping plane"""

        plane_normal = np.array(self.current_layer.experimental_clipping_planes[0].normal)
        slider_value = self.clipping_plane_slider.value()
        new_position_1 = np.array([0, 0, 0]) + slider_value[0] * plane_normal
        new_position_1 = (
            int(new_position_1[0] * self.viewer.dims.range[-3].step),
            (new_position_1[1] * self.viewer.dims.range[-2].step),
            int(new_position_1[2] * self.viewer.dims.range[-1].step),
        )
        self.current_layer.experimental_clipping_planes[0].position = new_position_1
        new_position_2 = np.array([0, 0, 0]) + slider_value[1] * plane_normal
        new_position_2 = (
            int(new_position_2[0] * self.viewer.dims.range[-3].step),
            (new_position_2[1] * self.viewer.dims.range[-2].step),
            int(new_position_2[2] * self.viewer.dims.range[-1].step),
        )

        self.current_layer.experimental_clipping_planes[1].position = new_position_2

        # Emit the visible nodes so that the TrackPoints and TrackGraph can update too.
        if not self.tracks_viewer.track_df.empty:
            p1, p2 = (
                self.current_layer.experimental_clipping_planes[0].position,
                self.current_layer.experimental_clipping_planes[1].position,
            )

            scale = self.current_layer.scale[1:]  # skip the time scale
            p1 = [int(p1[0] / scale[0]), p1[1] / scale[1], p1[2] / scale[2]]
            p2 = [int(p2[0] / scale[0]), p2[1] / scale[1], p2[2] / scale[2]]
            plane_nodes = self.filter_labels_with_clipping_planes(
                self.viewer.dims.current_step[0], p1, p2, plane_normal
            )

            plane_nodes += [
                self.tracks_viewer.tracks.get_track_id(node)
                for node in self.tracks_viewer.selected_nodes
                if self.tracks_viewer.tracks.get_time(node)
                == self.viewer.dims.current_step[0]
            ]  # also include the selected nodes for clarity, even if they are out of plane.

            visible_nodes = self.tracks_viewer.filter_visible_nodes()

            self.tracks_viewer.tracking_layers.update_visible(
                visible_nodes=visible_nodes, plane_nodes=plane_nodes
            )

    def filter_labels_with_clipping_planes(
        self, t: int, p1: list[int], p2: list[int], normal: tuple[float], tolerance=10
    ) -> list[int]:
        """
        Filter labels in a pandas DataFrame based on clipping planes.

        Parameters:
        - df: pd.DataFrame, with columns 't', 'x', 'y', 'z' representing coordinates of labels.
        - p1: tuple or np.ndarray, position of the first clipping plane.
        - p2: tuple or np.ndarray, position of the second clipping plane.
        - normal: tuple or np.ndarray, the normal vector of the clipping planes.

        Returns:
        - A list of node ids that are within the bounds of the clipping planes.
        """
        # Normalize the normal vector
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)

        # Calculate signed distances to both planes

        df = self.tracks_viewer.track_df[self.tracks_viewer.track_df['t'] == t]
        coords = df[
            ['z', 'y', 'x']
        ].to_numpy()  # Extract txyz coordinates as a NumPy array
        p1 = np.array(p1)
        p2 = np.array(p2)

        dist_to_p1 = np.dot(coords - p1, normal)
        dist_to_p2 = np.dot(coords - p2, normal)

        # Filter rows where points are between the planes
        mask = (dist_to_p1 >= -tolerance) & (dist_to_p2 <= tolerance)
        filtered_df = df[mask]

        return filtered_df['node_id'].tolist()
