import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)
from superqt import QLabeledDoubleSlider, QRangeSlider

import finn
from finn.layers.image._image_constants import (
    VolumeDepiction,
)
from finn.layers.utils.plane import ClippingPlane
from finn.utils.action_manager import action_manager
from finn.utils.translations import trans


class QtLayerDepiction(QFormLayout):
    def __init__(self, parent) -> None:
        super().__init__()
        self.parent = parent
        self.layer = parent.layer

        self.depictionComboBox = QComboBox(self.parent)
        depiction_options = [d.value for d in VolumeDepiction]
        self.depictionComboBox.addItems(depiction_options)
        index = self.depictionComboBox.findText(
            self.layer.depiction, Qt.MatchFlag.MatchFixedString
        )

        self.depictionComboBox.setCurrentIndex(index)
        self.depictionComboBox.currentTextChanged.connect(self.changeDepiction)
        self.depictionLabel = QLabel(trans._("depiction:"), self.parent)
        self.layer.events.depiction.connect(self._on_depiction_change)
        self.layer.plane.events.thickness.connect(self._on_plane_thickness_change)
        self.layer.events.plane.connect(self._update_plane_slider)

        # plane controls
        self.planeNormalButtons = PlaneNormalButtons(self.parent)
        self.planeNormalLabel = QLabel(trans._("plane normal:"), self.parent)
        action_manager.bind_button(
            "napari:orient_plane_normal_along_z",
            self.planeNormalButtons.zButton,
        )
        action_manager.bind_button(
            "napari:orient_plane_normal_along_y",
            self.planeNormalButtons.yButton,
        )
        action_manager.bind_button(
            "napari:orient_plane_normal_along_x",
            self.planeNormalButtons.xButton,
        )
        action_manager.bind_button(
            "napari:orient_plane_normal_along_view_direction_no_gen",
            self.planeNormalButtons.obliqueButton,
        )

        self.planeNormalButtons.xButton.clicked.connect(
            lambda: self._set_orientation("x")
        )
        self.planeNormalButtons.yButton.clicked.connect(
            lambda: self._set_orientation("y")
        )
        self.planeNormalButtons.zButton.clicked.connect(
            lambda: self._set_orientation("z")
        )
        self.planeNormalButtons.obliqueButton.clicked.connect(
            lambda: self._set_orientation("oblique")
        )

        self.clippingPlaneCheckbox = QCheckBox(trans._("clipping plane"), self.parent)
        self.clippingPlaneCheckbox.setStyleSheet("""
            font-size:11px
        """)

        self.clippingPlaneCheckbox.stateChanged.connect(self.activateClippingPlane)
        self.clippingPlaneSlider = QRangeSlider(Qt.Orientation.Horizontal, self.parent)
        self.clippingPlaneSlider.setStyleSheet("""
            QSlider::groove:horizontal:disabled {
                background: #353B43;  /* gray background */
            }
            QSlider::handle:horizontal:disabled {
                background: #4C545E;  /* Greyed-out handles */
            }
        """)
        self.clippingPlaneSlider.setFocusPolicy(Qt.NoFocus)
        self.clippingPlaneSlider.setMinimum(1)
        self.clippingPlaneSlider.setMaximum(self.layer.data.shape[-1])
        self.clippingPlaneSlider.setSingleStep(1)
        self.clippingPlaneSlider.setTickInterval(1)
        self.clippingPlaneSlider.valueChanged.connect(self.changeClippingPlaneRange)
        self.clippingPlaneSlider.setEnabled(False)

        self.planeSliderLabel = QLabel("plane slider position", self.parent)
        self.planeSlider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, self.parent)
        self.planeSlider.setMinimum(0)
        self.planeSlider.setMaximum(self.layer.data.shape[-1])
        self.planeSlider.setFocusPolicy(Qt.NoFocus)
        self.planeSlider.valueChanged.connect(self.changePlanePosition)
        self.planeThicknessSlider = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal, self.parent
        )
        self.planeThicknessLabel = QLabel(trans._("plane thickness:"), self.parent)
        self.planeThicknessSlider.setFocusPolicy(Qt.NoFocus)
        self.planeThicknessSlider.setMinimum(1)
        self.planeThicknessSlider.setMaximum(50)
        self.planeThicknessSlider.setValue(self.layer.plane.thickness)
        self.planeThicknessSlider.valueChanged.connect(self.changePlaneThickness)

        self.layout().addRow(self.depictionLabel, self.depictionComboBox)
        self.layout().addRow(self.planeNormalLabel, self.planeNormalButtons)
        self.layout().addRow(self.planeThicknessLabel, self.planeThicknessSlider)
        self.layout().addRow(self.planeSliderLabel, self.planeSlider)
        self.layout().addRow(self.clippingPlaneCheckbox, self.clippingPlaneSlider)

        self._set_orientation("z")
        self._on_ndisplay_changed()

    def changeDepiction(self, text):
        self.layer.depiction = text
        self._update_plane_parameter_visibility()

    def changePlaneThickness(self, value: float):
        self.layer.plane.thickness = value

    def _update_plane_parameter_visibility(self):
        """Hide plane rendering controls if they aren't needed."""
        depiction = VolumeDepiction(self.layer.depiction)
        plane_visible = (
            depiction == VolumeDepiction.PLANE
            and self.parent.ndisplay == 3
            and self.layer.ndim >= 3
        )
        clipping_plane_visible = (
            depiction == VolumeDepiction.VOLUME
            and self.parent.ndisplay == 3
            and self.layer.ndim >= 3
        )

        self.planeNormalButtons.setVisible(plane_visible or clipping_plane_visible)
        self.planeNormalLabel.setVisible(plane_visible or clipping_plane_visible)
        self.planeThicknessSlider.setVisible(plane_visible)
        self.planeThicknessLabel.setVisible(plane_visible)
        self.planeSlider.setVisible(plane_visible)
        self.planeSliderLabel.setVisible(plane_visible)

        self.clippingPlaneCheckbox.setVisible(clipping_plane_visible)
        self.clippingPlaneSlider.setVisible(clipping_plane_visible)

    def _on_depiction_change(self):
        """Receive layer model depiction change event and update combobox."""
        with self.layer.events.depiction.blocker():
            index = self.depictionComboBox.findText(
                self.layer.depiction, Qt.MatchFlag.MatchFixedString
            )
            self.depictionComboBox.setCurrentIndex(index)
            self._update_plane_parameter_visibility()

    def _on_plane_thickness_change(self):
        with self.layer.plane.events.blocker():
            self.planeThicknessSlider.setValue(self.layer.plane.thickness)

    def _update_plane_slider(self):
        """Updates the value of the plane slider when the user used the shift+drag method to shift the plane or when switching between different layers"""

        new_position = np.array(self.layer.plane.position)
        plane_normal = np.array(self.layer.plane.normal)
        slider_value = np.dot(new_position, plane_normal) / np.dot(
            plane_normal, plane_normal
        )
        self.planeSlider.valueChanged.disconnect(self.changePlanePosition)
        self.planeSlider.setValue(int(slider_value))
        self.planeSlider.valueChanged.connect(self.changePlanePosition)

    def _compute_plane_range(self) -> tuple[float, float]:
        """Compute the range of the plane and clipping plane sliders

        returns:
            tuple[float, float], the minimum and maximum values of the slider
        """

        normal = np.array(self.layer.plane.normal)
        Lx, Ly, Lz = self.layer.data.shape[-3:]

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

        return (min_position, max_position)

    def _set_orientation(self, orientation: str) -> None:
        """Set the range of the (clipping) plane sliders based on the orientation.
        args:
            orientation: str, the direction in which the plane should
                slide. Can be 'x', 'y', 'z', or 'oblique'.
        """

        if orientation == "x":
            clip_range = (0, self.layer.data.shape[-1])

        elif orientation == "y":
            clip_range = (0, self.layer.data.shape[-2])

        elif orientation == "z":
            clip_range = (0, self.layer.data.shape[-3])

        else:  # oblique view
            clip_range = self._compute_plane_range()
            self.layer.experimental_clipping_planes[0].normal = self.layer.plane.normal
            self.layer.experimental_clipping_planes[1].normal = (
                -self.layer.plane.normal[-3],
                -self.layer.plane.normal[-2],
                -self.layer.plane.normal[-1],
            )

        self.planeSlider.setMinimum(clip_range[0])
        self.planeSlider.setMaximum(clip_range[1])
        self.clippingPlaneSlider.setMinimum(clip_range[0])
        self.clippingPlaneSlider.setMaximum(clip_range[1])
        min_value = int(clip_range[0] + (1 / 3) * (clip_range[1] - clip_range[0]))
        max_value = int(clip_range[0] + (2 / 3) * (clip_range[1] - clip_range[0]))
        self.clippingPlaneSlider.setValue((min_value, max_value))
        self.planeSlider.setMinimum(clip_range[0])
        self.planeSlider.setMaximum(clip_range[1])

    def changePlanePosition(self, value: float):
        plane_normal = np.array(self.layer.plane.normal)
        new_position = np.array([0, 0, 0]) + value * plane_normal
        self.layer.plane.position = tuple(new_position)

    def activateClippingPlane(self, state):
        if state:
            self.layer.experimental_clipping_planes[0].enabled = True
            self.layer.experimental_clipping_planes[1].enabled = True
            self.clippingPlaneSlider.setEnabled(True)
        else:
            self.layer.experimental_clipping_planes[0].enabled = False
            self.layer.experimental_clipping_planes[1].enabled = False
            self.clippingPlaneSlider.setEnabled(False)
        self.layer.events.experimental_clipping_planes()

    def changeClippingPlaneRange(self, value):
        viewer = finn.viewer.current_viewer()

        if len(self.layer.experimental_clipping_planes) == 0:
            plane = self.layer.plane
            self.layer.experimental_clipping_planes.append(
                ClippingPlane(
                    normal=plane.normal,
                    position=plane.position,
                    enabled=False,
                )
            )
            self.layer.experimental_clipping_planes.append(
                ClippingPlane(
                    normal=[-n for n in plane.normal],
                    position=plane.position,
                    enabled=False,
                )
            )

        plane_normal = np.array(self.layer.experimental_clipping_planes[0].normal)
        new_position_1 = np.array([0, 0, 0]) + value[0] * plane_normal
        new_position_1 = (
            int(new_position_1[0] * viewer.dims.range[-3].step),
            (new_position_1[1] * viewer.dims.range[-2].step),
            int(new_position_1[2] * viewer.dims.range[-1].step),
        )
        self.layer.experimental_clipping_planes[0].position = new_position_1
        new_position_2 = np.array([0, 0, 0]) + value[1] * plane_normal
        new_position_2 = (
            int(new_position_2[0] * viewer.dims.range[-3].step),
            (new_position_2[1] * viewer.dims.range[-2].step),
            int(new_position_2[2] * viewer.dims.range[-1].step),
        )

        self.layer.experimental_clipping_planes[1].position = new_position_2
        self.layer.events.experimental_clipping_planes()

    def _on_ndisplay_changed(self):
        """Update widget visibility based on 2D and 3D visualization modes."""
        self._update_plane_parameter_visibility()
        if self.parent.ndisplay == 2:
            self.depictionComboBox.hide()
            self.depictionLabel.hide()
        else:
            self.depictionComboBox.show()
            self.depictionLabel.show()


class PlaneNormalButtons(QWidget):
    """Qt buttons for controlling plane orientation.

        Attributes
    ----------
    xButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the x axis.
    yButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the y axis.
    zButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the z axis.
    obliqueButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the camera view direction.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.xButton = QPushButton("x")
        self.yButton = QPushButton("y")
        self.zButton = QPushButton("z")
        self.obliqueButton = QPushButton(trans._("oblique"))

        self.layout().addWidget(self.xButton)
        self.layout().addWidget(self.yButton)
        self.layout().addWidget(self.zButton)
        self.layout().addWidget(self.obliqueButton)
