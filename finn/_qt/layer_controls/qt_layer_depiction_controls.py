import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)
from superqt import QRangeSlider

from finn.layers.image._image_constants import (
    VolumeDepiction,
)
from finn.layers.utils.plane import ClippingPlane
from finn.utils.action_manager import action_manager
from finn.utils.translations import trans


class QtLayerDepiction(QFormLayout):
    """Qt controls for the image and labels layer depiction.

    Parameters
    ----------
    parent : finn._qt.layer_controls.QtImageControls or finn._qt.layer_controls.QtLabelsControls
        An instance of QtImageControls or QtLabelsControls holding the napari layer.

    Attributes
    ----------
    parent : finn._qt.layer_controls.QtImageControls or finn._qt.layer_controls.QtLabelsControls
        An instance of QtImageControls or QtLabelsControls holding the napari layer.
    layer : finn.layers.Image
        An instance of a napari Image layer.

    planeNormalButtons : PlaneNormalButtons
        QPushButtons for controlling the plane normal.
    planeNormalLabel : qtpy.QtWidgets.QLabel
        Label for the plane normal buttons.
    clippingPlaneCheckbox : qtpy.QtWidgets.QCheckBox
        Checkbox for enabling the clipping plane.
    clippingPlaneSlider : superqt.QRangeSlider
        QRangeSlider for selecting the range of the clipping plane.
    """

    def __init__(self, parent) -> None:
        super().__init__()
        self.parent = parent
        self.layer = parent.layer

        # plane normal buttons
        self.planeNormalButtons = PlaneNormalButtons(self.parent)
        self.planeNormalLabel = QLabel(trans._("plane normal:"), self.parent)

        # bind functions to set the plane normal according to the button pressed
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

        # connect button press to updating the span of the plane and clipping plane sliders
        self.planeNormalButtons.xButton.clicked.connect(
            lambda: self._set_plane_slider_min_max("x")
        )
        self.planeNormalButtons.yButton.clicked.connect(
            lambda: self._set_plane_slider_min_max("y")
        )
        self.planeNormalButtons.zButton.clicked.connect(
            lambda: self._set_plane_slider_min_max("z")
        )
        self.planeNormalButtons.obliqueButton.clicked.connect(
            lambda: self._set_plane_slider_min_max("oblique")
        )

        # button to activate/deactivate the clipping slider and clipping planes in the 'volume' depiction
        self.clippingPlaneCheckbox = QCheckBox(trans._("clipping plane"), self.parent)
        self.clippingPlaneCheckbox.setStyleSheet("""
            font-size:11px
        """)
        self.clippingPlaneCheckbox.stateChanged.connect(self._activateClippingPlane)

        # clipping plane slider to set range of experimental_clipping_planes
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
        self.clippingPlaneSlider.valueChanged.connect(self.changeClippingPlanePositions)
        self.clippingPlaneSlider.setEnabled(False)

        # combine widgets
        self.layout().addRow(self.planeNormalLabel, self.planeNormalButtons)
        self.layout().addRow(self.clippingPlaneCheckbox, self.clippingPlaneSlider)
        self._set_plane_slider_min_max(
            "z"
        )  # set initial span of the sliders based on the size of the z axis (which is the default plane normal)

    def changePlaneThickness(self, value: float):
        """Change the number of slices to be rendered in the plane.
        args: value: float, the new thickness of the plane.
        """
        self.layer.plane.thickness = value

    def changePlanePosition(self, value: float):
        """Change the position of the plane
        args:
            value: float, the new position of the plane.
        """
        plane_normal = np.array(self.layer.plane.normal)
        new_position = np.array([0, 0, 0]) + value * plane_normal
        self.layer.plane.position = tuple(new_position)

    def changeClippingPlanePositions(self, value: tuple[int, int]):
        """Set lower and upper positions of the clipping plane."""

        if not self.layer.ndim >= 3:
            return

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
            int(new_position_1[0] * self.layer.scale[-3]),
            (new_position_1[1] * self.layer.scale[-2]),
            int(new_position_1[2] * self.layer.scale[-1]),
        )
        self.layer.experimental_clipping_planes[0].position = new_position_1
        new_position_2 = np.array([0, 0, 0]) + value[1] * plane_normal
        new_position_2 = (
            int(new_position_2[0] * self.layer.scale[-3]),
            (new_position_2[1] * self.layer.scale[-2]),
            int(new_position_2[2] * self.layer.scale[-1]),
        )

        self.layer.experimental_clipping_planes[1].position = new_position_2
        self.layer.events.experimental_clipping_planes()

    def _update_plane_parameter_visibility(self):
        """Hide plane rendering controls if they are not needed."""
        depiction = VolumeDepiction(self.layer.depiction)

        clipping_plane_visible = (
            depiction == VolumeDepiction.VOLUME
            and self.parent.ndisplay == 3
            and self.layer.ndim >= 3
        )

        self.planeNormalButtons.setVisible(clipping_plane_visible)
        self.planeNormalLabel.setVisible(clipping_plane_visible)
        self.clippingPlaneCheckbox.setVisible(clipping_plane_visible)
        self.clippingPlaneSlider.setVisible(clipping_plane_visible)

    def _compute_plane_range(self) -> tuple[float, float]:
        """Compute the total span of the plane and clipping plane sliders. Used in the special case of the oblique view.

        returns:
            tuple[float, float], the minimum and maximum possible values of the slider
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

    def _set_plane_slider_min_max(self, orientation: str) -> None:
        """Set the minimum and maximum values of the plane and clipping plane sliders based on the orientation. Also set the initial values of the slider.
        args:
            orientation: str, the direction in which the plane should
                slide. Can be 'x', 'y', 'z', or 'oblique'.
        """

        if not self.layer.ndim >= 3:
            return
        if orientation == "x":
            clip_range = (0, self.layer.data.shape[-1])

        elif orientation == "y":
            clip_range = (0, self.layer.data.shape[-2])

        elif orientation == "z":
            clip_range = (0, self.layer.data.shape[-3])

        else:  # oblique view
            clip_range = self._compute_plane_range()

        # Set the minimum and maximum values of the clipping plane slider
        self.clippingPlaneSlider.setMinimum(clip_range[0])
        self.clippingPlaneSlider.setMaximum(clip_range[1])

        # Set the initial values of the clipping plane slider to 1/3 and 2/3 of the range
        min_initial_value = int(clip_range[0] + (1 / 3) * (clip_range[1] - clip_range[0]))
        max_initial_value = int(clip_range[0] + (2 / 3) * (clip_range[1] - clip_range[0]))
        self.clippingPlaneSlider.setValue((min_initial_value, max_initial_value))

    def _activateClippingPlane(self, state):
        """Activate or deactivate the clipping plane based on the checkbox state.
        args:
            state: bool, the state of the checkbox.
        """
        if state:
            self.layer.experimental_clipping_planes[0].enabled = True
            self.layer.experimental_clipping_planes[1].enabled = True
            self.clippingPlaneSlider.setEnabled(True)
        else:
            self.layer.experimental_clipping_planes[0].enabled = False
            self.layer.experimental_clipping_planes[1].enabled = False
            self.clippingPlaneSlider.setEnabled(False)
        self.layer.events.experimental_clipping_planes()

    def _on_ndisplay_changed(self):
        """Update widget visibility based on 2D and 3D visualization modes."""
        self._update_plane_parameter_visibility()

    def disconnect(self):
        """Disconnect all event connections (e.g. when layer is removed)."""

        # break circular references
        self.parent = None
        self.layer = None


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
