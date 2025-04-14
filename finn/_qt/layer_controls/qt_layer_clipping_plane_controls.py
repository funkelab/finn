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
from superqt import QLabeledSlider

from finn.layers.utils.plane import ClippingPlane
from finn.utils.action_manager import action_manager
from finn.utils.translations import trans


class QtLayerClippingPlanes(QFormLayout):
    """Qt controls for the image and labels layer clipping planes.

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

        # make sure the clipping planes are initialized
        if len(self.layer.experimental_clipping_planes) == 0:
            self.layer.experimental_clipping_planes.append(
                ClippingPlane(
                    normal=(np.float64(1.0), np.float64(0.0), np.float64(0.0)),
                    position=(0.0, 0.0, 0.0),
                    enabled=False,
                )
            )
            self.layer.experimental_clipping_planes.append(
                ClippingPlane(
                    normal=[
                        -n for n in self.layer.experimental_clipping_planes[0].normal
                    ],
                    position=(0.0, 0.0, 0.0),
                    enabled=False,
                )
            )

        # clipping plane sliders to set width and position of experimental_clipping_planes
        self.clippingPlaneWidthLabel = QLabel("Clipping Plane Width")
        self.clippingPlaneWidthLabel.setEnabled(False)
        self.clippingPlaneWidthSlider = QLabeledSlider(
            Qt.Orientation.Horizontal, self.parent
        )
        self.clippingPlaneWidthSlider.setMinimum(1)
        self.clippingPlaneWidthSlider.setMaximum(self.layer.data.shape[-1])
        self.clippingPlaneWidthSlider.setValue(
            self.layer.data.shape[-1] // 2
        )  # half the width of the stack
        self.clippingPlaneWidthSlider.valueChanged.connect(self.updateWidthCenter)
        self.clippingPlaneWidthSlider.setEnabled(False)
        self.clippingPlaneWidthSlider.setStyleSheet("""
            QSlider::groove:horizontal:disabled {
                background: #353B43;  /* gray background */
            }
            QSlider::handle:horizontal:disabled {
                background: #4C545E;  /* Greyed-out handles */
            }
            QSlider::sub-page:horizontal:disabled {
                background: #353B43;  /* Darker gray for the 'bright' part (left of the handle) */
            }
        """)

        self.clippingPlaneCenterLabel = QLabel("Clipping Plane Center")
        self.clippingPlaneCenterLabel.setEnabled(False)
        self.clippingPlaneCenterSlider = QLabeledSlider(
            Qt.Orientation.Horizontal, self.parent
        )
        self.clippingPlaneCenterSlider.setMinimum(1)
        self.clippingPlaneCenterSlider.setMaximum(self.layer.data.shape[-1])
        self.clippingPlaneCenterSlider.setValue(
            self.layer.data.shape[-1] // 2
        )  # center of the stack
        self.clippingPlaneCenterSlider.valueChanged.connect(self.updateWidthCenter)
        self.clippingPlaneCenterSlider.setEnabled(False)
        self.clippingPlaneCenterSlider.setStyleSheet("""
            QSlider::groove:horizontal:disabled {
                background: #353B43;  /* gray background */
            }
            QSlider::handle:horizontal:disabled {
                background: #4C545E;  /* Greyed-out handles */
            }
            QSlider::sub-page:horizontal:disabled {
                background: #353B43;  /* Darker gray for the 'bright' part (left of the handle) */
            }
        """)

        # button to activate/deactivate the clipping planes and their sliders
        self.clippingPlaneLabel = QLabel("Activate clipping plane")
        self.clippingPlaneCheckbox = QCheckBox("", self.parent)
        self.clippingPlaneCheckbox.stateChanged.connect(self._activateClippingPlane)

        # combine widgets
        self.layout().addRow(self.planeNormalLabel, self.planeNormalButtons)
        self.layout().addRow(self.clippingPlaneLabel, self.clippingPlaneCheckbox)
        self.layout().addRow(self.clippingPlaneWidthLabel, self.clippingPlaneWidthSlider)
        self.layout().addRow(
            self.clippingPlaneCenterLabel, self.clippingPlaneCenterSlider
        )
        self._set_plane_slider_min_max(
            "z"
        )  # set initial span of the sliders based on the size of the z axis (which is the default plane normal)

    def updateWidthCenter(self) -> None:
        """Update the width and center of the clipping planes based on the current slider values."""

        center = self.clippingPlaneCenterSlider.value()
        width = self.clippingPlaneWidthSlider.value()

        self.set_clipping_plane_positions(width, center)

    def set_clipping_plane_positions(self, width: int, center: int) -> None:
        """Set the positions of the clipping planes based on the slider width and center values."""

        if self.layer.ndim < 3:
            return

        # Ensure the width is odd (to center around the middle slice)
        if width % 2 == 0:
            width += 1

        position1 = center - width // 2
        position2 = (center + width // 2) + 1

        if position1 < self.clippingPlaneCenterSlider.minimum():
            position1 = self.clippingPlaneCenterSlider.minimum()
        if position2 > self.clippingPlaneCenterSlider.maximum():
            position2 = self.clippingPlaneCenterSlider.maximum()

        plane_normal = np.array(self.layer.experimental_clipping_planes[0].normal)
        new_position1 = np.array([0, 0, 0]) + position1 * plane_normal
        new_position1 = (
            int(new_position1[0] * self.layer.scale[-3]),
            int(new_position1[1] * self.layer.scale[-2]),
            int(new_position1[2] * self.layer.scale[-1]),
        )
        self.layer.experimental_clipping_planes[0].position = new_position1

        new_position2 = np.array([0, 0, 0]) + position2 * plane_normal
        new_position2 = (
            int(new_position2[0] * self.layer.scale[-3]),
            int(new_position2[1] * self.layer.scale[-2]),
            int(new_position2[2] * self.layer.scale[-1]),
        )
        self.layer.experimental_clipping_planes[1].position = new_position2

        self.layer.events.experimental_clipping_planes()

    def _update_plane_parameter_visibility(self):
        """Hide plane rendering controls if they are not needed."""

        clipping_plane_visible = self.parent.ndisplay == 3 and self.layer.ndim >= 3

        self.planeNormalButtons.setVisible(clipping_plane_visible)
        self.planeNormalLabel.setVisible(clipping_plane_visible)
        self.clippingPlaneLabel.setVisible(clipping_plane_visible)
        self.clippingPlaneCheckbox.setVisible(clipping_plane_visible)
        self.clippingPlaneWidthSlider.setVisible(clipping_plane_visible)
        self.clippingPlaneWidthLabel.setVisible(clipping_plane_visible)
        self.clippingPlaneCenterLabel.setVisible(clipping_plane_visible)
        self.clippingPlaneCenterSlider.setVisible(clipping_plane_visible)

    def _compute_plane_range(self) -> tuple[float, float]:
        """Compute the total span of the plane and clipping plane sliders. Used in the special case of the oblique view.

        returns:
            tuple[float, float], the minimum and maximum possible values of the slider
        """

        normal = np.array(self.layer.experimental_clipping_planes[0].normal)
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

        return (int(min_position), int(max_position))

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

        # Set the minimum and maximum values of the clipping plane sliders
        self.clippingPlaneCenterSlider.setMinimum(clip_range[0])
        self.clippingPlaneCenterSlider.setMaximum(clip_range[1] + 1)
        self.clippingPlaneWidthSlider.setMaximum(clip_range[1] + 1)

        width = clip_range[1] // 2
        center = clip_range[0] + width // 2

        self.clippingPlaneWidthSlider.setValue(width)
        self.clippingPlaneCenterSlider.setValue(center)

    def _activateClippingPlane(self, state):
        """Activate or deactivate the clipping plane based on the checkbox state.
        args:
            state: bool, the state of the checkbox.
        """
        if state:
            self.layer.experimental_clipping_planes[0].enabled = True
            self.layer.experimental_clipping_planes[1].enabled = True
            self.clippingPlaneWidthSlider.setEnabled(True)
            self.clippingPlaneCenterSlider.setEnabled(True)
            self.clippingPlaneWidthLabel.setEnabled(True)
            self.clippingPlaneCenterLabel.setEnabled(True)
        else:
            self.layer.experimental_clipping_planes[0].enabled = False
            self.layer.experimental_clipping_planes[1].enabled = False
            self.clippingPlaneWidthSlider.setEnabled(False)
            self.clippingPlaneCenterSlider.setEnabled(False)
            self.clippingPlaneWidthLabel.setEnabled(False)
            self.clippingPlaneCenterLabel.setEnabled(False)
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
