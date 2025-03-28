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

    depictionComboBox : qtpy.QtWidgets.QComboBox
        QComboBox for selecting the depiction of the layer.
    depictionLabel : qtpy.QtWidgets.QLabel
        Label for the depiction combobox.
    planeNormalButtons : PlaneNormalButtons
        QPushButtons for controlling the plane normal.
    planeNormalLabel : qtpy.QtWidgets.QLabel
        Label for the plane normal buttons.
    clippingPlaneCheckbox : qtpy.QtWidgets.QCheckBox
        Checkbox for enabling the clipping plane.
    clippingPlaneSlider : superqt.QRangeSlider
        QRangeSlider for selecting the range of the clipping plane.
    planeSliderLabel : qtpy.QtWidgets.QLabel
        Label for the plane slider.
    planeSlider : superqt.QLabeledDoubleSlider
        QLabeledDoubleSlider for selecting the position of the plane.
    planeThicknessSlider : superqt.QLabeledDoubleSlider
        QLabeledDoubleSlider for selecting the thickness of the plane.
    planeThicknessLabel : qtpy.QtWidgets.QLabel
        Label for the plane thickness slider.
    """

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

        # plane slider to set the position of the plane in the 'plane' depiction.
        self.planeSliderLabel = QLabel("plane slider position", self.parent)
        self.planeSlider = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal, self.parent
        )  # we need a double slider because in the oblique orientation, we can have a negative value for the plane position
        self.planeSlider.setMinimum(0)
        self.planeSlider.setMaximum(self.layer.data.shape[-1])
        self.planeSlider.setFocusPolicy(Qt.NoFocus)
        self.planeSlider.valueChanged.connect(self.changePlanePosition)

        # plane thickness controls
        self.planeThicknessSlider = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal, self.parent
        )
        self.planeThicknessLabel = QLabel(trans._("plane thickness:"), self.parent)
        self.planeThicknessSlider.setFocusPolicy(Qt.NoFocus)
        self.planeThicknessSlider.setMinimum(1)
        self.planeThicknessSlider.setMaximum(50)
        self.planeThicknessSlider.setValue(self.layer.plane.thickness)
        self.planeThicknessSlider.valueChanged.connect(self.changePlaneThickness)

        # combine widgets
        self.layout().addRow(self.depictionLabel, self.depictionComboBox)
        self.layout().addRow(self.planeNormalLabel, self.planeNormalButtons)
        self.layout().addRow(self.planeThicknessLabel, self.planeThicknessSlider)
        self.layout().addRow(self.planeSliderLabel, self.planeSlider)
        self.layout().addRow(self.clippingPlaneCheckbox, self.clippingPlaneSlider)

        self._set_plane_slider_min_max(
            "z"
        )  # set initial span of the sliders based on the size of the z axis (which is the default plane normal)

    def changeDepiction(self, text: str):
        """Change the depiction of the layer between 'plane' and 'volume'.
        args:
            text: str, the new depiction of the layer. Can be 'plane' or 'volume'.
        """
        self.layer.depiction = text
        self._update_plane_parameter_visibility()

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
        """Change the value of the plane thickness slider"""
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

        # Set the minimum and maximum values of the plane slider
        self.planeSlider.setMinimum(clip_range[0])
        self.planeSlider.setMaximum(clip_range[1])

        # Set the minimum and maximum values of the clipping plane slider
        self.clippingPlaneSlider.setMinimum(clip_range[0])
        self.clippingPlaneSlider.setMaximum(clip_range[1])

        # Set the initial value of the plane slider to the middle of the range
        self.planeSlider.setValue(int((clip_range[0] + clip_range[1]) / 2))

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
        if self.parent.ndisplay == 2:
            self.depictionComboBox.hide()
            self.depictionLabel.hide()
        else:
            self.depictionComboBox.show()
            self.depictionLabel.show()

    def disconnect(self):
        """Disconnect all event connections (e.g. when layer is removed)."""

        if self.layer is not None:  # check if layer still exists
            self.layer.events.depiction.disconnect(self._on_depiction_change)
            self.layer.plane.events.thickness.disconnect(self._on_plane_thickness_change)
            self.layer.events.plane.disconnect(self._update_plane_slider)

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
