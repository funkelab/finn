from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel
from superqt import QLabeledDoubleSlider

import finn
import finn.layers
from finn._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from finn._qt.layer_controls.qt_layer_depiction_controls import QtLayerDepiction
from finn._qt.utils import qt_signals_blocked
from finn.layers.image._image_constants import (
    ImageRendering,
    Interpolation,
)
from finn.utils.translations import trans


class QtImageControls(QtBaseImageControls):
    """Qt view and controls for the napari Image layer.

    Parameters
    ----------
    layer : finn.layers.Image
        An instance of a napari Image layer.

    Attributes
    ----------
    layer : finn.layers.Image
        An instance of a napari Image layer.
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    panzoom_button : finn._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    transform_button : finn._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.
    attenuationSlider : qtpy.QtWidgets.QSlider
        Slider controlling attenuation rate for `attenuated_mip` mode.
    attenuationLabel : qtpy.QtWidgets.QLabel
        Label for the attenuation slider widget.
    interpComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the interpolation mode for image display.
    interpLabel : qtpy.QtWidgets.QLabel
        Label for the interpolation dropdown menu.
    isoThresholdSlider : qtpy.QtWidgets.QSlider
        Slider controlling the isosurface threshold value for rendering.
    isoThresholdLabel : qtpy.QtWidgets.QLabel
        Label for the isosurface threshold slider widget.
    renderComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the rendering mode for image display.
    renderLabel : qtpy.QtWidgets.QLabel
        Label for the rendering mode dropdown menu.
    """

    layer: "finn.layers.Image"
    PAN_ZOOM_ACTION_NAME = "activate_image_pan_zoom_mode"
    TRANSFORM_ACTION_NAME = "activate_image_transform_mode"

    def __init__(self, layer) -> None:
        super().__init__(layer)

        self.layer.events.interpolation2d.connect(self._on_interpolation_change)
        self.layer.events.interpolation3d.connect(self._on_interpolation_change)
        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self.layer.events.attenuation.connect(self._on_attenuation_change)
        # self.layer.events.depiction.connect(self._on_depiction_change)
        # self.layer.plane.events.thickness.connect(self._on_plane_thickness_change)
        # self.layer.events.plane.connect(self._update_plane_slider)

        self.interpComboBox = QComboBox(self)
        self.interpComboBox.currentTextChanged.connect(self.changeInterpolation)
        self.interpComboBox.setToolTip(
            trans._(
                "Texture interpolation for display.\nnearest and linear are most performant."
            )
        )
        self.interpLabel = QLabel(trans._("interpolation:"))

        renderComboBox = QComboBox(self)
        rendering_options = [i.value for i in ImageRendering]
        renderComboBox.addItems(rendering_options)
        index = renderComboBox.findText(
            self.layer.rendering, Qt.MatchFlag.MatchFixedString
        )
        renderComboBox.setCurrentIndex(index)
        renderComboBox.currentTextChanged.connect(self.changeRendering)
        self.renderComboBox = renderComboBox
        self.renderLabel = QLabel(trans._("rendering:"))

        # self.depictionComboBox = QComboBox(self)
        # depiction_options = [d.value for d in VolumeDepiction]
        # self.depictionComboBox.addItems(depiction_options)
        # index = self.depictionComboBox.findText(
        # self.layer.depiction, Qt.MatchFlag.MatchFixedString
        # )
        # self.depictionComboBox.setCurrentIndex(index)
        # self.depictionComboBox.currentTextChanged.connect(self.changeDepiction)
        # self.depictionLabel = QLabel(trans._("depiction:"))

        # # plane controls
        # self.planeNormalButtons = PlaneNormalButtons(self)
        # self.planeNormalLabel = QLabel(trans._("plane normal:"))
        # action_manager.bind_button(
        #     "napari:orient_plane_normal_along_z",
        #     self.planeNormalButtons.zButton,
        # )
        # action_manager.bind_button(
        #     "napari:orient_plane_normal_along_y",
        #     self.planeNormalButtons.yButton,
        # )
        # action_manager.bind_button(
        #     "napari:orient_plane_normal_along_x",
        #     self.planeNormalButtons.xButton,
        # )
        # action_manager.bind_button(
        #     "napari:orient_plane_normal_along_view_direction_no_gen",
        #     self.planeNormalButtons.obliqueButton,
        # )

        # self.planeNormalButtons.xButton.clicked.connect(
        #     lambda: self._set_orientation("x")
        # )
        # self.planeNormalButtons.yButton.clicked.connect(
        #     lambda: self._set_orientation("y")
        # )
        # self.planeNormalButtons.zButton.clicked.connect(
        #     lambda: self._set_orientation("z")
        # )
        # self.planeNormalButtons.obliqueButton.clicked.connect(
        #     lambda: self._set_orientation("oblique")
        # )

        # self.clippingPlaneCheckbox = QCheckBox(trans._("clipping plane"))
        # self.clippingPlaneCheckbox.setStyleSheet("""
        #     font-size:11px
        # """)

        # self.clippingPlaneCheckbox.stateChanged.connect(self.activateClippingPlane)
        # self.clippingPlaneSlider = QRangeSlider(Qt.Orientation.Horizontal, self)
        # self.clippingPlaneSlider.setStyleSheet("""
        #     QSlider::groove:horizontal:disabled {
        #         background: #353B43;  /* gray background */
        #     }
        #     QSlider::handle:horizontal:disabled {
        #         background: #4C545E;  /* Greyed-out handles */
        #     }
        # """)
        # self.clippingPlaneSlider.setFocusPolicy(Qt.NoFocus)
        # self.clippingPlaneSlider.setMinimum(1)
        # self.clippingPlaneSlider.setMaximum(self.layer.data.shape[-1])
        # self.clippingPlaneSlider.setSingleStep(1)
        # self.clippingPlaneSlider.setTickInterval(1)
        # self.clippingPlaneSlider.valueChanged.connect(self.changeClippingPlaneRange)

        # self.planeSliderLabel = QLabel("plane slider position")
        # self.planeSlider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, self)
        # self.planeSlider.setMinimum(0)
        # self.planeSlider.setMaximum(self.layer.data.shape[-1])
        # self.planeSlider.setFocusPolicy(Qt.NoFocus)
        # self.planeSlider.valueChanged.connect(self.changePlanePosition)
        # self.planeThicknessSlider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, self)
        # self.planeThicknessLabel = QLabel(trans._("plane thickness:"))
        # self.planeThicknessSlider.setFocusPolicy(Qt.NoFocus)
        # self.planeThicknessSlider.setMinimum(1)
        # self.planeThicknessSlider.setMaximum(50)
        # self.planeThicknessSlider.setValue(self.layer.plane.thickness)
        # self.planeThicknessSlider.valueChanged.connect(self.changePlaneThickness)

        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cmin, cmax = self.layer.contrast_limits_range
        sld.setMinimum(cmin)
        sld.setMaximum(cmax)
        sld.setValue(self.layer.iso_threshold)
        sld.valueChanged.connect(self.changeIsoThreshold)
        self.isoThresholdSlider = sld
        self.isoThresholdLabel = QLabel(trans._("iso threshold:"))

        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(0.5)
        sld.setSingleStep(0.001)
        sld.setValue(self.layer.attenuation)
        sld.setDecimals(3)
        sld.valueChanged.connect(self.changeAttenuation)
        self.attenuationSlider = sld
        self.attenuationLabel = QLabel(trans._("attenuation:"))

        # self._on_ndisplay_changed()
        # self._set_orientation("z")

        colormap_layout = QHBoxLayout()
        if hasattr(self.layer, "rgb") and self.layer.rgb:
            colormap_layout.addWidget(QLabel("RGB"))
            self.colormapComboBox.setVisible(False)
            self.colorbarLabel.setVisible(False)
        else:
            colormap_layout.addWidget(self.colorbarLabel)
            colormap_layout.addWidget(self.colormapComboBox)
        colormap_layout.addStretch(1)

        self.layout().addRow(self.button_grid)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._("blending:"), self.blendComboBox)
        self.layout().addRow(trans._("contrast limits:"), self.contrastLimitsSlider)
        self.layout().addRow(trans._("auto-contrast:"), self.autoScaleBar)
        self.layout().addRow(trans._("gamma:"), self.gammaSlider)
        self.layout().addRow(trans._("colormap:"), colormap_layout)
        self.layout().addRow(self.interpLabel, self.interpComboBox)

        self.depictionControls = QtLayerDepiction(self)
        for i in range(self.depictionControls.layout().rowCount()):
            label_item = self.depictionControls.layout().itemAt(i, QFormLayout.LabelRole)
            field_item = self.depictionControls.layout().itemAt(i, QFormLayout.FieldRole)

            label_widget = label_item.widget() if label_item else None
            field_widget = field_item.widget() if field_item else None

            if label_widget and field_widget:
                self.layout().addRow(label_widget, field_widget)
            elif field_widget:  # If there's no label, just add the field
                self.layout().addRow(field_widget)
        # self.layout().addRow(self.depictionLabel, self.depictionComboBox)
        # self.layout().addRow(self.planeNormalLabel, self.planeNormalButtons)
        # self.layout().addRow(self.planeThicknessLabel, self.planeThicknessSlider)
        # self.layout().addRow(self.planeSliderLabel, self.planeSlider)

        # self.layout().addRow(self.clippingPlaneCheckbox, self.clippingPlaneSlider)

        self.layout().addRow(self.renderLabel, self.renderComboBox)
        self.layout().addRow(self.isoThresholdLabel, self.isoThresholdSlider)
        self.layout().addRow(self.attenuationLabel, self.attenuationSlider)

    def changeInterpolation(self, text):
        """Change interpolation mode for image display.

        Parameters
        ----------
        text : str
            Interpolation mode used by vispy. Must be one of our supported
            modes:
            'bessel', 'bicubic', 'linear', 'blackman', 'catrom', 'gaussian',
            'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
            'nearest', 'spline16', 'spline36'
        """
        if self.ndisplay == 2:
            self.layer.interpolation2d = text
        else:
            self.layer.interpolation3d = text

    def changeRendering(self, text):
        """Change rendering mode for image display.

        Parameters
        ----------
        text : str
            Rendering mode used by vispy.
            Selects a preset rendering mode in vispy that determines how
            volume is displayed:
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maximum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * attenuated_mip: attenuated maximum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
        """
        self.layer.rendering = text
        self._update_rendering_parameter_visibility()

    # def changeDepiction(self, text):
    #     self.layer.depiction = text
    #     self._update_plane_parameter_visibility()

    # def changePlaneThickness(self, value: float):
    #     self.layer.plane.thickness = value

    # def _update_plane_slider(self):
    #     """Updates the value of the plane slider when the user used the shift+drag method to shift the plane or when switching between different layers"""

    #     new_position = np.array(self.layer.plane.position)
    #     plane_normal = np.array(self.layer.plane.normal)
    #     slider_value = np.dot(new_position, plane_normal) / np.dot(
    #         plane_normal, plane_normal
    #     )
    #     self.planeSlider.valueChanged.disconnect(self.changePlanePosition)
    #     self.planeSlider.setValue(int(slider_value))
    #     self.planeSlider.valueChanged.connect(self.changePlanePosition)

    # def _compute_plane_range(self) -> tuple[float, float]:
    #     """Compute the range of the plane and clipping plane sliders

    #     returns:
    #         tuple[float, float], the minimum and maximum values of the slider
    #     """

    #     normal = np.array(self.layer.plane.normal)
    #     Lx, Ly, Lz = self.layer.data.shape[-3:]

    #     # Define the corners of the 3D image bounding box
    #     corners = np.array(
    #         [
    #             [0, 0, 0],
    #             [Lx, 0, 0],
    #             [0, Ly, 0],
    #             [0, 0, Lz],
    #             [Lx, Ly, 0],
    #             [Lx, 0, Lz],
    #             [0, Ly, Lz],
    #             [Lx, Ly, Lz],
    #         ]
    #     )

    #     # Project the corners onto the normal vector
    #     projections = np.dot(corners, normal)

    #     # The range of possible positions is given by the min and max projections
    #     min_position = np.min(projections)
    #     max_position = np.max(projections)

    #     return (min_position, max_position)

    # def _set_orientation(self, orientation: str) -> None:
    #     """Set the range of the (clipping) plane sliders based on the orientation.
    #     args:
    #         orientation: str, the direction in which the plane should
    #             slide. Can be 'x', 'y', 'z', or 'oblique'.
    #     """

    #     if orientation == "x":
    #         clip_range = (0, self.layer.data.shape[-1])

    #     elif orientation == "y":
    #         clip_range = (0, self.layer.data.shape[-2])

    #     elif orientation == "z":
    #         clip_range = (0, self.layer.data.shape[-3])

    #     else:  # oblique view
    #         clip_range = self._compute_plane_range()
    #         self.layer.experimental_clipping_planes[0].normal = self.layer.plane.normal
    #         self.layer.experimental_clipping_planes[1].normal = (
    #             -self.layer.plane.normal[-3],
    #             -self.layer.plane.normal[-2],
    #             -self.layer.plane.normal[-1],
    #         )

    #     self.planeSlider.setMinimum(clip_range[0])
    #     self.planeSlider.setMaximum(clip_range[1])
    #     self.clippingPlaneSlider.setMinimum(clip_range[0])
    #     self.clippingPlaneSlider.setMaximum(clip_range[1])
    #     min_value = int(clip_range[0] + (1 / 3) * (clip_range[1] - clip_range[0]))
    #     max_value = int(clip_range[0] + (2 / 3) * (clip_range[1] - clip_range[0]))
    #     self.clippingPlaneSlider.setValue((min_value, max_value))
    #     self.planeSlider.setMinimum(clip_range[0])
    #     self.planeSlider.setMaximum(clip_range[1])

    # def changePlanePosition(self, value: float):
    #     plane_normal = np.array(self.layer.plane.normal)
    #     new_position = np.array([0, 0, 0]) + value * plane_normal
    #     self.layer.plane.position = tuple(new_position)

    # def activateClippingPlane(self, state):
    #     if state:
    #         self.layer.experimental_clipping_planes[0].enabled = True
    #         self.layer.experimental_clipping_planes[1].enabled = True
    #         self.clippingPlaneSlider.setEnabled(True)
    #     else:
    #         self.layer.experimental_clipping_planes[0].enabled = False
    #         self.layer.experimental_clipping_planes[1].enabled = False
    #         self.clippingPlaneSlider.setEnabled(False)

    # def changeClippingPlaneRange(self, value):
    #     viewer = finn.viewer.current_viewer()

    #     if len(self.layer.experimental_clipping_planes) == 0:
    #         plane = self.layer.plane
    #         self.layer.experimental_clipping_planes.append(
    #             ClippingPlane(
    #                 normal=plane.normal,
    #                 position=plane.position,
    #                 enabled=False,
    #             )
    #         )
    #         self.layer.experimental_clipping_planes.append(
    #             ClippingPlane(
    #                 normal=[-n for n in plane.normal],
    #                 position=plane.position,
    #                 enabled=False,
    #             )
    #         )

    #     plane_normal = np.array(self.layer.experimental_clipping_planes[0].normal)
    #     new_position_1 = np.array([0, 0, 0]) + value[0] * plane_normal
    #     new_position_1 = (
    #         int(new_position_1[0] * viewer.dims.range[-3].step),
    #         (new_position_1[1] * viewer.dims.range[-2].step),
    #         int(new_position_1[2] * viewer.dims.range[-1].step),
    #     )
    #     self.layer.experimental_clipping_planes[0].position = new_position_1
    #     new_position_2 = np.array([0, 0, 0]) + value[1] * plane_normal
    #     new_position_2 = (
    #         int(new_position_2[0] * viewer.dims.range[-3].step),
    #         (new_position_2[1] * viewer.dims.range[-2].step),
    #         int(new_position_2[2] * viewer.dims.range[-1].step),
    #     )

    #     self.layer.experimental_clipping_planes[1].position = new_position_2

    def changeIsoThreshold(self, value):
        """Change isosurface threshold on the layer model.

        Parameters
        ----------
        value : float
            Threshold for isosurface.
        """
        with self.layer.events.blocker(self._on_iso_threshold_change):
            self.layer.iso_threshold = value

    def _on_contrast_limits_change(self):
        with self.layer.events.blocker(self._on_iso_threshold_change):
            cmin, cmax = self.layer.contrast_limits_range
            self.isoThresholdSlider.setMinimum(cmin)
            self.isoThresholdSlider.setMaximum(cmax)
        return super()._on_contrast_limits_change()

    def _on_iso_threshold_change(self):
        """Receive layer model isosurface change event and update the slider."""
        with self.layer.events.iso_threshold.blocker():
            self.isoThresholdSlider.setValue(self.layer.iso_threshold)

    def changeAttenuation(self, value):
        """Change attenuation rate for attenuated maximum intensity projection.

        Parameters
        ----------
        value : Float
            Attenuation rate for attenuated maximum intensity projection.
        """
        with self.layer.events.blocker(self._on_attenuation_change):
            self.layer.attenuation = value

    def _on_attenuation_change(self):
        """Receive layer model attenuation change event and update the slider."""
        with self.layer.events.attenuation.blocker():
            self.attenuationSlider.setValue(self.layer.attenuation)

    def _on_interpolation_change(self, event):
        """Receive layer interpolation change event and update dropdown menu.

        Parameters
        ----------
        event : finn.utils.event.Event
            The napari event that triggered this method.
        """
        interp_string = event.value.value

        with (
            self.layer.events.interpolation.blocker(),
            self.layer.events.interpolation2d.blocker(),
            self.layer.events.interpolation3d.blocker(),
        ):
            if self.interpComboBox.findText(interp_string) == -1:
                self.interpComboBox.addItem(interp_string)
            self.interpComboBox.setCurrentText(interp_string)

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        with self.layer.events.rendering.blocker():
            index = self.renderComboBox.findText(
                self.layer.rendering, Qt.MatchFlag.MatchFixedString
            )
            self.renderComboBox.setCurrentIndex(index)
            self._update_rendering_parameter_visibility()

    # def _on_depiction_change(self):
    #     """Receive layer model depiction change event and update combobox."""
    #     with self.layer.events.depiction.blocker():
    #         index = self.depictionComboBox.findText(
    #             self.layer.depiction, Qt.MatchFlag.MatchFixedString
    #         )
    #         self.depictionComboBox.setCurrentIndex(index)
    #         self._update_plane_parameter_visibility()

    # def _on_plane_thickness_change(self):
    #     with self.layer.plane.events.blocker():
    #         self.planeThicknessSlider.setValue(self.layer.plane.thickness)

    def _update_rendering_parameter_visibility(self):
        """Hide isosurface rendering parameters if they aren't needed."""
        rendering = ImageRendering(self.layer.rendering)
        iso_threshold_visible = rendering == ImageRendering.ISO
        self.isoThresholdLabel.setVisible(iso_threshold_visible)
        self.isoThresholdSlider.setVisible(iso_threshold_visible)
        attenuation_visible = rendering == ImageRendering.ATTENUATED_MIP
        self.attenuationSlider.setVisible(attenuation_visible)
        self.attenuationLabel.setVisible(attenuation_visible)

    # def _update_plane_parameter_visibility(self):
    #     """Hide plane rendering controls if they aren't needed."""
    #     depiction = VolumeDepiction(self.layer.depiction)
    #     plane_visible = (
    #         depiction == VolumeDepiction.PLANE
    #         and self.ndisplay == 3
    #         and self.layer.ndim >= 3
    #     )
    #     clipping_plane_visible = (
    #         depiction == VolumeDepiction.VOLUME
    #         and self.ndisplay == 3
    #         and self.layer.ndim >= 3
    #     )
    #     self.planeNormalButtons.setVisible(plane_visible or clipping_plane_visible)
    #     self.planeNormalLabel.setVisible(plane_visible or clipping_plane_visible)
    #     self.planeThicknessSlider.setVisible(plane_visible)
    #     self.planeThicknessLabel.setVisible(plane_visible)
    #     self.planeSlider.setVisible(plane_visible)
    #     self.planeSliderLabel.setVisible(plane_visible)

    #     self.clippingPlaneCheckbox.setVisible(clipping_plane_visible)
    #     self.clippingPlaneSlider.setVisible(clipping_plane_visible)

    def _update_interpolation_combo(self):
        interp_names = [i.value for i in Interpolation.view_subset()]
        interp = (
            self.layer.interpolation2d
            if self.ndisplay == 2
            else self.layer.interpolation3d
        )
        with qt_signals_blocked(self.interpComboBox):
            self.interpComboBox.clear()
            self.interpComboBox.addItems(interp_names)
            self.interpComboBox.setCurrentText(interp)

    def _on_ndisplay_changed(self):
        """Update widget visibility based on 2D and 3D visualization modes."""
        self._update_interpolation_combo()
        self.depictionControls._on_ndisplay_changed()
        if self.ndisplay == 2:
            self.isoThresholdSlider.hide()
            self.isoThresholdLabel.hide()
            self.attenuationSlider.hide()
            self.attenuationLabel.hide()
            self.renderComboBox.hide()
            self.renderLabel.hide()
        else:
            self.renderComboBox.show()
            self.renderLabel.show()
            self._update_rendering_parameter_visibility()
        super()._on_ndisplay_changed()


# class PlaneNormalButtons(QWidget):
#     """Qt buttons for controlling plane orientation.

#         Attributes
#     ----------
#     xButton : qtpy.QtWidgets.QPushButton
#         Button which orients a plane normal along the x axis.
#     yButton : qtpy.QtWidgets.QPushButton
#         Button which orients a plane normal along the y axis.
#     zButton : qtpy.QtWidgets.QPushButton
#         Button which orients a plane normal along the z axis.
#     obliqueButton : qtpy.QtWidgets.QPushButton
#         Button which orients a plane normal along the camera view direction.
#     """

#     def __init__(self, parent=None) -> None:
#         super().__init__(parent=parent)
#         self.setLayout(QHBoxLayout())
#         self.layout().setSpacing(2)
#         self.layout().setContentsMargins(0, 0, 0, 0)

#         self.xButton = QPushButton("x")
#         self.yButton = QPushButton("y")
#         self.zButton = QPushButton("z")
#         self.obliqueButton = QPushButton(trans._("oblique"))

#         self.layout().addWidget(self.xButton)
#         self.layout().addWidget(self.yButton)
#         self.layout().addWidget(self.zButton)
#         self.layout().addWidget(self.obliqueButton)
