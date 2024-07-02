import contextlib
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QHBoxLayout

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import (
    qt_signals_blocked,
    set_widgets_enabled_with_opacity,
)
from napari._qt.widgets._slider_compat import QSlider
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari._qt.widgets.qt_mode_buttons import (
    QtModePushButton,
    QtModeRadioButton,
)
from napari.layers.points._points_constants import (
    SYMBOL_TRANSLATION,
    SYMBOL_TRANSLATION_INVERTED,
    Mode,
)
from napari.utils.action_manager import action_manager
from napari.utils.events import disconnect_events
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtGraphControls(QtLayerControls):
    """Qt view and controls for the napari Points layer.

    Parameters
    ----------
    layer : napari.layers.Points
        An instance of a napari Points layer.

    Attributes
    ----------
    addition_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add points to layer.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete points from layer.
    borderColorEdit : QColorSwatchEdit
        Widget to select display color for points borders.
    faceColorEdit : QColorSwatchEdit
        Widget to select display color for points faces.
    layer : napari.layers.Points
        An instance of a napari Points layer.
    outOfSliceCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to render out of slice.
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button for pan/zoom mode.
    select_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select points from layer.
    sizeSlider : qtpy.QtWidgets.QSlider
        Slider controlling size of points.
    symbolComboBox : qtpy.QtWidgets.QComboBox
        Drop down list of symbol options for points markers.

    Raises
    ------
    ValueError
        Raise error if points mode is not recognized.
        Points mode must be one of: ADD, PAN_ZOOM, or SELECT.
    """

    layer: 'napari.layers.Graph'

    def __init__(self, layer) -> None:
        super().__init__(layer)
