
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QHBoxLayout

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.widgets.qt_mode_buttons import QtModeRadioButton
from napari.layers.graph._graph_constants import Mode
from napari.utils.action_manager import action_manager

if TYPE_CHECKING:
    import napari.layers


class QtGraphControls(QtLayerControls):
    """Qt view and controls for the napari Graph layer.
    """

    layer: 'napari.layers.Graph'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        self.select_button = QtModeRadioButton(
            layer,
            'select_points',
            Mode.SELECT,
        )
        action_manager.bind_button(
            'napari:activate_graph_select_mode', self.select_button
        )
        # self.addition_button = QtModeRadioButton(layer, 'add_points', Mode.ADD)
        # action_manager.bind_button(
        #     'napari:activate_graph_add_mode', self.addition_button
        # )
        self.panzoom_button = QtModeRadioButton(
            layer,
            'pan',
            Mode.PAN_ZOOM,
            checked=True,
        )
        action_manager.bind_button(
            'napari:activate_graph_pan_zoom_mode', self.panzoom_button
        )
        # self.delete_button = QtModePushButton(
        #     layer,
        #     'delete_shape',
        # )
        # action_manager.bind_button(
        #     'napari:delete_selected_graph_points', self.delete_button
        # )
        
        self._EDIT_BUTTONS = (
            self.select_button,
            # self.addition_button,
            # self.delete_button,
        )

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        # button_row.addWidget(self.delete_button)
        # button_row.addWidget(self.addition_button)
        button_row.addWidget(self.select_button)
        button_row.addWidget(self.panzoom_button)
        button_row.setContentsMargins(0, 0, 0, 5)
        button_row.setSpacing(4)

        self.layout().addRow(button_row)
        