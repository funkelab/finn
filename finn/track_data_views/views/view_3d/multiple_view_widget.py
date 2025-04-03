import time

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QSplitter,
)

import finn
from finn.components.viewer_model import ViewerModel
from finn.layers import Labels, Layer, Points, Shapes, Tracks
from finn.qt import QtViewer
from finn.track_data_views.views.layers.contour_labels import ContourLabels
from finn.track_data_views.views.layers.track_graph import TrackGraph
from finn.track_data_views.views.layers.track_labels import TrackLabels
from finn.track_data_views.views.layers.track_points import TrackPoints
from finn.track_data_views.views_coordinator.tracks_viewer import TracksViewer
from finn.utils.action_manager import action_manager
from finn.utils.events.event import WarningEmitter


def copy_layer(layer: Layer, name: str = ""):
    if isinstance(
        layer, TrackGraph
    ):  # instead of showing the tracks (not very useful on 3D data because they are collapsed to a single frame), use an empty shapes layer as substitute to ensure that the layer indices in the orthogonal viewer models match with those in the main viewer
        res_layer = Shapes(
            data=[],
        )

    elif isinstance(layer, TrackLabels):
        res_layer = ContourLabels(
            data=layer.data,
            name=layer.name,
            colormap=layer.colormap,
            opacity=layer.opacity,
            scale=layer.scale,
        )
        layer.update_group_labels.connect(res_layer.set_group_labels)
        res_layer.group_labels = layer.group_labels
    elif isinstance(layer, TrackPoints):
        res_layer = Points(
            data=layer.data,
            name=layer.name,
            symbol=layer.symbol,
            face_color=layer.face_color,
            size=layer.size,
            properties=layer.properties,
            border_color=layer.border_color,
            scale=layer.scale,
            blending="translucent_no_depth",
        )
    else:
        res_layer = Layer.create(*layer.as_layer_data_tuple())

    res_layer.metadata["viewer_name"] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ("thumbnail", "name"):
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


action_manager.bind_shortcut("finn:move_point", "T")


class own_partial:
    """
    Workaround for deepcopy not copying partial functions
    (Qt widgets are not serializable)
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str | None = None,
        layer_type: str | None = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class MultipleViewerWidget(QSplitter):
    """The main widget of the example."""

    def __init__(self, viewer: finn.Viewer):
        super().__init__()
        self.viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(self.viewer)
        self.viewer_model1 = ViewerModel(title="model1")
        self.viewer_model2 = ViewerModel(title="model2")
        self._block = False
        self.qt_viewer1 = QtViewerWrap(viewer, self.viewer_model1)
        self.qt_viewer2 = QtViewerWrap(viewer, self.viewer_model2)
        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Vertical)
        viewer_splitter.addWidget(self.qt_viewer1)
        viewer_splitter.addWidget(self.qt_viewer2)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        self.addWidget(viewer_splitter)

        # add existing layers
        for i, layer in enumerate(self.viewer.layers):
            self.viewer_model1.layers.insert(i, copy_layer(layer, "model1"))
            self.viewer_model2.layers.insert(i, copy_layer(layer, "model2"))
            for name in get_property_names(layer):
                getattr(layer.events, name).connect(
                    own_partial(self._property_sync, name)
                )
            if isinstance(layer, Labels):
                layer.events.set_data.connect(self._set_data_refresh)
                self.viewer_model1.layers[layer.name].events.set_data.connect(
                    self._set_data_refresh
                )
                self.viewer_model2.layers[layer.name].events.set_data.connect(
                    self._set_data_refresh
                )

            # connect data and paint events
            if layer.name != ".cross" and not isinstance(layer, Tracks):
                # model 1
                self.viewer_model1.layers[layer.name].events.data.connect(self._sync_data)
                self.viewer_model1.layers[layer.name].events.mode.connect(self._sync_mode)
                if isinstance(self.viewer_model1.layers[layer.name], Labels):
                    self.viewer_model1.layers[layer.name].events.paint.connect(
                        self._sync_paint
                    )
                    self.viewer_model1.layers[layer.name].events.selected_label.connect(
                        self._sync_selected_label
                    )
                    self.viewer_model1.layers[layer.name].mouse_drag_callbacks.append(
                        self._sync_click
                    )

                # model 2
                self.viewer_model2.layers[layer.name].events.data.connect(self._sync_data)
                self.viewer_model2.layers[layer.name].events.mode.connect(self._sync_mode)
                if isinstance(self.viewer_model2.layers[layer.name], Labels):
                    self.viewer_model2.layers[layer.name].events.paint.connect(
                        self._sync_paint
                    )
                    self.viewer_model2.layers[layer.name].events.selected_label.connect(
                        self._sync_selected_label
                    )
                    self.viewer_model2.layers[layer.name].mouse_drag_callbacks.append(
                        self._sync_click
                    )
            layer.events.name.connect(self._sync_name)
            self._order_update()

        # connect to events
        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(self._layer_selection_changed)
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_model1.dims.events.current_step.connect(self._point_update)
        self.viewer_model2.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)
        self.viewer_model1.events.status.connect(self._status_update)
        self.viewer_model2.events.status.connect(self._status_update)

    def update(self):
        self.tracks_viewer.update_selection()

    def _status_update(self, event):
        self.viewer.status = event.value

    def _reset_view(self):
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()

    def _reset_layers(self):
        self.viewer_model1.layers.clear()
        self.viewer_model2.layers.clear()

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            self.viewer_model1.layers.selection.active = None
            self.viewer_model2.layers.selection.active = None
            return

        if event.value.name in self.viewer_model1.layers:
            self.viewer_model1.layers.selection.active = self.viewer_model1.layers[
                event.value.name
            ]
        if event.value.name in self.viewer_model2.layers:
            self.viewer_model2.layers.selection.active = self.viewer_model2.layers[
                event.value.name
            ]

    def _point_update(self, event):
        try:
            for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
                if model.dims is event.source:
                    continue
                model.dims.current_step = event.value
        except IndexError:
            "Layer was already removed! This error likely occurs because two actions are called at the same time."

    def _order_update(self):
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model1.dims.order = order
            self.viewer_model2.dims.order = order
            return

        order[-3:] = order[-2], order[-3], order[-1]
        self.viewer_model1.dims.order = order
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.viewer_model2.dims.order = order

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events"""

        if (
            event.value.name not in self.viewer_model1.layers
            and event.value.name not in self.viewer_model2.layers
        ):
            self.viewer_model1.layers.insert(
                event.index, copy_layer(event.value, "model1")
            )
            self.viewer_model2.layers.insert(
                event.index, copy_layer(event.value, "model2")
            )

            for name in get_property_names(event.value):
                getattr(event.value.events, name).connect(
                    own_partial(self._property_sync, name)
                )

            if isinstance(event.value, Labels):
                event.value.events.set_data.connect(self._set_data_refresh)
                self.viewer_model1.layers[event.value.name].events.set_data.connect(
                    self._set_data_refresh
                )
                self.viewer_model2.layers[event.value.name].events.set_data.connect(
                    self._set_data_refresh
                )

            if isinstance(event.value, TrackPoints):
                event.value.events.border_color.connect(self._sync_shown_points)

            # connect data and paint events
            if event.value.name != ".cross" and not isinstance(event.value, TrackGraph):
                # model 1
                self.viewer_model1.layers[event.value.name].events.data.connect(
                    self._sync_data
                )
                self.viewer_model1.layers[event.value.name].events.mode.connect(
                    self._sync_mode
                )

                if isinstance(self.viewer_model1.layers[event.value.name], Labels):
                    self.viewer_model1.layers[
                        event.value.name
                    ].events.selected_label.connect(self._sync_selected_label)

                    if isinstance(event.value, TrackLabels):
                        self.viewer_model1.layers[
                            event.value.name
                        ].mouse_drag_callbacks.append(self._sync_click)

                        self.viewer_model1.layers[event.value.name].events.paint.connect(
                            self._sync_paint
                        )

                        self.viewer_model1.layers[event.value.name].bind_key("z")(
                            self.tracks_viewer.undo
                        )
                        self.viewer_model1.layers[event.value.name].bind_key("r")(
                            self.tracks_viewer.redo
                        )
                        self.viewer_model1.layers[
                            event.value.name
                        ].undo = self.tracks_viewer.undo
                        self.viewer_model1.layers[
                            event.value.name
                        ].redo = self.tracks_viewer.redo

                if isinstance(event.value, TrackPoints):
                    self.viewer_model1.layers[
                        event.value.name
                    ].mouse_drag_callbacks.append(self._sync_point_click)
                    self.viewer_model1.layers[event.value.name].bind_key("z")(
                        self.tracks_viewer.undo
                    )
                    self.viewer_model1.layers[event.value.name].bind_key("r")(
                        self.tracks_viewer.redo
                    )
                    self.viewer_model1.layers[
                        event.value.name
                    ].undo = self.tracks_viewer.undo
                    self.viewer_model1.layers[
                        event.value.name
                    ].redo = self.tracks_viewer.redo

                # model 2
                self.viewer_model2.layers[event.value.name].events.data.connect(
                    self._sync_data
                )
                self.viewer_model2.layers[event.value.name].events.mode.connect(
                    self._sync_mode
                )

                if isinstance(self.viewer_model2.layers[event.value.name], Labels):
                    self.viewer_model2.layers[
                        event.value.name
                    ].events.selected_label.connect(self._sync_selected_label)

                    if isinstance(event.value, TrackLabels):
                        self.viewer_model2.layers[
                            event.value.name
                        ].mouse_drag_callbacks.append(self._sync_click)

                        self.viewer_model2.layers[event.value.name].events.paint.connect(
                            self._sync_paint
                        )
                        self.viewer_model2.layers[event.value.name].bind_key("z")(
                            self.tracks_viewer.undo
                        )
                        self.viewer_model2.layers[event.value.name].bind_key("r")(
                            self.tracks_viewer.redo
                        )
                        self.viewer_model2.layers[
                            event.value.name
                        ].undo = self.tracks_viewer.undo
                        self.viewer_model2.layers[
                            event.value.name
                        ].redo = self.tracks_viewer.redo

                if isinstance(event.value, TrackPoints):
                    self.viewer_model2.layers[
                        event.value.name
                    ].mouse_drag_callbacks.append(self._sync_point_click)
                    self.viewer_model2.layers[event.value.name].bind_key("z")(
                        self.tracks_viewer.undo
                    )
                    self.viewer_model2.layers[event.value.name].bind_key("r")(
                        self.tracks_viewer.redo
                    )
                    self.viewer_model2.layers[
                        event.value.name
                    ].undo = self.tracks_viewer.undo
                    self.viewer_model2.layers[
                        event.value.name
                    ].redo = self.tracks_viewer.redo

            event.value.events.name.connect(self._sync_name)

            self._order_update()

    def _sync_selected_label(self, event):
        """Sync the selected label between Label instances"""

        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if event.source.name in model.layers:
                layer = model.layers[event.source.name]
                if layer is event.source:
                    return
                try:
                    self._block = True
                    layer.selected_label = event.source.selected_label
                finally:
                    self._block = False

    def _sync_mode(self, event):
        """Sync the tool mode between source viewer and other viewer models"""

        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if event.source.name in model.layers:
                layer = model.layers[event.source.name]
                if layer is event.source:
                    continue
                try:
                    self._block = True
                    layer.mode = event.source.mode
                finally:
                    self._block = False

    def _sync_point_click(self, layer, event):
        """Retrieve the label that was clicked on and forward it to the TrackLabels instance if present"""

        name = layer.name
        if (
            event.type == "mouse_press"
            and name in self.viewer.layers
            and isinstance(self.viewer.layers[name], TrackPoints)
        ):
            # differentiate between click and drag
            mouse_press_time = time.time()
            dragged = False
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True
                yield
            if dragged and time.time() - mouse_press_time < 0.5:
                dragged = False  # suppress micro drag events and treat them as click
            if not dragged:
                point_index = layer.get_value(
                    event.position,
                    view_direction=event.view_direction,
                    dims_displayed=event.dims_displayed,
                    world=True,
                )
                self.viewer.layers[name].process_point_click(point_index, event)

    def _sync_click(self, layer, event):
        """Retrieve the label that was clicked on and forward it to the TrackLabels instance if present"""

        name = layer.name
        if (
            event.type == "mouse_press"
            and layer.mode == "pan_zoom"
            and name in self.viewer.layers
            and isinstance(self.viewer.layers[name], TrackLabels)
        ):
            # differentiate between click and drag
            mouse_press_time = time.time()
            dragged = False
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True
                yield
            if dragged and time.time() - mouse_press_time < 0.5:
                dragged = False  # suppress micro drag events and treat them as click
            # on release
            if not dragged:
                label = layer.get_value(
                    event.position,
                    view_direction=event.view_direction,
                    dims_displayed=event.dims_displayed,
                    world=True,
                )

                # Process the click event on the TrackLabels instance
                self.viewer.layers[name].process_click(event, label)

    def _sync_paint(self, event):
        """Forward the paint event to the TrackLabels, if present"""

        if event.source.name in self.viewer.layers and isinstance(
            self.viewer.layers[event.source.name], TrackLabels
        ):
            self.viewer.layers[event.source.name]._on_paint(event)

    def _sync_name(self, event):
        """sync name of layers"""

        try:
            index = self.viewer.layers.index(event.source)
            self.viewer_model1.layers[index].name = event.source.name
            self.viewer_model2.layers[index].name = event.source.name
        except IndexError:
            return

    def _sync_data(self, event):
        """sync data modification from additional viewers"""

        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if event.source.name in model.layers:
                layer = model.layers[event.source.name]
                if layer is event.source:
                    self.viewer.layers[
                        event.source.name
                    ].selected_data = event.source.selected_data
                    self.viewer.layers[event.source.name]._update_data(event)
                    continue
                try:
                    self._block = True
                    layer.data = event.source.data

                finally:
                    self._block = False

    def _sync_shown_points(self, event):
        """Sync the visible points between TrackPoints layer and orthogonal views"""

        for model in [self.viewer_model1, self.viewer_model2]:
            if event.source.name in model.layers:
                layer = model.layers[event.source.name]
                try:
                    self._block = True
                    layer.shown = event.source.shown
                    layer.border_color = event.source.border_color
                    layer.size = event.source.size
                    layer.refresh()
                finally:
                    self._block = False

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if event.source.name in model.layers:
                layer = model.layers[event.source.name]
                if layer is event.source:
                    continue
                try:
                    self._block = True
                    layer.refresh()
                finally:
                    self._block = False

    def _layer_removed(self, event):
        """remove layer in all viewers"""

        layer_name = event.value.name
        if layer_name in self.viewer_model1.layers:
            self.viewer_model1.layers.pop(layer_name)
        if layer_name in self.viewer_model2.layers:
            self.viewer_model2.layers.pop(layer_name)

    def _layer_moved(self, event):
        """update order of layers"""

        dest_index = (
            event.new_index if event.new_index < event.index else event.new_index + 1
        )
        self.viewer_model1.layers.move(event.index, dest_index)
        self.viewer_model2.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""

        if event.source.name not in self.viewer.layers:
            return
        try:
            self._block = True
            if event.source.name in self.viewer_model1.layers:
                setattr(
                    self.viewer_model1.layers[event.source.name],
                    name,
                    getattr(event.source, name),
                )
            if event.source.name in self.viewer_model2.layers:
                setattr(
                    self.viewer_model2.layers[event.source.name],
                    name,
                    getattr(event.source, name),
                )
        finally:
            self._block = False
