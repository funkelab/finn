from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from funtracks.features import Feature
from funtracks.user_actions import UserUpdateSegmentation
from psygnal import Signal

import finn
from finn.utils import DirectLabelColormap
from finn.utils.action_manager import action_manager
from finn.utils.notifications import show_info

if TYPE_CHECKING:
    import funlib.persistence as fp

    from finn.track_data_views.views_coordinator.project_viewer import ProjectViewer
    from finn.utils.events import Event


class Alphamap:
    updated = Signal()

    def __init__(self, alpha_dict):
        self._alpha_dict = alpha_dict

    def map(self, value):
        return self._alpha_dict[value]

    def update(self, alpha_dict):
        self._alpha_dict.update(alpha_dict)
        self.updated.emit()


def new_label(layer: TrackLabels):
    """A function to override the default finn labels new_label function.
    Must be registered (see end of this file)"""
    layer.events.selected_label.disconnect(layer._ensure_valid_label)
    _new_label(layer, new_track_id=True)
    layer.events.selected_label.connect(layer._ensure_valid_label)


def _new_label(layer: TrackLabels, new_track_id=True):
    """A function to get a new label for a given TrackLabels layer. Should properly
    go on the class, but needs to be registered to override the default finn function
    in the action manager. This helper is abstracted out because we want to do the same
    thing without making a new track id in the layer, and with the new track id in the
    overriden action.

    Args:
        layer (TrackLabels): A TrackLabels layer from which get a new label for drawing a
            new segmentation. Updates the selected_label attribute.
        new_track_id (bool, optional): If you should also generate a new track id and set
            it to the selected_track attribute. Defaults to True.
    """

    if isinstance(layer.data, np.ndarray):
        new_selected_label = np.max(layer.data) + 1
        if layer.selected_label == new_selected_label:
            show_info(
                "Current selected label is not being used. You will need to use it first "
                "to be able to set the current select label to the next one available"
            )
        else:
            if new_track_id:
                new_selected_track = layer.project_viewer.project.get_next_track_id()
                layer.selected_track = new_selected_track
            layer.selected_label = new_selected_label
            layer.colormap.color_dict[new_selected_label] = (
                layer.project_viewer.colormap.map(layer.selected_track)
            )
            # to refresh, otherwise you paint with a transparent label until you
            # release the mouse
            layer.colormap = DirectLabelColormap(color_dict=layer.colormap.color_dict)
    else:
        show_info("Calculating empty label on non-numpy array is not supported")


class TrackLabels(finn.layers.Labels):
    """Extended labels layer that holds the track information and emits
    and responds to dynamics visualization signals"""

    @property
    def _type_string(self) -> str:
        return (
            "labels"  # to make sure that the layer is treated as labels layer for saving
        )

    def __init__(
        self,
        viewer: finn.Viewer,
        data: fp.Array,
        name: str,
        opacity: float,
        project_viewer: ProjectViewer,
    ):
        self.project_viewer = project_viewer
        self.project = project_viewer.project
        self.selected_track = None
        self.default_opacity = opacity
        self.default_color = [0.5, 0.5, 0.5, 1.0]
        self.not_visible_opacity = 0.2
        self.highighted_opacity = 1.0
        self.color_by: Feature | None
        self.colormap_seed = 1010101
        if len(self.project.solution) == 0:
            self.color_by: Feature | None = None
        else:
            self.color_by = self.project.cand_graph.features.track_id
        self.alphamap = Alphamap(alpha_dict=defaultdict(lambda: self.default_opacity))
        self.alphamap.updated.connect(self.refresh_colormap)

        super().__init__(
            # this will load the whole array into memory :)
            data=data.data.compute(),
            name=name,
            opacity=opacity,
            colormap=self._get_colormap(),
            scale=data.voxel_size,
            multiscale=False,
        )

        self.viewer = viewer

        # Key bindings (should be specified both on the viewer (in tracks_viewer)
        # and on the layer to overwrite finn defaults)
        self.bind_key("q")(self.project_viewer.toggle_display_mode)
        self.bind_key("a")(self.project_viewer.create_edge)
        self.bind_key("d")(self.project_viewer.delete_node)
        self.bind_key("Delete")(self.project_viewer.delete_node)
        self.bind_key("b")(self.project_viewer.delete_edge)
        # self.bind_key("s")(self.tracks_viewer.set_split_node)
        # self.bind_key("e")(self.tracks_viewer.set_endpoint_node)
        # self.bind_key("c")(self.tracks_viewer.set_linear_node)
        self.bind_key("z")(self.project_viewer.undo)
        self.bind_key("r")(self.project_viewer.redo)

        # Connect click events to node selection
        @self.mouse_drag_callbacks.append
        def click(_, event):
            if (
                event.type == "mouse_press"
                and self.mode == "pan_zoom"
                and not (
                    self.project_viewer.mode == "lineage"
                    and self.viewer.dims.ndisplay == 3
                )
            ):  # disable selecting in lineage mode in 3D
                label = self.get_value(
                    event.position,
                    view_direction=event.view_direction,
                    dims_displayed=event.dims_displayed,
                    world=True,
                )
                # check opacity (=visibility) in the colormap
                if label is not None and label != 0 and self.colormap.map(label)[-1] != 0:
                    append = "Shift" in event.modifiers
                    self.project_viewer.selected_nodes.add(label, append)

        # Listen to paint events and changing the selected label
        self.events.paint.connect(self._on_paint)
        self.project_viewer.selected_nodes.list_updated.connect(
            self.update_selected_label
        )
        self.events.selected_label.connect(self._ensure_valid_label)
        self.events.mode.connect(self._check_mode)
        self.viewer.dims.events.current_step.connect(self._ensure_valid_label)

    def _get_feature_colormap(self, feature: Feature | None) -> DirectLabelColormap:
        nodes = self.project.cand_graph.nodes
        if feature is None:
            colormap = finn.utils.colormaps.label_colormap(
                49,
                seed=self.colormap_seed,
                background_value=0,
            )
            feature_values = nodes
        else:
            feature_values = self.project.cand_graph.get_feature_values(nodes, feature)
            if feature == self.project.cand_graph.features.track_id:
                colormap = self.project_viewer.colormap
            else:
                colormap = finn.utils.colormaps.ensure_colormap("viridis")
        colors = []
        for node, val in zip(nodes, feature_values, strict=False):
            color = colormap.map(val) if val is not None else self.default_color
            alpha = (
                self.alphamap.map(node) if node is not None else self.not_visible_opacity
            )
            color[-1] = alpha
            colors.append(color)

        return DirectLabelColormap(
            color_dict={
                **dict(zip(nodes, colors, strict=True)),
                None: [0, 0, 0, 0],
            }
        )

    def refresh_colormap(self):
        self.colormap = self._get_colormap()

    def _get_colormap(self):
        return self._get_feature_colormap(self.color_by)

    def _check_mode(self):
        """Check if the mode is valid and call the ensure_valid_label function"""
        # here disconnecting the event listener is still necessary because
        # self.mode = paint triggers the event internally and it is not blocked with
        # event.blocker()
        self.events.mode.disconnect(self._check_mode)
        if self.mode == "polygon":
            show_info("Please use the paint tool to update the label")
            self.mode = "paint"

        self._ensure_valid_label()
        self.events.mode.connect(self._check_mode)

    def redo(self):
        """Overwrite the redo functionality of the labels layer and invoke redo action on
        the tracks_viewer.tracks_controller first
        """

        self.project_viewer.redo()

    def undo(self):
        """Overwrite undo function and invoke undo action on the
        tracks_viewer.tracks_controller
        """

        self.project_viewer.undo()

    def _parse_paint_event(self, event_val):
        """_summary_

        Args:
            event_val (list[tuple]): A list of paint "atoms" generated by the labels
                layer. Each atom is a 3-tuple of arrays containing:
                - a numpy multi-index, pointing to the array elements that were
                changed (a tuple with len ndims)
                - the values corresponding to those elements before the change
                - the value after the change
        Returns:
            tuple(int, list[tuple]): The new value, and a list of node update actions
                defined by the time point and node update item
                Each "action" is a 2-tuple containing:
                - a numpy multi-index, pointing to the array elements that were
                changed (a tuple with len ndims)
                - the value before the change
        """

        new_value = event_val[-1][-1]
        ndim = len(event_val[-1][0])
        concatenated_indices = tuple(
            np.concatenate([ev[0][dim] for ev in event_val]) for dim in range(ndim)
        )
        concatenated_values = np.concatenate([ev[1] for ev in event_val])
        old_values = np.unique(concatenated_values)
        actions = []
        for old_value in old_values:
            mask = concatenated_values == old_value
            indices = tuple(concatenated_indices[dim][mask] for dim in range(ndim))
            time_points = np.unique(indices[0])
            for time in time_points:
                time_mask = indices[0] == time
                actions.append(
                    (tuple(indices[dim][time_mask] for dim in range(ndim)), old_value)
                )
        return new_value, actions

    def _revert_paint(self, event):
        """Revert a paint event after it fails validation (no motile tracker Actions have
        been created). This keeps the view synced with the backend data.
        """
        super().undo()

    def _on_paint(self, event):
        """Listen to the paint event and check which track_ids have changed"""

        with self.events.selected_label.blocker():
            # current_timepoint = self.viewer.dims.current_step[
            #     0
            # ]  # also pass on the current time point to know which node to select later
            new_value, updated_pixels = self._parse_paint_event(event.value)

            action = UserUpdateSegmentation(self.project, new_value, updated_pixels)
            self.project_viewer.history.add_new_action(action)

    def _refresh(self):
        """Refresh the data in the labels layer"""
        # TODO: DON'T COMPUTE
        self.data = self.project_viewer.project.segmentation.data.compute()
        self.refresh_colormap()
        self.refresh()

    def update_label_colormap(self, visible: list[int] | str) -> None:
        """Updates the opacity of the label colormap to highlight the selected label
        and optionally hide cells not belonging to the current lineage

        Visible is a list of visible node id
        """
        nodes = self.project.cand_graph.nodes
        with self.events.selected_label.blocker():
            highlighted = self.project_viewer.selected_nodes

            # update the opacity of the cyclic label colormap values according to
            # whether nodes are visible/invisible/highlighted
            if visible == "all":
                alpha_dict = {node: self.default_opacity for node in nodes}
            else:
                # make them all invisible
                alpha_dict = {node: self.not_visible_opacity for node in nodes}
                # set the visible ones to default opacity
                for node in visible:
                    # find the index in the colormap
                    alpha_dict[node] = self.default_opacity

            for node in highlighted:
                alpha_dict[node] = self.highighted_opacity

            self.alphamap.update(alpha_dict)
            self.refresh()

    def new_colormap(self):
        """Override existing function to generate new colormap on tracks_viewer and
        emit refresh signal to update colors in all layers/widgets"""
        self.colormap_seed = random.uniform(0, 1)
        self.project_viewer.colormap = finn.utils.colormaps.label_colormap(
            49,
            seed=self.colormap_seed,
            background_value=0,
        )
        self.project_viewer._refresh()

    def update_selected_label(self):
        """Update the selected label in the labels layer"""

        self.events.selected_label.disconnect(self._ensure_valid_label)
        if len(self.project_viewer.selected_nodes) > 0:
            self.selected_label = int(self.project_viewer.selected_nodes[0])
        self.events.selected_label.connect(self._ensure_valid_label)

    def _ensure_valid_label(self, event: Event | None = None):
        """Make sure a valid label is selected, because it is not allowed to paint with a
        label that already exists at a different timepoint.
        Scenarios:
        1. If a node with the selected label value (node id) exists at a different time
            point, check if there is any node with the same track_id at the current time
            point
            1.a if there is a node with the same track id, select that one, so that it
                can be used to update an existing node
            1.b if there is no node with the same track id, create a new node id and
                paint with the track_id of the selected label.
              This can be used to add a new node with the same track id at a time point
              where it does not (yet) exist (anymore).
        2. if there is no existing node with this value in the graph, it is assume that
            you want to add a node with the current track id
        Retrieve the track_id from self.current_track_id and use it to find if there are
        any nodes of this track id at current time point
        3. If no node with this label exists yet, it is valid and can be used to start a
            new track id. Therefore, create a new node id and map a new color.
            Add it to the dictionary.
        4. If a node with the label exists at the current time point, it is valid and
            can be used to update the existing node in a paint event. No action is needed
        """

        if self.project_viewer.project is not None and self.mode in (
            "fill",
            "paint",
            "erase",
            "pick",
        ):
            self.events.selected_label.disconnect(self._ensure_valid_label)

            current_timepoint = self.viewer.dims.current_step[0]
            # if a node with the given label is already in the graph
            graph = self.project_viewer.project.cand_graph
            if self.project_viewer.project.cand_graph.has_node(self.selected_label):
                # Update the track id
                self.selected_track = graph.get_track_id(self.selected_label)
                existing_time = graph.get_time(self.selected_label)
                if existing_time == current_timepoint:
                    # we are changing the existing node. This is fine
                    pass
                else:
                    # if there is already a node in that track in this frame, edit that
                    # instead
                    # get all nodes in track
                    nodes_in_track = graph.get_elements_with_feature(
                        graph.features.track_id, self.selected_track
                    )
                    times = list(graph.get_times(nodes_in_track))
                    if current_timepoint in times:
                        current_time_idx = times.index(current_timepoint)
                        node_in_current_time = nodes_in_track[current_time_idx]
                        self.selected_label = int(node_in_current_time)
                    else:
                        # use a new label, but the same track id
                        _new_label(self, new_track_id=False)
                        self.colormap = DirectLabelColormap(
                            color_dict=self.colormap.color_dict
                        )

            # the current node does not exist in the graph.
            # Use the current selected_track as the track id (will be a new track if a
            # new label was found with "m")
            # Check that the track id is not already in this frame.
            else:
                # if there is already a node in that track in this frame, edit that
                # instead
                nodes_in_track = graph.get_elements_with_feature(
                    graph.features.track_id, self.selected_track
                )
                times = list(graph.get_times(nodes_in_track))
                if current_timepoint in times:
                    current_time_idx = times.index(current_timepoint)
                    node_in_current_time = nodes_in_track[current_time_idx]
                    self.selected_label = int(node_in_current_time)

            self.events.selected_label.connect(self._ensure_valid_label)

    @finn.layers.Labels.n_edit_dimensions.setter
    def n_edit_dimensions(self, n_edit_dimensions):
        # Overriding the setter to disable editing in time dimension
        if n_edit_dimensions > self.project_viewer.project.ndim - 1:
            n_edit_dimensions = self.project_viewer.project.ndim - 1
        self._n_edit_dimensions = n_edit_dimensions
        self.events.n_edit_dimensions()


# This is to override the default finn function to get a new label for the labels layer
action_manager.register_action(
    name="finn:new_label",
    command=new_label,
    keymapprovider=TrackLabels,
    description="",
)
