from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from funtracks.actions import SetFeatureValues
from funtracks.actions._base import ActionGroup
from funtracks.user_actions import UserAddNode

import finn
from finn.track_data_views.node_type import NodeType
from finn.utils.notifications import show_info

if TYPE_CHECKING:
    from finn.track_data_views.views_coordinator.project_viewer import ProjectViewer


class TrackPoints(finn.layers.Points):
    """Extended points layer that holds the track information and emits and
    responds to dynamics visualization signals
    """

    @property
    def _type_string(self) -> str:
        return (
            "points"  # to make sure that the layer is treated as points layer for saving
        )

    def __init__(
        self,
        name: str,
        project_viewer: ProjectViewer,
        show_cands=True,
    ):
        self.project_viewer = project_viewer
        self.show_cands = show_cands
        self.graph = (
            self.project_viewer.project.solution
            if not show_cands
            else self.project_viewer.project.cand_graph
        )
        self.nodes = list(self.graph.nodes)
        node_index_dict, track_ids, positions, symbols, colors = self._get_points_data(
            self.nodes
        )
        self.node_index_dict = node_index_dict
        self.current_track_id = None
        self.continue_track = True  # TODO: update these from UI somehow

        self.default_size = 5

        super().__init__(
            data=positions,
            name=name,
            symbol=symbols,
            face_color=colors,
            size=self.default_size,
            properties={
                "node_id": self.nodes,
                "track_id": track_ids,
            },  # TODO: use features
            border_color=[1, 1, 1, 1],
            blending="translucent_no_depth",
        )

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

        # Connect to click events to select nodes
        @self.mouse_drag_callbacks.append
        def click(layer, event):
            if event.type == "mouse_press":
                # is the value passed from the click event?
                point_index = layer.get_value(
                    event.position,
                    view_direction=event.view_direction,
                    dims_displayed=event.dims_displayed,
                    world=True,
                )
                if point_index is not None:
                    node_id = self.nodes[point_index]
                    append = "Shift" in event.modifiers
                    self.project_viewer.selected_nodes.add(node_id, append)

        # listen to updates of the data
        self.events.data.connect(self._update_data)

        # connect to changing the point size in the UI
        self.events.current_size.connect(
            lambda: self.set_point_size(size=self.current_size)
        )

        # listen to updates in the selected data (from the point selection tool)
        # to update the nodes in self.tracks_viewer.selected_nodes
        self.selected_data.events.items_changed.connect(self._update_selection)

    def set_point_size(self, size: int) -> None:
        """Sets a new default point size"""

        self.default_size = size
        self._refresh()

    def _get_points_data(self, nodes: list[int]):
        graph = self.project_viewer.project.cand_graph
        node_index_dict = {node: idx for idx, node in enumerate(nodes)}
        track_ids = graph.get_track_ids(nodes)
        times = np.expand_dims(np.array(graph.get_times(nodes)), axis=1)
        positions = np.array(graph.get_positions(nodes))
        point_data = np.concat([times, positions], axis=1)

        statemap = {
            0: NodeType.END,
            1: NodeType.CONTINUE,
            2: NodeType.SPLIT,
        }
        symbolmap = self.project_viewer.symbolmap
        symbols = [symbolmap[statemap[graph.out_degree(node)]] for node in nodes]
        colors = [
            self.project_viewer.colormap.map(track_id)
            if track_id is not None
            else [1, 1, 1, 1]
            for track_id in track_ids
        ]
        return node_index_dict, track_ids, point_data, symbols, colors

    def _refresh(self):
        """Refresh the data in the points layer"""

        self.events.data.disconnect(
            self._update_data
        )  # do not listen to new events until updates are complete
        self.graph = (
            self.project_viewer.project.solution
            if self.show_cands
            else self.project_viewer.project.cand_graph
        )
        self.nodes = list(self.graph.nodes)
        node_index_dict, track_ids, positions, symbols, colors = self._get_points_data(
            self.nodes
        )
        self.node_index_dict = node_index_dict
        self.data = positions
        self.symbol = symbols
        self.face_color = colors
        self.properties = {"node_id": self.nodes, "track_id": track_ids}
        self.size = self.default_size
        self.border_color = [1, 1, 1, 1]

        self.events.data.connect(
            self._update_data
        )  # reconnect listening to update events

    def _create_node_attrs(self, new_point: np.array) -> tuple[np.array, dict]:
        """Create attributes for a new node at given time point"""

        t = int(new_point[0])
        if self.current_track_id is None and self.continue_track:
            track_id = self.project_viewer.project.get_next_track_id()
        else:
            track_id = self.current_track_id
        features = self.graph.features
        attributes = {
            features.position: new_point[1:],
            features.time: t,
            features.track_id: track_id,
        }
        return attributes

    def _update_data(self, event):
        """Calls the tracks controller with to update the data in the Tracks object and
        dispatch the update
        """

        if event.action == "added":
            # we only want to allow this update if there is no seg layer
            if self.project_viewer.tracking_layers.seg_layer is None:
                new_point = event.value[-1]
                node_id = self.project_viewer.project.get_next_node_id()
                attributes = self._create_node_attrs(new_point)
                action = UserAddNode(self.project_viewer.project, node_id, attributes)
                self.project_viewer.history.add_new_action(action)
            else:
                show_info(
                    "Mixed point and segmentation nodes not allowed: add points by "
                    "drawing on segmentation layer"
                )
                self._refresh()

        if event.action == "removed":
            self.project_viewer.delete_node(event)

        if event.action == "changed":
            # we only want to allow this update if there is no seg layer
            features = self.project_viewer.project.cand_graph.features
            if self.project_viewer.tracking_layers.seg_layer is None:
                actions = []
                for ind in self.selected_data:
                    point = self.data[ind]
                    pos = point[1:]
                    node_id = self.properties["node_id"][ind]
                    attributes = {features.position: pos}
                    actions.append(
                        SetFeatureValues(
                            self.project_viewer.project,
                            node_id,
                        )
                    )
                group = ActionGroup(self.project_viewer.project, actions)
                self.project_viewer.history.add_new_action(group)
            else:
                self._refresh()  # refresh to move points back where they belong

    def _update_selection(self):
        """Replaces the list of selected_nodes with the selection provided by the user"""

        selected_points = self.selected_data
        self.project_viewer.selected_nodes.reset()
        for point in selected_points:
            node_id = self.nodes[point]
            self.project_viewer.selected_nodes.add(node_id, True)

        if len(selected_points) > 0:
            self.current_track_id = self.project_viewer.project.cand_graph.get_track_id(
                selected_points[-1]
            )

    def update_point_outline(self, visible: list[int] | str) -> None:
        """Update the outline color of the selected points and visibility according to
        display mode

        Args:
            visible (list[int] | str): A list of track ids, or "all"
        """
        # filter out the non-selected tracks if in lineage mode
        if visible == "all":
            self.shown[:] = True
        else:
            indices = np.where(np.isin(self.properties["track_id"], visible))[0].tolist()
            self.shown[:] = False
            self.shown[indices] = True
        # set border color for selected item
        self.border_color = [1, 1, 1, 1]
        self.size = self.default_size
        for node in self.project_viewer.selected_nodes:
            index = self.node_index_dict[node]
            self.border_color[index] = (
                0,
                1,
                1,
                1,
            )
            self.size[index] = math.ceil(self.default_size + 0.3 * self.default_size)
        if len(self.project_viewer.selected_nodes._list):
            track_id = self.project_viewer.project.cand_graph.get_track_id(
                self.project_viewer.selected_nodes._list[-1]
            )
            # if track_id is not None:
            self.current_track_id = track_id
        self.refresh()
