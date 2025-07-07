from __future__ import annotations

from funtracks import Project
from funtracks.actions.action_history import ActionHistory
from funtracks.user_actions import UserDeleteEdge, UserDeleteNode, UserSelectEdge

import finn
from finn.track_data_views.node_type import NodeType
from finn.track_data_views.views.layers.tracks_layer_group import TracksLayerGroup
from finn.track_data_views.views.tree_view.tree_widget_utils import (
    extract_lineage_tree,
)
from finn.track_data_views.views_coordinator.node_selection_list import NodeSelectionList


class ProjectViewer:
    """Purposes of the ProjectViewer:
    - Emit signals that all widgets should use to update selection or update
        the currently displayed Tracks object
    - Storing the currently displayed tracks
    - Store shared rendering information like colormaps (or symbol maps)
    """

    def __init__(
        self,
        project: Project,
        viewer: finn.Viewer | None = None,
    ):
        self.viewer = finn.Viewer() if viewer is None else viewer
        self.project = project
        self.history = ActionHistory()

        self.colormap = finn.utils.colormaps.label_colormap(
            49,
            seed=0.5,
            background_value=0,
        )
        self.symbolmap: dict[NodeType, str] = {
            NodeType.END: "x",
            NodeType.CONTINUE: "disc",
            NodeType.SPLIT: "triangle_up",
            NodeType.NOT_SELECTED: "hbar",
        }
        self.mode = "all"
        self.visible: list | str = []
        self.selected_nodes = NodeSelectionList()
        self.tracking_layers = TracksLayerGroup(self)
        self.selected_nodes.list_updated.connect(self.update_selection)

        self.set_keybinds()

    def set_keybinds(self):
        # TODO: separate and document keybinds (and maybe allow user to choose)
        self.viewer.bind_key("q")(self.toggle_display_mode)
        self.viewer.bind_key("a")(self.create_edge)
        self.viewer.bind_key("d")(self.delete_node)
        self.viewer.bind_key("Delete")(self.delete_node)
        self.viewer.bind_key("b")(self.delete_edge)
        # self.viewer.bind_key("s")(self.set_split_node)
        # self.viewer.bind_key("e")(self.set_endpoint_node)
        # self.viewer.bind_key("c")(self.set_linear_node)
        self.viewer.bind_key("z")(self.undo)
        self.viewer.bind_key("r")(self.redo)

    def _refresh(self, node: str | None = None, refresh_view: bool = False) -> None:
        """Call refresh function on finn layers and the submit signal that tracks are
        updated. Restore the selected_nodes, if possible
        """

        if len(self.selected_nodes) > 0 and any(
            not self.project.cand_graph.has_node(node) for node in self.selected_nodes
        ):
            self.selected_nodes.reset()

        self.tracking_layers._refresh()

        self.tracks_updated.emit(refresh_view)

        # if a new node was added, we would like to select this one now (call this after
        # emitting the signal, because if the node is a new node, we have to update the
        # table in the tree widget first, or it won't be present)
        if node is not None:
            self.selected_nodes.add(node)

        # restore selection and/or highlighting in all finn Views (finn Views do not
        # know about their selection ('all' vs 'lineage'), but TracksViewer does)
        self.update_selection()

    def toggle_display_mode(self, event=None) -> None:
        """Toggle the display mode between available options"""

        if self.mode == "lineage":
            self.set_display_mode("all")
        else:
            self.set_display_mode("lineage")

    def set_display_mode(self, mode: str) -> None:
        """Update the display mode and call to update colormaps for points, labels, and
        tracks
        """

        # toggle between 'all' and 'lineage'
        if mode == "lineage":
            self.mode = "lineage"
            self.viewer.text_overlay.text = "Toggle Display [Q]\n Lineage"
        else:
            self.mode = "all"
            self.viewer.text_overlay.text = "Toggle Display [Q]\n All"

        self.viewer.text_overlay.visible = True
        visible_tracks = self.filter_visible_nodes()
        self.tracking_layers.update_visible(visible_tracks, self.visible)

    def filter_visible_nodes(self) -> list[int]:
        """Construct a list of node ids that should be displayed"""
        solution = self.project.solution

        if self.mode == "lineage":
            if self.project is None or len(solution) == 0:
                return []
            # if no nodes are selected, check which nodes were previously visible and
            # filter those
            if len(self.selected_nodes) == 0 and self.visible is not None:
                prev_visible = [node for node in self.visible if solution.has_node(node)]
                self.visible = []
                for node_id in prev_visible:
                    self.visible += extract_lineage_tree(solution, node_id)
                    if set(prev_visible).issubset(self.visible):
                        break
            else:
                self.visible = []
                for node in self.selected_nodes:
                    self.visible += extract_lineage_tree(solution, node)

            return list({self.project.cand_graph.get_track_ids(self.visible)})
        self.visible = "all"
        return "all"

    def update_selection(self) -> None:
        """Sets the view and triggers visualization updates in other components"""
        self.set_finn_view()
        visible_tracks = self.filter_visible_nodes()
        self.tracking_layers.update_visible(visible_tracks, self.visible)

    def set_finn_view(self) -> None:
        """Adjust the current_step of the viewer to jump to the last item of the
        selected_nodes list
        """
        if len(self.selected_nodes) > 0:
            node = self.selected_nodes[-1]
            self.tracking_layers.center_view(node)

    def delete_node(self, event=None):
        """Calls the user action to delete currently selected nodes"""
        for node in self.selected_nodes._list:
            action = UserDeleteNode(self.project, node)
            self.history.add_new_action(action)

    def delete_edge(self, event=None):
        """Calls the user action to delete an edge between the two currently
        selected nodes
        """

        if len(self.selected_nodes) == 2:
            node1 = self.selected_nodes[0]
            node2 = self.selected_nodes[1]

            time1 = self.project.get_time(node1)
            time2 = self.project.get_time(node2)

            if time1 > time2:
                node1, node2 = node2, node1
            action = UserDeleteEdge(self.project, (node1, node2))
            self.history.add_new_action(action)

    def create_edge(self, event=None):
        """Calls the tracks controller to add an edge between the two currently selected
        nodes
        """

        if len(self.selected_nodes) == 2:
            node1 = self.selected_nodes[0]
            node2 = self.selected_nodes[1]

            time1 = self.project.get_time(node1)
            time2 = self.project.get_time(node2)

            if time1 > time2:
                node1, node2 = node2, node1
            self.history.add_new_action(UserSelectEdge(self.project, (node1, node2), {}))

    def undo(self, event=None):
        self.history.undo()

    def redo(self, event=None):
        self.history.redo()
