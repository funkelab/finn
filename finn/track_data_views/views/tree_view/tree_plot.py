from typing import Any

import networkx as nx
import numpy as np
import pygfx as gfx
from funtracks.data_model import SolutionTracks
from qtpy.QtWidgets import QVBoxLayout, QWidget
from wgpu.gui.auto import WgpuCanvas

from finn.track_data_views.views_coordinator.node_selection_list import NodeSelectionList
from finn.utils.colormaps import Colormap


class TreePlot(QWidget):
    PointSize = 3
    SelectedSize = 5
    HighlightColor = (0.9, 0.05, 0.8, 1.0)

    def __init__(
        self,
        color_map: Colormap,
        selection: NodeSelectionList,
        parent=None,
    ):
        super().__init__(parent=parent)

        self.color_map = color_map
        self.selection = selection
        self.solution = None

        # pygfx stuff
        self.layout = QVBoxLayout(self)
        self.canvas = WgpuCanvas()
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = self._create_scene()
        self.camera = gfx.OrthographicCamera(110, 110, maintain_aspect=False)
        self.controller_xy = gfx.PanZoomController(register_events=self.renderer)
        self.controller_xy.add_camera(self.camera)
        self.controller_x = gfx.PanZoomController(
            register_events=self.renderer, enabled=False
        )
        self.controller_x.add_camera(self.camera, include_state={"x", "width"})
        self.controller_y = gfx.PanZoomController(
            register_events=self.renderer, enabled=False
        )
        self.controller_y.add_camera(self.camera, include_state={"y", "height"})
        self.layout.addWidget(self.canvas)
        self.canvas.request_draw(self.redraw)
        # self.setMinimumHeight(200)

        self.solution_changed = False
        self.selection_changed = False
        self.selection.list_updated.connect(self.on_selection_changed)

    def on_selection_changed(self):
        self.selection_changed = True
        self.canvas.request_draw()

    def on_solution_changed(self, solution: SolutionTracks):
        self.solution = solution
        self.solution_changed = True

        num_nodes = self.solution.graph.number_of_nodes()
        sizes = np.ones((num_nodes,), dtype=np.float32) * TreePlot.PointSize
        positions = np.zeros((num_nodes, 3), dtype=np.float32)
        colors = np.ones((num_nodes, 4), dtype=np.float32)
        edge_colors = np.ones((num_nodes, 4), dtype=np.float32)

        self.points = gfx.Points(
            gfx.Geometry(
                positions=positions,
                colors=colors,
                edge_colors=edge_colors,
                sizes=sizes,
            ),
            gfx.PointsMarkerMaterial(
                marker="circle",
                color_mode="vertex",
                edge_color_mode="vertex",
                size_mode="vertex",
                size_space="world",
            ),
        )

        self.scene = self._create_scene()
        self.scene.add(self.points)
        self.canvas.request_draw()

    def redraw(self):
        if self.solution_changed:
            self._compute_layout()
            self.solution_changed = False
        if self.selection_changed:
            self._apply_selection()
            self.selection_changed = False

        self.renderer.render(self.scene, self.camera)

    def _create_scene(self):
        # add other visual items here
        return gfx.Scene()

    def _compute_layout(self):
        tracklet_ids = self._get_sorted_track_ids(self.solution.graph)

        tracklet_id_to_index = {
            tracklet_id: index for index, tracklet_id in enumerate(tracklet_ids)
        }

        self.node_id_to_index = {
            node_id: i for i, node_id in enumerate(self.solution.nodes())
        }
        for i, node_id in enumerate(self.solution.nodes()):
            self.points.geometry.positions.data[i, :2] = self._get_position(
                node_id, tracklet_id_to_index
            )
            color = self._get_color(node_id)
            self.points.geometry.colors.data[i] = color
            self.points.geometry.edge_colors.data[i] = color

    def _apply_selection(self):
        changed_indices = []
        for node_id in self.selection:
            index = self.node_id_to_index[node_id]
            # increase size
            self.points.geometry.sizes.data[index] = TreePlot.SelectedSize
            # highlight edge
            self.points.geometry.edge_colors.data[index] = TreePlot.HighlightColor
            changed_indices.append(index)

        self.points.geometry.sizes.update_indices(changed_indices)
        self.points.geometry.edge_colors.update_indices(changed_indices)

        self._show_selection()

    def _show_selection(self):
        if not self.selection:
            return

        focus_node_id = self.selection[-1]
        index = self.node_id_to_index[focus_node_id]
        position = self.points.geometry.positions.data[index]
        state = self.camera.get_state()
        camera_view = (
            state["position"][:2] - [state["width"] / 2, state["height"] / 2],
            state["position"][:2] + [state["width"] / 2, state["height"] / 2],
        )
        if not (
            all(position[:2] > camera_view[0]) and all(position[:2] < camera_view[1])
        ):
            self.camera.world.position = position

    def _get_position(self, node_id, tracklet_id_to_index):
        tracklet_id = self.solution.get_track_id(node_id)
        index = tracklet_id_to_index[tracklet_id]
        t = self.solution.get_time(node_id)
        return index * 10, t * 10

    def _get_color(self, node_id):
        tracklet_id = self.solution.get_track_id(node_id)
        return self.color_map.map(tracklet_id)

    def _get_sorted_track_ids(
        self, graph: nx.DiGraph, tracklet_id_key: str = "track_id"
    ) -> list[Any]:
        """
        Extract the lineage tree plot order of the tracklet_ids on the graph,
        ensuring that each tracklet_id is placed in between its daughter
        tracklet_ids and adjacent to its parent track id.

        Args:
            graph (nx.DiGraph): graph with a tracklet_id attribute on it.
            tracklet_id_key (str): tracklet_id key on the graph.

        Returns:
            list[Any] of ordered tracklet_ids.
        """

        # Create tracklet_id to parent_tracklet_id mapping (0 if tracklet has no parent)
        tracklet_to_parent_tracklet = {}
        for node, data in graph.nodes(data=True):
            tracklet = data[tracklet_id_key]
            if tracklet in tracklet_to_parent_tracklet:
                continue
            predecessor = next(graph.predecessors(node), None)
            if predecessor is not None:
                parent_tracklet_id = graph.nodes[predecessor][tracklet_id_key]
            else:
                parent_tracklet_id = 0
            tracklet_to_parent_tracklet[tracklet] = parent_tracklet_id

        # Final sorted order of roots
        roots = sorted(
            [tid for tid, ptid in tracklet_to_parent_tracklet.items() if ptid == 0]
        )
        x_axis_order = list(roots)

        # Find the children of each of the starting points, and work down the tree.
        while len(roots) > 0:
            children_list = []
            for tracklet_id in roots:
                children = [
                    tid
                    for tid, ptid in tracklet_to_parent_tracklet.items()
                    if ptid == tracklet_id
                ]
                for i, child in enumerate(children):
                    [children_list.append(child)]
                    x_axis_order.insert(x_axis_order.index(tracklet_id) + i, child)
            roots = children_list

        return x_axis_order
