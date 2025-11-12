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
    # for some reason, lines are much thicker than points even when the size
    # is the same, so we go with a much lower value here then for points
    LineSize = 0.3
    SelectedSize = 5
    HoverColor = (0.6, 0.6, 0.7)
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
        self.canvas.add_event_handler(self.on_pointer_move, "pointer_move")
        self.canvas.add_event_handler(self.on_pointer_down, "pointer_down")
        # self.setMinimumHeight(200)

        self.solution_changed = False
        self.selection_changed = False
        self.selection.list_updated.connect(self.on_selection_changed)

    def on_selection_changed(self):
        print("The selection changed!")
        self.selection_changed = True
        self.canvas.request_draw()

    def on_solution_changed(self, solution: SolutionTracks):
        self.solution = solution
        self.solution_changed = True

        num_nodes = self.solution.graph.number_of_nodes()
        num_edges = self.solution.graph.number_of_edges()

        node_sizes = np.ones((num_nodes,), dtype=np.float32) * TreePlot.PointSize
        node_positions = np.zeros((num_nodes, 3), dtype=np.float32)
        node_colors = np.ones((num_nodes, 4), dtype=np.float32)
        node_outline_colors = np.ones((num_nodes, 4), dtype=np.float32)
        node_ids = np.zeros((num_nodes,), dtype=np.uint32)

        edge_positions = np.zeros((num_edges * 3, 3), dtype=np.float32)
        edge_positions[2::3] = np.nan  # every third entry has to be NaN
        edge_colors = np.ones((num_edges * 3, 4), dtype=np.float32)
        edge_ids = np.zeros((num_edges, 2), dtype=np.uint32)

        self.nodes = gfx.Points(
            gfx.Geometry(
                positions=node_positions,
                colors=node_colors,
                edge_colors=node_outline_colors,
                sizes=node_sizes,
                node_ids=node_ids,
            ),
            gfx.PointsMarkerMaterial(
                marker="circle",
                color_mode="vertex",
                edge_color_mode="vertex",
                size_mode="vertex",
                size_space="world",
                depth_write=False,
            ),
            render_order=2,
        )
        self.nodes.material.pick_write = True

        self.hover_node = gfx.Points(
            gfx.Geometry(
                positions=np.zeros((1, 3), dtype=np.float32),
            ),
            gfx.PointsGaussianBlobMaterial(
                color=TreePlot.HoverColor,
                size=6.0 * TreePlot.PointSize,
                size_space="world",
                depth_write=False,
            ),
            render_order=0,
            visible=False,
        )

        self.edges = gfx.Line(
            gfx.Geometry(
                positions=edge_positions,
                colors=edge_colors,
                edge_ids=edge_ids,
            ),
            gfx.LineMaterial(
                thickness=TreePlot.LineSize,
                thickness_space="world",
                color_mode="vertex",
                depth_write=False,
                aa=True,
            ),
            render_order=1,
        )
        self.edges.material.pick_write = True

        self.scene = self._create_scene()
        self.scene.add(self.nodes)
        self.scene.add(self.edges)
        self.scene.add(self.hover_node)
        self.canvas.request_draw()

    def on_pointer_move(self, event):
        # TODO :)
        pass

    def on_pointer_move(self, event):
        position = event["x"], event["y"]
        info = self.renderer.get_pick_info(position)
        print(info)
        world_object = info["world_object"]
        if isinstance(world_object, gfx.Points):
            point_index = info["vertex_index"]
            self.hover_node.visible = True
            self.hover_node.geometry.positions.data[0] = (
                self.nodes.geometry.positions.data[point_index]
            )
            self.hover_node.geometry.positions.update_indices([0])
        else:
            self.hover_node.visible = False

        # TODO
        # if isinstance(world_object, gfx.Line):
        #     edge_index = info["vertex_index"] // 3
        #     (u, v) = self.lines.geometry.edge_ids.data[edge_index]

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
            self.nodes.geometry.positions.data[i, :2] = self._get_position(
                node_id, tracklet_id_to_index
            )
            color = self._get_color(node_id)
            self.nodes.geometry.colors.data[i] = color
            self.nodes.geometry.edge_colors.data[i] = color
            self.nodes.geometry.node_ids.data[i] = node_id

        for i, (u, v) in enumerate(self.solution.edges()):
            self.edges.geometry.positions.data[i * 3, :2] = np.array(
                self._get_position(u, tracklet_id_to_index)
            )
            self.edges.geometry.positions.data[i * 3 + 1, :2] = np.array(
                self._get_position(v, tracklet_id_to_index)
            )
            self.edges.geometry.colors.data[i * 3] = self._get_color(u)
            self.edges.geometry.colors.data[i * 3 + 1] = self._get_color(v)
            self.edges.geometry.edge_ids.data[i] = (u, v)

    def _apply_selection(self):
        changed_indices = []
        for node_id in self.selection:
            index = self.node_id_to_index[node_id]
            # increase size
            self.nodes.geometry.sizes.data[index] = TreePlot.SelectedSize
            # highlight edge
            self.nodes.geometry.edge_colors.data[index] = TreePlot.HighlightColor
            changed_indices.append(index)

        self.nodes.geometry.sizes.update_indices(changed_indices)
        self.nodes.geometry.edge_colors.update_indices(changed_indices)

        self._show_selection()

    def _show_selection(self):
        if not self.selection:
            return

        focus_node_id = self.selection[-1]
        index = self.node_id_to_index[focus_node_id]
        position = self.nodes.geometry.positions.data[index]
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
