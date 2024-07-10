from vispy import gloo

from napari._vispy.visuals.graph import GraphVisual

import numpy as np

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.utils.gl import BLENDING_MODES
from napari._vispy.utils.text import update_text
from napari._vispy.visuals.points import PointsVisual
from napari.settings import get_settings
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.events import disconnect_events


class VispyGraphLayer(VispyBaseLayer):
    node: GraphVisual

    def __init__(self, layer) -> None:
        node = GraphVisual()
        super().__init__(layer, node)

        self._on_data_change()

    def _on_data_change(self) -> None:
        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switched for vispy's x / y ordering
        if len(self.layer.viewed_nodes) == 0:
            # always pass one invisible point to avoid issues
            locations = np.zeros((1, self.layer._slice_input.ndisplay))
        else:
            locations = self.layer.data.node_attrs[self.layer.viewed_nodes].position
        size = self.layer.size
        border_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        face_color = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        border_width = np.zeros(1)
        symbol = ['o']

        set_data = self.node.points_markers.set_data

        border_kw = {
            'edge_width': border_width,
            'edge_width_rel': None,
        }
        print("LOCATIONS ", locations)
        set_data(
            locations[:, ::-1],
            size=size,
            symbol=symbol,
            # edge_color is the name of the vispy marker visual kwarg
            edge_color=border_color,
            face_color=face_color,
            **border_kw,
        )


        self._set_graph_edges_data()
        self.reset()

    def _set_graph_edges_data(self) -> None:
        """Sets the LineVisual with the graph edges data"""
        subvisual = self.node.edges
        edges = self.layer.viewed_edges
        start_nodes = np.array(edges[:,0])
        print(start_nodes.shape)
        end_nodes = np.array(edges[:, 1])
        print(end_nodes.shape)

        if len(edges) == 0:
            subvisual.visible = False
            return

        subvisual.visible = True
        start_node_locations = self.layer.data.node_attrs[start_nodes].position[:, ::-1]
        print("Start node locations\n", start_node_locations)
        end_node_locations = self.layer.data.node_attrs[end_nodes].position[:,::-1]
        print("End node locations\n", end_node_locations)
        flat_edges = np.stack((start_node_locations, end_node_locations))  # (N x 2, D)
        print("Flat edges shape:", flat_edges.shape)
        # flat_edges = flat_edges[:, ::-1]

        # edge_color = self.layer._view_edge_color

        # clearing up buffer, there was a vispy error otherwise
        subvisual._line_visual._pos_vbo = gloo.VertexBuffer()
        subvisual.set_data(
            flat_edges,
            color='white',
            width=1,
        )
    

    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()