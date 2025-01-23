from vispy import gloo

from finn._vispy.visuals.graph import GraphVisual

import numpy as np

from finn._vispy.layers.base import VispyBaseLayer
from finn.utils.colormaps.standardize_color import transform_color
from finn.utils.events import disconnect_events
import time


class VispyGraphLayer(VispyBaseLayer):
    node: GraphVisual

    def __init__(self, layer) -> None:
        node = GraphVisual()
        super().__init__(layer, node)

        self._on_data_change()

        self.layer.events.highlight.connect(self._on_highlight_change)

    def _on_data_change(self) -> None:
        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switched for vispy's x / y ordering
        if self.layer.viewed_nodes.size == 0:
            # always pass one invisible point to avoid issues
            locations = np.zeros((1, self.layer._slice_input.ndisplay))
        else:
            locations = self.layer.data.node_attrs[self.layer.viewed_nodes].position
        size = self.layer.size
        border_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        face_color = self.layer._view_face_color
        border_width = np.zeros(1)
        symbol = ['o']

        set_data = self.node.node_markers.set_data

        border_kw = {
            'edge_width': border_width,
            'edge_width_rel': None,
        }
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


    def _get_edge_endpoints(self, edges) -> np.ndarray:
        print(edges)
        start_nodes = np.array(edges[:,0])
        end_nodes = np.array(edges[:, 1])

        start_node_locations = self.layer.data.node_attrs[start_nodes].position[:, ::-1]
        end_node_locations = self.layer.data.node_attrs[end_nodes].position[:,::-1]
        print(end_node_locations)
        return np.stack((start_node_locations, end_node_locations), axis=1)

    def _set_graph_edges_data(self) -> None:
        """Sets the LineVisual with the graph edges data"""
        subvisual = self.node.edge_markers
        edges = self.layer.viewed_edges

        if len(edges) == 0:
            subvisual.visible = False
            return

        subvisual.visible = True
        edges = self._get_edge_endpoints(edges)
        
        flat_edges = edges.reshape((-1, edges.shape[-1]))# (N x 2, D)

        # edge_color = self.layer._view_edge_color

        # clearing up buffer, there was a vispy error otherwise
        subvisual._line_visual._pos_vbo = gloo.VertexBuffer()
        subvisual.set_data(
            flat_edges,
            color='white',
            width=1,
        )

    def _highlight_nodes(self):
        if len(self.layer.highlighted_nodes) > 0:
            # Color the hovered or selected points
            locations = self.layer.data.node_attrs[self.layer.highlighted_nodes].position
            size = self.layer.size
            symbol = ['o']
            border_width = np.array([1])

            scale = self.layer.scale[-1]
            highlight_thickness = 5
            scaled_highlight = highlight_thickness * self.layer.scale_factor
            scaled_size = (size + border_width) * scale
            highlight_color = np.array([[1.0, 0.0, 1.0, 1.0]], dtype=np.float32)

            self.node.node_selection_markers.set_data(
                locations[:, ::-1],
                size=scaled_size,
                symbol=symbol,
                edge_width=scaled_highlight * 10,
                edge_color=highlight_color,
                face_color=transform_color('transparent'),
            )

            self.node.node_selection_markers.visible = True
        else:
            self.node.node_selection_markers.visible = False
        

    def _higlight_edges(self):
        print(self.layer.highlighted_edges)
        print(self.layer.highlighted_edges.size)
        if self.layer.highlighted_edges.size > 0:
            print(self.layer.highlighted_edges.ndim)
            # Color the hovered or selected points
            print(self.layer.highlighted_edges.shape)
            edges = self._get_edge_endpoints(self.layer.highlighted_edges)
            flat_edges = edges.reshape((-1, edges.shape[-1]))# (N x 2, D)

            highlight_color = np.array([[1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            print(f"Setting data in edge highlight ({edges})")
            self.node.edge_selection_markers.set_data(
                flat_edges,
                color=highlight_color,
                width=2,
            )
            self.node.edge_selection_markers.visible = True
        else:
            self.node.edge_selection_markers.visible = False

    def _on_highlight_change(self):
        self._highlight_nodes()
        self._higlight_edges()
        self.node.update()
        
    
    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
