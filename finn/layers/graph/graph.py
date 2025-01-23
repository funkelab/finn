import time
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
    Union,
)

import numpy as np
import numpy.typing as npt

from finn.layers.base import Layer, no_op
from finn.layers.base._base_constants import ActionType
from finn.layers.graph._graph_constants import Mode
from finn.layers.points._points_constants import PointsProjectionMode
from finn.layers.graph._graph_actions import select
from finn.layers.graph._slice import _GraphSliceRequest
from finn.utils.colormaps import AVAILABLE_COLORMAPS
from finn.utils.events import Event

DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class Graph(Layer):
    """
    Graph layer used to display spatial graphs.

    Parameters
    ----------
    data : sg.SpatialGraph
        The spatial graph to display
    name: str
        The name of the layer
    """
    _modeclass = Mode
    _projectionclass = PointsProjectionMode

    _drag_modes: ClassVar[dict[Mode, Callable[['Graph', Event], Any]]] = {
        Mode.PAN_ZOOM: no_op,
        Mode.SELECT: select,
    }

    _move_modes: ClassVar[dict[Mode, Callable[['Graph', Event], Any]]] = {
        Mode.PAN_ZOOM: no_op,
        Mode.SELECT: no_op,
    }
    _cursor_modes: ClassVar[dict[Mode, str]] = {
        Mode.PAN_ZOOM: 'standard',
        Mode.SELECT: 'standard',
    }

    def __init__(
        self,
        graph,
        name,

    ) -> None:
        self._data = graph

        super().__init__(
            data=self.data,
            ndim=self.data.ndims,
            name=name,
        )

        self.events.add(
            highlight=Event,
        )

        self._mode = Mode.PAN_ZOOM

        self.size = 5
        # an upper bound on the node ids, used for getting new ids for new nodes
        self.node_id_upper_bound = np.max(self._data.nodes)

        self.node_properties = list(self.data.node_attr_dtypes.keys())
        # a dictionary from attributes to value for new nodes
        self._new_node_attrs = {
            key: np.ndarray([0], dtype=self.to_numpy_dtype(dtype)) for key, dtype in self.data.node_attr_dtypes.items()
        } # TODO: expose in UI
        self._current_face_color_property = self.node_properties[1]  # TODO: Expose in UI
        self.colormap = AVAILABLE_COLORMAPS["viridis"]  # TODO: Expose in UI

        # the nodes and edges currently in the view
        self.viewed_nodes = np.array([], dtype=self.data.node_dtype)
        self.viewed_edges = np.zeros(shape=(0, 2), dtype=self.data.node_dtype)

        # the currently selected nodes and edges
        self.selected_nodes = set()
        self.selected_edges = set()  


        self.highlighted_nodes  = np.array([], dtype=self.data.node_dtype)
        self.highlighted_edges = np.empty(shape=(2,0), dtype=self.data.node_dtype)
#
        self._projection_mode = PointsProjectionMode.ALL # TODO: Expose in UI

        # Trigger generation of view slice and thumbnail
        self.refresh()

    def to_numpy_dtype(self, dtype: str):
        if "[" in dtype:
            dtype = dtype.split("[")[0]
        return dtype

    @property
    def data(self):
        # user writes own docstring
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def num_nodes(self):
        return len(self._data)

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        # TODO: get total roi of spatial graph efficiently
        if self.num_nodes == 0:
            extrema = np.full((2, self.ndim), np.nan)
        else:
            maxs = np.max(self.data.node_attrs.position, axis=0)
            mins = np.min(self.data.node_attrs.position, axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema.astype(float)

    @property
    def _extent_data_augmented(self) -> npt.NDArray:
        # _extent_data is a property that returns a new/copied array, which
        # is safe to modify below
        extent = self._extent_data
        if self.size == 0:
            return extent

        max_point_size = self.size
        extent[0] -= max_point_size / 2
        extent[1] += max_point_size / 2
        return extent

    def _get_ndim(self) -> int:
        return self.data.ndims

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        return self._get_base_state()

    def _set_view_slice(self):
        # start_time = time.time()
        """Sets the view given the indices to slice with."""
        request = _GraphSliceRequest(
            slice_input=self._slice_input,
            data=self.data,
            world_to_data=self._data_to_world.inverse,
            projection_mode=self.projection_mode,
        )
        response = request()
        self.viewed_nodes = response.indices
        self.viewed_edges = response.edges_indices
        end_time = time.time()
        # print(f"Setting view slice took {end_time - start_time} seconds")


    def _get_node_2d(self, position) -> Optional[tuple[int, float]]:
        """ID of the point closest to a given 2D position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : int or None
            Index of point that is at the current coordinate if any.
        """
        nodes, dists = self.data.query_nearest_nodes(np.array(position), k=1, return_distances=True)
        if len(nodes) > 0:
            node = nodes[0]
            dist = dists[0]
            # print("Nearest node to point", position, "has position", self.data.node_attrs[node].position)
            return node, dist
        else:
            return None
        
    def _get_edge_2d(self, position) -> Optional[tuple[np.ndarray, int]]:
        """ID of the endpoints of the edge closest to a given 2D position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : tuple[int, int] or None
            Index of point that is at the current coordinate if any.
        """
        edges, dists = self.data.query_nearest_edges(np.array(position), k=1, return_distances=True)
        if len(edges) > 0:
            edge = edges[0]
            dist = dists[0]
            # print("Nearest node to point", position, "has position", self.data.node_attrs[node].position)
            return edge, dist
        else:
            return None

    def _get_value(self, position) -> Optional[Union[int, np.ndarray]]:
        """ID of the point or edge closest to a given 2D position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : int or np.ndarray or None
            ID of point or edge closeest to the current coordinate if any.
        """
        node_info = self._get_node_2d(position)
        edge_info = self._get_edge_2d(position)
        if node_info is None and edge_info is None:
            return None
        elif node_info is None:
            return edge_info[0]
        elif edge_info is None:
            return node_info[0]
        else:
            node_id, node_dist = node_info
            edge_id, edge_dist = edge_info
            if node_dist <= edge_dist:
                return node_id
            else:
                return edge_id

    
    def _get_value_3d(
        self,
        start_point: Optional[np.ndarray],
        end_point: Optional[np.ndarray],
        dims_displayed: list[int],
    ) ->  Optional[int]:
        """Get the layer data value along a ray

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value
            The data value along the supplied ray.
        """
        pass
        

    def _update_thumbnail(self) -> None:
        """Update thumbnail with current points and colors."""
        # TODO: this is an empty thumbnail
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    @property
    def _view_face_color(self) -> np.ndarray:
        """Get the face colors of the points in view

        Returns
        -------
        view_face_color : (N x 4) np.ndarray
            RGBA color array for the face colors of the N points in view.
            If there are no points in view, returns array of length 0.
        """
        if self.viewed_nodes.size == 0:
            return np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        property = self._current_face_color_property
        values = getattr(self.data.node_attrs[self.viewed_nodes], property)
        values = values.astype(np.float32) / np.max(values)
        color = self.colormap.map(values)
        return color

    def _set_highlight(self, force: bool = False) -> None:
        """Render highlights of shapes including boundaries, vertices,
        interaction boxes, and the drag selection box when appropriate.
        Highlighting only occurs in Mode.SELECT.

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`
        """
        highlighted_nodes = list(self.selected_nodes.intersection(set(self.viewed_nodes)))
        viewed_edge_set = set(tuple(e) for e in self.viewed_edges)
        print(f"{viewed_edge_set=}")
        print(f"{self.selected_edges=}")
        highlighted_edges = list(self.selected_edges.intersection(viewed_edge_set))
        print(f"{highlighted_edges=}")
        if self._value is not None:
            if isinstance(self._value, int | np.uint):
                highlighted_nodes.append(self._value)
            else:
                highlighted_edges.append(self._value)
        
        print(f"after append {highlighted_edges=}")
        self.highlighted_nodes = highlighted_nodes
        self.higlighted_edges = highlighted_edges
        self.events.highlight()

    def get_new_node_ids(self, k=1):
        node_ids = [self.node_id_upper_bound + i for i in range(1, k + 1)]
        self.node_id_upper_bound += k
        return node_ids
    
    def get_new_node_attrs(self, location) -> dict:
        attrs = {}
        for key, dtype in self._data.node_attr_dtypes.items():
            if key == self._data.position_attr:
                value = location
            else:
                value = self._new_node_attrs[key]
            attrs[key] = np.array([value], dtype=dtype)
        return attrs

    def add_node(self, location: np.ndarray):
        """Adds a node at the given coordinates.

        Parameters
        ----------
        location : array
            Location of node to add to the layer data.
        """
        self.events.data(
            value=self.data,
            action=ActionType.ADDING,
        )
        nodes = self.get_new_node_ids(k=1)
        attrs = self.get_new_node_attrs(location)
        self._data.add_nodes(nodes, **attrs)
        self.events.data(
            value=self.data,
            action=ActionType.ADDED,
            data_indices=nodes,
        )
