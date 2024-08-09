import time
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
)

import numpy as np
import numpy.typing as npt

from napari.layers.base import Layer, no_op
from napari.layers.graph._graph_constants import Mode
from napari.layers.points._points_constants import PointsProjectionMode
from napari.layers.graph._graph_actions import select
from napari.layers.graph._slice import _GraphSliceRequest
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.events import Event

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
        data,
        name,

    ) -> None:
        self._data = data

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
        self.node_properties = list(self.data.node_attr_dtypes.keys())
        self._current_face_color_property = self.node_properties[1]
        self.colormap = AVAILABLE_COLORMAPS["viridis"]
        self.viewed_nodes = np.array([], dtype=self.data.node_dtype)
        self.viewed_edges = np.zeros(shape=(0, 2), dtype=self.data.node_dtype)
        self.selected_data = set()  # currently just nodes
        self.highlighted_nodes  = np.array([], dtype=self.data.node_dtype)
        self._projection_mode = PointsProjectionMode.ALL

        # Trigger generation of view slice and thumbnail
        self.refresh()

    @property
    def data(self):
        # user writes own docstring
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def num_nodes(self):
        return self._data.num_nodes

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



    def _get_node(self, position) -> Optional[int]:
        """Index of the point at a given 2D position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : int or None
            Index of point that is at the current coordinate if any.
        """
        nodes = self.data._rtree.nearest(np.array(position), k=1)
        if len(nodes) > 0:
            node = nodes[0]
            # print("Nearest node to point", position, "has position", self.data.node_attrs[node].position)
            return node
        else:
            return None

    def _get_value(self, position) -> Optional[int]:
        """Index of the point at a given 2D position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : int or None
            Index of point that is at the current coordinate if any.
        """
        return self._get_node(position)

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
        self.highlighted_nodes = np.array(list(self.selected_data.intersection(set(self.viewed_nodes))))

        self.events.highlight()
