from typing import Any, Dict, Optional, Tuple, Union, cast
import spatial_graph as sg
import numpy as npt
from numpy.typing import ArrayLike
from psygnal.containers import Selection

from napari.layers.base._base_constants import ActionType
from napari.layers.graph._slice import _GraphSliceRequest, _GraphSliceResponse
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.utils.events import Event
from napari.utils.translations import trans

import numbers
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from copy import copy, deepcopy
from itertools import cycle
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import ArrayLike
from psygnal.containers import Selection
from scipy.stats import gmean

from napari.layers.base import Layer, no_op
from napari.layers.base._base_constants import ActionType
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)
from napari.layers.points._points_constants import (
    Mode,
    PointsProjectionMode,
    Shading,
)
from napari.layers.points._points_mouse_bindings import add, highlight, select
from napari.layers.points._points_utils import (
    _create_box_from_corners_3d,
    coerce_symbols,
    create_box,
    fix_data_points,
    points_to_squares,
)
from napari.layers.points._slice import _PointSliceRequest, _PointSliceResponse
from napari.layers.utils._color_manager_constants import ColorMode
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils.color_manager import ColorManager
from napari.layers.utils.color_transformations import ColorType
from napari.layers.utils.interactivity_utils import (
    displayed_plane_from_nd_line_segment,
)
from napari.layers.utils.layer_utils import (
    _features_to_properties,
    _FeatureTable,
    _unique_element,
)
from napari.layers.utils.text_manager import TextManager
from napari.utils.colormaps import Colormap, ValidColormapArg
from napari.utils.colormaps.standardize_color import hex_to_name, rgb_to_hex
from napari.utils.events import Event
from napari.utils.events.custom_types import Array
from napari.utils.events.migrations import deprecation_warning_event
from napari.utils.geometry import project_points_onto_plane, rotate_points
from napari.utils.migrations import add_deprecated_property, rename_argument
from napari.utils.status_messages import generate_layer_coords_status
from napari.utils.transforms import Affine
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components.dims import Dims

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

        # self.events.add(
        #     size=Event,
        #     ...
        # )

        self.size = 5

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
        print("Extrema: ", extrema)
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
        """Sets the view given the indices to slice with."""
        print("Slice input", self._slice_input)
        request = _GraphSliceRequest(
            slice_input=self._slice_input,
            data=self.data,
            world_to_data=self._data_to_world.inverse,
            projection_mode=self.projection_mode,
        )
        print(request)
        response = request()
        self.viewed_nodes = response.indices
        self.viewed_edges = response.edges_indices

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
        # TODO: Query closest point to given point
        low = np.array(position)
        high = np.array(position)
        for dim in self._slice_input.displayed:
            low[dim] -= self.size
            high[dim] += self.size
        nodes = self.data.query_in_roi(np.array([low, high]))
        return nodes[0]

    def _update_thumbnail(self) -> None:
        """Update thumbnail with current points and colors."""
        # TODO: this is an empty thumbnail
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped