from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np
import spatial_graph as sg
from numpy.typing import ArrayLike
import random

from napari.layers.base._slice import _next_request_id
from napari.layers.points._points_constants import PointsProjectionMode
from napari.layers.points._slice import _PointSliceResponse
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.utils.transforms import Affine


@dataclass(frozen=True)
class _GraphSliceResponse():
    """Contains all the output data of slicing an graph layer.
    Attributes
    ----------
    indices : array like
        Indices of the sliced *nodes* data.
    edge_indices : array like
        Indices of the sliced nodes for each *edge*.
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """
    indices: np.ndarray = field(repr=False)
    edges_indices: ArrayLike = field(repr=False)
    slice_input: _SliceInput
    request_id: int


@dataclass(frozen=True)
class _GraphSliceRequest:
    """A callable that stores all the input data needed to slice a graph layer.
    This should be treated a deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.
    In general, the calling an instance of this may take a long time, so you may
    want to run it off the main thread.
    Attributes
    ----------
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : BaseGraph
        The layer's data field, which is the main input to slicing.
    world_to_data : Affine
        The slicing coordinates and margins in data space.
    """
    slice_input: _SliceInput
    data: sg.SpatialGraph = field(repr=False)
    world_to_data : Affine
    projection_mode: PointsProjectionMode
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _GraphSliceResponse:
        # Return early if no data
        if self.data.num_nodes == 0:
            return _GraphSliceResponse(
                indices=[],
                edges_indices=[],
                scale=np.empty(0),
                slice_input=self.slice_input,
                request_id=self.id,
            )

        not_disp = list(self.slice_input.not_displayed)
        if not not_disp:
            node_indices = self.data.nodes
            edges = self.data.edges
            return _GraphSliceResponse(
                indices=node_indices,
                edges_indices=edges,
                slice_input=self.slice_input,
                request_id=self.id,
            )
        data_slice = self.slice_input.data_slice(self.world_to_data)
        print("Data slice", data_slice)
        point, m_left, m_right = data_slice.as_array()

        if self.projection_mode == 'none':
            low = point.copy()
            high = point.copy()
        else:
            low = point - m_left
            high = point + m_right

        # assume slice thickness of 1 in data pixels
        # (same as before thick slices were implemented)
        too_thin_slice = np.isclose(high, low)
        low[too_thin_slice] -= 0.5
        high[too_thin_slice] += 0.5
        node_indices, edges_indices = self._get_slice_data(
            low, high
        )

        return _GraphSliceResponse(
            indices=node_indices,
            edges_indices=edges_indices,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _get_slice_data(
        self,
        low: np.ndarray,
        high: np.ndarray,
    ) -> Tuple[np.ndarray, ArrayLike]:
        """
        Slices data according to displayed indices
        while ignoring not initialized nodes from graph.
        Args:
            low: the lower bound of the slice in all dimensions
            high: the upper bound of the slice in all dimensions

        Returns: tuple(node_ids, edge_ids,)
        """
        # TODO: unbounded roi in certain dimensions
        low = np.nan_to_num(low, nan=-100)
        high = np.nan_to_num(high, nan=100)
        roi = np.array([low, high])
        print("Roi", roi)
        nodes, edges = self.data.query_in_roi(roi, edge_inclusion="incident")
        edges = np.array([[random.choice(nodes), random.choice(nodes)] for _ in range(100)]).copy()
        print("Nodes", nodes)
        print("Edges! ", edges)
        return nodes, edges