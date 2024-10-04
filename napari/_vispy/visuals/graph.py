from __future__ import annotations

from vispy.visuals import LineVisual
from vispy.scene.visuals import Compound, Line, Text

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin
from napari._vispy.visuals.markers import Markers


class GraphVisual(ClippingPlanesMixin, Compound):
    def __init__(self) -> None:
        super().__init__(
            [
                Markers(),
                Markers(),
                Line(),
                Text(),
            ]
        )
        self.scaling = True
        # connect='segments' indicates you need start point and end point for
        # each segment, rather than just a list of points. This mode means you
        # don't need segments to be sorted to display a line.
        self.add_subvisual(LineVisual(connect='segments'))
        self.add_subvisual(LineVisual(connect='segments'))

    @property
    def node_markers(self) -> Markers:
        """Points markers visual"""
        return self._subvisuals[0]

    @property
    def node_selection_markers(self) -> Markers:
        """Highlight markers visual"""
        return self._subvisuals[1]

    @property
    def highlight_lines(self) -> Line:
        """Highlight lines visual"""
        return self._subvisuals[2]

    @property
    def text(self) -> Text:
        """Text labels visual"""
        return self._subvisuals[3]

    @property
    def edge_markers(self) -> Line:
        """Edges visual"""
        return self._subvisuals[4]

    @property
    def edge_selection_markers(self) -> Line:
        """Edges visual"""
        return self._subvisuals[5]

    @property
    def scaling(self) -> bool:
        """
        Scaling property for both the markers visuals. If set to true,
        the points rescale based on zoom (i.e: constant world-space size)
        """
        return self.node_markers.scaling == 'visual'

    @scaling.setter
    def scaling(self, value: bool) -> None:
        scaling_txt = 'visual' if value else 'fixed'
        self.node_markers.scaling = scaling_txt
        self.node_selection_markers.scaling = scaling_txt

    @property
    def antialias(self) -> float:
        return self.node_markers.antialias

    @antialias.setter
    def antialias(self, value: float) -> None:
        self.node_markers.antialias = value
        self.node_selection_markers.antialias = value

    @property
    def spherical(self) -> bool:
        return self.node_markers.spherical

    @spherical.setter
    def spherical(self, value: bool) -> None:
        self.node_markers.spherical = value

    @property
    def canvas_size_limits(self) -> tuple[int, int]:
        return self.node_markers.canvas_size_limits

    @canvas_size_limits.setter
    def canvas_size_limits(self, value: tuple[int, int]) -> None:
        self.node_markers.canvas_size_limits = value
        self.node_selection_markers.canvas_size_limits = value
