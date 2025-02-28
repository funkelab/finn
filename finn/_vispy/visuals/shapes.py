from vispy.scene.visuals import Compound, Line, Markers, Mesh, Text

from finn._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin


class ShapesVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for shapes visualization with
    clipping planes functionality

    Components:
        - Mesh for shape faces (vispy.MeshVisual)
        - Mesh for highlights (vispy.MeshVisual)
        - Lines for highlights (vispy.LineVisual)
        - Vertices for highlights (vispy.MarkersVisual)
        - Text labels (vispy.TextVisual)
    """

    def __init__(self) -> None:
        super().__init__([Mesh(), Mesh(), Line(), Markers(), Text()])

    @property
    def shape_faces(self) -> Mesh:
        """Mesh for shape faces"""
        return self._subvisuals[0]

    @property
    def shape_highlights(self) -> Mesh:
        """Mesh for shape highlights"""
        return self._subvisuals[1]

    @property
    def highlight_lines(self) -> Line:
        """Lines for shape highlights"""
        return self._subvisuals[2]

    @property
    def highlight_vertices(self) -> Markers:
        """Vertices for shape highlights"""
        return self._subvisuals[3]

    @property
    def text(self) -> Text:
        """Text labels"""
        return self._subvisuals[4]
