from finn._pydantic_compat import Field
from finn.utils.color import ColorValue
from finn.utils.events import EventedModel

_DEFAULT_COLOR = ColorValue("black")


class SurfaceWireframe(EventedModel):
    """
    Wireframe representation of the edges of a surface mesh.

    Attributes
    ----------
    visible : bool
        Whether the wireframe is displayed.
    color : ColorValue
        The color of the wireframe lines.
        See ``ColorValue.validate`` for supported values.
    width : float
        The width of the wireframe lines.
    """

    visible: bool = False
    color: ColorValue = Field(default_factory=lambda: _DEFAULT_COLOR)
    width: float = 1
