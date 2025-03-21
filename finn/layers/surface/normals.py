from enum import Enum, auto

from finn._pydantic_compat import Field
from finn.utils.color import ColorValue
from finn.utils.events import EventedModel


class NormalMode(Enum):
    FACE = auto()
    VERTEX = auto()


_DEFAULT_COLOR = ColorValue("black")


class Normals(EventedModel):
    """
    Represents face or vertex normals of a surface mesh.

    Attributes
    ----------
    mode: str
        Which normals to display (face or vertex). Immutable Field.
    visible : bool
        Whether the normals are displayed.
    color : str, array-like
        The color of the normal lines.
        See ``ColorValue.validate`` for supported values.
    width : float
        The width of the normal lines.
    length : float
        The length of the face normal lines.
    """

    mode: NormalMode = Field(NormalMode.FACE, allow_mutation=False)
    visible: bool = False
    color: ColorValue = Field(default_factory=lambda: _DEFAULT_COLOR)
    width: float = 1
    length: float = 5


class SurfaceNormals(EventedModel):
    """
    Represents both face and vertex normals for a surface mesh.
    """

    face: Normals = Field(
        Normals(mode=NormalMode.FACE, color="orange"), allow_mutation=False
    )
    vertex: Normals = Field(
        Normals(mode=NormalMode.FACE, color="blue"), allow_mutation=False
    )
