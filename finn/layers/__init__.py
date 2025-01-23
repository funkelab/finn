"""Layers are the viewable objects that can be added to a viewer.

Custom layers must inherit from Layer and pass along the
`visual node <https://vispy.org/api/vispy.scene.visuals.html>`_
to the super constructor.
"""

import inspect as _inspect

from finn.layers.base import Layer
from finn.layers.image import Image
from finn.layers.labels import Labels
from finn.layers.points import Points
from finn.layers.shapes import Shapes
from finn.layers.surface import Surface
from finn.layers.tracks import Tracks
from finn.layers.vectors import Vectors
from finn.utils.misc import all_subclasses as _all_subcls

# isabstract check is to exclude _ImageBase class
NAMES: set[str] = {
    subclass.__name__.lower()
    for subclass in _all_subcls(Layer)
    if not _inspect.isabstract(subclass)
}

__all__ = [
    'NAMES',
    'Image',
    'Labels',
    'Layer',
    'Points',
    'Shapes',
    'Surface',
    'Tracks',
    'Vectors',
]
