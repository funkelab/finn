import logging

from qtpy import API_NAME
from vispy import app

# set vispy application to the appropriate qt backend
app.use_app(API_NAME)
del app

# set vispy logger to show warning and errors only
vispy_logger = logging.getLogger("vispy")
vispy_logger.setLevel(logging.WARNING)


from finn._vispy.camera import VispyCamera
from finn._vispy.canvas import VispyCanvas
from finn._vispy.overlays.axes import VispyAxesOverlay
from finn._vispy.overlays.interaction_box import (
    VispySelectionBoxOverlay,
    VispyTransformBoxOverlay,
)
from finn._vispy.overlays.labels_polygon import VispyLabelsPolygonOverlay
from finn._vispy.overlays.scale_bar import VispyScaleBarOverlay
from finn._vispy.overlays.text import VispyTextOverlay
from finn._vispy.utils.quaternion import quaternion2euler_degrees
from finn._vispy.utils.visual import create_vispy_layer, create_vispy_overlay

__all__ = [
    "VispyAxesOverlay",
    "VispyCamera",
    "VispyCanvas",
    "VispyLabelsPolygonOverlay",
    "VispyScaleBarOverlay",
    "VispySelectionBoxOverlay",
    "VispyTextOverlay",
    "VispyTransformBoxOverlay",
    "create_vispy_layer",
    "create_vispy_overlay",
    "quaternion2euler_degrees",
]
