from typing import Callable

from finn.layers.base._base_constants import Mode
from finn.layers.tracks.tracks import Tracks
from finn.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from finn.utils.translations import trans


def register_tracks_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(Tracks, description, repeatable)


def register_tracks_mode_action(
    description: str,
) -> Callable[[Callable], Callable]:
    return register_layer_attr_action(Tracks, description, 'mode')


@register_tracks_mode_action(trans._('Transform'))
def activate_tracks_transform_mode(layer: Tracks) -> None:
    layer.mode = str(Mode.TRANSFORM)


@register_tracks_mode_action(trans._('Pan/zoom'))
def activate_tracks_pan_zoom_mode(layer: Tracks) -> None:
    layer.mode = str(Mode.PAN_ZOOM)


tracks_fun_to_mode = [
    (activate_tracks_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_tracks_transform_mode, Mode.TRANSFORM),
]
