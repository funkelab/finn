
from typing import Callable

from finn.layers.graph._graph_constants import Mode
from finn.layers.graph.graph import Graph
from finn.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from finn.utils.translations import trans


def register_graph_mode_action(
    description: str,
) -> Callable[[Callable], Callable]:
    return register_layer_attr_action(Graph, description, 'mode')

def register_graph_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(Graph, description, repeatable)


@register_graph_mode_action(trans._('Select graph elements'))
def activate_graph_select_mode(layer: Graph) -> None:
    layer.mode = Mode.SELECT

@register_graph_mode_action(trans._('Pan/zoom'))
def activate_graph_pan_zoom_mode(layer: Graph) -> None:
    layer.mode = Mode.PAN_ZOOM


@register_graph_action(trans._('Delete selected graph elements'))
def delete_selected_graph(layer: Graph) -> None:
    """Delete all selected points."""
    layer.remove_selected()

