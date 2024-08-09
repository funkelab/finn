
from typing import TypeVar
_T = TypeVar('_T')


def _toggle_selected(selection: set[_T], value: _T) -> set[_T]:
    """Add or remove value from the selection set.

    This function returns a copy of the existing selection.

    Parameters
    ----------
    selection : set
        Set of selected data points to be modified.
    value : int
        Index of point to add or remove from selected data set.

    Returns
    -------
    selection: set
        Updated selection.
    """
    selection = set(selection)
    if value in selection:
        selection.remove(value)
    else:
        selection.add(value)
    return selection

def select(layer, event):
    """Select points.

    Clicking on a point will select that point. If holding shift while clicking
    that point will be added to or removed from the existing selection
    depending on whether it is selected or not.

    Clicking and dragging a point that is already selected will drag all the
    currently selected points.

    Clicking and dragging on an empty part of the canvas (i.e. not on a point)
    will create a drag box that will select all points inside it when finished.
    Holding shift throughout the entirety of this process will add those points
    to any existing selection, otherwise these will become the only selected
    points.
    """
    # on press
    modify_selection = (
        'Shift' in event.modifiers or 'Control' in event.modifiers
    )

    # Get value under the cursor, for points, this is the index of the highlighted
    # if any, or None.
    value = layer.get_value(
        position=event.position,
    )
    # if modifying selection add / remove any from existing selection
    if modify_selection:
        if value is not None:
            layer.selected_data = _toggle_selected(layer.selected_data, value)
    else:
        if value is not None:
            # If the current index is not in the current list make it the only
            # index selected, otherwise don't change the selection so that
            # the current selection can be dragged together.
            if value not in layer.selected_data:
                layer.selected_data = {value}
        else:
            layer.selected_data = set()
    # reset the selection box data and highlights
    layer._set_highlight()
