from __future__ import annotations

from typing import TYPE_CHECKING

from finn._qt.containers.qt_axis_model import AxisList, QtAxisListModel
from finn.components.layerlist import LayerList
from finn.utils.events import SelectableEventedList
from finn.utils.translations import trans
from finn.utils.tree import Group

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]


def create_view(obj: SelectableEventedList | Group, parent: QWidget | None = None):
    """Create a `QtListView`, or `QtNodeTreeView` for `obj`.

    Parameters
    ----------
    obj : SelectableEventedList or Group
        The python object for which to creat a QtView.
    parent : QWidget, optional
        Optional parent widget, by default None

    Returns
    -------
    Union[QtListView, QtNodeTreeView]
        A view instance appropriate for `obj`.
    """
    from finn._qt.containers import QtLayerList, QtListView, QtNodeTreeView

    if isinstance(obj, LayerList):
        return QtLayerList(obj, parent=parent)
    if isinstance(obj, Group):
        return QtNodeTreeView(obj, parent=parent)
    if isinstance(obj, SelectableEventedList):
        return QtListView(obj, parent=parent)
    raise TypeError(
        trans._(
            "Cannot create Qt view for obj: {obj}",
            deferred=True,
            obj=obj,
        )
    )


def create_model(obj: SelectableEventedList | Group, parent: QWidget | None = None):
    """Create a `QtListModel`, or `QtNodeTreeModel` for `obj`.

    Parameters
    ----------
    obj : SelectableEventedList or Group
        The python object for which to creat a QtView.
    parent : QWidget, optional
        Optional parent widget, by default None

    Returns
    -------
    Union[QtListModel, QtNodeTreeModel]
        A model instance appropriate for `obj`.
    """
    from finn._qt.containers import (
        QtLayerListModel,
        QtListModel,
        QtNodeTreeModel,
    )

    if isinstance(obj, LayerList):
        return QtLayerListModel(obj, parent=parent)
    if isinstance(obj, Group):
        return QtNodeTreeModel(obj, parent=parent)
    if isinstance(obj, AxisList):
        return QtAxisListModel(obj, parent=parent)
    if isinstance(obj, SelectableEventedList):
        return QtListModel(obj, parent=parent)
    raise TypeError(
        trans._(
            "Cannot create Qt model for obj: {obj}",
            deferred=True,
            obj=obj,
        )
    )
