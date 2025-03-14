from unittest.mock import Mock

import pytest
from magicgui import magic_factory, magicgui
from magicgui.widgets import Container
from qtpy.QtWidgets import QWidget

import finn
from finn._app_model import get_app_model
from finn._qt._qplugins._qnpe2 import _get_widget_viewer_param
from finn.viewer import Viewer


class ErrorWidget:
    pass


class QWidget_example(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()


class QWidget_string_annnot(QWidget):
    def __init__(self, test: "finn.viewer.Viewer"):
        super().__init__()  # pragma: no cover


class Container_example(Container):
    def __init__(self, test: Viewer):
        super().__init__()


@magic_factory
def magic_widget_example():
    """Example magic factory widget."""


def callable_example():
    @magicgui
    def magic_widget_example():
        """Example magic factory widget."""

    return magic_widget_example


class Widg2(QWidget):
    def __init__(self, napari_viewer) -> None:
        self.viewer = napari_viewer
        super().__init__()


def magicfunc(viewer: "finn.Viewer"):
    return viewer


dwidget_args = {
    "single_class": QWidget_example,
    "class_tuple": (QWidget_example, {"area": "right"}),
    "tuple_list": [(QWidget_example, {"area": "right"}), (Widg2, {})],
    "tuple_list2": [(QWidget_example, {"area": "right"}), Widg2],
    "bad_class": 1,
    "bad_tuple1": (QWidget_example, 1),
    "bad_double_tuple": ((QWidget_example, {}), (Widg2, {})),
}


@pytest.mark.parametrize(
    ("widget_callable", "param"),
    [
        (QWidget_example, "napari_viewer"),
        (QWidget_string_annnot, "test"),
        (Container_example, "test"),
    ],
)
def test_get_widget_viewer_param(widget_callable, param):
    """Test `_get_widget_viewer_param` returns correct parameter name."""
    out = _get_widget_viewer_param(widget_callable, "widget_name")
    assert out == param


def test_get_widget_viewer_param_error():
    """Test incorrect subclass raises error in `_get_widget_viewer_param`."""
    with pytest.raises(TypeError) as e:
        _get_widget_viewer_param(ErrorWidget, "widget_name")
    assert "'widget_name' must be `QtWidgets.QWidget`" in str(e)


def test_widget_hide_destroy(make_napari_viewer, qtbot):
    """Test that widget hide and destroy works."""
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(QWidget_example(viewer), name="test")
    dock_widget = viewer.window._dock_widgets["test"]

    # Check widget persists after hide
    widget = dock_widget.widget()
    dock_widget.title.hide_button.click()
    assert widget
    # Check that widget removed from `_dock_widgets` dict and parent
    # `QtViewerDockWidget` is `None` when closed
    dock_widget.destroyOnClose()
    assert "test" not in viewer.window._dock_widgets
    assert widget.parent() is None
    widget.deleteLater()
    widget.close()
    qtbot.wait(50)


@pytest.mark.parametrize(
    "Widget",
    [
        QWidget_example,
        Container_example,
        magic_widget_example,
        callable_example,
    ],
)
def test_widget_types_supported(
    make_napari_viewer,
    tmp_plugin,
    Widget,
):
    """Test all supported widget types correctly instantiated and call processor.

    The 4 parametrized `Widget`s represent the varing widget constructors and
    signatures that we want to support.
    """
    # Using the decorator as a function on the parametrized `Widget`
    # This allows `Widget` to be callable object that, when called, returns an
    # instance of a widget
    tmp_plugin.contribute.widget(display_name="Widget")(Widget)

    app = get_app_model()
    viewer = make_napari_viewer()

    # `side_effect` required so widget is added to window and then
    # cleaned up, preventing widget leaks
    viewer.window.add_dock_widget = Mock(side_effect=viewer.window.add_dock_widget)
    app.commands.execute_command("tmp_plugin:Widget")
    viewer.window.add_dock_widget.assert_called_once()
