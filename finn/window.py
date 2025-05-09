"""The Window class is the primary entry to the napari GUI.

Currently, this module is just a stub file that will simply pass through the
:class:`finn._qt.qt_main_window.Window` class.  In the future, this module
could serve to define a window Protocol that a backend would need to implement
to server as a graphical user interface for finn.
"""

__all__ = ["Window"]

from finn.utils.translations import trans

try:
    from finn._qt import Window

except ImportError as e:
    err = e

    class Window:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def close(self):
            pass

        def __getattr__(self, name):
            raise type(err)(
                trans._(
                    "An error occured when importing Qt dependencies.  Cannot show "
                    "napari window.  See cause above",
                )
            ) from err
