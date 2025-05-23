import os

from lazy_loader import attach as _attach

from finn._check_numpy_version import limit_numpy1x_threads_on_macos_arm

try:
    from finn._version import version as __version__
except ImportError:
    __version__ = "not-installed"

# Allows us to use pydata/sparse arrays as layer data
os.environ.setdefault("SPARSE_AUTO_DENSIFY", "1")
limit_numpy1x_threads_on_macos_arm()

del limit_numpy1x_threads_on_macos_arm
del os

# Add everything that needs to be accessible from the napari namespace here.
_proto_all_ = [
    "__version__",
    "components",
    "experimental",
    "layers",
    "qt",
    "types",
    "viewer",
    "utils",
]

_submod_attrs = {
    "_event_loop": ["gui_qt", "run"],
    "plugins.io": ["save_layers"],
    "utils": ["sys_info"],
    "utils.notifications": ["notification_manager"],
    "view_layers": [
        "view_image",
        "view_labels",
        "view_path",
        "view_points",
        "view_shapes",
        "view_surface",
        "view_tracks",
        "view_vectors",
        "imshow",
    ],
    "viewer": ["Viewer", "current_viewer"],
}

# All imports in __init__ are hidden inside of `__getattr__` to prevent
# importing the full chain of packages required when calling `import finn`.
#
# This has the biggest implications for running `napari` on the command line
# (or running `python -m napari`) since `finn.__init__` gets imported
# on the way to `finn.__main__`. Importing everything here has the
# potential to take a second or more, so we definitely don't want to import it
# just to access the CLI (which may not actually need any of the imports)


__getattr__, __dir__, __all__ = _attach(
    __name__, submodules=_proto_all_, submod_attrs=_submod_attrs
)
del _attach
