from finn._vispy.overlays.axes import VispyAxesOverlay
from finn.components import ViewerModel
from finn.components.overlays import AxesOverlay


def test_init_with_2d_display_of_2_dimensions():
    axes_model = AxesOverlay()
    viewer = ViewerModel()
    viewer.dims.ndim = 2
    viewer.dims.ndisplay = 3

    axes_view = VispyAxesOverlay(viewer=viewer, overlay=axes_model)

    assert tuple(axes_view.node.text.text) == ('1', '0')


def test_init_with_2d_display_of_3_dimensions():
    axes_model = AxesOverlay()
    viewer = ViewerModel()
    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 2

    axes_view = VispyAxesOverlay(viewer=viewer, overlay=axes_model)

    assert tuple(axes_view.node.text.text) == ('2', '1')


def test_init_with_3d_display_of_2_dimensions():
    axes_model = AxesOverlay()
    viewer = ViewerModel()
    viewer.dims.ndim = 2
    viewer.dims.ndisplay = 3

    axes_view = VispyAxesOverlay(viewer=viewer, overlay=axes_model)

    assert tuple(axes_view.node.text.text) == ('1', '0')


def test_init_with_3d_display_of_3_dimensions():
    axes_model = AxesOverlay()
    viewer = ViewerModel()
    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 3

    axes_view = VispyAxesOverlay(viewer=viewer, overlay=axes_model)

    assert tuple(axes_view.node.text.text) == ('2', '1', '0')
