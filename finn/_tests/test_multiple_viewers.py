import gc
from unittest.mock import patch


def test_multi_viewers_dont_clash(make_napari_viewer, qtbot):
    v1 = make_napari_viewer(title="v1")
    v2 = make_napari_viewer(title="v2")
    assert not v1.grid.enabled
    assert not v2.grid.enabled

    v1.window.activate()  # a click would do this in the actual gui
    v1.window._qt_viewer.viewerButtons.gridViewButton.click()

    assert not v2.grid.enabled
    assert v1.grid.enabled

    with patch.object(v1.window._qt_window, "_save_current_window_settings"):
        v1.close()
    with patch.object(v2.window._qt_window, "_save_current_window_settings"):
        v2.close()
    qtbot.wait(50)
    gc.collect()
