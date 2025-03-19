import numpy as np
from app_model.types import Action

from finn._app_model import get_app_model
from finn._app_model.constants import MenuId
from finn._app_model.context import LayerListContextKeys as LLCK
from finn.layers import Image


def test_update_menu_state_context(make_napari_viewer):
    """Test `_update_menu_state` correctly updates enabled/visible state."""
    app = get_app_model()
    viewer = make_napari_viewer()

    action = Action(
        id="dummy_id",
        title="dummy title",
        callback=lambda: None,
        menus=[{"id": MenuId.MENUBAR_FILE, "when": (LLCK.num_layers > 0)}],
        enablement=(LLCK.num_layers == 2),
    )
    app.register_action(action)

    dummy_action = viewer.window.file_menu.findAction("dummy_id")

    assert "dummy_id" in app.commands
    assert len(viewer.layers) == 0
    # `dummy_action` should be disabled & not visible as num layers == 0
    viewer.window._update_file_menu_state()
    assert not dummy_action.isVisible()
    assert not dummy_action.isEnabled()

    layer_a = Image(np.random.random((10, 10)))
    viewer.layers.append(layer_a)
    assert len(viewer.layers) == 1
    viewer.window._update_file_menu_state()
    # `dummy_action` should be visible but not enabled after adding layer
    assert dummy_action.isVisible()
    assert not dummy_action.isEnabled()

    layer_b = Image(np.random.random((10, 10)))
    viewer.layers.append(layer_b)
    assert len(viewer.layers) == 2
    # `dummy_action` should be enabled and visible after adding second layer
    viewer.window._update_file_menu_state()
    assert dummy_action.isVisible()
    assert dummy_action.isEnabled()
