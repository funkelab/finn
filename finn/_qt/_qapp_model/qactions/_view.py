"""Qt 'View' menu Actions."""

import sys

from app_model.types import (
    Action,
    KeyCode,
    KeyMod,
    StandardKeyBinding,
    SubmenuItem,
    ToggleRule,
)

from finn._app_model.constants import MenuGroup, MenuId
from finn._qt._qapp_model.qactions._toggle_action import ViewerToggleAction
from finn._qt.qt_main_window import Window
from finn._qt.qt_viewer import QtViewer
from finn.settings import get_settings
from finn.utils.translations import trans
from finn.viewer import Viewer

# View submenus
VIEW_SUBMENUS = [
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_AXES, title=trans._('Axes')),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCALEBAR, title=trans._('Scale Bar')),
    ),
]


# View actions
def _toggle_activity_dock(window: Window):
    window._status_bar._toggle_activity_dock()


def _get_current_fullscreen_status(window: Window):
    return window._qt_window.isFullScreen()


def _get_current_menubar_status(window: Window):
    return window._qt_window._toggle_menubar_visibility


def _get_current_play_status(qt_viewer: QtViewer):
    return bool(qt_viewer.dims.is_playing)


def _get_current_activity_dock_status(window: Window):
    return window._qt_window._activity_dialog.isVisible()


def _tooltip_visibility_toggle() -> None:
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


def _get_current_tooltip_visibility() -> bool:
    return get_settings().appearance.layer_tooltip_visibility


def _fit_to_view(viewer: Viewer):
    viewer.reset_view(reset_camera_angle=False)


def _zoom_in(viewer: Viewer):
    viewer.camera.zoom *= 1.5


def _zoom_out(viewer: Viewer):
    viewer.camera.zoom /= 1.5


Q_VIEW_ACTIONS: list[Action] = [
    Action(
        id='finn.window.view.toggle_fullscreen',
        title=trans._('Toggle Full Screen'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 1,
            }
        ],
        callback=Window._toggle_fullscreen,
        keybindings=[StandardKeyBinding.FullScreen],
        toggled=ToggleRule(get_current=_get_current_fullscreen_status),
    ),
    Action(
        id='finn.window.view.toggle_menubar',
        title=trans._('Toggle Menubar Visibility'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 2,
                'when': sys.platform != 'darwin',
            }
        ],
        callback=Window._toggle_menubar_visible,
        keybindings=[
            {
                'win': KeyMod.CtrlCmd | KeyCode.KeyM,
                'linux': KeyMod.CtrlCmd | KeyCode.KeyM,
            }
        ],
        # TODO: add is_mac global context keys (rather than boolean here)
        enablement=sys.platform != 'darwin',
        status_tip=trans._('Show/Hide Menubar'),
        toggled=ToggleRule(get_current=_get_current_menubar_status),
    ),
    Action(
        id='finn.window.view.toggle_play',
        title=trans._('Toggle Play'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 3,
            }
        ],
        callback=Window._toggle_play,
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyP}],
        toggled=ToggleRule(get_current=_get_current_play_status),
    ),
    Action(
        id='finn.viewer.fit_to_view',
        title=trans._('Fit to View'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_fit_to_view,
        keybindings=[StandardKeyBinding.OriginalSize],
    ),
    Action(
        id='finn.viewer.camera.zoom_in',
        title=trans._('Zoom In'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_zoom_in,
        keybindings=[StandardKeyBinding.ZoomIn],
    ),
    Action(
        id='finn.viewer.camera.zoom_out',
        title=trans._('Zoom Out'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_zoom_out,
        keybindings=[StandardKeyBinding.ZoomOut],
    ),
    Action(
        id='finn.window.view.toggle_activity_dock',
        title=trans._('Toggle Activity Dock'),
        menus=[
            {'id': MenuId.MENUBAR_VIEW, 'group': MenuGroup.RENDER, 'order': 11}
        ],
        callback=_toggle_activity_dock,
        toggled=ToggleRule(get_current=_get_current_activity_dock_status),
    ),
    # TODO: this could be made into a toggle setting Action subclass
    # using a similar pattern to the above ViewerToggleAction classes
    Action(
        id='finn.window.view.toggle_layer_tooltips',
        title=trans._('Toggle Layer Tooltips'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.RENDER,
                'order': 10,
            }
        ],
        callback=_tooltip_visibility_toggle,
        toggled=ToggleRule(get_current=_get_current_tooltip_visibility),
    ),
]

MENUID_DICT = {'axes': MenuId.VIEW_AXES, 'scale_bar': MenuId.VIEW_SCALEBAR}

toggle_action_details = [
    (
        'finn.window.view.toggle_viewer_axes',
        trans._('Axes Visible'),
        'axes',
        'visible',
    ),
    (
        'finn.window.view.toggle_viewer_axes_colored',
        trans._('Axes Colored'),
        'axes',
        'colored',
    ),
    (
        'finn.window.view.toggle_viewer_axes_labels',
        trans._('Axes Labels'),
        'axes',
        'labels',
    ),
    (
        'finn.window.view.toggle_viewer_axesdashed',
        trans._('Axes Dashed'),
        'axes',
        'dashed',
    ),
    (
        'finn.window.view.toggle_viewer_axes_arrows',
        trans._('Axes Arrows'),
        'axes',
        'arrows',
    ),
    (
        'finn.window.view.toggle_viewer_scale_bar',
        trans._('Scale Bar Visible'),
        'scale_bar',
        'visible',
    ),
    (
        'finn.window.view.toggle_viewer_scale_bar_colored',
        trans._('Scale Bar Colored'),
        'scale_bar',
        'colored',
    ),
    (
        'finn.window.view.toggle_viewer_scale_bar_ticks',
        trans._('Scale Bar Ticks'),
        'scale_bar',
        'ticks',
    ),
]

# Add `Action`s that toggle various viewer `axes` and `scale_bar` sub-attributes
# E.g., `toggle_viewer_scale_bar_ticks` toggles the sub-attribute `ticks` of the
# viewer attribute `scale_bar`
for cmd, cmd_title, viewer_attr, sub_attr in toggle_action_details:
    Q_VIEW_ACTIONS.append(
        ViewerToggleAction(
            id=cmd,
            title=cmd_title,
            viewer_attribute=viewer_attr,
            sub_attribute=sub_attr,
            menus=[{'id': MENUID_DICT[viewer_attr]}],
        )
    )
