"""Qt 'Help' menu Actions."""

import sys
from functools import partial
from webbrowser import open as web_open

from app_model.types import Action, KeyBindingRule, KeyCode, KeyMod
from packaging.version import parse

from finn import __version__
from finn._app_model.constants import MenuGroup, MenuId
from finn._qt.dialogs.qt_about import QtAbout
from finn._qt.qt_main_window import Window
from finn.utils.translations import trans


def _show_about(window: Window):
    QtAbout.showAbout(window._qt_window)


v = parse(__version__)
VERSION = 'dev' if v.is_devrelease or v.is_prerelease else str(v.base_version)

HELP_URLS: dict[str, str] = {
    'getting_started': f'https://finn.org/{VERSION}/tutorials/start_index.html',
    'tutorials': f'https://finn.org/{VERSION}/tutorials/index.html',
    'layers_guide': f'https://finn.org/{VERSION}/howtos/layers/index.html',
    'examples_gallery': f'https://finn.org/{VERSION}/gallery.html',
    'release_notes': f'https://finn.org/{VERSION}/release/release_{VERSION.replace(".", "_")}.html',
    'github_issue': 'https://github.com/napari/napari/issues',
    'homepage': 'https://finn.org',
}

Q_HELP_ACTIONS: list[Action] = [
    Action(
        id='finn.window.help.info',
        title=trans._('‎napari Info'),
        callback=_show_about,
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.RENDER}],
        status_tip=trans._('About napari'),
        keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.Slash)],
    ),
    Action(
        id='finn.window.help.about_macos',
        title=trans._('About napari'),
        callback=_show_about,
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'group': MenuGroup.RENDER,
                'when': sys.platform == 'darwin',
            }
        ],
        status_tip=trans._('About napari'),
    ),
    Action(
        id='finn.window.help.getting_started',
        title=trans._('Getting started'),
        callback=partial(web_open, url=HELP_URLS['getting_started']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='finn.window.help.tutorials',
        title=trans._('Tutorials'),
        callback=partial(web_open, url=HELP_URLS['tutorials']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='finn.window.help.layers_guide',
        title=trans._('Using Layers Guides'),
        callback=partial(web_open, url=HELP_URLS['layers_guide']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='finn.window.help.examples',
        title=trans._('Examples Gallery'),
        callback=partial(web_open, url=HELP_URLS['examples_gallery']),
        menus=[{'id': MenuId.MENUBAR_HELP}],
    ),
    Action(
        id='finn.window.help.release_notes',
        title=trans._('Release Notes'),
        callback=partial(web_open, url=HELP_URLS['release_notes']),
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'when': VERSION != 'dev',
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id='finn.window.help.github_issue',
        title=trans._('Report an issue on GitHub'),
        callback=partial(web_open, url=HELP_URLS['github_issue']),
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'when': VERSION == 'dev',
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id='finn.window.help.homepage',
        title=trans._('napari homepage'),
        callback=partial(web_open, url=HELP_URLS['homepage']),
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.NAVIGATION}],
    ),
]
