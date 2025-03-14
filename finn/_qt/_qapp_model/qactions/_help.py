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
VERSION = "dev" if v.is_devrelease or v.is_prerelease else str(v.base_version)

HELP_URLS: dict[str, str] = {
    "getting_started": "https://funkelab.github.io/motile_tracker/getting_started.html",
    "github_issue": "https://github.com/funkelab/motile_tracker/issues",
}

Q_HELP_ACTIONS: list[Action] = [
    Action(
        id="finn.window.help.info",
        title=trans._("‎finn Info"),
        callback=_show_about,
        menus=[{"id": MenuId.MENUBAR_HELP, "group": MenuGroup.RENDER}],
        status_tip=trans._("About finn"),
        keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.Slash)],
    ),
    Action(
        id="finn.window.help.about_macos",
        title=trans._("About finn"),
        callback=_show_about,
        menus=[
            {
                "id": MenuId.MENUBAR_HELP,
                "group": MenuGroup.RENDER,
                "when": sys.platform == "darwin",
            }
        ],
        status_tip=trans._("About finn"),
    ),
    Action(
        id="finn.window.help.getting_started",
        title=trans._("Getting started"),
        callback=partial(web_open, url=HELP_URLS["getting_started"]),
        menus=[{"id": MenuId.MENUBAR_HELP}],
    ),
    Action(
        id="finn.window.help.github_issue",
        title=trans._("Report an issue on GitHub"),
        callback=partial(web_open, url=HELP_URLS["github_issue"]),
        menus=[
            {
                "id": MenuId.MENUBAR_HELP,
                "when": VERSION == "dev",
                "group": MenuGroup.NAVIGATION,
            }
        ],
    ),
]
