from finn._qt.dialogs.qt_about import QtAbout
from finn._tests.utils import skip_local_popups


@skip_local_popups
def test_about(qtbot):
    wdg = QtAbout()
    qtbot.addWidget(wdg)
