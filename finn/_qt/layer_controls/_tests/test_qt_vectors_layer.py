import numpy as np

from finn._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from finn.layers import Vectors

_VECTORS = np.zeros((2, 2, 2))


def test_out_of_slice_display_checkbox(qtbot):
    layer = Vectors(_VECTORS)
    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)
    qtctrl.outOfSliceCheckBox.setChecked(True)
    assert layer.out_of_slice_display
    qtctrl.outOfSliceCheckBox.setChecked(False)
    assert not layer.out_of_slice_display
