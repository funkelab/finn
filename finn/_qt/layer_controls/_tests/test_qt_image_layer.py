import numpy as np

from finn._qt.layer_controls.qt_image_controls import QtImageControls
from finn.components.dims import Dims
from finn.layers import Image


def test_interpolation_combobox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl.interpComboBox
    opts = {combo.itemText(i) for i in range(combo.count())}
    assert opts == {"cubic", "linear", "kaiser", "nearest", "spline36"}
    # programmatically adding approved interpolation works
    layer.interpolation2d = "lanczos"
    assert combo.findText("lanczos") == 5


def test_rendering_combobox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl.renderComboBox
    opts = {combo.itemText(i) for i in range(combo.count())}
    rendering_options = {
        "translucent",
        "additive",
        "iso",
        "mip",
        "minip",
        "attenuated_mip",
        "average",
    }
    assert opts == rendering_options
    # programmatically updating rendering mode updates the combobox
    layer.rendering = "iso"
    assert combo.findText("iso") == combo.currentIndex()


def test_depiction_combobox_changes(qtbot):
    """Changing the model attribute should update the view."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtctrl.ndisplay = 3
    qtbot.addWidget(qtctrl)
    combo_box = qtctrl.depictionControls.depictionComboBox
    opts = {combo_box.itemText(i) for i in range(combo_box.count())}
    depiction_options = {
        "volume",
        "plane",
    }
    assert opts == depiction_options
    layer.depiction = "plane"
    assert combo_box.findText("plane") == combo_box.currentIndex()
    layer.depiction = "volume"
    assert combo_box.findText("volume") == combo_box.currentIndex()


def test_plane_controls_show_hide_on_depiction_change(qtbot):
    """Changing depiction mode should show/hide plane controls in 3D."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    qtctrl.ndisplay = 3

    layer.depiction = "volume"
    assert qtctrl.depictionControls.planeThicknessSlider.isHidden()
    assert qtctrl.depictionControls.planeThicknessLabel.isHidden()
    assert not qtctrl.depictionControls.clippingPlaneCheckbox.isHidden()
    assert not qtctrl.depictionControls.clippingPlaneSlider.isHidden()
    assert qtctrl.depictionControls.planeSlider.isHidden()
    assert qtctrl.depictionControls.planeSliderLabel.isHidden()

    layer.depiction = "plane"
    assert not qtctrl.depictionControls.planeThicknessSlider.isHidden()
    assert not qtctrl.depictionControls.planeThicknessLabel.isHidden()
    assert not qtctrl.depictionControls.planeNormalButtons.isHidden()
    assert not qtctrl.depictionControls.planeNormalLabel.isHidden()
    assert not qtctrl.depictionControls.planeSlider.isHidden()
    assert not qtctrl.depictionControls.planeSliderLabel.isHidden()
    assert qtctrl.depictionControls.clippingPlaneCheckbox.isHidden()
    assert qtctrl.depictionControls.clippingPlaneSlider.isHidden()


def test_plane_controls_show_hide_on_ndisplay_change(qtbot):
    """Changing ndisplay should show/hide plane controls if depicting a plane."""
    layer = Image(np.random.rand(10, 15, 20))
    layer.depiction = "plane"
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    assert qtctrl.ndisplay == 2
    assert qtctrl.depictionControls.planeThicknessSlider.isHidden()
    assert qtctrl.depictionControls.planeThicknessLabel.isHidden()
    assert qtctrl.depictionControls.planeNormalButtons.isHidden()
    assert qtctrl.depictionControls.planeNormalLabel.isHidden()

    qtctrl.ndisplay = 3
    assert not qtctrl.depictionControls.planeThicknessSlider.isHidden()
    assert not qtctrl.depictionControls.planeThicknessLabel.isHidden()
    assert not qtctrl.depictionControls.planeNormalButtons.isHidden()
    assert not qtctrl.depictionControls.planeNormalLabel.isHidden()


def test_plane_thickness_slider_value_change(qtbot):
    """Changing the model should update the view."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    layer.plane.thickness *= 2
    assert qtctrl.depictionControls.planeThicknessSlider.value() == layer.plane.thickness


def test_plane_slider_value_change(qtbot):
    """Test if updating the plane position updates the slider value."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    plane_normal = np.array(layer.plane.normal)
    new_position = np.array([0, 0, 0]) + 2 * plane_normal
    layer.plane.position = tuple(new_position)
    slider_value = np.dot(new_position, plane_normal) / np.dot(plane_normal, plane_normal)
    assert qtctrl.depictionControls.planeSlider.value() == slider_value


def test_plane_position_change(qtbot):
    """Test if updating the slider updates the plane position."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    test_value = 2
    qtctrl.depictionControls.planeSlider.setValue(test_value)
    plane_normal = np.array(layer.plane.normal)
    new_position = np.array([0, 0, 0]) + test_value * plane_normal
    assert layer.plane.position == tuple(new_position)


def test_clipping_plane_position_change(qtbot):
    """Test if updating the clipping plane slider updates the clipping plane positions"""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    test_value = (2, 4)
    qtctrl.depictionControls.changeClippingPlanePositions(test_value)

    plane_normal = np.array(layer.experimental_clipping_planes[0].normal)
    new_position_1 = tuple(np.array([0, 0, 0]) + test_value[0] * plane_normal)
    new_position_2 = tuple(np.array([0, 0, 0]) + test_value[1] * plane_normal)

    assert layer.experimental_clipping_planes[0].position == new_position_1
    assert layer.experimental_clipping_planes[1].position == new_position_2


def test_auto_contrast_buttons(qtbot):
    layer = Image(np.arange(8**3).reshape(8, 8, 8), contrast_limits=(0, 1))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    assert layer.contrast_limits == [0, 1]
    qtctrl.autoScaleBar._once_btn.click()
    assert layer.contrast_limits == [0, 63]

    # change slice
    dims = Dims(ndim=3, range=((0, 4, 1), (0, 8, 1), (0, 8, 1)), point=(1, 8, 8))
    layer._slice_dims(dims)
    # hasn't changed yet
    assert layer.contrast_limits == [0, 63]

    # with auto_btn, it should always change
    qtctrl.autoScaleBar._auto_btn.click()
    assert layer.contrast_limits == [64, 127]
    dims.point = (2, 8, 8)
    layer._slice_dims(dims)
    assert layer.contrast_limits == [128, 191]
    dims.point = (3, 8, 8)
    layer._slice_dims(dims)
    assert layer.contrast_limits == [192, 255]

    # once button turns off continuous
    qtctrl.autoScaleBar._once_btn.click()
    dims.point = (4, 8, 8)
    layer._slice_dims(dims)
    assert layer.contrast_limits == [192, 255]
