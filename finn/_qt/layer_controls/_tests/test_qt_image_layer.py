import numpy as np

from finn._qt.layer_controls.qt_image_controls import QtImageControls
from finn.components.dims import Dims
from finn.layers import Image
from finn.layers.utils.plane import ClippingPlane


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


def test_plane_controls_show_hide_on_ndisplay_change(qtbot):
    """Changing depiction mode should show/hide plane controls in 3D."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    qtctrl.ndisplay = 3

    assert not qtctrl.clippingPlaneControls.planeNormalLabel.isHidden()
    assert not qtctrl.clippingPlaneControls.planeNormalButtons.isHidden()

    assert not qtctrl.clippingPlaneControls.clippingPlaneLabel.isHidden()
    assert not qtctrl.clippingPlaneControls.clippingPlaneCheckbox.isHidden()

    assert not qtctrl.clippingPlaneControls.clippingPlaneWidthLabel.isHidden()
    assert not qtctrl.clippingPlaneControls.clippingPlaneWidthSlider.isHidden()

    assert not qtctrl.clippingPlaneControls.clippingPlaneCenterLabel.isHidden()
    assert not qtctrl.clippingPlaneControls.clippingPlaneCenterSlider.isHidden()

    qtctrl.ndisplay = 2
    assert qtctrl.clippingPlaneControls.planeNormalLabel.isHidden()
    assert qtctrl.clippingPlaneControls.planeNormalButtons.isHidden()

    assert qtctrl.clippingPlaneControls.clippingPlaneLabel.isHidden()
    assert qtctrl.clippingPlaneControls.clippingPlaneCheckbox.isHidden()

    assert qtctrl.clippingPlaneControls.clippingPlaneWidthLabel.isHidden()
    assert qtctrl.clippingPlaneControls.clippingPlaneWidthSlider.isHidden()

    assert qtctrl.clippingPlaneControls.clippingPlaneCenterLabel.isHidden()
    assert qtctrl.clippingPlaneControls.clippingPlaneCenterSlider.isHidden()


def test_set_clipping_plane_position(qtbot):
    """Test if updating the clipping plane slider updates the clipping plane positions"""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    width = 3
    center = 4
    qtctrl.clippingPlaneControls.set_clipping_plane_positions(width, center)

    position1 = center - width // 2
    position2 = (center + width // 2) + 1

    plane_normal = np.array(layer.experimental_clipping_planes[0].normal)
    new_position1 = np.array([0, 0, 0]) + position1 * plane_normal
    new_position1 = (
        int(new_position1[0] * layer.scale[-3]),
        int(new_position1[1] * layer.scale[-2]),
        int(new_position1[2] * layer.scale[-1]),
    )
    new_position2 = np.array([0, 0, 0]) + position2 * plane_normal
    new_position2 = (
        int(new_position2[0] * layer.scale[-3]),
        int(new_position2[1] * layer.scale[-2]),
        int(new_position2[2] * layer.scale[-1]),
    )

    assert layer.experimental_clipping_planes[0].position == new_position1
    assert layer.experimental_clipping_planes[1].position == new_position2


def test_compute_plane_range(qtbot):
    """Test the _compute_plane_range function."""
    layer_data = np.random.rand(10, 15, 20)
    layer = Image(layer_data)
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    # Set the plane normal
    layer.experimental_clipping_planes[0].normal = [1, 0, 0]  # Normal along the x-axis
    expected_range = (0, layer_data.shape[-3])  # Range along the x-axis

    # Call _compute_plane_range
    computed_range = qtctrl.clippingPlaneControls._compute_plane_range()
    assert computed_range == expected_range, (
        f"Expected {expected_range}, got {computed_range}"
    )

    # Test with a different plane normal
    layer.experimental_clipping_planes[0].normal = [0, 1, 0]  # Normal along the x-axis
    expected_range = (0, layer_data.shape[-2])  # Range along the y-axis
    computed_range = qtctrl.clippingPlaneControls._compute_plane_range()
    assert computed_range == expected_range, (
        f"Expected {expected_range}, got {computed_range}"
    )

    # Test with an oblique plane normal
    layer.experimental_clipping_planes[0].normal = [1, 1, 1]  # Normal along the x-axis
    computed_range = qtctrl.clippingPlaneControls._compute_plane_range()
    expected_range = (np.float64(0.0), np.float64(25.98))
    np.testing.assert_almost_equal(computed_range[0], expected_range[0], decimal=1)


def test_activate_clipping_plane(qtbot):
    """Test the _activateClippingPlane function."""
    layer_data = np.random.rand(10, 15, 20)
    layer = Image(layer_data)
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    # Ensure the experimental_clipping_planes are initialized
    layer.experimental_clipping_planes = [
        ClippingPlane(normal=[1, 0, 0], position=[0, 0, 0], enabled=False),
        ClippingPlane(normal=[-1, 0, 0], position=[0, 0, 0], enabled=False),
    ]

    # Activate the clipping plane
    qtctrl.clippingPlaneControls._activateClippingPlane(True)
    assert layer.experimental_clipping_planes[0].enabled is True
    assert layer.experimental_clipping_planes[1].enabled is True

    assert qtctrl.clippingPlaneControls.clippingPlaneWidthSlider.isEnabled() is True
    assert qtctrl.clippingPlaneControls.clippingPlaneCenterSlider.isEnabled() is True
    assert qtctrl.clippingPlaneControls.clippingPlaneWidthLabel.isEnabled() is True
    assert qtctrl.clippingPlaneControls.clippingPlaneCenterLabel.isEnabled() is True

    # Deactivate the clipping plane
    qtctrl.clippingPlaneControls._activateClippingPlane(False)
    assert layer.experimental_clipping_planes[0].enabled is False
    assert layer.experimental_clipping_planes[1].enabled is False

    assert qtctrl.clippingPlaneControls.clippingPlaneWidthSlider.isEnabled() is False
    assert qtctrl.clippingPlaneControls.clippingPlaneCenterSlider.isEnabled() is False
    assert qtctrl.clippingPlaneControls.clippingPlaneWidthLabel.isEnabled() is False
    assert qtctrl.clippingPlaneControls.clippingPlaneCenterLabel.isEnabled() is False


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
