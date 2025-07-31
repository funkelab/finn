import numpy as np

from finn.layers import Image, Labels
from finn.track_data_views.views.view_3d.multiple_view_widget import MultipleViewerWidget


def test_multiple_viewer_widget_initialization(make_napari_viewer, qtbot):
    """Test that the MultipleViewerWidget initializes correctly."""

    viewer = make_napari_viewer()
    viewer.dims.order = [0, 1, 2]
    viewer.dims.axis_labels = ["z", "y", "x"]
    widget = MultipleViewerWidget(viewer)
    qtbot.addWidget(widget)

    # Check that the widget initializes without errors
    assert widget.viewer == viewer
    assert widget.viewer_model1 is not None
    assert widget.viewer_model2 is not None
    assert widget.qt_viewer1 is not None
    assert widget.qt_viewer2 is not None


def test_add_remove_layer(make_napari_viewer, qtbot):
    """Test that layers are correctly added and removed in both viewer models."""

    viewer = make_napari_viewer()
    viewer.dims.order = [0, 1, 2]
    viewer.dims.axis_labels = ["z", "y", "x"]
    widget = MultipleViewerWidget(viewer)
    qtbot.addWidget(widget)

    widget = MultipleViewerWidget(viewer)

    # Test image layer
    layer = Image(np.zeros((2, 2, 2)))
    layer.name = "test_layer"
    viewer.add_layer(layer)

    # test labels layer
    labels = Labels(np.zeros((2, 2, 2), dtype=np.uint8))
    labels.name = "test_labels_layer"
    viewer.add_layer(labels)

    # Check that the layer was added correctly to both viewer models
    assert "test_layer" in widget.viewer_model1.viewer_model.layers
    assert "test_layer" in widget.viewer_model2.viewer_model.layers
    assert isinstance(widget.viewer_model1.viewer_model.layers["test_layer"], Image)
    assert isinstance(widget.viewer_model2.viewer_model.layers["test_layer"], Image)

    assert "test_labels_layer" in widget.viewer_model1.viewer_model.layers
    assert "test_labels_layer" in widget.viewer_model2.viewer_model.layers
    assert isinstance(
        widget.viewer_model1.viewer_model.layers["test_labels_layer"], Labels
    )
    assert isinstance(
        widget.viewer_model2.viewer_model.layers["test_labels_layer"], Labels
    )

    viewer.layers.remove(layer)

    assert "test_layer" not in widget.viewer_model1.viewer_model.layers
    assert "test_layer" not in widget.viewer_model2.viewer_model.layers


def test_move_layer(make_napari_viewer, qtbot):
    """Test moving a layer is synced to the viewer models."""

    viewer = make_napari_viewer()
    viewer.dims.order = [0, 1, 2]
    viewer.dims.axis_labels = ["z", "y", "x"]
    widget = MultipleViewerWidget(viewer)
    qtbot.addWidget(widget)

    widget = MultipleViewerWidget(viewer)

    layer1 = Image(np.zeros((2, 2, 2)))
    layer2 = Image(np.ones((2, 2, 2)))
    layer1.name = "layer1"
    layer2.name = "layer2"

    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    # Check that the layer was added to both viewer models
    assert viewer.layers[0].name == layer1.name
    assert viewer.layers[1].name == layer2.name

    assert widget.viewer_model1.viewer_model.layers[0].name == layer1.name
    assert widget.viewer_model1.viewer_model.layers[1].name == layer2.name
    assert widget.viewer_model2.viewer_model.layers[0].name == layer1.name
    assert widget.viewer_model2.viewer_model.layers[1].name == layer2.name

    # Move layer1 to the top
    viewer.layers.move(1, 0)
    assert viewer.layers[0].name == layer2.name
    assert viewer.layers[1].name == layer1.name

    # Check that the layer was moved in both viewer models
    assert widget.viewer_model1.viewer_model.layers[0].name == layer2.name
    assert widget.viewer_model1.viewer_model.layers[1].name == layer1.name
    assert widget.viewer_model2.viewer_model.layers[0].name == layer2.name
    assert widget.viewer_model2.viewer_model.layers[1].name == layer1.name


def test_rename_layer(make_napari_viewer, qtbot):
    """Test that layers are added correctly to the viewer models."""

    viewer = make_napari_viewer()
    viewer.dims.order = [0, 1, 2]
    viewer.dims.axis_labels = ["z", "y", "x"]
    widget = MultipleViewerWidget(viewer)
    qtbot.addWidget(widget)

    widget = MultipleViewerWidget(viewer)

    layer1 = Image(np.zeros((2, 2, 2)))
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    assert widget.viewer_model1.viewer_model.layers[0].name == layer1.name
    assert widget.viewer_model2.viewer_model.layers[0].name == layer1.name

    viewer.layers[0].name = "layer1_renamed"
    assert viewer.layers[0].name == "layer1_renamed"
    assert widget.viewer_model1.viewer_model.layers[0].name == "layer1_renamed"
    assert widget.viewer_model2.viewer_model.layers[0].name == "layer1_renamed"


def test_update_current_step(make_napari_viewer, qtbot):
    """Test if updating the current step in one viewer (model) updates the other."""

    viewer = make_napari_viewer()
    viewer.dims.order = [0, 1, 2]
    viewer.dims.axis_labels = ["z", "y", "x"]
    widget = MultipleViewerWidget(viewer)
    qtbot.addWidget(widget)

    widget = MultipleViewerWidget(viewer)

    layer1 = Image(np.zeros((2, 2, 2, 2)))
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    assert viewer.dims.current_step == (0, 0, 0, 0)
    assert widget.viewer_model1.viewer_model.dims.current_step == (0, 0, 0, 0)
    assert widget.viewer_model2.viewer_model.dims.current_step == (0, 0, 0, 0)

    # Update current step and check that the viewer models follow
    viewer.dims.current_step = (1, 1, 0, 0)
    assert viewer.dims.current_step == (1, 1, 0, 0)
    assert widget.viewer_model1.viewer_model.dims.current_step == (1, 1, 0, 0)
    assert widget.viewer_model2.viewer_model.dims.current_step == (1, 1, 0, 0)

    widget.viewer_model1.viewer_model.dims.current_step = (0, 0, 1, 1)
    assert viewer.dims.current_step == (0, 0, 1, 1)
    assert widget.viewer_model1.viewer_model.dims.current_step == (0, 0, 1, 1)
    assert widget.viewer_model2.viewer_model.dims.current_step == (0, 0, 1, 1)


def test_reset_view(make_napari_viewer, qtbot):
    """Test if resetting the view in one viewer (model) resets the other."""

    viewer = make_napari_viewer()
    viewer.dims.order = [0, 1, 2]
    viewer.dims.axis_labels = ["z", "y", "x"]
    widget = MultipleViewerWidget(viewer)
    qtbot.addWidget(widget)

    widget = MultipleViewerWidget(viewer)

    layer1 = Image(np.zeros((2, 2, 2)))
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    # change camera zoom on a viewer model
    widget.viewer_model1.viewer_model.camera.zoom = 5
    assert widget.viewer_model1.viewer_model.camera.zoom == 5

    # check that resetting the view on the viewer is propagated to the viewer model
    viewer.reset_view()
    assert widget.viewer_model1.viewer_model.camera.zoom == viewer.camera.zoom


def test_layer_selection(make_napari_viewer, qtbot):
    """Test syncing layer selection between the viewer and the viewer models."""

    viewer = make_napari_viewer()
    viewer.dims.order = [0, 1, 2]
    viewer.dims.axis_labels = ["z", "y", "x"]
    widget = MultipleViewerWidget(viewer)
    qtbot.addWidget(widget)

    widget = MultipleViewerWidget(viewer)

    layer1 = Image(np.zeros((2, 2, 2)))
    layer2 = Image(np.ones((2, 2, 2)))
    layer1.name = "layer1"
    layer2.name = "layer2"

    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    # Change the layer selection in the veiwer
    viewer.layers.selection.active = viewer.layers[0]
    assert (
        widget.viewer_model1.viewer_model.layers.selection.active
        == widget.viewer_model1.viewer_model.layers[0]
    )
    assert (
        widget.viewer_model2.viewer_model.layers.selection.active
        == widget.viewer_model2.viewer_model.layers[0]
    )

    viewer.layers.selection.active = viewer.layers[1]
    assert (
        widget.viewer_model1.viewer_model.layers.selection.active
        == widget.viewer_model1.viewer_model.layers[1]
    )
    assert (
        widget.viewer_model2.viewer_model.layers.selection.active
        == widget.viewer_model2.viewer_model.layers[1]
    )


def test_property_sync(make_napari_viewer, qtbot):
    """test if updating the data on a layer in the viewer updates the data in the viewer models and vice versa."""

    viewer = make_napari_viewer()
    viewer.dims.order = [0, 1, 2]
    viewer.dims.axis_labels = ["z", "y", "x"]
    widget = MultipleViewerWidget(viewer)
    qtbot.addWidget(widget)

    widget = MultipleViewerWidget(viewer)

    labels = Labels(np.zeros((2, 2, 2), dtype=np.uint8))
    viewer.add_layer(labels)

    viewer.layers[0].opacity = 0.5
    assert widget.viewer_model1.viewer_model.layers[0].opacity == 0.5
    assert widget.viewer_model2.viewer_model.layers[0].opacity == 0.5

    widget.viewer_model1.viewer_model.layers[0].visible = False
    assert not viewer.layers[0].visible
    assert not widget.viewer_model2.viewer_model.layers[0].visible

    viewer.layers[0].data[0][0][0] = 1
    assert widget.viewer_model1.viewer_model.layers[0].data[0][0][0] == 1
    assert widget.viewer_model1.viewer_model.layers[0].data[0][0][0] == 1

    widget.viewer_model1.viewer_model.layers[0].data[0][0][1] = 2
    assert viewer.layers[0].data[0][0][1] == 2
    assert widget.viewer_model1.viewer_model.layers[0].data[0][0][1] == 2
