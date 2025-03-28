import numpy as np
import pytest

from finn_builtins.example_data import (
    Fluo_N2DL_HeLa,
    Fluo_N2DL_HeLa_crop,
    Mouse_Embryo_Membrane,
    delete_all,
)


@pytest.mark.slow
def test_sample_data():
    delete_all()
    raw_layer_data, seg_layer_data = Mouse_Embryo_Membrane()
    raw_data = raw_layer_data[0]
    seg_data = seg_layer_data[0]
    shape = (117, 123, 127, 127)
    assert raw_data.shape == shape
    assert seg_data.shape == shape
    assert raw_data.dtype == np.uint16
    assert seg_data.dtype == np.uint16


@pytest.mark.slow
@pytest.mark.parametrize(
    ("ds_function", "img_shape", "point_shape"),
    [
        (Fluo_N2DL_HeLa, (92, 700, 1100), (8602, 3)),
        (Fluo_N2DL_HeLa_crop, (92, 210, 340), (1266, 3)),
    ],
)
def test_Fluo_N2DL_Hela(ds_function, img_shape, point_shape):
    delete_all()
    raw_layer_data, seg_layer_data, points_layer_data = ds_function()
    raw_data = raw_layer_data[0]
    seg_data = seg_layer_data[0]
    points = points_layer_data[0]
    assert raw_data.shape == img_shape
    assert seg_data.shape == img_shape
    assert raw_data.dtype == np.uint16
    assert seg_data.dtype == np.uint16
    assert points.shape == point_shape
