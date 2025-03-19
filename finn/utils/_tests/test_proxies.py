from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from finn.components.viewer_model import ViewerModel
from finn.utils._proxies import PublicOnlyProxy, ReadOnlyWrapper


def test_ReadOnlyWrapper_setitem():
    """test that ReadOnlyWrapper prevents setting items"""
    d = {"hi": 3}
    d_read_only = ReadOnlyWrapper(d)

    with pytest.raises(TypeError):
        d_read_only["hi"] = 5


def test_ReadOnlyWrapper_setattr():
    """test that ReadOnlyWrapper prevents setting attributes"""

    class TestClass:
        x = 3

    tc = TestClass()
    tc_read_only = ReadOnlyWrapper(tc)

    with pytest.raises(TypeError):
        tc_read_only.x = 5


@pytest.fixture
def _patched_root_dir():
    """Simulate a call from outside of napari"""
    with patch("finn.utils.misc.ROOT_DIR", new="/some/other/package"):
        yield


@pytest.mark.filterwarnings("ignore:Qt libs are available but")
def test_thread_proxy_guard(monkeypatch, single_threaded_executor):
    class X:
        a = 1

    monkeypatch.setenv("NAPARI_ENSURE_PLUGIN_MAIN_THREAD", "True")

    x = X()
    x_proxy = PublicOnlyProxy(x)

    f = single_threaded_executor.submit(x.__setattr__, "a", 2)
    f.result()
    assert x.a == 2

    f = single_threaded_executor.submit(x_proxy.__setattr__, "a", 3)
    with pytest.raises(RuntimeError):
        f.result()
    assert x.a == 2


@pytest.mark.usefixtures("_patched_root_dir")
def test_public_proxy_limited_to_napari():
    """Test that the recursive public proxy goes no farther than finn."""
    viewer = ViewerModel()
    viewer.add_points(None)
    pv = PublicOnlyProxy(viewer)
    assert not isinstance(pv.layers[0].data, PublicOnlyProxy)


@pytest.mark.usefixtures("_patched_root_dir")
def test_array_from_proxy_objects():
    """Test that the recursive public proxy goes no farther than finn."""
    viewer = ViewerModel()
    viewer.add_points(None)
    pv = PublicOnlyProxy(viewer)
    assert isinstance(np.array(pv.dims.displayed, dtype=int), np.ndarray)


def test_viewer_method():
    viewer = PublicOnlyProxy(ViewerModel())
    assert viewer.add_points() is not None


def test_unwrap_setattr():
    """Check that objects added with __setattr__ of an object wrapped with
    PublicOnlyProxy are unwrapped before setting the attribute.
    """

    @dataclass
    class Sample:
        attribute = "aaa"

    sample = Sample()
    public_only_sample = PublicOnlyProxy(sample)

    text = "bbb"
    wrapped_text = PublicOnlyProxy(text)

    public_only_sample.attribute = wrapped_text
    attribute = sample.attribute  # use original, not wrapped object

    # check that the attribute in the unwrapped sample is itself not the
    # wrapped text, but the original text.
    assert id(text) == id(attribute)
