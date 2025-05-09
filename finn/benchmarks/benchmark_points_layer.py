# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import os

import numpy as np
from packaging.version import parse as parse_version

import finn
from finn.components import Dims
from finn.layers import Points

from .utils import Skip

NAPARI_0_4_19 = parse_version(finn.__version__) <= parse_version("0.4.19")


class Points2DSuite:
    """Benchmarks for the Points layer with 2D data"""

    params = [2**i for i in range(4, 18, 2)]

    if "PR" in os.environ:
        skip_params = [(2**i,) for i in range(8, 18, 2)]

    def setup(self, n):
        np.random.seed(0)
        self.data = np.random.random((n, 2))
        self.layer = Points(self.data)

    def time_create_layer(self, n):
        """Time to create layer."""
        Points(self.data)

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value((0,) * 2)

    def time_add(self, n):
        self.layer.add(self.data)

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class Points3DSuite:
    """Benchmarks for the Points layer with 3D data."""

    params = [2**i for i in range(4, 18, 2)]
    if "PR" in os.environ:
        skip_params = [(2**i,) for i in range(6, 18, 2)]

    def setup(self, n):
        np.random.seed(0)
        self.data = np.random.random((n, 3))
        self.layer = Points(self.data)

    def time_create_layer(self, n):
        """Time to create layer."""
        Points(self.data)

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value((0,) * 3)

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class PointsSlicingSuite:
    """Benchmarks for slicing the Points layer with 3D data."""

    params = [True, False]
    timeout = 300
    skip_params = Skip(always=lambda _: NAPARI_0_4_19)

    def setup(self, flatten_slice_axis):
        np.random.seed(0)
        size = 20000 if "PR" in os.environ else 20000000
        self.data = np.random.uniform(size=(size, 3), low=0, high=500)
        if flatten_slice_axis:
            self.data[:, 0] = np.round(self.data[:, 0])
        self.layer = Points(self.data)
        self.dims = Dims(ndim=3, point=(249, 0, 0))

    def time_slice_points(self, flatten_slice_axis):
        """Time to take one slice of points"""
        self.layer._make_slice_request(self.dims)()


class PointsToMaskSuite:
    """Benchmarks for creating a binary image mask from points."""

    param_names = ["num_points", "mask_shape", "point_size"]
    params = [
        [64, 256, 1024, 4096, 16384],
        [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ],
        [5, 10],
    ]

    skip_params = Skip(
        if_in_pr=lambda num_points, mask_shape, points_size: num_points > 256
        or mask_shape[0] > 512
    )

    def setup(self, num_points, mask_shape, point_size):
        np.random.seed(0)
        data = np.random.random((num_points, len(mask_shape))) * mask_shape
        self.layer = Points(data, size=point_size)

    def time_to_mask(self, num_points, mask_shape, point_size):
        self.layer.to_mask(shape=mask_shape)


if __name__ == "__main__":
    from utils import run_benchmark

    run_benchmark()
