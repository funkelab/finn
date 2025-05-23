from enum import auto
from typing import Literal

from finn.utils.misc import StringEnum

InterpolationStr = Literal[
    "bessel",
    "cubic",
    "linear",
    "blackman",
    "catrom",
    "gaussian",
    "hamming",
    "hanning",
    "hermite",
    "kaiser",
    "lanczos",
    "mitchell",
    "nearest",
    "spline16",
    "spline36",
    "custom",
]


class Interpolation(StringEnum):
    """INTERPOLATION: Vispy interpolation mode.

    The spatial filters used for interpolation are from vispy's
    spatial filters. The filters are built in the file below:

    https://github.com/vispy/vispy/blob/main/vispy/glsl/build-spatial-filters.py
    """

    BESSEL = auto()
    CUBIC = auto()
    LINEAR = auto()
    BLACKMAN = auto()
    CATROM = auto()
    GAUSSIAN = auto()
    HAMMING = auto()
    HANNING = auto()
    HERMITE = auto()
    KAISER = auto()
    LANCZOS = auto()
    MITCHELL = auto()
    NEAREST = auto()
    SPLINE16 = auto()
    SPLINE36 = auto()
    CUSTOM = auto()

    value: InterpolationStr

    @classmethod
    def view_subset(
        cls,
    ) -> tuple[
        "Interpolation",
        "Interpolation",
        "Interpolation",
        "Interpolation",
        "Interpolation",
    ]:
        return (
            cls.CUBIC,
            cls.LINEAR,
            cls.KAISER,
            cls.NEAREST,
            cls.SPLINE36,
        )

    def __str__(self) -> InterpolationStr:
        return self.value


class ImageRendering(StringEnum):
    """Rendering: Rendering mode for the layer.

    Selects a preset rendering mode in vispy

    * translucent: voxel colors are blended along the view ray until
      the result is opaque.
    * mip: maximum intensity projection. Cast a ray and display the
      maximum value that was encountered.
    * minip: minimum intensity projection. Cast a ray and display the
      minimum value that was encountered.
    * attenuated_mip: attenuated maximum intensity projection. Cast a
      ray and attenuate values based on integral of encountered values,
      display the maximum value that was encountered after attenuation.
      This will make nearer objects appear more prominent.
    * additive: voxel colors are added along the view ray until
      the result is saturated.
    * iso: isosurface. Cast a ray until a certain threshold is
      encountered. At that location, lighning calculations are
      performed to give the visual appearance of a surface.
    * average: average intensity projection. Cast a ray and display the
      average of values that were encountered.
    """

    TRANSLUCENT = auto()
    ADDITIVE = auto()
    ISO = auto()
    MIP = auto()
    MINIP = auto()
    ATTENUATED_MIP = auto()
    AVERAGE = auto()


ImageRenderingStr = Literal[
    "translucent",
    "additive",
    "iso",
    "mip",
    "minip",
    "attenuated_mip",
    "average",
]


class ImageProjectionMode(StringEnum):
    """
    Projection mode for aggregating a thick nD slice onto displayed dimensions.

        * NONE: ignore slice thickness, only using the dims point
        * SUM: sum data across the thick slice
        * MEAN: average data across the thick slice
        * MAX: display the maximum value across the thick slice
        * MIN: display the minimum value across the thick slice
    """

    NONE = auto()
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()
