from funtracks.features.feature_set import FeatureSet
from funtracks.features.regionprops_annotator import RPFeature
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

import finn
from finn import Viewer
from finn.layers import Layer
from finn.track_application_menus.region_annotator_widget import RegionAnnotatorWidget
from finn.track_application_menus.scale_widget import ScaleWidget


class MeasurementSetupWidget(QWidget):
    """Widget to combine regionprops features, intensity image selection, and scaling
    information widgets."""

    def __init__(self, viewer: Viewer, input_layer: finn.layers.Layer | None = None):
        super().__init__()
        self.viewer: Viewer = viewer
        self.input_layer = input_layer

        # add a widget for choosing the features to measure
        seg = isinstance(self.input_layer, finn.layers.Labels)
        self.feature_widget = RegionAnnotatorWidget(self.viewer, ndims=3, seg=seg)

        # add a widget for entering scaling information
        self.scale_widget = ScaleWidget(scaling=(1, 1, 1))
        self.scale_widget.scaling_updated.connect(self.apply_scaling)

        layout = QVBoxLayout()
        layout.addWidget(self.feature_widget)
        layout.addWidget(self.scale_widget)
        self.setLayout(layout)

    def update_input_layer(self, layer: Layer) -> None:
        """Update the feature and scale widget vbased on dims, scaling, and layer type
        of the new layer

        Args:
            layer [Layer]: layer that is the new input layer (should be either Points or
            Labels)
        """

        self.input_layer = layer
        ndims = self.input_layer.data.ndim if self.input_layer is not None else 3
        seg = isinstance(self.input_layer, finn.layers.Labels)
        self.feature_widget.update(ndims=ndims, seg=seg)
        self.scale_widget.update_scaling(self.input_layer.scale)

    def get_features(self) -> list[RPFeature]:
        """Return the features to be measured from the feature_widget"""

        return self.feature_widget.get_selected_features()

    def update_features(self, features: FeatureSet) -> None:
        """Update the selected features in the feature widget"""

        feature_list = features._features
        self.feature_widget.set_selected_features(feature_list)

    def get_intensity_image(self):
        """Return the selected intensity image as np.ndarray, if available"""

        return (
            self.feature_widget.intensity_image
            if self.feature_widget.intensity_image is not None
            else None
        )

    def get_scaling(self) -> tuple[float]:
        """Return the scaling information from the scaling widget"""

        return self.scale_widget.get_scaling()

    def update_scaling(self, scale: tuple[float]) -> None:
        """Update the scaling information in the scale widget"""

        self.scale_widget.update_scaling(scale)

    def apply_scaling(self) -> None:
        """Apply the scaling to the input layer and the intensity image, if present"""

        scale = self.get_scaling()
        if self.input_layer is not None:
            self.input_layer.scale = scale
        if self.feature_widget.intensity_image is not None:
            self.feature_widget.intensity_image.scale = scale
