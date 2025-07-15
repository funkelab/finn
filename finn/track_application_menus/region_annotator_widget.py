from funtracks.features.regionprops_annotator import (
    Area,
    Circularity,
    EllipsoidAxes,
    Intensity,
    Perimeter,
    RPFeature,
)
from napari import Viewer
from qtpy.QtWidgets import QCheckBox, QGroupBox, QVBoxLayout, QWidget

from finn.layers import Image
from finn.track_application_menus.layer_dropdown import LayerDropdown


class RegionAnnotatorWidget(QWidget):
    """Widget for choosing regionprops features to measure."""

    def __init__(self, viewer: Viewer, ndims: int, seg: bool = False):
        super().__init__()

        self.viewer = viewer
        self.ndims = ndims
        self.enable_intensity = False
        self.feature_classes = [
            Area(ndim=ndims),
            EllipsoidAxes(ndim=ndims),
            Circularity(ndim=ndims),
            Perimeter(ndim=ndims),
            Intensity(ndim=ndims),
        ]
        self.measurement_checkboxes = {}

        # Create checkbox for each feature.
        layout = QVBoxLayout()
        for feature in self.feature_classes:
            checkbox = QCheckBox(feature.display_name)
            checkbox.setChecked(False)
            checkbox.setEnabled(seg)
            self.measurement_checkboxes[feature.key] = checkbox
            layout.addWidget(checkbox)

        # dropdown menu to select an Image layer for optional intensity measurements
        self.intensity_box = QGroupBox("Intensity image")
        self.image_dropdown = LayerDropdown(self.viewer, (Image))
        self.image_dropdown.layer_changed.connect(self._update_intensity_image)
        if self.image_dropdown.get_current_layer() in self.viewer.layers:
            self.intensity_image = self.viewer.layers[
                self.image_dropdown.get_current_layer()
            ]
        else:
            self.intensity_image = None
        intensity_box_layout = QVBoxLayout()
        intensity_box_layout.addWidget(self.image_dropdown)
        self.intensity_box.setLayout(intensity_box_layout)

        # wrap in a group box
        box = QGroupBox("Features to measure")
        box.setLayout(layout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        main_layout.addWidget(self.intensity_box)

        self.setLayout(main_layout)

    def _update_intensity_image(self, selected_layer: str) -> None:
        """Update the intensity image layer
        Args:
            selected_layer [str]: the name of the layer that is selected to be the
            intensity layer.
        """

        if selected_layer == "":
            self.intensity_image = None
            self.update_checkbox_availability(False)
        else:
            self.intensity_image = self.viewer.layers[selected_layer]
            self.update_checkbox_availability(True)

    def get_selected_features(self) -> list[RPFeature]:
        """Return a list of the features that have been selected"""

        features = []
        for feature in self.feature_classes:
            checkbox = self.measurement_checkboxes[feature.key]
            if checkbox.isChecked() and checkbox.isEnabled():
                features.append(feature)

        return features

    def set_selected_features(self, features: list[RPFeature]) -> None:
        """Update the checkbox state for the features, depending on whether they are in
         the list.
        Args:
            features (list[RPFeature]): list of RPFeatures that are included.
        """

        included_features = [feature.key for feature in features]
        for key, checkbox in self.measurement_checkboxes.items():
            checkbox.setChecked(key in included_features)

    def update_checkbox_availability(self, enable: bool = False) -> None:
        """Activate/deactivate the checkbox for the intensity feature
        Args:
            enable [bool]: value for the checkbox.setEnabled
        """

        self.enable_intensity = enable
        self.measurement_checkboxes["intensity"].setEnabled(enable)

    def update(self, ndims: int = 3, seg: bool = False) -> None:
        """Update the feature names according to the dimensions of the data, and activate/
        deactivate them according to whether a segmentation is provided (required for
        regionprops features)

        Args:
            ndims [int]: number of dimensions of the data, to set the display name of the
            features.
            seg [bool]: whether a segmentation is provided, to enable/disable the
            checkbox.
        """

        # if the dimensions have changed, create new feature classes with the new dims.
        if ndims != self.ndims:
            self.feature_classes = [
                Area(ndim=ndims),
                Intensity(ndim=ndims),
                EllipsoidAxes(ndim=ndims),
                Circularity(ndim=ndims),
                Perimeter(ndim=ndims),
            ]

        # set the display name and enable/disable the checkboxes.
        for feature in self.feature_classes:
            checkbox = self.measurement_checkboxes[feature.key]
            checkbox.setText(feature.display_name)
            checkbox.setEnabled(seg)
            if feature.key == "intensity":
                checkbox.setEnabled(seg and self.enable_intensity)
