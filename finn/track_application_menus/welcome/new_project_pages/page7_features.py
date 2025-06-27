import os
from typing import Any

from funtracks.features._base import Feature
from funtracks.features.measurement_features import featureset
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from finn.track_application_menus.welcome.new_project_pages.page1_goals import Page1
from finn.track_application_menus.welcome.new_project_pages.page2_raw_data import Page2
from finn.track_application_menus.welcome.new_project_pages.page3_seg_data import Page3
from finn.track_application_menus.welcome.new_project_pages.page4_dimensions import Page4
from finn.track_application_menus.welcome.new_project_pages.page5_csv_import import Page5


class FeatureWidget(QWidget):
    """Widget allowing the user to choose which features to include"""

    def __init__(
        self,
        ndim: int,
        mappable_columns: list[str] | None = [],
        data_type: str = "segmentation",
    ):
        super().__init__()

        self.feature_instances = [feature_cls(ndim=ndim) for feature_cls in featureset]

        self.features_layout = QVBoxLayout()

        self.show_mapping = bool(mappable_columns)
        self.choose_column_label = QLabel("Choose from column")
        self.features_layout.addWidget(self.choose_column_label)
        self.choose_column_label.setVisible(self.show_mapping)

        self.measurement_checkboxes = {}
        self.measurement_comboboxes = {}

        # For each feature:
        for feature in self.feature_instances:
            row_layout = QHBoxLayout()
            checkbox = QCheckBox(feature.value_names)
            checkbox.setChecked(False)
            row_layout.addWidget(checkbox)
            self.measurement_checkboxes[feature.attr_name] = checkbox
            checkbox.setEnabled(data_type == "segmentation")

            combobox = QComboBox()
            combobox.addItem("None")
            combobox.setVisible(self.show_mapping)
            combobox.setEnabled(data_type == "segmentation")

            row_layout.addWidget(combobox)
            self.measurement_comboboxes[feature.attr_name] = combobox

            self.features_layout.addLayout(row_layout)

        self.setLayout(self.features_layout)

    def _update_features(
        self,
        ndim: int,
        mappable_columns: list[str] | None = [],
        include_intensity: bool = False,
        data_type: str = "segmentation",
    ):
        self.show_mapping = bool(mappable_columns)
        self.choose_column_label.setVisible(self.show_mapping)
        self.feature_instances = [feature_cls(ndim=ndim) for feature_cls in featureset]

        for feature in self.feature_instances:
            attr_name = feature.attr_name
            # Update checkbox label if needed
            if attr_name in self.measurement_checkboxes:
                checkbox = self.measurement_checkboxes[attr_name]
                checkbox.setText(feature.value_names)
                # Enable/disable intensity checkbox if needed
                if attr_name == "intensity":
                    checkbox.setEnabled(include_intensity)
                checkbox.setEnabled(data_type == "segmentation")

            # Update combobox visibility and options
            if attr_name in self.measurement_comboboxes:
                combobox = self.measurement_comboboxes[attr_name]
                combobox.setVisible(self.show_mapping)
                if self.show_mapping:
                    # Save current selection
                    current = combobox.currentText()
                    combobox.blockSignals(True)
                    combobox.clear()
                    combobox.addItem("None")
                    combobox.addItems(mappable_columns)
                    # Restore selection if still available, else set to "None"
                    if current in mappable_columns:
                        combobox.setCurrentText(current)
                    else:
                        combobox.setCurrentText("None")
                    combobox.blockSignals(False)
                else:
                    combobox.setCurrentText("None")
                # Enable/disable intensity combobox if needed
                if attr_name == "intensity":
                    combobox.setEnabled(include_intensity)
                combobox.setEnabled(data_type == "segmentation")

    def get_selected_features(self) -> list[dict[str : Feature | str | bool]]:
        """For each feature, check whether to include it, and optionally if it should be
        imported from an existing column.
        Returns:
            dict[str: str | bool]: dictionary with feature name, boolean to include it,
            and which column to take it from.
        """
        features = []
        for feature in self.feature_instances:
            attr_name = feature.attr_name
            checkbox = self.measurement_checkboxes[attr_name]
            include = checkbox.isChecked() and checkbox.isEnabled()
            from_column = None
            if (
                hasattr(self, "measurement_comboboxes")
                and attr_name in self.measurement_comboboxes
            ):
                combobox = self.measurement_comboboxes[attr_name]
                if combobox.isVisible() and combobox.isEnabled():
                    from_column = combobox.currentText()
                    if from_column == "None":
                        from_column = None

            features.append(
                {"feature": feature, "include": include, "from_column": from_column}
            )

        return features


class Page7(QWidget):
    """Page 7 of the project dialog, to enter features to measure or import"""

    def __init__(
        self, page1: Page1, page2: Page2, page3: Page3, page4: Page4, page5: Page5
    ):
        super().__init__()

        self.page1 = page1
        self.page1.choice_updated.connect(self.update_features)
        mappable_columns = (
            self.page5.get_unmapped_columns()
            if self.page1.get_choice() == "curate_tracks"
            else []
        )
        self.page2 = page2
        self.page2.validity_changed.connect(self.update_features)
        self.include_intensity = True if self.page2.get_path() is not None else False
        self.page3 = page3
        data_type = self.page3.data_type
        self.page3.type_updated.connect(self.update_features)
        self.page4 = page4
        self.page4.dim_updated.connect(self.update_features)
        self.page5 = page5
        self.page5.mapping_updated.connect(self.update_features)

        if self.page4.incl_z:
            ndim = 4
        else:
            ndim = 3

        self.feature_widget = FeatureWidget(
            ndim=ndim,
            mappable_columns=mappable_columns,
            data_type=data_type,
        )

        # wrap in a group box
        layout = QVBoxLayout(self)
        feature_group_box = QGroupBox("Node Features")
        feature_group_layout = QVBoxLayout(feature_group_box)
        feature_group_layout.addWidget(QLabel("Choose which features to include"))
        feature_group_layout.addWidget(self.feature_widget)
        feature_group_box.setLayout(feature_group_layout)
        layout.addWidget(feature_group_box)
        self.setLayout(layout)

    def update_features(self):
        """Update the features based on the selected dimensions"""

        if self.page1.get_choice() != "curate_tracks":
            mappable_columns = []
        else:
            mappable_columns = self.page5.get_unmapped_columns()
        if self.page2.get_path() is None or not os.path.exists(self.page2.get_path()):
            self.include_intensity = False
        else:
            self.include_intensity = True

        if self.page4.incl_z:
            ndim = 4
        else:
            ndim = 3

        self.feature_widget._update_features(
            ndim,
            mappable_columns=mappable_columns,
            include_intensity=self.include_intensity,
        )

    def get_settings(self) -> dict[str:Any]:
        """Get the features selected by the user in the FeatureWidget"""

        return {"features": self.feature_widget.get_selected_features()}
