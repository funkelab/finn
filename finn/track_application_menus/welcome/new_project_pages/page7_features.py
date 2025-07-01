import os
from typing import Any

from funtracks.features._base import Feature, FeatureType
from funtracks.features.measurement_features import featureset
from psygnal import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from finn._qt.qt_resources import QColoredSVGIcon
from finn.track_application_menus.welcome.new_project_pages.page1_goals import Page1
from finn.track_application_menus.welcome.new_project_pages.page2_raw_data import Page2
from finn.track_application_menus.welcome.new_project_pages.page3_seg_data import Page3
from finn.track_application_menus.welcome.new_project_pages.page4_dimensions import Page4
from finn.track_application_menus.welcome.new_project_pages.page5_csv_import import Page5


class FeatureWidget(QWidget):
    """Widget allowing the user to choose which features to include"""

    validity_changed = Signal()

    def __init__(
        self,
        ndim: int,
        mappable_columns: list[str] | None = [],
        data_type: str = "segmentation",
    ):
        super().__init__()

        self.feature_instances = [feature_cls(ndim=ndim) for feature_cls in featureset]
        self.data_type = data_type
        self.ndim = ndim
        self.features_layout = QVBoxLayout()
        self.is_valid = True

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
            checkbox.setEnabled(self.data_type == "segmentation")

            combobox = QComboBox()
            combobox.addItem("None")
            combobox.setVisible(self.show_mapping)
            combobox.setEnabled(self.data_type == "segmentation")

            row_layout.addWidget(combobox)
            self.measurement_comboboxes[feature.attr_name] = combobox

            self.features_layout.addLayout(row_layout)

        self.extra_features_layout = QVBoxLayout()
        self.extra_features = []  # List of (QLineEdit, QComboBox)
        self.mappable_columns = list(mappable_columns)  # Save for later use

        # Add "+" button for extra features
        self.add_feature_btn = QPushButton("+")
        self.add_feature_btn.setFixedSize(20, 20)
        self.add_feature_btn.clicked.connect(self.add_extra_feature)
        self.add_feature_btn.setEnabled(self._can_add_extra_feature())

        self.features_layout.addLayout(self.extra_features_layout)
        self.features_layout.addWidget(self.add_feature_btn)

        self.setLayout(self.features_layout)

    def _update_features(
        self,
        ndim: int,
        mappable_columns: list[str] | None = [],
        include_intensity: bool = False,
        data_type: str = "segmentation",
    ):
        self.show_mapping = bool(mappable_columns)
        self.mappable_columns = list(mappable_columns)  # Save for later use
        self.choose_column_label.setVisible(self.show_mapping)
        self.ndim = ndim
        self.feature_instances = [feature_cls(ndim=ndim) for feature_cls in featureset]
        self.data_type = data_type
        enable_features = self.data_type == "segmentation" or (
            self.data_type == "points" and bool(mappable_columns)
        )

        if not mappable_columns:
            self.clear_extra_features()

        self.add_feature_btn.setEnabled(self._can_add_extra_feature())

        for feature in self.feature_instances:
            attr_name = feature.attr_name
            # Update checkbox label if needed
            if attr_name in self.measurement_checkboxes:
                checkbox = self.measurement_checkboxes[attr_name]
                checkbox.setText(feature.value_names)
                # Enable/disable intensity checkbox if needed
                checkbox.setEnabled(enable_features)
                if attr_name == "intensity":
                    checkbox.setEnabled(include_intensity)

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
                combobox.setEnabled(enable_features)
                # Enable/disable intensity combobox if needed
                if attr_name == "intensity":
                    combobox.setEnabled(include_intensity)

    def _can_add_extra_feature(self):
        # Only enable if there are unmapped columns and no combo is set to None
        used_columns = {
            combo.currentText()
            for _, combo, *_ in self.extra_features
            if combo.currentText() != "None"
        }
        available = [col for col in self.mappable_columns if col not in used_columns]
        return bool(available)

    def clear_extra_features(self):
        # Remove all extra feature widgets and layouts
        for edit, combo, row_layout, delete_btn in self.extra_features:
            edit.deleteLater()
            combo.deleteLater()
            delete_btn.deleteLater()
            while row_layout.count():
                item = row_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            self.extra_features_layout.removeItem(row_layout)
        self.extra_features = []

    def add_extra_feature(self):
        # Find the columns that are available to map (should be numerical)
        used_columns = {combo.currentText() for _, combo, *_ in self.extra_features}
        available = [col for col in self.mappable_columns if col not in used_columns]

        if not available:
            return

        feature_name_edit = QLineEdit(available[0])
        combo = QComboBox()
        combo.addItem("None")
        for col in self.mappable_columns:
            if col not in used_columns:
                combo.addItem(col)
        combo.setCurrentText(available[0])

        def on_combo_changed():
            self.add_feature_btn.setEnabled(self._can_add_extra_feature())
            feature_name_edit.setText(combo.currentText())

        feature_name_edit.textChanged.connect(self.validate_unique_feature_names)
        combo.currentTextChanged.connect(on_combo_changed)

        # Delete button
        delete_icon = QColoredSVGIcon.from_resources("delete").colored("white")
        delete_btn = QPushButton(icon=delete_icon)
        delete_btn.setToolTip("Remove this feature")
        delete_btn.setFixedSize(20, 20)

        row_layout = QHBoxLayout()
        row_layout.addWidget(feature_name_edit)
        row_layout.addWidget(combo)
        row_layout.addWidget(delete_btn)
        self.extra_features_layout.addLayout(row_layout)
        self.extra_features.append((feature_name_edit, combo, row_layout, delete_btn))

        def remove_feature():
            feature_name_edit.deleteLater()
            combo.deleteLater()
            delete_btn.deleteLater()
            while row_layout.count():
                item = row_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            self.extra_features_layout.removeItem(row_layout)
            self.extra_features = [
                ef for ef in self.extra_features if ef[0] != feature_name_edit
            ]
            self.add_feature_btn.setEnabled(self._can_add_extra_feature())

        delete_btn.clicked.connect(remove_feature)

        self.add_feature_btn.setEnabled(self._can_add_extra_feature())

    def validate_unique_feature_names(self):
        # Collect all attr_names used and check for duplicates
        base_names = {feature.attr_name for feature in self.feature_instances}
        extra_names = [edit.text().strip() for edit, *_ in self.extra_features]
        all_names = list(base_names) + extra_names
        duplicates = {name for name in extra_names if all_names.count(name) > 1 and name}
        # change color of text edit when a duplicate name is used
        for edit, *_ in self.extra_features:
            if edit.text().strip() in duplicates:
                edit.setStyleSheet("background-color: #c62828;")  # light red
            else:
                edit.setStyleSheet("")

        # send a signal to notify parent widget.
        if bool(duplicates):
            self.is_valid = False
        else:
            self.is_valid = True

        self.validity_changed.emit()

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
                if combobox.isVisibleTo(self) and combobox.isEnabled():
                    from_column = combobox.currentText()
                    if from_column == "None":
                        from_column = None

            # features can only be computed when a segmentation is provided.
            feature.computed = self.data_type == "segmentation"
            features.append(
                {"feature": feature, "include": include, "from_column": from_column}
            )

        for edit, combo, *_ in self.extra_features:
            col = combo.currentText()
            if col != "None":
                features.append(
                    {
                        "feature": Feature(
                            attr_name=edit.text(),
                            value_names=edit.text(),
                            feature_type=FeatureType.NODE.value,
                            valid_ndim=(self.ndim,),
                            computed=False,
                            required=False,
                        ),
                        "include": True,
                        "from_column": col,
                    }
                )

        return features


class Page7(QWidget):
    """Page 7 of the project dialog, to enter features to measure or import"""

    validity_changed = Signal()

    def __init__(
        self, page1: Page1, page2: Page2, page3: Page3, page4: Page4, page5: Page5
    ):
        super().__init__()

        self.is_valid = True
        self.page1 = page1
        self.page1.choice_updated.connect(self.update_features)
        choice = self.page1.get_choice()

        mappable_columns = (
            self.page5.get_unmapped_columns(numerical_only=True)
            if choice == "curate_tracks"
            else []
        )

        if choice == "curate_tracks":
            self.points_label = QLabel(
                "<i>Measuring features is only supported when you "
                "provide segmentation data.<br><br>.You can load existing features from CSV for "
                "viewing purposes only.</i>"
            )
        else:
            self.points_label = QLabel(
                "<i>Measuring features is only supported when you "
                "provide segmentation data.</i>"
            )

        self.page2 = page2
        self.page2.validity_changed.connect(self.update_features)
        self.include_intensity = True if self.page2.get_path() is not None else False
        self.page3 = page3
        data_type = self.page3.data_type
        self.page3.validity_changed.connect(self.update_features)
        self.page4 = page4
        self.page4.dim_updated.connect(self.update_features)
        self.page5 = page5
        self.page5.mapping_updated.connect(self.update_features)

        self.points_label.setVisible(self.page3.data_type == "points")

        if self.page4.incl_z:
            ndim = 4
        else:
            ndim = 3

        self.feature_widget = FeatureWidget(
            ndim=ndim,
            mappable_columns=mappable_columns,
            data_type=data_type,
        )
        self.feature_widget.validity_changed.connect(self.validate)

        # wrap in a group box
        layout = QVBoxLayout(self)
        layout.addWidget(self.points_label)
        feature_group_box = QGroupBox("Node Features")
        feature_group_layout = QVBoxLayout(feature_group_box)
        feature_group_layout.addWidget(QLabel("Choose which features to include"))
        feature_group_layout.addWidget(self.feature_widget)
        feature_group_box.setLayout(feature_group_layout)
        layout.addWidget(feature_group_box)
        self.setLayout(layout)

    def update_features(self):
        """Update the features based on the selected dimensions"""

        choice = self.page1.get_choice()
        if choice == "curate_tracks":
            self.points_label = QLabel(
                "<i>Measuring features is only supported when you "
                "provide segmentation data.<br><br>.You can load existing features from CSV for "
                "viewing purposes only.</i>"
            )
        else:
            self.points_label = QLabel(
                "<i>Measuring features is only supported when you "
                "provide segmentation data.</i>"
            )
        self.points_label.setVisible(self.page3.data_type == "points")

        mappable_columns = (
            self.page5.get_unmapped_columns(numerical_only=True)
            if choice == "curate_tracks"
            else []
        )

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
            data_type=self.page3.data_type,
        )

    def get_settings(self) -> dict[str:Any]:
        """Get the features selected by the user in the FeatureWidget"""

        return {"features": self.feature_widget.get_selected_features()}

    def validate(self) -> None:
        """Check if the feature widget is valid"""

        if self.feature_widget.is_valid:
            self.is_valid = True
        else:
            self.is_valid = False

        self.validity_changed.emit()
