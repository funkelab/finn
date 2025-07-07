import os
from typing import Any

from funtracks.features._base import Feature, FeatureType
from funtracks.features.edge_features import edge_features
from funtracks.features.measurement_features import measurement_features
from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

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
        mappable_columns: list[str],
        data_type: str = "segmentation",
    ):
        super().__init__()

        self.node_feature_instances = [
            feature_cls(ndim=ndim) for feature_cls in measurement_features
        ]
        self.edge_feature_instances = [
            feature_cls() for feature_cls in edge_features if feature_cls().computed
        ]
        self.data_type = data_type
        self.ndim = ndim
        self.features_layout = QVBoxLayout()
        self.features_layout.addWidget(QLabel("Node Features"))
        self.edge_features_layout = QVBoxLayout()
        self.edge_features_layout.addWidget(QLabel("Edge Features"))
        self.edge_features_layout.setContentsMargins(40, 0, 0, 0)
        self.is_valid = True

        self.show_mapping = bool(mappable_columns)
        self.measurement_checkboxes = {}
        self.measurement_comboboxes = {}
        self.edge_measurement_checkboxes = {}

        # For each feature:
        for feature in self.node_feature_instances:
            row_layout = QHBoxLayout()
            checkbox = QCheckBox(feature.display_name)
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

        for feature in self.edge_feature_instances:
            row_layout = QHBoxLayout()
            checkbox = QCheckBox(feature.display_name)
            checkbox.setChecked(False)
            row_layout.addWidget(checkbox)
            self.edge_measurement_checkboxes[feature.attr_name] = checkbox
            checkbox.setEnabled(self.data_type == "segmentation")
            self.edge_features_layout.addLayout(row_layout)

        self.extra_features_layout = QVBoxLayout()
        self.extra_features = []  # List of (QLineEdit, QComboBox)
        self.mappable_columns = list(mappable_columns)

        # Add "+" button for extra features
        self.add_feature_btn = QPushButton("+")
        self.add_feature_btn.setFixedSize(20, 20)
        self.add_feature_btn.clicked.connect(self._add_extra_feature)
        self.add_feature_btn.setEnabled(self._can_add_extra_feature())

        self.features_layout.addLayout(self.extra_features_layout)
        self.features_layout.addWidget(self.add_feature_btn)

        main_layout = QHBoxLayout()
        main_layout.addLayout(self.features_layout)
        main_layout.addLayout(self.edge_features_layout)
        main_layout.addStretch()
        main_layout.setAlignment(self.features_layout, Qt.AlignTop)
        main_layout.setAlignment(self.edge_features_layout, Qt.AlignTop)

        self.setLayout(main_layout)

    def _update_features(
        self,
        ndim: int,
        mappable_columns: list[str],
        include_intensity: bool = False,
        data_type: str = "segmentation",
    ):
        """Updates the feature widgets according to the data provided by the user."""

        self.show_mapping = bool(mappable_columns)  # if a csv is present
        self.mappable_columns = list(mappable_columns)
        self.ndim = ndim
        self.node_feature_instances = [
            feature_cls(ndim=ndim) for feature_cls in measurement_features
        ]
        self.edge_feature_instances = [
            feature_cls() for feature_cls in edge_features if feature_cls().computed
        ]
        self.data_type = data_type
        enable_features = self.data_type == "segmentation" or (
            self.data_type == "points" and bool(mappable_columns)
        )  # Measuring node features is not possible without a segmentation. When
        # providing tracking data from csv, we do allow mapping a column to a readonly
        # feature.

        if not mappable_columns:
            self._clear_extra_features()

        self.add_feature_btn.setEnabled(self._can_add_extra_feature())

        for feature in self.node_feature_instances:
            attr_name = feature.attr_name
            # Update checkbox label if needed
            if attr_name in self.measurement_checkboxes:
                checkbox = self.measurement_checkboxes[attr_name]
                checkbox.setText(feature.display_name)
                checkbox.setEnabled(enable_features)
                # Enable/disable intensity checkbox if needed
                if attr_name == "intensity":
                    checkbox.setEnabled(enable_features and include_intensity)

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
                combobox.setEnabled(enable_features)
                if attr_name == "intensity":
                    combobox.setEnabled(include_intensity and enable_features)

        for feature in self.edge_feature_instances:
            attr_name = feature.attr_name
            # Update checkbox label if needed
            if attr_name in self.edge_measurement_checkboxes:
                checkbox = self.edge_measurement_checkboxes[attr_name]
                checkbox.setText(feature.display_name)
                # Enable/disable checkbox if needed
                checkbox.setEnabled(True)  # distance features should always be allowed
                if attr_name == "iou":
                    checkbox.setEnabled(data_type == "segmentation")  # iou only if data
                    # type is 'segmentation'.

    def _can_add_extra_feature(self) -> bool:
        """Returns True if there are any unmapped csv columns available that the user
        could select, else returns False."""

        # Only enable if there are unmapped columns and no combo is set to None
        used_columns = {
            combo.currentText()
            for _, combo, *_ in self.extra_features
            if combo.currentText() != "None"
        }
        available = [col for col in self.mappable_columns if col not in used_columns]
        return bool(available)

    def _clear_extra_features(self) -> None:
        """Clears all the optional extra feature widgets and layouts"""

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

    def _add_extra_feature(self) -> None:
        """Add a widget allowing the user to select an extra, optional readonly feature
        from a dropdown menu listing the available csv columns."""

        # Find the columns that are available to map (should be numerical)
        used_columns = {combo.currentText() for _, combo, *_ in self.extra_features}
        available = [col for col in self.mappable_columns if col not in used_columns]

        # return early if nothing was found.
        if not available:
            return

        # Creata line edit allowing the user to select a name for this feature.
        feature_name_edit = QLineEdit(available[0])

        # Create a combobox to select a column
        combo = QComboBox()
        combo.addItem("None")
        for col in self.mappable_columns:
            if col not in used_columns:
                combo.addItem(col)
        combo.setCurrentText(available[0])

        def on_combo_changed():
            self.add_feature_btn.setEnabled(self._can_add_extra_feature())
            feature_name_edit.setText(combo.currentText())

        # Connect to signals, line edit cannot have a name that is already in use for any
        # of the default supported features. Combobox updates should check
        feature_name_edit.textChanged.connect(self.validate_unique_feature_names)
        combo.currentTextChanged.connect(on_combo_changed)

        # Add a delete button to remove the extra feature from the list
        delete_icon = QColoredSVGIcon.from_resources("delete").colored("white")
        delete_btn = QPushButton(icon=delete_icon)
        delete_btn.setToolTip("Remove this feature")
        delete_btn.setFixedSize(20, 20)

        # Combine everythig in a layout
        row_layout = QHBoxLayout()
        row_layout.addWidget(feature_name_edit)
        row_layout.addWidget(combo)
        row_layout.addWidget(delete_btn)
        self.extra_features_layout.addLayout(row_layout)
        self.extra_features.append((feature_name_edit, combo, row_layout, delete_btn))

        # Implement removing a widget and layout when the delete button is clicked.
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

        # Enable/Disable the + button
        self.add_feature_btn.setEnabled(self._can_add_extra_feature())

    def validate_unique_feature_names(self) -> None:
        """Collect all attr_names used and check for duplicates, make the widget invalid
        if any are found."""

        base_names = {feature.attr_name for feature in self.node_feature_instances}
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
        for feature in self.node_feature_instances:
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

            # features can only be computed when a segmentation is provided. If not, make
            # the feature a readonly feature.
            feature.computed = self.data_type == "segmentation"
            features.append(
                {"feature": feature, "include": include, "from_column": from_column}
            )

        for feature in self.edge_feature_instances:
            attr_name = feature.attr_name
            checkbox = self.edge_measurement_checkboxes[attr_name]
            include = checkbox.isChecked() and checkbox.isEnabled()
            from_column = None
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

        self.page1 = page1
        self.page2 = page2
        self.include_intensity = self.page2.get_path() is not None
        self.page3 = page3
        self.page4 = page4
        self.page4.dim_updated.connect(self.update_features)
        self.page5 = page5
        self.page5.mapping_updated.connect(self.update_features)

        self.is_valid = True

        # Collapsible help widget
        instructions = QLabel(
            "<qt><i>Please select the features that you would like to measure. Node "
            "features, such as size, shape, or intensity apply to point detections "
            "without any segmentation. Note that some features, such as sphericity "
            "and surface area are expensive and therefore slow to compute. If you are "
            "loading existing tracking data for which you already have measured "
            "features, you can select a column in the csv file instead of recomputing "
            "the feature. If you want to view additional features that are not in the "
            "list, you can add them with the + button, but note that it will not be "
            "possible to recompute these features when you update the segmentation of "
            "a node or if you make new nodes.</i></qt>"
        )
        instructions.setWordWrap(True)
        collapsible_widget = QCollapsible("Explanation")
        collapsible_widget.layout().setContentsMargins(0, 0, 0, 0)
        collapsible_widget.layout().setSpacing(0)
        collapsible_widget.addWidget(instructions)
        collapsible_widget.collapse(animate=False)

        # Label with additional information when the user selected points as datatype
        choice = self.page1.get_choice()
        if choice == "curate_tracks":
            self.points_label = QLabel(
                "<i>Measuring features is only supported when you "
                "provide segmentation data. You can load existing features from "
                "CSV for viewing purposes only.</i>"
            )
        else:
            self.points_label = QLabel(
                "<i>Measuring features is only supported when you "
                "provide segmentation data.</i>"
            )

        self.points_label.setWordWrap(True)
        self.points_label.setVisible(self.page3.data_type == "points")

        # Add widget to select features
        ndim = 4 if self.page4.incl_z else 3
        mappable_columns = (
            self.page5.get_unmapped_columns(numerical_only=True)
            if choice == "curate_tracks"
            else []
        )

        self.feature_widget = FeatureWidget(
            ndim=ndim,
            mappable_columns=mappable_columns,
            data_type=self.page3.data_type,
        )

        self.feature_widget.validity_changed.connect(self.validate)

        # Create a content widget for the scroll area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.addWidget(self.points_label)
        content_layout.addWidget(QLabel("Choose which features to include"))
        content_layout.addWidget(self.feature_widget)

        # Wrap in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)

        # Create a group box and its own layout
        box = QGroupBox("Node and Edge Features")
        box_layout = QVBoxLayout()
        box_layout.addWidget(collapsible_widget)
        box_layout.addWidget(scroll_area)
        box.setLayout(box_layout)

        # Set the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def update_features(self):
        """Update the features based on the data and its dimensions"""

        choice = self.page1.get_choice()
        if choice == "curate_tracks":
            self.points_label.setText(
                "<i>Measuring features is only supported when you "
                "provide segmentation data.<br><br>You can load existing features from "
                "CSV for viewing purposes only.</i>"
            )
        else:
            self.points_label.setText(
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

        ndim = 4 if self.page4.incl_z else 3
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
