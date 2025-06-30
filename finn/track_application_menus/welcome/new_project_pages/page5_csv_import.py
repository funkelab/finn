from psygnal import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from finn.track_application_menus.csv_widget import CSVWidget
from finn.track_application_menus.welcome.new_project_pages.page3_seg_data import Page3
from finn.track_application_menus.welcome.new_project_pages.page4_dimensions import Page4


class Page5(QWidget):
    """Page5, to import external tracks from CSV"""

    validity_changed = Signal()
    mapping_updated = Signal()

    def __init__(self, page3: Page3, page4: Page4):
        super().__init__()
        self.is_valid = False
        self.page3 = page3
        self.page4 = page4
        self.page3.validity_changed.connect(self._update_has_segmentation)
        self.page4.dim_updated.connect(self._update_incl_z)

        self.incl_z = self.page4.incl_z
        self.has_segmentation = self.page3.data_type == "segmentation"

        # CSVWidget for tracks
        self.csv_widget = CSVWidget(
            add_segmentation=self.has_segmentation, incl_z=self.incl_z
        )
        self.csv_widget.validity_changed.connect(self.validate)
        self.csv_widget.mapping_updated.connect(self._emit_mapping_updated)

        # wrap everything in a group
        import_data_group = QGroupBox("Import external tracking data")
        self.csv_layout = QVBoxLayout()
        self.csv_layout.addWidget(self.csv_widget)
        import_data_group.setLayout(self.csv_layout)
        import_data_layout = QVBoxLayout()
        import_data_layout.addWidget(import_data_group)
        self.setLayout(import_data_layout)

    def _emit_mapping_updated(self):
        """Emit a signal when the mapping is updated in the CSVWidget."""

        self.mapping_updated.emit()

    def _update_has_segmentation(self):
        """Update whether the user has segmentation data based on the selection in page3."""

        if self.page3.data_type == "segmentation":
            self.has_segmentation = True
        else:
            self.has_segmentation = False

        self.csv_widget.update_field_map(seg=self.has_segmentation, incl_z=self.incl_z)

    def _update_incl_z(self):
        """Update whether the user wants to include z-dimension based on the selection in"
        " page4."""

        self.incl_z = self.page4.incl_z
        self.csv_widget.update_field_map(seg=self.has_segmentation, incl_z=self.incl_z)

    def validate(self) -> None:
        """Check whether all required information was filled out and then emit a signal"""

        self.is_valid = self.csv_widget.is_valid
        self.validity_changed.emit()

    def _get_tracks_path(self) -> str:
        """Get the path to the CSV file containing the tracks"""
        return self.csv_widget.get_path()

    def _get_mapping(self) -> str:
        """Get the mapping from feature name to csv field name"""
        if self.csv_widget.csv_field_widget is not None:
            # Return the mapping from feature name to csv field name
            return self.csv_widget.csv_field_widget.get_name_map()
        return None

    def get_unmapped_columns(self, numerical_only: bool = False) -> list[str]:
        """Get the columns that were not mapped to any feature
        args:
            numerical_only (bool, default False): whether to return only columns with a
             numerical dtype
        """

        if self.csv_widget is not None:
            return self.csv_widget.get_unmapped_columns(numerical_only)
        return []

    def get_settings(self) -> dict[str:str]:
        """Get the settings, CSV path and mapping, from this widget."""

        return {
            "tracks_path": self._get_tracks_path(),
            "column_mapping": self._get_mapping(),
        }
