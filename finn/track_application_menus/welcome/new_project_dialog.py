from typing import Any

from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from finn.track_application_menus.welcome.create_project import (
    DialogValueError,
    create_project,
)
from finn.track_application_menus.welcome.new_project_pages import (
    Page1,
    Page2,
    Page3,
    Page4,
    Page5,
    Page6,
    Page7,
    Page8,
)


class NewProjectDialog(QDialog):
    """Dialog to create a new project"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.resize(600, 500)
        self.stacked = QStackedWidget(self)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stacked)

        # --- Page 1: Project goals --- #
        self.page1 = Page1()
        layout1 = QVBoxLayout()
        layout1.addWidget(self.page1)

        # Page 1: Add Next/Cancel button
        btn_layout1 = QHBoxLayout()
        btn_layout1.addStretch()
        self.cancel_btn1 = QPushButton("Cancel")
        self.next_btn1 = QPushButton("Next")
        self.next_btn1.setEnabled(True)
        btn_layout1.addWidget(self.cancel_btn1)
        btn_layout1.addWidget(self.next_btn1)
        layout1.addLayout(btn_layout1)

        page1_widget = QWidget()
        page1_widget.setLayout(layout1)
        self.stacked.addWidget(page1_widget)

        # --- Page 2: Intensity Data Selection --- #
        self.page2 = Page2(page1=self.page1)
        self.page2.validity_changed.connect(self._validate_page2)
        layout2 = QVBoxLayout()
        layout2.addWidget(self.page2)

        # Add Prev/Next/Cancel buttons
        btn_layout2 = QHBoxLayout()
        self.prev_btn1 = QPushButton("Previous")
        self.cancel_btn2 = QPushButton("Cancel")
        self.next_btn2 = QPushButton("Next")
        btn_layout2.addStretch()
        btn_layout2.addWidget(self.cancel_btn2)
        btn_layout2.addWidget(self.prev_btn1)
        btn_layout2.addWidget(self.next_btn2)
        layout2.addLayout(btn_layout2)

        page2_widget = QWidget()
        page2_widget.setLayout(layout2)
        self.stacked.addWidget(page2_widget)

        # --- Page 3: Detection data selection --- #
        self.page3 = Page3(page1=self.page1, page2=self.page2)
        self.page3.validity_changed.connect(self._validate_page3)

        layout3 = QVBoxLayout()
        layout3.addWidget(self.page3)

        # Add Prev/Next/Cancel buttons
        btn_layout3 = QHBoxLayout()
        self.prev_btn2 = QPushButton("Previous")
        self.cancel_btn3 = QPushButton("Cancel")
        self.next_btn3 = QPushButton("Next")
        btn_layout3.addStretch()
        btn_layout3.addWidget(self.cancel_btn3)
        btn_layout3.addWidget(self.prev_btn2)
        btn_layout3.addWidget(self.next_btn3)
        layout3.addLayout(btn_layout3)

        page3_widget = QWidget()
        page3_widget.setLayout(layout3)
        self.stacked.addWidget(page3_widget)

        # --- Page 4: Image data dimensions --- #
        self.page4 = Page4(self.page2, self.page3)
        self.page4.validity_changed.connect(self._validate_page4)

        layout4 = QVBoxLayout()
        layout4.addWidget(self.page4)

        # Add Prev/Next/Cancel buttons
        btn_layout4 = QHBoxLayout()
        self.prev_btn3 = QPushButton("Previous")
        self.next_btn4 = QPushButton("Next")
        self.cancel_btn4 = QPushButton("Cancel")

        btn_layout4.addStretch()
        btn_layout4.addWidget(self.cancel_btn4)
        btn_layout4.addWidget(self.prev_btn3)
        btn_layout4.addWidget(self.next_btn4)
        layout4.addLayout(btn_layout4)

        page4_widget = QWidget()
        page4_widget.setLayout(layout4)
        self.stacked.addWidget(page4_widget)

        # --- Page 5: Importing external tracks from csv --- #
        self.page5 = Page5(self.page3, self.page4)
        self.page5.validity_changed.connect(self._validate_page5)

        layout5 = QVBoxLayout()
        layout5.addWidget(self.page5)

        # Add Prev/Next/Cancel buttons
        btn_layout5 = QHBoxLayout()
        self.prev_btn4 = QPushButton("Previous")
        self.next_btn5 = QPushButton("Next")
        self.cancel_btn5 = QPushButton("Cancel")

        btn_layout5.addStretch()
        btn_layout5.addWidget(self.cancel_btn5)
        btn_layout5.addWidget(self.prev_btn4)
        btn_layout5.addWidget(self.next_btn5)
        layout5.addLayout(btn_layout5)

        page5_widget = QWidget()
        page5_widget.setLayout(layout5)
        self.stacked.addWidget(page5_widget)

        # --- Page 6: Project and candidate graph parameters --- #
        self.page6 = Page6()

        layout6 = QVBoxLayout()
        layout6.addWidget(self.page6)

        # Add Prev/Next/Cancel buttons
        btn_layout6 = QHBoxLayout()
        self.prev_btn5 = QPushButton("Previous")
        self.next_btn6 = QPushButton("Next")
        self.cancel_btn6 = QPushButton("Cancel")

        btn_layout6.addStretch()
        btn_layout6.addWidget(self.cancel_btn6)
        btn_layout6.addWidget(self.prev_btn5)
        btn_layout6.addWidget(self.next_btn6)
        layout6.addLayout(btn_layout6)

        page6_widget = QWidget()
        page6_widget.setLayout(layout6)
        self.stacked.addWidget(page6_widget)

        # --- Page 7: Features to measure --- #
        self.page7 = Page7(
            page1=self.page1,
            page2=self.page2,
            page3=self.page3,
            page4=self.page4,
            page5=self.page5,
        )
        self.page7.validity_changed.connect(self._validate_page7)
        layout7 = QVBoxLayout()
        layout7.addWidget(self.page7)

        # Add Prev/Next/Cancel buttons
        btn_layout7 = QHBoxLayout()
        self.prev_btn6 = QPushButton("Previous")
        self.next_btn7 = QPushButton("Next")
        self.cancel_btn7 = QPushButton("Cancel")

        btn_layout7.addStretch()
        btn_layout7.addWidget(self.cancel_btn7)
        btn_layout7.addWidget(self.prev_btn6)
        btn_layout7.addWidget(self.next_btn7)
        layout7.addLayout(btn_layout7)

        page7_widget = QWidget()
        page7_widget.setLayout(layout7)

        self.stacked.addWidget(page7_widget)

        # --- Page 8: Save Project --- #
        self.page8 = Page8()
        layout8 = QVBoxLayout()
        layout8.addWidget(self.page8)
        self.page8.validity_changed.connect(self._validate_page8)

        # Add Prev/Ok/Cancel buttons
        btn_layout8 = QHBoxLayout()
        self.prev_btn7 = QPushButton("Previous")
        self.ok_btn = QPushButton("OK")
        self.cancel_btn8 = QPushButton("Cancel")

        btn_layout8.addStretch()
        btn_layout8.addWidget(self.cancel_btn8)
        btn_layout8.addWidget(self.prev_btn7)
        btn_layout8.addWidget(self.ok_btn)
        layout8.addLayout(btn_layout8)

        page8_widget = QWidget()
        page8_widget.setLayout(layout8)
        self.stacked.addWidget(page8_widget)

        # --- Navigation --- #
        self.next_btn1.clicked.connect(self._go_to_page2)
        self.next_btn2.clicked.connect(self._go_to_page3)
        self.next_btn3.clicked.connect(self._go_to_page4)
        self.next_btn4.clicked.connect(self._go_to_page5_or_6)
        self.next_btn5.clicked.connect(lambda: self.stacked.setCurrentIndex(5))
        self.next_btn6.clicked.connect(self._go_to_page7)
        self.next_btn7.clicked.connect(self._go_to_page8)

        self.prev_btn1.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.prev_btn2.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        self.prev_btn3.clicked.connect(lambda: self.stacked.setCurrentIndex(2))
        self.prev_btn4.clicked.connect(lambda: self.stacked.setCurrentIndex(3))
        self.prev_btn5.clicked.connect(self._go_to_page4_or_5)
        self.prev_btn6.clicked.connect(lambda: self.stacked.setCurrentIndex(5))
        self.prev_btn7.clicked.connect(lambda: self.stacked.setCurrentIndex(6))

        self.ok_btn.clicked.connect(self._on_ok_clicked)

        # Connect cancel buttons to close the dialog
        self.cancel_btn1.clicked.connect(self._cancel)
        self.cancel_btn2.clicked.connect(self._cancel)
        self.cancel_btn3.clicked.connect(self._cancel)
        self.cancel_btn4.clicked.connect(self._cancel)
        self.cancel_btn5.clicked.connect(self._cancel)
        self.cancel_btn6.clicked.connect(self._cancel)
        self.cancel_btn7.clicked.connect(self._cancel)
        self.cancel_btn8.clicked.connect(self._cancel)

        self.stacked.setCurrentIndex(0)

    def _validate_page2(self):
        """Validate inputs on page 2 and enable/disable the NEXT button to page3."""
        self.next_btn2.setEnabled(self.page2.is_valid)

    def _validate_page3(self):
        """Validate inputs on page 3 and enable/disable the NEXT button to page4."""
        self.next_btn3.setEnabled(self.page3.is_valid)

    def _validate_page4(self):
        """Validate inputs on page 4 and enable/disable the NEXT button to page5."""
        self.next_btn4.setEnabled(self.page4.is_valid)

    def _validate_page5(self):
        """Validate inputs on page 5 and enable/disable the NEXT button to page6."""
        self.next_btn5.setEnabled(self.page5.is_valid)

    def _validate_page7(self):
        """Validate inputs on page 7 and enable/disable the OK button."""
        self.next_btn7.setEnabled(self.page7.is_valid)

    def _validate_page8(self):
        """Validate inputs on page 8 and enable/disable the OK button."""
        self.ok_btn.setEnabled(self.page8.is_valid)

    def _go_to_page2(self):
        """Go to page 2 and validate it to enable/disable the NEXT button."""
        self.stacked.setCurrentIndex(1)
        self.page2.validate()

    def _go_to_page3(self):
        """Go to page 3 and validate it to enable/disable the NEXT button."""
        self.stacked.setCurrentIndex(2)
        self.page3.validate()

    def _go_to_page4(self):
        """Go to page 4 and validate it to enable/disable the NEXT button."""
        self.stacked.setCurrentIndex(3)
        self.page4.update_table()

    def _go_to_page4_or_5(self):
        """Go to page 5 to track from scratch, otherwise go to page 4."""
        if self.page1.get_choice() == "curate_tracks":
            self.stacked.setCurrentIndex(4)
        else:
            self.stacked.setCurrentIndex(3)

    def _go_to_page5_or_6(self):
        """Go to page 5 to use external tracks, otherwise go to page 6."""
        if self.page1.get_choice() == "curate_tracks":
            self.stacked.setCurrentIndex(4)
            self.page5.validate()
        else:
            self.stacked.setCurrentIndex(5)

    def _go_to_page7(self):
        """Go to page 7 and make sure the mapping on page5 is updated first."""
        self.page5.mapping_updated.emit()  # ensure the mapping is updated
        self.stacked.setCurrentIndex(6)

    def _go_to_page8(self):
        """Go to page 8 and validate it to update the button"""
        self.page8.validate()
        self.stacked.setCurrentIndex(7)

    def _on_ok_clicked(self):
        """Collect all information entered by the user and try to build a project. Throws
        an error if invalid/incompatible information was entered."""
        try:
            project_info = self._get_project_info()
            self.project = create_project(project_info)
        except DialogValueError as e:
            if e.show_dialog:
                QMessageBox.warning(self, "Error", str(e))
            return
        self.accept()

    def _get_project_info(self) -> dict[str:Any]:
        """Collect the information from the different dialog pages.
        Returns:
            dict[str: Any] with information from the different pages as follows:
            Page 3:
                - data_type [str]: either 'segmentation' or 'points'
            Page 4:
                - intensity_image [da.Array | None] : intensity data
                - segmentation_image [da.Array | None] : segmentation data
                - points_data [pd.DataFrame]: point detection data
                - ndim [int]: the number of dimensions (incl time) of the data (3, 4, or
                    5)
                - axes [dict]:
                    dimensions [tuple[str]]: dimension names (e.g. 'time', 'z')
                    raw_indices [tuple[int]]: index of each dimension in the raw data
                    seg_indices [tuple[int]]: index of each dimension in the seg data
                    axis_names [tuple(str)]: dimension names assigned by the user
                    units (tuple[str]): units for each dimension, e.g. 'µm'
                    scaling [tuple(float)]: spatial calibration in the same order as the
                        dimensions
            Page 5:
                - tracks_path [str | None]: path to where the tracking data csv file is
                    stored (if provided)
                - tracks_mapping [dict[str: str] | None] : mapping of the csv column headers to
                    the required tracking information (dimensions, ids)
            Page 6:
                - project_params [ProjectParams]: parameters for the project
                - cand_graph_params [CandGraphParams]: parameters for the candidate graph
            Page 7:
                - features (list[dict[str: str|bool]]): list of features to measure,
                    each with 'feature_name', 'include' (bool), and 'from_column'
                    (str or None)
            Page 8:
                - title [str]: name of the project,
                - directory [str]: path to directory where the project should be saved
        """

        choice = self.page1.get_choice()

        project_info = {
            "tracks_path": None,
            "column_mapping": None,
        }

        project_info = project_info | self.page3.get_settings()
        project_info = project_info | self.page4.get_settings()
        if choice == "curate_tracks":
            project_info = project_info | self.page5.get_settings()
        project_info = project_info | self.page6.get_settings()
        project_info = project_info | self.page7.get_settings()
        project_info = project_info | self.page8.get_settings()

        return project_info

    def _cancel(self):
        """Reject the dialog, go back to welcome menu"""
        self.reject()
