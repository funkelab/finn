from typing import Any

from funtracks.params._base import Params
from funtracks.params.cand_graph_params import CandGraphParams
from funtracks.params.project_params import ProjectParams
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible


class ParamsWidget(QWidget):
    """Widget for entering solver parameters"""

    def __init__(self, params: Params, group_title: str):
        super().__init__()

        layout = QVBoxLayout(self)

        # Solver parameters
        self.solver_params_group = QGroupBox(group_title)
        solver_layout = QVBoxLayout(self.solver_params_group)

        self.params = params()
        for name, field in self.params.model_fields.items():
            value = getattr(self.params, name)
            if isinstance(value, bool):
                checkbox = QCheckBox(field.title)
                checkbox.setChecked(value)
                checkbox.param_name = name
                checkbox.toggled.connect(self._on_param_changed)
                checkbox.setToolTip(field.description)
                solver_layout.addWidget(checkbox)
            elif isinstance(value, int):
                spinbox = QSpinBox()
                spinbox.setValue(value)
                spinbox.setMinimum(0)
                spinbox.setToolTip(field.description)
                spinbox.param_name = name
                spinbox.valueChanged.connect(self._on_param_changed)
                solver_layout.addWidget(QLabel(field.title + ":"))
                solver_layout.addWidget(spinbox)
            elif isinstance(value, float):
                double_spinbox = QDoubleSpinBox()
                double_spinbox.setValue(value)
                double_spinbox.setSingleStep(0.01)
                double_spinbox.setMinimum(0.0)
                double_spinbox.setToolTip(field.description)
                double_spinbox.param_name = name
                double_spinbox.valueChanged.connect(self._on_param_changed)
                solver_layout.addWidget(QLabel(field.title + ":"))
                solver_layout.addWidget(double_spinbox)

        self.solver_params_group.setLayout(solver_layout)
        layout.addWidget(self.solver_params_group)

    def _on_param_changed(self):
        """Update the Params object when a parameter is changed"""

        sender = self.sender()
        if hasattr(sender, "param_name"):
            value = None
            if isinstance(sender, QCheckBox):
                value = sender.isChecked()
            elif isinstance(sender, QSpinBox | QDoubleSpinBox):
                value = sender.value()
            if value is not None:
                setattr(self.params, sender.param_name, value)

    def get_param_values(self) -> Params:
        """Return a Params object with the parameter values entered by the user"""

        return self.params


class Page6(QWidget):
    """Page 6 of the Project dialog, to enter project parameters"""

    def __init__(self):
        super().__init__()

        # Collapsible help widget
        instructions = QLabel(
            "<qt><i>Please specify the parameters that apply to your "
            "data. You can still change this afterwards.</i></qt>"
        )
        instructions.setWordWrap(True)
        collapsible_widget = QCollapsible("Explanation")
        collapsible_widget.layout().setContentsMargins(0, 0, 0, 0)
        collapsible_widget.layout().setSpacing(0)
        collapsible_widget.addWidget(instructions)
        collapsible_widget.collapse(animate=False)

        # Project params widget
        self.project_params_widget = ParamsWidget(ProjectParams, "Project Parameters")

        # Project Cand graph widget
        self.cand_graph_params_widget = ParamsWidget(
            CandGraphParams, "Candidate Graph Parameters"
        )

        # Create a content widget for the scroll area
        params_widgets = QWidget()
        params_layout = QVBoxLayout(params_widgets)
        params_layout.addWidget(self.project_params_widget)
        params_layout.addWidget(self.cand_graph_params_widget)

        # Wrap in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(params_widgets)

        # Wrap in a group box
        box = QGroupBox("Project and Candidate Graph Parameters")
        box_layout = QVBoxLayout()
        box_layout.addWidget(collapsible_widget)
        box_layout.addWidget(scroll_area)
        box.setLayout(box_layout)

        # Set the box to the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def get_settings(self) -> dict[str:Any]:
        """Get the Param settings entered by the user"""

        settings = {
            "project_params": self.project_params_widget.get_param_values(),
            "cand_graph_params": self.cand_graph_params_widget.get_param_values(),
        }
        return settings
