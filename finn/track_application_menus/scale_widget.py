from psygnal import Signal
from qtpy.QtWidgets import QDoubleSpinBox, QGroupBox, QLabel, QVBoxLayout, QWidget


class ScaleWidget(QWidget):
    """Voxel dimension widget

    Lets the user specify voxel dimensions using double spin boxes
    """

    scaling_updated = Signal()

    def __init__(self, scaling: tuple):
        super().__init__()

        self.scaling = scaling

        if len(self.scaling) not in (3, 4):
            raise ValueError("Dimensions must be 3 (tyx) or 4 (tzyx)")

        main_layout = QVBoxLayout()
        voxel_dimension_box = QGroupBox("Voxel dimensions")
        voxel_dimension_layout = QVBoxLayout()

        self.x_spin = QDoubleSpinBox()
        self.y_spin = QDoubleSpinBox()
        self.x_spin.setMinimum(0.01)
        self.x_spin.setSingleStep(0.01)
        self.x_spin.setDecimals(2)
        self.x_spin.setKeyboardTracking(False)
        self.y_spin.setMinimum(0.01)
        self.y_spin.setSingleStep(0.01)
        self.y_spin.setDecimals(2)
        self.y_spin.setKeyboardTracking(False)

        self.x_spin.setValue(self.scaling[2])
        self.y_spin.setValue(self.scaling[1])

        self.x_spin.valueChanged.connect(self.emit_scaling_updated)
        self.y_spin.valueChanged.connect(self.emit_scaling_updated)

        voxel_dimension_layout.addWidget(QLabel("Y (µm):"))
        voxel_dimension_layout.addWidget(self.y_spin)
        voxel_dimension_layout.addWidget(QLabel("X (µm):"))
        voxel_dimension_layout.addWidget(self.x_spin)

        self.z_spin = QDoubleSpinBox()
        self.z_spin.setMinimum(0.01)
        self.z_spin.setSingleStep(0.01)
        self.z_spin.setDecimals(2)
        self.z_spin.setKeyboardTracking(False)
        self.z_label = QLabel("Z (µm):")
        voxel_dimension_layout.addWidget(self.z_label)
        voxel_dimension_layout.addWidget(self.z_spin)
        self.z_spin.setVisible(len(self.scaling) == 4)
        self.z_label.setVisible(len(self.scaling) == 4)
        if len(self.scaling) == 4:
            self.z_spin.setValue(self.scaling[3])
        self.z_spin.valueChanged.connect(self.emit_scaling_updated)

        voxel_dimension_box.setLayout(voxel_dimension_layout)
        main_layout.addWidget(voxel_dimension_box)
        self.setLayout(main_layout)

    def get_scaling(self) -> tuple[float]:
        """Return the scaling information in the spinboxes"""

        if len(self.scaling) == 3:
            self.scaling = (1, self.y_spin.value(), self.x_spin.value())

        if len(self.scaling) == 4:
            self.scaling = (
                1,
                self.z_spin.value(),
                self.y_spin.value(),
                self.x_spin.value(),
            )

        return self.scaling

    def update_scaling(self, scale: tuple[float]) -> None:
        """Update the scaling information in the widget"""

        self.x_spin.setValue(scale[-1])
        self.y_spin.setValue(scale[-2])

        if len(scale) == 4:
            self.z_spin.setVisible(True)
            self.z_spin.setValue(scale[-3])
            self.z_label.setVisible(True)
        else:
            self.z_spin.setVisible(False)
            self.z_label.setVisible(False)

        self.scaling = scale

    def emit_scaling_updated(self) -> None:
        """Emit the scaling updated signal"""

        self.scaling_updated.emit()
