import PyQt6.QtWidgets as qwt

from gui.widgets.common import HWidget
from gui.widgets.utils import set_color_on_widget


class NumOfInstancesRow(qwt.QWidget):
    """Simple widget which contains input for the number of instances of the
    workers that will be spawned once some job starts.
    """

    def __init__(self) -> None:
        super().__init__()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        num_of_instances_row = HWidget()
        layout.addWidget(num_of_instances_row)
        num_of_instances_row.layout().setContentsMargins(0, 0, 0, 0)
        num_of_instances_row.setMaximumWidth(200)
        num_of_instances_row.layout().addWidget(qwt.QLabel(
            text='number or instances'
        ))
        self.num_of_instances_input = qwt.QLineEdit()
        num_of_instances_row.layout().addWidget(self.num_of_instances_input)
        self.num_of_instances_input.setText(str(2))

    @property
    def num_of_instances_value(self) -> str:
        return self.num_of_instances_input.text()
