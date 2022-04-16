import PyQt6.QtWidgets as qwt
from enums import DATA_TYPE

from gui.widgets.common import HWidget


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


class DataTypeRadioButtons(qwt.QWidget):
    """Simple widget containing data types on which some process regarding MRI
    GAN can be done, e.g. if train is selected and we are currently extracting
    landmarks, only on train dataset landmarks will be extacted.
    """

    def __init__(self) -> None:
        super().__init__()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = qwt.QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        ext_buttons_row = HWidget()
        layout.addWidget(ext_buttons_row)
        ext_buttons_row.setMaximumWidth(200)
        ext_buttons_row.layout().setContentsMargins(0, 0, 0, 0)
        self.data_btn_bg = qwt.QButtonGroup(ext_buttons_row)
        for idx, dt in enumerate(DATA_TYPE):
            btn = qwt.QRadioButton(dt.value)
            # set train as checked button
            if idx == 0:
                btn.setChecked(True)
            self.data_btn_bg.addButton(btn)
            ext_buttons_row.layout().addWidget(btn)

    @property
    def selected_data_type(self) -> DATA_TYPE:
        for but in self.data_btn_bg.buttons():
            if but.isChecked():
                return DATA_TYPE[but.text().upper()]
