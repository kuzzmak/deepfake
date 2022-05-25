from pathlib import Path
from typing import Any, List, Optional, Union

import cv2 as cv
import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from configs.mri_gan_config import MRIGANConfig
from enums import DATA_TYPE, LAYOUT, MRI_GAN_DATASET, WIDGET_TYPE
from gui.widgets.common import (
    Button,
    GroupBox,
    HWidget,
    HorizontalSpacer,
    NoMarginLayout,
    PlayIcon,
    VWidget,
)


class CustomInputRow(qwt.QWidget):

    def __init__(self, name: str, default_value: str = '') -> None:
        """Widget for adding another row in the `Step` widget.

        Parameters
        ----------
        name : str
            name of the property being added
        default_value : str, optional
            default value in the input field, by default ''
        """
        super().__init__()

        self._name = name
        self._default_value = default_value

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)
        row = HWidget()
        layout.addWidget(row)
        row.layout().setContentsMargins(0, 0, 0, 0)
        row.layout().addWidget(qwt.QLabel(text=self._name))
        self._input = qwt.QLineEdit()
        row.layout().addWidget(self._input)
        self._input.setText(self._default_value)

    @property
    def input_value(self) -> str:
        return self._input.text()


class NumOfInstancesRow(qwt.QWidget):
    """Simple widget which contains input for the number of instances of the
    workers that will be spawned once some job starts.
    """

    def __init__(self) -> None:
        super().__init__()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)
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


class Step(qwt.QWidget):

    def __init__(self, gb_name: str, start_btn_name: str) -> None:
        """Widget containing entry for number of instances which controlls how
        many processes or threads will be spawned when some job is starting,
        selector for what kind of data will be used: train, test, valid or
        everything, button for starting the job and button for configuring
        paths for the job. This four components are base, possible to add
        additional widgets and functionality from the outside.

        Parameters
        ----------
        gb_name : str
            name of the step
        start_btn_name : str
            text on the button which starts job
        """
        super().__init__()

        self._init_ui(gb_name, start_btn_name)

    def _init_ui(self, gb_name: str, start_btn_name) -> None:
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        gb = GroupBox(gb_name, LAYOUT.HORIZONTAL)
        layout.addWidget(gb)
        gb.setMaximumWidth(400)

        self.left_part = VWidget()
        gb.layout().addWidget(self.left_part)

        self._num_of_instances = NumOfInstancesRow()
        self.left_part.layout().addWidget(
            self._num_of_instances
        )

        self.radio_btns = DataTypeRadioButtons()
        self.left_part.layout().addWidget(
            self.radio_btns
        )

        right_part = VWidget()
        gb.layout().addWidget(right_part)

        self.start_btn = Button(start_btn_name)
        right_part.layout().addWidget(
            self.start_btn
        )
        self.start_btn.setIcon(PlayIcon())

        self.configure_paths_btn = Button('configure paths')
        right_part.layout().addWidget(
            self.configure_paths_btn
        )

    @property
    def selected_data_type(self) -> DATA_TYPE:
        return self.radio_btns.selected_data_type

    @property
    def num_of_instances(self) -> str:
        return self._num_of_instances.num_of_instances_value

    def add_field(self, name: str, default_value: str = '') -> None:
        attr_name = f'_custom_{name}'
        setattr(self, attr_name, CustomInputRow(name, default_value))
        self.left_part.layout().insertWidget(0, getattr(self, attr_name))


class GenerateFrameLabelsCSVStep(Step):
    """Step for generating frame labels CSV. Normal `Step` widget with the
    addition of the selector for MRI GAN dataset.
    """

    def __init__(self, gb_name: str, start_btn_name: str) -> None:
        super().__init__(gb_name, start_btn_name)

        self.__init_ui()

    def __init_ui(self) -> None:
        dataset_type_row = HWidget()
        dataset_type_row.layout().setContentsMargins(0, 0, 0, 0)
        self.dataset_btn_bg = qwt.QButtonGroup(dataset_type_row)
        for idx, dt in enumerate(MRI_GAN_DATASET):
            btn = qwt.QRadioButton(dt.value)
            if idx == 0:
                btn.setChecked(True)
            self.dataset_btn_bg.addButton(btn)
            dataset_type_row.layout().addWidget(btn)
        self.left_part.layout().insertWidget(0, dataset_type_row)

    @property
    def selected_mri_gan_dataset(self) -> MRI_GAN_DATASET:
        for but in self.dataset_btn_bg.buttons():
            if but.isChecked():
                return MRI_GAN_DATASET[but.text().upper()]


class Parameter(qwt.QWidget):
    """Widget for showing input or selection for some model parameter.

    Parameters
    ----------
    name : str
        name of the parameter in GUI
    default_values : Optional[List[Any]], optional
        put default values in input, by default None
    widget_type : WIDGET_TYPE, optional
        is this widget input widget or radio buttons, by default
            WIDGET_TYPE.INPUT
    """

    def __init__(
        self,
        name: str,
        default_values: Optional[List[Any]] = None,
        widget_type: WIDGET_TYPE = WIDGET_TYPE.INPUT,
    ) -> None:
        super().__init__()

        self._name = name
        self._default_values = default_values
        self._wt = widget_type

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout(LAYOUT.HORIZONTAL)
        self.setLayout(layout)
        if self._name:
            layout.addWidget(qwt.QLabel(text=self._name))
            layout.addItem(HorizontalSpacer())

        if self._wt == WIDGET_TYPE.INPUT:
            self._input = qwt.QLineEdit()
            self._input.setMaximumWidth(100)
            layout.addWidget(self._input)
            if self._default_values is None:
                return
            vals = self._default_values
            if len(vals) == 0:
                return
            self._input.setText(str(vals[0]))

        elif self._wt == WIDGET_TYPE.RADIO_BUTTON:
            if self._default_values is None:
                return
            vals = self._default_values
            if len(vals) == 0:
                return
            self.btn_bg = qwt.QButtonGroup(self)
            for idx, val in enumerate(vals):
                btn = qwt.QRadioButton(val)
                if idx == 0:
                    btn.setChecked(True)
                self.btn_bg.addButton(btn)
                layout.addWidget(btn)

    @property
    def value(self) -> Any:
        if self._wt == WIDGET_TYPE.INPUT:
            return self._input.text()
        elif self._wt == WIDGET_TYPE.RADIO_BUTTON:
            for but in self.btn_bg.buttons():
                if but.isChecked():
                    return but.text()
        return None


class DFDetectorParameter(Parameter):
    """Special `Parameter` widget for the train deepfake detector worker
    parameters.

    Parameters
    ----------
    name : str
        name of the parameter in GUI
    config_key : Optional[str], optional
        key for the parameter from `mri_gan_config.yaml` file,
            by default None
    default_values : Optional[List[Any]], optional
        put default values in input, by default None
    widget_type : WIDGET_TYPE, optional
        is this widget input widget or radio buttons, by default
            WIDGET_TYPE.INPUT
    """

    def __init__(
        self,
        name: str,
        config_key: Optional[str] = None,
        default_values: Optional[List[Any]] = None,
        widget_type: WIDGET_TYPE = WIDGET_TYPE.INPUT,
    ) -> None:
        super().__init__(name, default_values, widget_type)

        self._config_key = config_key

        self.__init_ui()

    def __init_ui(self):
        if self._wt != WIDGET_TYPE.INPUT:
            return
        if self._config_key is None:
            raise Exception(
                'Config key must be present when using input widget.'
            )
        self._input.setText(
            str(
                MRIGANConfig
                .get_instance()
                .get_deep_fake_training_params()[self._config_key]
            )
        )


class MRIGANParemeter(Parameter):
    """Special `Parameter` widget for the train MRI GAN worker parameters.

    Parameters
    ----------
    name : str
        name of the parameter in GUI
    config_key : Optional[str], optional
        key for the parameter from `mri_gan_config.yaml` file,
            by default None
    default_values : Optional[List[Any]], optional
        put default values in input, by default None
    widget_type : WIDGET_TYPE, optional
        is this widget input widget or radio buttons, by default
            WIDGET_TYPE.INPUT
    """

    def __init__(
        self,
        name: str,
        config_key: Optional[str] = None,
        default_values: Optional[List[Any]] = None,
        widget_type: WIDGET_TYPE = WIDGET_TYPE.INPUT,
    ) -> None:
        super().__init__(name, default_values, widget_type)

        self._config_key = config_key

        self.__init_ui()

    def __init_ui(self):
        if self._wt != WIDGET_TYPE.INPUT:
            return
        if self._config_key is None:
            raise Exception(
                'Config key must be present when using input widget.'
            )
        self._input.setText(
            str(
                MRIGANConfig
                .get_instance()
                .get_mri_gan_model_params()[self._config_key]
            )
        )


class DragAndDrop(qwt.QLabel):
    """Widget representing a drag and drop zone for images.
    """

    # image_path_sig = qtc.pyqtSignal(str)

    def __init__(self, text: str = ''):
        super().__init__()

        self._text = text
        self._file_path = None

        self._init_ui()
        # self.image_path_sig.connect(self._set_image)

    @property
    def file_path(self) -> Union[Path, None]:
        if self._file_path is not None:
            return Path(self._file_path)
        return None

    def _init_ui(self) -> None:
        self.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.setText(f'\n\n {self._text} \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')
        self.setAcceptDrops(True)

    def setPixmap(self, image: qtg.QPixmap):
        super().setPixmap(image)

    def dragEnterEvent(self, event: qtc.QEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event: qtc.QEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: qtc.QEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                self._open_video(url)

            # event.setDropAction(qtc.Qt.DropAction.CopyAction)
            # file_path = event.mimeData().urls()[0].toLocalFile()
            # self._set_image(file_path)
            event.accept()
        else:
            event.ignore()

    def _open_video(self, filename: qtc.QUrl) -> None:
        path = filename.toLocalFile()
        self._file_path = path
        cap = cv.VideoCapture(path)
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        # width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        # height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        ret, frame = cap.read()
        if not ret:
            return
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = qtg.QImage(
            frame, frame.shape[1],
            frame.shape[0],
            qtg.QImage.Format.Format_RGB888,
        )
        pix = qtg.QPixmap.fromImage(img)
        pix = pix.scaled(
            self.size(),
            qtc.Qt.AspectRatioMode.KeepAspectRatio,
            qtc.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)

    @qtc.pyqtSlot(str)
    def _set_image(self, file_path: str):
        pixmap = qtg.QPixmap(file_path)
        pixmap = pixmap.scaled(
            128,
            128,
            qtc.Qt.AspectRatioMode.KeepAspectRatio,
            qtc.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
