import logging
from typing import Union

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt
import torch
from torchvision import transforms
from torch.nn import MSELoss

from common_structures import TensorCommObject
from core.dataset.configuration import DatasetConfiguration
from core.model.configuration import ModelConfiguration
from core.optimizer.configuration import DEFAULT_ADAM_CONF
from core.trainer.configuration import TrainerConfiguration
from enums import DEVICE, MODEL, OPTIMIZER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import HWidget, VWidget, VerticalSpacer
from gui.widgets.preview.configuration import PreviewConfiguration
from gui.widgets.preview.preview import Preview
from trainer_thread import Worker

logger = logging.getLogger(__name__)


class ModelSelector(qwt.QWidget):

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = qwt.QVBoxLayout()

        models_gb = qwt.QGroupBox()
        models_gb.setTitle('Available deepfake models')
        models_gb_layout = qwt.QVBoxLayout(models_gb)

        bg = qwt.QButtonGroup(models_gb)
        bg.idPressed.connect(self._model_changed)

        original = qwt.QRadioButton('Original', models_gb)
        models_gb_layout.addWidget(original)
        bg.addButton(original)

        layout.addWidget(models_gb)
        policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Minimum,
            qwt.QSizePolicy.Fixed,
        )
        self.setSizePolicy(policy)
        self.setLayout(layout)

    @qtc.pyqtSlot(int)
    def _model_changed(self, index: int) -> None:
        print('model index: ', index)


class TrainingConfiguration(qwt.QWidget):

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = qwt.QVBoxLayout()

        models_gb = qwt.QGroupBox()
        models_gb.setTitle('Training configuration')
        models_gb_layout = qwt.QVBoxLayout(models_gb)

        self.input_A_directory_btn = qwt.QPushButton(text='Input A directory')
        self.input_A_directory_btn.setToolTip('Not yet selected.')
        self.input_A_directory_btn.clicked.connect(
            lambda: self._select_input_directory('A'))
        models_gb_layout.addWidget(self.input_A_directory_btn)

        self.input_B_directory_btn = qwt.QPushButton(text='Input B directory')
        self.input_B_directory_btn.setToolTip('Not yet selected.')
        self.input_B_directory_btn.clicked.connect(
            lambda: self._select_input_directory('B'))
        models_gb_layout.addWidget(self.input_B_directory_btn)

        batch_size_row = HWidget()
        batch_size_row.layout().addWidget(qwt.QLabel(text="Batch size: "))
        self.batch_size_input = qwt.QLineEdit()
        batch_size_row.layout().addWidget(self.batch_size_input)
        models_gb_layout.addWidget(batch_size_row)

        epochs_row = HWidget()
        epochs_row.layout().addWidget(qwt.QLabel(text='Number of epochs: '))
        self.epochs_input = qwt.QLineEdit()
        epochs_row.layout().addWidget(self.epochs_input)
        models_gb_layout.addWidget(epochs_row)

        optimizer_gb = qwt.QGroupBox()
        optimizer_gb.setTitle('Optimizer configuration')
        optimizer_gb_layout = qwt.QVBoxLayout(optimizer_gb)

        self.optimizer_selection = qwt.QComboBox()
        self.optimizer_selection.addItem('Adam', OPTIMIZER.ADAM)
        optimizer_gb_layout.addWidget(self.optimizer_selection)

        self.optimizer_options = qwt.QStackedWidget()
        optimizer_gb_layout.addWidget(self.optimizer_options)

        ####################
        # ADAM CONFIGURATION
        ####################
        self.adam_options = VWidget()
        self.optimizer_options.addWidget(self.adam_options)
        # learning rate
        lr_row = HWidget()
        lr_row.layout().addWidget(qwt.QLabel(text='learning rate'))
        self.adam_lr_input = qwt.QLineEdit()
        self.adam_lr_input.setText(str(5e-5))
        lr_row.layout().addWidget(self.adam_lr_input)
        self.adam_options.layout().addWidget(lr_row)
        # betas
        betas_row = HWidget()
        betas_row.layout().addWidget(qwt.QLabel(text="Beta 1"))
        self.adam_beta1_input = qwt.QLineEdit()
        self.adam_beta1_input.setText(str(0.5))
        betas_row.layout().addWidget(self.adam_beta1_input)
        betas_row.layout().addWidget(qwt.QLabel(text="Beta 2"))
        self.adam_beta2_input = qwt.QLineEdit()
        self.adam_beta2_input.setText(str(0.999))
        betas_row.layout().addWidget(self.adam_beta2_input)
        self.adam_options.layout().addWidget(betas_row)

        layout.addWidget(models_gb)
        layout.addWidget(optimizer_gb)
        self.setLayout(layout)

    @qtc.pyqtSlot(str)
    def _select_input_directory(self, side: str):
        """Function for selecting input folder for side A or side B.

        Args:
            side (str): for which side folder is being selected
        """
        directory = qwt.QFileDialog.getExistingDirectory(
            self,
            'getExistingDirectory',
            './',
        )
        if not directory:
            logger.warning(f'No directory selected for side {side}.')
            return
        else:
            logger.info(
                f'Selected input directory ({directory}) for ' +
                f'side {side}.'
            )
            btn = getattr(self, f'input_{side}_directory_btn')
            btn.setToolTip(directory)

    def optimizer_learning_rate(self) -> Union[float, None]:
        optim = self.optimizer_selection.currentData().value.lower()
        optim_lr_input = getattr(self, f'{optim}_lr_input')
        text = optim_lr_input.text()
        try:
            val = float(text)
        except ValueError:
            logger.error(
                'Learning rate can not be something that is not a number.'
            )
            val = None
        return val


class TrainingTab(BaseWidget):

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = qwt.QHBoxLayout()

        left_part = qwt.QWidget()
        left_part.setMaximumWidth(300)
        # left_part.setAutoFillBackground(True)
        # p = left_part.palette()
        # p.setColor(left_part.backgroundRole(), qtc.Qt.red)
        # left_part.setPalette(p)
        left_part_layout = qwt.QVBoxLayout()
        left_part.setLayout(left_part_layout)

        model_selector = ModelSelector()
        left_part_layout.addWidget(model_selector)

        self.training_conf = TrainingConfiguration()
        left_part_layout.addWidget(self.training_conf)

        left_part_layout.addItem(VerticalSpacer)

        button_row = qwt.QWidget()
        button_row_layout = qwt.QHBoxLayout()
        button_row.setLayout(button_row_layout)

        start_btn = qwt.QPushButton(text='Start')
        start_btn.clicked.connect(self._start)
        button_row_layout.addWidget(start_btn)

        stop_btn = qwt.QPushButton(text='Stop')
        stop_btn.clicked.connect(self._stop)
        button_row_layout.addWidget(stop_btn)

        left_part_layout.addWidget(button_row)
        layout.addWidget(left_part)

        self.preview = Preview()
        layout.addWidget(self.preview)
        self.setLayout(layout)

    def _start(self):
        lr = self.training_conf.optimizer_learning_rate()
        if lr is None:
            return
        # print(self.adam_lr_input.text())
        # device = DEVICE.CPU
        # if torch.cuda.is_available():
        #     device = DEVICE.CUDA

        # input_shape = (3, 128, 128)

        # model_conf = ModelConfiguration(MODEL.ORIGINAL)

        # optimizer_conf = DEFAULT_ADAM_CONF
        # optimizer_conf.optimizer_args['lr'] = 5e-5
        # optimizer_conf.optimizer_args['betas'] = (0.5, 0.999)

        # data_transforms = transforms.Compose([transforms.ToTensor()])
        # dataset_conf = DatasetConfiguration(
        #     # faces_path=r'C:\Users\tonkec\Documents\deepfake\data\gen_faces\metadata',
        #     metadata_path=r'C:\Users\kuzmi\Documents\deepfake\data\gen_faces\temp',
        #     input_shape=input_shape[1],
        #     batch_size=32,
        #     load_into_memory=True,
        #     data_transforms=data_transforms,
        # )

        # comm_obj = TensorCommObject()
        # comm_obj.data_sig = self.preview.refresh_data_sig
        # preview_conf = PreviewConfiguration(True, comm_obj)

        # conf = TrainerConfiguration(
        #     device=device,
        #     input_shape=input_shape,
        #     epochs=10,
        #     criterion=MSELoss(),
        #     model_conf=model_conf,
        #     optimizer_conf=optimizer_conf,
        #     dataset_conf=dataset_conf,
        #     preview_conf=preview_conf,
        # )

        # self.thread = qtc.QThread()
        # self.worker = Worker(conf)
        # self.worker.moveToThread(self.thread)
        # self.thread.started.connect(self.worker.run)
        # self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)
        # self.thread.start()

    def _stop(self):
        ...
