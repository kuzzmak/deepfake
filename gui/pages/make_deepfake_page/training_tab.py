import logging

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt
import torch
from torchvision import transforms
from torch.nn import MSELoss

from common_structures import TensorCommObject
from core.dataset.configuration import DatasetConfiguration
from core.model.configuration import ModelConfiguration
from core.optimizer.configuration import OptimizerConfiguration
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
        self.batch_size_input.setText(str(32))
        batch_size_row.layout().addWidget(self.batch_size_input)
        models_gb_layout.addWidget(batch_size_row)

        epochs_row = HWidget()
        epochs_row.layout().addWidget(qwt.QLabel(text='Number of epochs: '))
        self.epochs_input = qwt.QLineEdit()
        self.epochs_input.setText(str(10))
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
        betas_row.layout().addWidget(qwt.QLabel(text="beta 1"))
        self.adam_beta1_input = qwt.QLineEdit()
        self.adam_beta1_input.setText(str(0.5))
        betas_row.layout().addWidget(self.adam_beta1_input)
        betas_row.layout().addWidget(qwt.QLabel(text="beta 2"))
        self.adam_beta2_input = qwt.QLineEdit()
        self.adam_beta2_input.setText(str(0.999))
        betas_row.layout().addWidget(self.adam_beta2_input)
        self.adam_options.layout().addWidget(betas_row)
        # eps
        eps_row = HWidget()
        eps_row.layout().addWidget(qwt.QLabel(text='eps'))
        self.adam_eps_input = qwt.QLineEdit()
        self.adam_eps_input.setText(str(1e-8))
        eps_row.layout().addWidget(self.adam_eps_input)
        self.adam_options.layout().addWidget(eps_row)
        # weight_decay
        weight_decay_row = HWidget()
        weight_decay_row.layout().addWidget(qwt.QLabel(text='weight decay'))
        self.adam_weight_decay_input = qwt.QLineEdit()
        self.adam_weight_decay_input.setText(str(0))
        weight_decay_row.layout().addWidget(self.adam_weight_decay_input)
        self.adam_options.layout().addWidget(weight_decay_row)

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

    @property
    def batch_size(self) -> str:
        """How many examples will be processed in one step.

        Returns:
            str: value from input
        """
        return self.batch_size_input.text()

    @property
    def epochs(self) -> str:
        """How many epoch training process will last.

        Returns:
            str: value from input
        """
        return self.epochs_input.text()

    @property
    def selected_optimizer(self) -> OPTIMIZER:
        """Currently selected optimizer.

        Returns:
            OPTIMIZER: optimizer enum
        """
        return self.optimizer_selection.currentData()

    @property
    def adam_beta1(self) -> str:
        """First beta constant for Adam optimizer.

        Returns:
            str: value from input
        """
        return self.adam_beta1_input.text()

    @property
    def adam_beta2(self) -> str:
        """Second beta constant for Adam optimizer.

        Returns:
            str: value from input
        """
        return self.adam_beta2_input.text()

    @property
    def adam_eps(self) -> str:
        """Constant for numerical stability for Adam optimizer.

        Returns:
            str: value from input
        """
        return self.adam_eps_input.text()

    @property
    def adam_weight_decay(self) -> str:
        """Constant for weight decay for Adam optimizer.

        Returns:
            str: value from input
        """
        return self.adam_weight_decay_input.text()

    @property
    def optimizer_learning_rate(self) -> str:
        """Learning rate for the currently selected optimizer.

        Returns:
            str: value from input
        """
        optim = self.selected_optimizer.value.lower()
        optim_lr_input: qwt.QLineEdit = getattr(self, f'{optim}_lr_input')
        text = optim_lr_input.text()
        return text


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

    def _optimizer_options(self) -> dict:
        opts = {}
        lr = self.training_conf.optimizer_learning_rate
        opts['lr'] = float(lr)
        if self.training_conf.selected_optimizer == OPTIMIZER.ADAM:
            beta1 = float(self.training_conf.adam_beta1)
            beta2 = float(self.training_conf.adam_beta2)
            opts['betas'] = (beta1, beta2)
            eps = float(self.training_conf.adam_eps)
            opts['eps'] = eps
            weight_decay = float(self.training_conf.adam_weight_decay)
            opts['weight_decay'] = weight_decay

        return opts

    def _start(self):
        optimizer_conf = OptimizerConfiguration(
            self.training_conf.selected_optimizer,
            self._optimizer_options(),
        )
        device = DEVICE.CPU
        if torch.cuda.is_available():
            device = DEVICE.CUDA

        input_shape = (3, 128, 128)

        model_conf = ModelConfiguration(MODEL.ORIGINAL)

        data_transforms = transforms.Compose([transforms.ToTensor()])
        dataset_conf = DatasetConfiguration(
            # faces_path=r'C:\Users\tonkec\Documents\deepfake\data\gen_faces\metadata',
            metadata_path_A=r'C:\Users\kuzmi\Documents\deepfake\data\face_A\metadata_sorted',
            metadata_path_B=r'C:\Users\kuzmi\Documents\deepfake\data\face_B\metadata',
            input_shape=input_shape[1],
            batch_size=int(self.training_conf.batch_size),
            load_into_memory=True,
            data_transforms=data_transforms,
        )

        comm_obj = TensorCommObject()
        comm_obj.data_sig = self.preview.refresh_data_sig
        preview_conf = PreviewConfiguration(True, comm_obj)

        conf = TrainerConfiguration(
            device=device,
            input_shape=input_shape,
            epochs=int(self.training_conf.epochs),
            criterion=MSELoss(),
            model_conf=model_conf,
            optimizer_conf=optimizer_conf,
            dataset_conf=dataset_conf,
            preview_conf=preview_conf,
        )

        self.thread = qtc.QThread()
        self.worker = Worker(conf)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _stop(self):
        ...
