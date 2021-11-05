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
from gui.widgets.common import HWidget, VWidget
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
        original.setChecked(True)
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
        self.setLayout(layout)

        model_selector = ModelSelector()
        layout.addWidget(model_selector)
        model_selector.layout().setContentsMargins(0, 0, 0, 0)

        models_gb = qwt.QGroupBox()
        layout.addWidget(models_gb)
        models_gb.setTitle('Training configuration')
        models_gb_layout = qwt.QVBoxLayout(models_gb)

        self.input_A_directory_btn = qwt.QPushButton(text='Input A directory')
        models_gb_layout.addWidget(self.input_A_directory_btn)
        self.input_A_directory_btn.setToolTip('Not yet selected.')
        self.input_A_directory_btn.clicked.connect(
            lambda: self._select_input_directory('A'))

        self.input_B_directory_btn = qwt.QPushButton(text='Input B directory')
        models_gb_layout.addWidget(self.input_B_directory_btn)
        self.input_B_directory_btn.setToolTip('Not yet selected.')
        self.input_B_directory_btn.clicked.connect(
            lambda: self._select_input_directory('B'))

        batch_size_row = HWidget()
        models_gb_layout.addWidget(batch_size_row)
        batch_size_row.layout().addWidget(qwt.QLabel(text="Batch size: "))
        self.batch_size_input = qwt.QLineEdit()
        self.batch_size_input.setText(str(32))
        batch_size_row.layout().addWidget(self.batch_size_input)

        epochs_row = HWidget()
        models_gb_layout.addWidget(epochs_row)
        epochs_row.layout().addWidget(qwt.QLabel(text='Number of epochs: '))
        self.epochs_input = qwt.QLineEdit()
        self.epochs_input.setText(str(10))
        epochs_row.layout().addWidget(self.epochs_input)

        load_data_into_memory_row = HWidget()
        models_gb_layout.addWidget(load_data_into_memory_row)
        self.ldim_chk = qwt.QCheckBox(
            text='load datasets into memory (RAM or GPU)'
        )
        self.ldim_chk.setChecked(True)
        load_data_into_memory_row.layout().addWidget(self.ldim_chk)

        optimizer_gb = qwt.QGroupBox()
        layout.addWidget(optimizer_gb)
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
        self.adam_options.layout().addWidget(lr_row)
        lr_row.layout().addWidget(qwt.QLabel(text='learning rate'))
        self.adam_lr_input = qwt.QLineEdit()
        self.adam_lr_input.setText(str(5e-5))
        lr_row.layout().addWidget(self.adam_lr_input)
        # betas
        betas_row = HWidget()
        self.adam_options.layout().addWidget(betas_row)
        betas_row.layout().addWidget(qwt.QLabel(text="beta 1"))
        self.adam_beta1_input = qwt.QLineEdit()
        self.adam_beta1_input.setText(str(0.5))
        betas_row.layout().addWidget(self.adam_beta1_input)
        betas_row.layout().addWidget(qwt.QLabel(text="beta 2"))
        self.adam_beta2_input = qwt.QLineEdit()
        self.adam_beta2_input.setText(str(0.999))
        betas_row.layout().addWidget(self.adam_beta2_input)
        # eps
        eps_row = HWidget()
        self.adam_options.layout().addWidget(eps_row)
        eps_row.layout().addWidget(qwt.QLabel(text='eps'))
        self.adam_eps_input = qwt.QLineEdit()
        self.adam_eps_input.setText(str(1e-8))
        eps_row.layout().addWidget(self.adam_eps_input)
        # weight_decay
        weight_decay_row = HWidget()
        self.adam_options.layout().addWidget(weight_decay_row)
        weight_decay_row.layout().addWidget(qwt.QLabel(text='weight decay'))
        self.adam_weight_decay_input = qwt.QLineEdit()
        self.adam_weight_decay_input.setText(str(0))
        weight_decay_row.layout().addWidget(self.adam_weight_decay_input)

        ####################
        # IMAGE AUGMENTATION
        ####################
        image_augmentation_gb = qwt.QGroupBox()
        layout.addWidget(image_augmentation_gb)
        image_augmentation_gb.setTitle('Image augmentations')
        image_augmentation_gb_layout = qwt.QVBoxLayout(image_augmentation_gb)

        self.flip_image_chk = qwt.QCheckBox(text='flip')
        image_augmentation_gb_layout.addWidget(self.flip_image_chk)

        self.sharpen_chk = qwt.QCheckBox(text='sharpen')
        image_augmentation_gb_layout.addWidget(self.sharpen_chk)

        add_light_row = HWidget()
        image_augmentation_gb_layout.addWidget(add_light_row)
        add_light_row.layout().setContentsMargins(0, 0, 0, 0)
        self.add_light_chk = qwt.QCheckBox(text='add light')
        add_light_row.layout().addWidget(self.add_light_chk)
        self.add_light_input = qwt.QLineEdit()
        add_light_row.layout().addWidget(self.add_light_input)

        add_saturation_row = HWidget()
        image_augmentation_gb_layout.addWidget(add_saturation_row)
        add_saturation_row.layout().setContentsMargins(0, 0, 0, 0)
        self.add_saturation_chk = qwt.QCheckBox(text='add saturation')
        add_saturation_row.layout().addWidget(self.add_saturation_chk)
        self.add_saturation_input = qwt.QLineEdit()
        add_saturation_row.layout().addWidget(self.add_saturation_input)

        add_gaussian_blur_row = HWidget()
        image_augmentation_gb_layout.addWidget(add_gaussian_blur_row)
        add_gaussian_blur_row.layout().setContentsMargins(0, 0, 0, 0)
        self.add_gaussian_blur_chk = qwt.QCheckBox(text='add gaussian blur')
        add_gaussian_blur_row.layout().addWidget(self.add_gaussian_blur_chk)
        self.add_gaussian_blur_input = qwt.QLineEdit()
        add_gaussian_blur_row.layout().addWidget(self.add_gaussian_blur_input)

        add_bilateral_blur_row = HWidget()
        image_augmentation_gb_layout.addWidget(add_bilateral_blur_row)
        add_bilateral_blur_row.layout().setContentsMargins(0, 0, 0, 0)
        self.add_bilateral_blur_chk = qwt.QCheckBox(text='add bilateral blur')
        add_bilateral_blur_row.layout().addWidget(self.add_bilateral_blur_chk)
        add_bilateral_blur_row.layout().addWidget(qwt.QLabel(text='d'))
        self.add_bilateral_blur_d_input = qwt.QLineEdit()
        self.add_bilateral_blur_d_input.setMaximumWidth(25)
        add_bilateral_blur_row.layout().addWidget(
            self.add_bilateral_blur_d_input
        )
        add_bilateral_blur_row.layout().addWidget(qwt.QLabel(text='color'))
        self.add_bilateral_blur_color_input = qwt.QLineEdit()
        self.add_bilateral_blur_color_input.setMaximumWidth(25)
        add_bilateral_blur_row.layout().addWidget(
            self.add_bilateral_blur_color_input
        )
        add_bilateral_blur_row.layout().addWidget(qwt.QLabel(text='space'))
        self.add_bilateral_blur_space_input = qwt.QLineEdit()
        self.add_bilateral_blur_space_input.setMaximumWidth(25)
        add_bilateral_blur_row.layout().addWidget(
            self.add_bilateral_blur_space_input
        )

        erode_row = HWidget()
        image_augmentation_gb_layout.addWidget(erode_row)
        erode_row.layout().setContentsMargins(0, 0, 0, 0)
        self.erode_chk = qwt.QCheckBox(text='erode')
        erode_row.layout().addWidget(self.erode_chk)
        erode_row.layout().addWidget(qwt.QLabel(text='kernel shape'))
        self.erode_input = qwt.QLineEdit()
        erode_row.layout().addWidget(self.erode_input)

        dilate_row = HWidget()
        image_augmentation_gb_layout.addWidget(dilate_row)
        dilate_row.layout().setContentsMargins(0, 0, 0, 0)
        self.dilate_chk = qwt.QCheckBox(text='dilate')
        dilate_row.layout().addWidget(self.dilate_chk)
        dilate_row.layout().addWidget(qwt.QLabel(text='kernel shape'))
        self.dilate_input = qwt.QLineEdit()
        dilate_row.layout().addWidget(self.dilate_input)

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
    def load_datasets_into_memory(self) -> bool:
        """Should datasets A and B be loaded into memory (RAM if no grphics
        card is available or GPU is it's available)

        Returns:
            bool: True if datasets should be loaded into memory, False
                otherwise
        """
        return self.ldim_chk.isChecked()

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

    stop_training_sig = qtc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = qwt.QHBoxLayout()

        left_part = qwt.QWidget()
        left_part.setMaximumWidth(350)
        left_part_layout = qwt.QVBoxLayout()
        left_part.setLayout(left_part_layout)

        scroll = qwt.QScrollArea()
        scroll.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        self.training_conf = TrainingConfiguration()
        scroll.setWidget(self.training_conf)
        left_part_layout.addWidget(scroll)

        button_row = qwt.QWidget()
        button_row_layout = qwt.QHBoxLayout()
        button_row.setLayout(button_row_layout)

        self.start_btn = qwt.QPushButton(text='Start')
        self.start_btn.clicked.connect(self._start)
        button_row_layout.addWidget(self.start_btn)

        self.stop_btn = qwt.QPushButton(text='Stop')
        self.stop_btn.clicked.connect(self._stop)
        self.enable_widget(self.stop_btn, True)
        button_row_layout.addWidget(self.stop_btn)

        left_part_layout.addWidget(button_row)
        layout.addWidget(left_part)

        self.preview = Preview(4)
        self.preview.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.preview)
        self.setLayout(layout)

    def _optimizer_options(self) -> dict:
        """Constructs optimizer options based on the type of optimizer that's
        selected.

        Returns:
            dict: optimizer options
        """
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
        """Initiates training process.
        """
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
            metadata_path_A=r'C:\Users\kuzmi\Documents\deepfake\data\face_A\metadata_sorted',
            metadata_path_B=r'C:\Users\kuzmi\Documents\deepfake\data\face_B\metadata',
            input_shape=input_shape[1],
            batch_size=int(self.training_conf.batch_size),
            load_into_memory=self.training_conf.load_datasets_into_memory,
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
        self.stop_training_sig.connect(self.worker.stop_training)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        self.thread.finished.connect(
            lambda: self.enable_widget(self.start_btn, True)
        )
        self.thread.finished.connect(
            lambda: self.enable_widget(self.stop_btn, False)
        )
        self.enable_widget(self.start_btn, False)
        self.enable_widget(self.stop_btn, True)

    def _stop(self):
        """Stops training process.
        """
        self.stop_training_sig.emit()
