import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2 as cv
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt
import torch.nn as nn
from torchvision import transforms

from common_structures import TensorCommObject
from configs.app_config import APP_CONFIG
from core import loss
from core.dataset.configuration import DatasetConfiguration
from core.image.augmentation import ImageAugmentation
from core.model.configuration import ModelConfiguration
from core.optimizer.configuration import OptimizerConfiguration
from core.trainer.configuration import TrainerConfiguration
from core.worker import FSTrainerWorker, Worker
from core.worker.trainer_thread import TrainingWorker
from enums import (
    CONNECTION,
    DEVICE,
    INTERPOLATION,
    JOB_TYPE,
    MODEL,
    OPTIMIZER,
    SIGNAL_OWNER,
)
from gui.pages.make_deepfake_page.tabs.training.fs_model.options import Options
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import HWidget, NoMarginLayout, RadioButtons, VWidget
from gui.widgets.preview.configuration import PreviewConfiguration
from gui.widgets.preview.preview import Preview
from utils import parse_tuple

logger = logging.getLogger(__name__)


class ModelSelector(qwt.QWidget):

    model_changed_sig = qtc.pyqtSignal(MODEL)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = NoMarginLayout()
        self.setLayout(layout)

        models_gb = qwt.QGroupBox()
        models_gb.setTitle('Available deepfake models')
        models_gb_layout = qwt.QVBoxLayout(models_gb)

        model_radio_buttons = RadioButtons(['original', 'fs'])
        models_gb_layout.addWidget(model_radio_buttons)
        model_radio_buttons.selection_changed_sig.connect(self._model_changed)

        layout.addWidget(models_gb)
        policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Policy.Minimum,
            qwt.QSizePolicy.Policy.Fixed,
        )
        self.setSizePolicy(policy)

    @qtc.pyqtSlot(list)
    def _model_changed(self, model: List[str]) -> None:
        self.model_changed_sig.emit(MODEL[model[0].upper()])


class TrainingConfiguration(qwt.QWidget):

    def __init__(self):
        super().__init__()
        # list of possible augmentations, when implementing a new augmentation,
        # after creating a widget, augmentation name should be appended to this
        # list in order to correctly augment images
        self.possible_augmentations = []
        self._init_ui()

        self._input_A_dir = None
        self._input_B_dir = None

    def _init_ui(self):
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)

        dataset_conf_gb = qwt.QGroupBox()
        layout.addWidget(dataset_conf_gb)
        dataset_conf_gb.setTitle('Dataset configuration')
        dataset_conf_gb_layout = qwt.QVBoxLayout(dataset_conf_gb)

        self.input_A_directory_btn = qwt.QPushButton(text='Input A directory')
        dataset_conf_gb_layout.addWidget(self.input_A_directory_btn)
        self.input_A_directory_btn.setToolTip('Not yet selected.')
        self.input_A_directory_btn.clicked.connect(
            lambda: self._select_input_directory('A')
        )

        self.input_B_directory_btn = qwt.QPushButton(text='Input B directory')
        dataset_conf_gb_layout.addWidget(self.input_B_directory_btn)
        self.input_B_directory_btn.setToolTip('Not yet selected.')
        self.input_B_directory_btn.clicked.connect(
            lambda: self._select_input_directory('B')
        )

        input_size_row = HWidget()
        dataset_conf_gb_layout.addWidget(input_size_row)
        input_size_row.layout().setContentsMargins(0, 0, 0, 0)
        input_size_row.layout().addWidget(qwt.QLabel(text='input shape'))
        self.input_shape_input = qwt.QLineEdit()
        self.input_shape_input.setText('3, 64, 64')
        input_size_row.layout().addWidget(self.input_shape_input)

        output_size_row = HWidget()
        dataset_conf_gb_layout.addWidget(output_size_row)
        output_size_row.layout().setContentsMargins(0, 0, 0, 0,)
        output_size_row.layout().addWidget(qwt.QLabel(text='output shape'))
        self.output_shape_input = qwt.QLineEdit()
        self.output_shape_input.setText('3, 128, 128')
        output_size_row.layout().addWidget(self.output_shape_input)

        models_gb = qwt.QGroupBox()
        layout.addWidget(models_gb)
        models_gb.setTitle('Training configuration')
        models_gb_layout = qwt.QVBoxLayout(models_gb)

        batch_size_row = HWidget()
        models_gb_layout.addWidget(batch_size_row)
        batch_size_row.layout().setContentsMargins(0, 0, 0, 0)
        batch_size_row.layout().addWidget(qwt.QLabel(text="batch size: "))
        self.batch_size_input = qwt.QLineEdit()
        self.batch_size_input.setText(str(64))
        batch_size_row.layout().addWidget(self.batch_size_input)

        epochs_row = HWidget()
        models_gb_layout.addWidget(epochs_row)
        epochs_row.layout().setContentsMargins(0, 0, 0, 0)
        epochs_row.layout().addWidget(qwt.QLabel(text='number of epochs: '))
        self.epochs_input = qwt.QLineEdit()
        self.epochs_input.setText(str(10))
        epochs_row.layout().addWidget(self.epochs_input)

        loss_function_row = HWidget()
        models_gb_layout.addWidget(loss_function_row)
        loss_function_row.layout().setContentsMargins(0, 0, 0, 0)
        loss_function_row.layout().addWidget(qwt.QLabel(text='loss function'))
        self.loss_dropdown = qwt.QComboBox()
        loss_function_row.layout().addWidget(self.loss_dropdown)
        self.loss_dropdown.addItem('MSE', 'MSE')
        self.loss_dropdown.addItem('DSSIM', 'DSSIM')
        self.loss_dropdown.setCurrentIndex(1)

        device_row = HWidget()
        models_gb_layout.addWidget(device_row)
        device_row.layout().setContentsMargins(0, 0, 0, 0)
        device_row.layout().addWidget(qwt.QLabel(text='device'))
        self.device_bg = qwt.QButtonGroup(device_row)
        for device in APP_CONFIG.app.core.devices:
            btn = qwt.QRadioButton(device.value, models_gb)
            btn.setChecked(True)
            device_row.layout().addWidget(btn)
            self.device_bg.addButton(btn)

        layout.addWidget(models_gb)

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

        self.flip_chk = qwt.QCheckBox(text='flip')
        image_augmentation_gb_layout.addWidget(self.flip_chk)
        self.possible_augmentations.append('flip')

        self.sharpen_chk = qwt.QCheckBox(text='sharpen')
        image_augmentation_gb_layout.addWidget(self.sharpen_chk)
        self.possible_augmentations.append('sharpen')

        light_row = HWidget()
        image_augmentation_gb_layout.addWidget(light_row)
        light_row.layout().setContentsMargins(0, 0, 0, 0)
        self.light_chk = qwt.QCheckBox(text='light')
        self.light_chk.setToolTip('Add light to the image')
        light_row.layout().addWidget(self.light_chk)
        light_row.layout().addWidget(qwt.QLabel(text='gamma'))
        self.light_input = qwt.QLineEdit()
        light_row.layout().addWidget(self.light_input)
        self.possible_augmentations.append('light')

        saturation_row = HWidget()
        image_augmentation_gb_layout.addWidget(saturation_row)
        saturation_row.layout().setContentsMargins(0, 0, 0, 0)
        self.saturation_chk = qwt.QCheckBox(text='saturation')
        saturation_row.layout().addWidget(self.saturation_chk)
        self.saturation_input = qwt.QLineEdit()
        saturation_row.layout().addWidget(self.saturation_input)
        self.possible_augmentations.append('saturation')

        gaussian_blur_row = HWidget()
        image_augmentation_gb_layout.addWidget(gaussian_blur_row)
        gaussian_blur_row.layout().setContentsMargins(0, 0, 0, 0)
        self.gaussian_blur_chk = qwt.QCheckBox(text='gaussian blur')
        gaussian_blur_row.layout().addWidget(self.gaussian_blur_chk)
        self.gaussian_blur_input = qwt.QLineEdit()
        gaussian_blur_row.layout().addWidget(self.gaussian_blur_input)
        self.possible_augmentations.append('gaussian_blur')

        bilateral_blur_row = HWidget()
        image_augmentation_gb_layout.addWidget(bilateral_blur_row)
        bilateral_blur_row.layout().setContentsMargins(0, 0, 0, 0)
        self.bilateral_blur_chk = qwt.QCheckBox(text='bilateral blur')
        bilateral_blur_row.layout().addWidget(self.bilateral_blur_chk)
        bilateral_blur_row.layout().addWidget(qwt.QLabel(text='d'))
        self.bilateral_blur_d_input = qwt.QLineEdit()
        self.bilateral_blur_d_input.setMaximumWidth(25)
        bilateral_blur_row.layout().addWidget(
            self.bilateral_blur_d_input
        )
        bilateral_blur_row.layout().addWidget(qwt.QLabel(text='color'))
        self.bilateral_blur_color_input = qwt.QLineEdit()
        self.bilateral_blur_color_input.setMaximumWidth(25)
        bilateral_blur_row.layout().addWidget(
            self.bilateral_blur_color_input
        )
        bilateral_blur_row.layout().addWidget(qwt.QLabel(text='space'))
        self.bilateral_blur_space_input = qwt.QLineEdit()
        self.bilateral_blur_space_input.setMaximumWidth(25)
        bilateral_blur_row.layout().addWidget(
            self.bilateral_blur_space_input
        )
        self.possible_augmentations.append('bilateral_blur')

        erode_row = HWidget()
        image_augmentation_gb_layout.addWidget(erode_row)
        erode_row.layout().setContentsMargins(0, 0, 0, 0)
        self.erode_chk = qwt.QCheckBox(text='erode')
        erode_row.layout().addWidget(self.erode_chk)
        erode_row.layout().addWidget(qwt.QLabel(text='kernel shape'))
        self.erode_input = qwt.QLineEdit()
        erode_row.layout().addWidget(self.erode_input)
        self.possible_augmentations.append('erode')

        dilate_row = HWidget()
        image_augmentation_gb_layout.addWidget(dilate_row)
        dilate_row.layout().setContentsMargins(0, 0, 0, 0)
        self.dilate_chk = qwt.QCheckBox(text='dilate')
        dilate_row.layout().addWidget(self.dilate_chk)
        dilate_row.layout().addWidget(qwt.QLabel(text='kernel shape'))
        self.dilate_input = qwt.QLineEdit()
        dilate_row.layout().addWidget(self.dilate_input)
        self.possible_augmentations.append('dilate')

        warp_row = HWidget()
        image_augmentation_gb_layout.addWidget(warp_row)
        warp_row.layout().setContentsMargins(0, 0, 0, 0)
        self.warp_chk = qwt.QCheckBox(text='warp')
        warp_row.layout().addWidget(self.warp_chk)
        self.interpolations_dropdown = qwt.QComboBox()
        warp_row.layout().addWidget(self.interpolations_dropdown)
        for index, inter in enumerate(INTERPOLATION):
            self.interpolations_dropdown.addItem(inter.name, inter.value)
            if inter == INTERPOLATION.CUBIC:
                self.interpolations_dropdown.setCurrentIndex(index)
        self.possible_augmentations.append('warp')

        show_augmentations_btn = qwt.QPushButton(
            text='see augmentations on example image'
        )
        show_augmentations_btn.clicked.connect(self._show_augmentations)
        image_augmentation_gb_layout.addWidget(show_augmentations_btn)

    def _show_augmentations(self):
        augs, errs = self.augmentations()
        if len(errs) > 0:
            for err in errs:
                logger.error(err)
            return
        image = cv.imread(
            APP_CONFIG.app.resources.face_example_path,
            cv.IMREAD_COLOR,
        )
        for aug in augs:
            image = aug(image)
        cv.imshow('augmentations on face example', image)
        cv.waitKey()

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

        logger.info(
            f'Selected input directory ({directory}) for ' +
            f'side {side}.'
        )
        btn = getattr(self, f'input_{side}_directory_btn')
        btn.setToolTip(directory)

        setattr(self, f'_input_{side}_dir', directory)

    @property
    def input_A_directory(self) -> Union[str, None]:
        return getattr(self, '_input_A_dir')

    @property
    def input_B_directory(self) -> Union[str, None]:
        return getattr(self, '_input_B_dir')

    @property
    def input_shape(self) -> str:
        return self.input_shape_input.text()

    @property
    def output_shape(self) -> str:
        return self.output_shape_input.text()

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
    def device(self) -> DEVICE:
        """Currently selected device on which training will commence.

        Returns:
            DEVICE: cpu or cuda
        """
        for but in self.device_bg.buttons():
            if but.isChecked():
                return DEVICE[but.text().upper()]

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

    def augmentations(
        self
    ) -> Tuple[List[Union[Callable, partial]], List[str]]:
        """Getter for different image augmentation functions. It goes through
        avaiilable augmentation functions and checks if the user checked it
        and inputed valid value. If everything was fine, list of image
        augmentation functions is returned.

        Returns:
            Tuple[List[Union[Callable, partial]], List[str]]: first element of
                the tuple is a list of image augmentation functions if all user
                inputs for parameters were valid and the second value in tuple
                is a list of errors that tell user which parameters are not
                valid
        """
        augs = []
        input_errors = []
        for aug in self.possible_augmentations:
            # checkbox which is supposed to be selected if some particular
            # augmentation is supposed to be used
            chk: qwt.QCheckBox = getattr(self, f'{aug}_chk')
            if not chk.isChecked():
                continue
            # augmentations with no parameters
            if aug in ['flip', 'sharpen']:
                augs.append(getattr(ImageAugmentation, aug))
            # specific augmentation with multiple parameters
            elif aug == 'bilateral_blur':
                props = ['d', 'color', 'space']
                prop_values = []
                for prop in props:
                    p: qwt.QLineEdit = getattr(
                        self,
                        f'bilateral_blur_{prop}_input',
                    )
                    try:
                        prop_value = int(p.text())
                    except ValueError:
                        input_errors.append(
                            'Unable to parse input for bilateral blur' +
                            f' property: {prop}.'
                        )
                        break
                    prop_values.append(prop_value)

                if len(prop_values) != 3:
                    continue
                # partial augmentation function that only needs and image,
                # other parameters are filled from the prop_values list
                func = partial(ImageAugmentation.bilateral_blur, *prop_values)
                augs.append(func)
            # augmentation with dropdown
            elif aug == 'warp':
                input_value = self.interpolations_dropdown.currentIndex()
                # TODO fix this non existing function
                func = partial(ImageAugmentation.warp, input_value)
                augs.append(func)
                continue
            # augmentations with one parameter
            else:
                aug_input: qwt.QLineEdit = getattr(self, f'{aug}_input')

                if aug in ['erode', 'dilate']:
                    split = aug_input.text().split(',')
                    split = [s.strip() for s in split]
                    if len(split) != 2:
                        input_errors.append(
                            f'Unable to parse input for {aug}, ' +
                            'invalid kernel shape.'
                        )
                        continue

                    try:
                        input_value = tuple([int(s) for s in split])
                    except ValueError:
                        input_errors.append(
                            f'Unable to parse input for {aug}.'
                        )
                        continue
                else:
                    try:
                        input_value = float(aug_input.text())
                    except ValueError:
                        # something in input widget that is not a number
                        input_errors.append(
                            f'Unable to parse input for {aug}.')
                        continue
                # partial augmentation function that only needs and image
                func = partial(
                    getattr(ImageAugmentation, aug),
                    input_value,
                )
                augs.append(func)

        if len(input_errors) > 0:
            return [], input_errors
        return augs, []

    @property
    def loss_function(self) -> nn.Module:
        """Constructs loss function based on the loss function user selected.

        Returns:
            nn.Module: loss function
        """
        selected = self.loss_dropdown.currentIndex()
        loss_fn = self.loss_dropdown.itemData(selected, 0)
        loss_fn = getattr(loss, loss_fn)()
        return loss_fn


class TrainingTab(BaseWidget):

    stop_training_sig = qtc.pyqtSignal()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self._selected_model = MODEL.ORIGINAL
        self._threads: Dict[JOB_TYPE, Tuple[qtc.QThread, Worker]] = dict()

        self._init_ui()

    def _init_ui(self):
        layout = qwt.QHBoxLayout()

        left_part = qwt.QWidget()
        left_part.setMaximumWidth(400)
        left_part_layout = qwt.QVBoxLayout()
        left_part.setLayout(left_part_layout)

        model_selector = ModelSelector()
        model_selector.model_changed_sig.connect(self._change_training_options)
        left_part_layout.addWidget(model_selector)

        scroll = qwt.QScrollArea()
        left_part_layout.addWidget(scroll)
        scroll.setVerticalScrollBarPolicy(
            qtc.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        scroll.setHorizontalScrollBarPolicy(
            qtc.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidgetResizable(True)

        self._stacked_wgt = qwt.QStackedWidget()
        scroll.setWidget(self._stacked_wgt)

        self._training_conf = TrainingConfiguration()
        self._stacked_wgt.addWidget(self._training_conf)

        self._fs_options = Options()
        self._stacked_wgt.addWidget(self._fs_options)

        self._model_option_mappings = {
            MODEL.ORIGINAL: self._training_conf,
            MODEL.FS: self._fs_options,
        }

        button_row = qwt.QWidget()
        button_row_layout = qwt.QHBoxLayout()
        button_row.setLayout(button_row_layout)

        self.start_btn = qwt.QPushButton(text='Start')
        self.start_btn.clicked.connect(self._start)
        button_row_layout.addWidget(self.start_btn)

        self.stop_btn = qwt.QPushButton(text='Stop')
        self.stop_btn.clicked.connect(self._stop)
        self.enable_widget(self.stop_btn, False)
        button_row_layout.addWidget(self.stop_btn)

        left_part_layout.addWidget(button_row)
        layout.addWidget(left_part)

        self.preview = Preview(4)
        self.preview.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.preview)
        self.setLayout(layout)

    def _change_training_options(self, model: MODEL) -> None:
        self._stacked_wgt.setCurrentWidget(self._model_option_mappings[model])
        self._selected_model = model

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

    def _make_trainer_configuration(self) -> Union[TrainerConfiguration, None]:
        """Tries to construct trainer configuration. If anything user inputed
        was wrong, configuration can not be made.

        Returns
        -------
        Union[TrainerConfiguration, None]
            trainer configuration is every input was valid, None otherwise
        """
        augs, errs = self.training_conf.augmentations()
        if len(errs) > 0:
            for err in errs:
                logger.error(err)
            return None

        optimizer_conf = OptimizerConfiguration(
            self.training_conf.selected_optimizer,
            self._optimizer_options(),
        )

        input_shape = parse_tuple(self.training_conf.input_shape)
        if None in input_shape:
            logger.error(
                'Input shape has values which are not ' +
                'parsable to a numeric type.'
            )
            return None
        if len(input_shape) != 3:
            logger.error('Input shape is invalid, must have 3 numbers')
            return None

        output_shape = parse_tuple(self.training_conf.output_shape)
        if None in output_shape:
            logger.error(
                'Output shape has values which are not ' +
                'parsable to a numeric type.'
            )
            return None
        if len(output_shape) != 3:
            logger.error('Ouput shape is invalid, must have 3 numbers')
            return None

        model_conf = ModelConfiguration(MODEL.ORIGINAL)

        data_transforms = transforms.Compose([transforms.ToTensor()])
        dataset_conf = DatasetConfiguration(
            path_A=self.training_conf.input_A_directory,
            path_B=self.training_conf.input_B_directory,
            input_size=input_shape[1],
            output_size=output_shape[1],
            batch_size=int(self.training_conf.batch_size),
            image_augmentations=augs,
            data_transforms=data_transforms,
        )

        comm_obj = TensorCommObject()
        comm_obj.data_sig = self.preview.refresh_data_sig
        preview_conf = PreviewConfiguration(True, comm_obj)

        criterion = self.training_conf.loss_function

        conf = TrainerConfiguration(
            device=self.training_conf.device,
            input_shape=input_shape,
            epochs=int(self.training_conf.epochs),
            criterion=criterion,
            model_conf=model_conf,
            optimizer_conf=optimizer_conf,
            dataset_conf=dataset_conf,
            preview_conf=preview_conf,
        )
        return conf

    def _start(self):
        """Initiates training process.
        """
        if self._selected_model == MODEL.ORIGINAL:
            conf = self._make_trainer_configuration()
            if conf is None:
                return

            self.training_thread = qtc.QThread()
            self.worker = TrainingWorker(
                conf,
                self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
            )
            self.stop_training_sig.connect(self.worker.stop_training)
            self.worker.moveToThread(self.training_thread)
            self.training_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.training_thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.training_thread.finished.connect(
                self.training_thread.deleteLater
            )
            self.training_thread.start()
            self.training_thread.finished.connect(
                lambda: self.enable_widget(self.start_btn, True)
            )
            self.training_thread.finished.connect(
                lambda: self.enable_widget(self.stop_btn, False)
            )
            self.enable_widget(self.start_btn, False)
            self.enable_widget(self.stop_btn, True)

        elif self._selected_model == MODEL.FS:
            thread = qtc.QThread()
            worker = FSTrainerWorker(
                batch_size=self._fs_options.batch_size,
                dataset_root=r'C:\Users\tonkec\Desktop\vggface2_crop_arcfacealign_224',
                gdeep=self._fs_options.gdeep,
                message_worker_sig=self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
            )
            self.stop_training_sig.connect(
                lambda: worker.conn_q.put(CONNECTION.STOP),
            )
            worker.moveToThread(thread)
            self._threads[JOB_TYPE.TRAIN_FS_DF_MODEL] = (thread, worker)
            thread.started.connect(worker.run)
            thread.start()
            worker.finished.connect(self._on_fs_trainer_worker_finished)
            self.enable_widget(self.start_btn, False)
            self.enable_widget(self.stop_btn, True)


    @qtc.pyqtSlot()
    def _on_fs_trainer_worker_finished(self) -> None:
        val = self._threads.get(JOB_TYPE.TRAIN_FS_DF_MODEL, None)
        if val is not None:
            thread, _ = val
            thread.quit()
            thread.wait()
            self._threads.pop(JOB_TYPE.TRAIN_FS_DF_MODEL, None)
        self.enable_widget(self.start_btn, True)
        self.enable_widget(self.stop_btn, False)


    def _stop(self):
        """Stops training process.
        """
        self.stop_training_sig.emit()
