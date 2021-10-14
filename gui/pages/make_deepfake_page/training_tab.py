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
from enums import DEVICE, MODEL
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import VerticalSpacer
from gui.widgets.preview.configuration import PreviewConfiguration
from gui.widgets.preview.preview import Preview
from trainer_thread import Worker


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
            qwt.QSizePolicy.Fixed,
            qwt.QSizePolicy.Fixed,
        )
        self.setSizePolicy(policy)
        self.setLayout(layout)

    @qtc.pyqtSlot(int)
    def _model_changed(self, index: int) -> None:
        print('model index: ', index)


class TrainingTab(BaseWidget):

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = qwt.QHBoxLayout()

        left_part = qwt.QWidget()
        left_part.setMaximumWidth(300)
        left_part.setAutoFillBackground(True)
        p = left_part.palette()
        p.setColor(left_part.backgroundRole(), qtc.Qt.red)
        left_part.setPalette(p)
        left_part_layout = qwt.QVBoxLayout()
        left_part.setLayout(left_part_layout)

        model_selector = ModelSelector()
        left_part_layout.addWidget(model_selector)
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
        device = DEVICE.CPU
        if torch.cuda.is_available():
            device = DEVICE.CUDA

        input_shape = (3, 128, 128)

        model_conf = ModelConfiguration(MODEL.ORIGINAL)

        optimizer_conf = DEFAULT_ADAM_CONF
        optimizer_conf.optimizer_args['lr'] = 5e-5
        optimizer_conf.optimizer_args['betas'] = (0.5, 0.999)

        data_transforms = transforms.Compose([transforms.ToTensor()])
        dataset_conf = DatasetConfiguration(
            # faces_path=r'C:\Users\tonkec\Documents\deepfake\data\gen_faces\metadata',
            metadata_path=r'C:\Users\kuzmi\Documents\deepfake\data\gen_faces\metadata',
            input_shape=input_shape[1],
            batch_size=32,
            load_into_memory=True,
            data_transforms=data_transforms,
        )

        comm_obj = TensorCommObject()
        comm_obj.data_sig = self.preview.refresh_data_sig
        preview_conf = PreviewConfiguration(True, comm_obj)

        conf = TrainerConfiguration(
            device=device,
            input_shape=input_shape,
            epochs=1,
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
