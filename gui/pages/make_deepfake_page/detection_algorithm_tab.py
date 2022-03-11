import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from config import APP_CONFIG
from core.dictionary import Dictionary
from core.extractor import Extractor, ExtractorConfiguration
from enums import (
    BODY_KEY,
    DEVICE,
    FACE_DETECTION_ALGORITHM,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    NUMBER_TYPE,
    SIGNAL_OWNER,
    IMAGE_SORT,
)
from gui.pages.make_deepfake_page.image_viewer_sorter import ImageViewerSorter
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    MinimalSizePolicy,
    VerticalSpacer,
    HorizontalSpacer,
)
from message.message import Body, Message
from resources.icons import icons  # noqa: F401
from utils import get_file_paths_from_dir, parse_number

logger = logging.getLogger(__name__)


class FaceExtractionWorker(qtc.QObject):

    finished = qtc.pyqtSignal()

    def __init__(
        self,
        input_dir: str,
        fda: FACE_DETECTION_ALGORITHM,
        device: DEVICE = DEVICE.CPU,
    ):
        super().__init__()
        self._input_dir = input_dir
        self._fda = fda
        self._device = device

    def run(self) -> None:
        conf = ExtractorConfiguration(
            input_dir=self._input_dir,
            fda=self._fda,
            device=self._device,
        )
        extractor = Extractor(conf)
        extractor.run()
        self.finished.emit()


class DetectionAlgorithmTab(BaseWidget):

    input_picture_added_sig = qtc.pyqtSignal()
    output_picture_added_sig = qtc.pyqtSignal()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self.faces_directory = None
        self.algorithm_selected_value = FACE_DETECTION_ALGORITHM.S3FD
        self.image_sort_method = IMAGE_SORT.IMAGE_HASH

        self._image_hash_eps = 18
        self._current_tab = 0
        self._threads = []
        self._landmarks_dir = None
        self._alignments_dir = None
        self._landmarks_file_present = False
        self._alignments_file_present = False
        self._landmarks = None
        self._alignments = None

        self.init_ui()
        self.add_signals()

    def init_ui(self):
        layout = qwt.QHBoxLayout()

        #########################
        # LEFT PART OF THE WINDOW
        #########################
        left_part = qwt.QWidget()
        left_part_layout = qwt.QVBoxLayout()
        left_part.setLayout(left_part_layout)

        algorithm_gb = qwt.QGroupBox(
            title='Available face detection algorithms'
        )
        box_group_layout = qwt.QHBoxLayout(algorithm_gb)

        left_part_layout.addWidget(algorithm_gb)

        bg = qwt.QButtonGroup(algorithm_gb)
        bg.idPressed.connect(self.algorithm_selected)

        faceboxes_btn = qwt.QRadioButton(text='FaceBoxes', parent=algorithm_gb)
        box_group_layout.addWidget(faceboxes_btn)
        bg.addButton(faceboxes_btn)

        s3fd_btn = qwt.QRadioButton(text='S3FD', parent=algorithm_gb)
        s3fd_btn.setChecked(True)
        box_group_layout.addWidget(s3fd_btn)
        bg.addButton(s3fd_btn)

        device_gb = qwt.QGroupBox(title='Device')
        self.device_bg = qwt.QButtonGroup(device_gb)
        device_gb_layout = qwt.QHBoxLayout(device_gb)
        left_part_layout.addWidget(device_gb)
        for device in APP_CONFIG.app.core.devices:
            btn = qwt.QRadioButton(device.value, device_gb)
            btn.setChecked(True)
            self.device_bg.addButton(btn)
            device_gb_layout.addWidget(btn)

        input_directory_wgt = qwt.QWidget()
        input_directory_wgt_layout = qwt.QHBoxLayout()
        input_directory_wgt.setLayout(input_directory_wgt_layout)
        input_directory_wgt_layout.addWidget(qwt.QLabel(
            text='Directory with face images or \nmetadata directory')
        )
        input_directory_wgt_layout.addItem(HorizontalSpacer)
        self.select_input_faces_directory_btn = qwt.QPushButton(text='Select')
        self.select_input_faces_directory_btn.clicked.connect(
            self.select_faces_directory
        )
        self.select_input_faces_directory_btn.setFixedWidth(120)
        input_directory_wgt_layout.addWidget(
            self.select_input_faces_directory_btn
        )
        left_part_layout.addWidget(input_directory_wgt)

        sort_wgt = qwt.QWidget()
        sort_wgt_layout = qwt.QVBoxLayout()
        sort_wgt.setLayout(sort_wgt_layout)

        sort_gb = qwt.QGroupBox()
        sort_gb.setTitle('Sorting methods')
        sort_gb_layout = qwt.QHBoxLayout(sort_gb)

        sort_wgt_layout.addWidget(sort_gb)

        sort_bg = qwt.QButtonGroup(sort_gb)
        sort_bg.idPressed.connect(self._image_sort_method_selected)

        image_hash_rbtn = qwt.QRadioButton(text='image hash', parent=sort_gb)
        image_hash_rbtn.setChecked(True)
        sort_gb_layout.addWidget(image_hash_rbtn)
        sort_bg.addButton(image_hash_rbtn)

        some_other_rbtn = qwt.QRadioButton(text='some other', parent=sort_gb)
        sort_gb_layout.addWidget(some_other_rbtn)
        sort_bg.addButton(some_other_rbtn)

        left_part_layout.addWidget(sort_wgt)

        # widget containing possible options for a particular sorting method
        self.sort_method_wgt = qwt.QStackedWidget()
        self.sort_method_wgt.setSizePolicy(MinimalSizePolicy)
        sort_method_wgt_layout = self.sort_method_wgt.layout()
        sort_method_wgt_layout.setContentsMargins(0, 0, 0, 0)

        self.image_hash_method = qwt.QWidget()
        image_hash_method_layout = qwt.QVBoxLayout()
        image_hash_method_layout.setContentsMargins(0, 0, 0, 0)
        self.image_hash_method.setLayout(image_hash_method_layout)
        eps_row = qwt.QWidget()
        eps_row_layout = qwt.QHBoxLayout()
        eps_row.setLayout(eps_row_layout)
        eps_label = qwt.QLabel(text='eps')
        self.eps_edit = qwt.QLineEdit()
        self.eps_edit.setText(str(self._image_hash_eps))
        eps_row_layout.addWidget(eps_label)
        eps_row_layout.addItem(VerticalSpacer)
        eps_row_layout.addWidget(self.eps_edit)
        image_hash_method_layout.addWidget(eps_row)
        self.sort_method_wgt.addWidget(self.image_hash_method)

        self.some_other_method = qwt.QWidget()
        some_other_method_layout = qwt.QVBoxLayout()
        some_other_method_layout.setContentsMargins(0, 0, 0, 0)
        self.some_other_method.setLayout(some_other_method_layout)
        some_other_method_layout.addWidget(qwt.QLabel(text='something'))
        self.sort_method_wgt.addWidget(self.some_other_method)

        left_part_layout.addWidget(self.sort_method_wgt)
        left_part_layout.addItem(VerticalSpacer)

        button_row_wgt = qwt.QWidget()
        button_row_wgt_layout = qwt.QHBoxLayout()
        button_row_wgt_layout.setContentsMargins(0, 0, 0, 0)
        button_row_wgt.setLayout(button_row_wgt_layout)

        start_detection_btn = qwt.QPushButton(text='Start detection')
        start_detection_btn.clicked.connect(self.start_detection)
        # start_detection_btn.setFixedWidth(150)
        start_detection_btn.setIcon(qtg.QIcon(qtg.QPixmap(':/play.svg')))
        button_row_wgt_layout.addWidget(start_detection_btn)

        self.sort_btn = qwt.QPushButton(text='Sort')
        self.enable_widget(self.sort_btn, False)
        self.sort_btn.clicked.connect(self._sort)
        button_row_wgt_layout.addWidget(self.sort_btn)

        left_part_layout.addWidget(button_row_wgt)

        size_policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Fixed, qwt.QSizePolicy.Minimum)
        left_part.setSizePolicy(size_policy)

        ##########################
        # RIGHT PART OF THE WINDOW
        ##########################
        right_part = qwt.QWidget()
        right_part_layout = qwt.QVBoxLayout()
        right_part.setLayout(right_part_layout)

        signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER
            ]
        }
        self.input_image_viewer_sorter_wgt = ImageViewerSorter(signals)
        self.input_image_viewer_sorter_wgt \
            .image_viewer_images_ok \
            .images_loading_sig \
            .connect(self._images_loading_changed)
        # connect both image viewers to the function which updated alignments
        # and landmarks files when some image is removed
        self.input_image_viewer_sorter_wgt \
            .image_viewer_images_ok \
            .removed_image_paths_sig \
            .connect(self._update_landmarks_and_alignments)
        self.input_image_viewer_sorter_wgt \
            .image_viewer_images_not_ok \
            .removed_image_paths_sig \
            .connect(self._update_landmarks_and_alignments)

        right_part_layout.addWidget(self.input_image_viewer_sorter_wgt)

        layout.addWidget(left_part)
        # line dividing left anf right part of the window
        line = qwt.QFrame()
        line.setFrameShape(qwt.QFrame.VLine)
        line.setFrameShadow(qwt.QFrame.Sunken)
        layout.addWidget(line)
        layout.addWidget(right_part)
        self.setLayout(layout)

    @property
    def device(self) -> DEVICE:
        """Currently selected device on which face extraction process
        will be executed.

        Returns:
            DEVICE: cpu or cuda
        """
        for but in self.device_bg.buttons():
            if but.isChecked():
                return DEVICE[but.text().upper()]

    def add_signals(self):
        """Adds input picture viewer signal to message worker so detected
        faces can be shown in gui.
        """
        msg = Message(
            MESSAGE_TYPE.REQUEST,
            MESSAGE_STATUS.OK,
            SIGNAL_OWNER.DETECTION_ALGORITHM_TAB,
            SIGNAL_OWNER.MESSAGE_WORKER,
            Body(
                JOB_TYPE.ADD_SIGNAL,
                {
                    BODY_KEY.SIGNAL_OWNER:
                    SIGNAL_OWNER.DETECTION_ALGORITHM_TAB_INPUT_PICTURE_VIEWER,
                    BODY_KEY.SIGNAL: self.input_picture_added_sig,
                }
            )
        )
        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

    @qtc.pyqtSlot(list)
    def _update_landmarks_and_alignments(self, paths: List[Path]) -> None:
        # in order to remove some detected faces, both alignments.json and
        # landmarks.json file need to be present in selected directory so
        # data doesn't corrupt
        if not self._alignments_file_present or \
                not self._landmarks_file_present:
            return
        # TODO speed up this update in other thread
        file_names = list(map(lambda p: p.name, paths))
        [self._landmarks.remove(k) for k in file_names]
        self._landmarks.save(self._landmarks_dir)
        [self._alignments.remove(k) for k in file_names]
        self._alignments.save(self._alignments_dir)

    @qtc.pyqtSlot(bool)
    def _images_loading_changed(self, status: bool) -> None:
        """Disables some widgets when images are loading.

        Args:
            status (bool): loading status
        """
        self.enable_widget(self.sort_btn, not status)
        self.enable_widget(self.select_input_faces_directory_btn, not status)

    @property
    def image_viewer_sorter(self) -> ImageViewerSorter:
        return self.input_image_viewer_sorter_wgt

    @qtc.pyqtSlot()
    def _sort(self) -> None:
        """Emits signal to appropriate `ImageViewerSorter` to sort `Face`
        metadata in images_ok `ImageViewer`.
        """
        eps = parse_number(self.eps_edit.text())
        if eps is None:
            logger.warning(
                f'Unable to parse: "{self.eps_edit.text()}" to ' +
                f'{NUMBER_TYPE.INT.value}. Sorting will not be done.'
            )
            return
        self.image_viewer_sorter.sort_sig.emit(eps)

    @qtc.pyqtSlot()
    def _on_worker_finished(self):
        for thread, worker in self._threads:
            thread.quit()
            thread.wait()
        self._threads = []

    @qtc.pyqtSlot(int)
    def algorithm_selected(self, id: int) -> None:
        """Face detection algorithm selection changed.

        Parameters
        ----------
        id : int
            if of the button selected
        """
        if id == -2:
            self.algorithm_selected_value = FACE_DETECTION_ALGORITHM.FACEBOXES
        else:
            self.algorithm_selected_value = FACE_DETECTION_ALGORITHM.S3FD

    @qtc.pyqtSlot(int)
    def _image_sort_method_selected(self, id: int) -> None:
        """Triggers when some image sorting method is chosen.

        Args:
            id (int): id of the method in button group
        """
        if id == -2:
            self.image_sort_method = IMAGE_SORT.IMAGE_HASH
            self.sort_method_wgt.setCurrentWidget(self.image_hash_method)
        elif id == -3:
            # some other
            self.sort_method_wgt.setCurrentWidget(self.some_other_method)

    @qtc.pyqtSlot()
    def select_faces_directory(self) -> None:
        """Selects directory with images for detection or metadata directory
        with already detected faces.
        """
        directory = qwt.QFileDialog.getExistingDirectory(
            self,
            'Select faces or metadata directory',
            'data',
        )
        if not directory:
            logger.warning('No directory selected.')
            return
        image_paths = get_file_paths_from_dir(directory, ['p'])
        # if path exists and some face metadata exists in this folder,
        # update preview
        if image_paths and len(image_paths) > 0:
            # send found paths to image viewer so they load
            self.image_viewer_sorter.data_paths_sig.emit(image_paths)

        self.faces_directory = directory
        logger.info(f'Selected faces directory: {directory}.')

        # check if in selected directory already exist landmarks.json and
        # alignments.json files which are generated after face extraction
        # and alignment process
        directory = Path(directory)
        self._landmarks_dir = directory / 'landmarks.json'
        self._landmarks_file_present = os.path.exists(self._landmarks_dir)
        if not self._landmarks_file_present:
            logger.warning(
                'landmarks.json file not found in selected ' +
                'directory, maybe face detection was not run yet?'
            )
        else:
            logger.info('Found landmarks.json file in selected directory')
            logger.debug(
                'Loading landmarks file from: ' +
                f'{str(self._landmarks_dir)}.'
            )
            self._landmarks = Dictionary.load(self._landmarks_dir)
            logger.debug('Landmarks file loaded.')

        self._alignments_dir = directory / 'alignments.json'
        self._alignments_file_present = os.path.exists(self._alignments_dir)
        if not self._alignments_file_present:
            logger.warning(
                'alignments.json file not found in selected ' +
                'directory, maybe alignment was not run yet?'
            )
        else:
            logger.info('Found alignments.json file in selected directory')
            logger.debug(
                'Loading alignments file from: ' +
                f'{str(self._alignments_dir)}.'
            )
            self._alignments = Dictionary.load(self._alignments_dir)
            logger.debug('Alignments file loaded.')

    def start_detection(self) -> None:
        """Sends message with faces directories to make deepfake page.
        """
        if self.faces_directory is None:
            logger.error(
                'Unable to start face detection, ' +
                'no directory is selected.'
            )
            return
        thread = qtc.QThread()
        worker = FaceExtractionWorker(
            self.faces_directory,
            self.algorithm_selected_value,
            self.device,
        )
        self._threads.append((thread, worker))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_worker_finished)
        thread.start()
