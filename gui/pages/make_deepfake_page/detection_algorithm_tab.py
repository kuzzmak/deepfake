import logging
from typing import Dict, List, Optional

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from config import APP_CONFIG
from core.face import Face
from enums import (
    BODY_KEY,
    DATA_TYPE,
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
from gui.widgets.picture_viewer import ImageViewer, StandardItem
from message.message import Body, Message
from resources.icons import icons  # noqa: F401
from serializer.face_serializer import FaceSerializer
from utils import get_file_paths_from_dir, parse_number
from worker.io_worker import Worker as IOWorker

logger = logging.getLogger(__name__)


class DetectionAlgorithmTab(BaseWidget):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
    ):
        super().__init__(signals)

        self.input_faces_directory = APP_CONFIG.app.input_faces_directory
        self.output_faces_directory = APP_CONFIG.app.output_faces_directory
        self.algorithm_selected_value = FACE_DETECTION_ALGORITHM.S3FD
        self.image_sort_method = IMAGE_SORT.IMAGE_HASH

        self._image_hash_eps = 18
        self._current_tab = 0

        self.init_ui()
        # self.add_signals()

    def init_ui(self):
        layout = qwt.QHBoxLayout()

        #########################
        # LEFT PART OF THE WINDOW
        #########################
        left_part = qwt.QWidget()
        left_part_layout = qwt.QVBoxLayout()
        left_part.setLayout(left_part_layout)

        algorithm_selection_wgt = qwt.QWidget()
        algorithm_selection_wgt_layout = qwt.QHBoxLayout()
        algorithm_selection_wgt.setLayout(algorithm_selection_wgt_layout)
        size_policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Minimum,
            qwt.QSizePolicy.Maximum,
        )
        algorithm_selection_wgt.setSizePolicy(size_policy)

        algorithm_gb = qwt.QGroupBox()
        algorithm_gb.setTitle('Available face detection algorithms')
        box_group_layout = qwt.QHBoxLayout(algorithm_gb)

        algorithm_selection_wgt_layout.addWidget(algorithm_gb)

        bg = qwt.QButtonGroup(algorithm_gb)
        bg.idPressed.connect(self.algorithm_selected)

        mtcnn_btn = qwt.QRadioButton(text='MTCNN', parent=algorithm_gb)
        box_group_layout.addWidget(mtcnn_btn)
        bg.addButton(mtcnn_btn)

        faceboxes_btn = qwt.QRadioButton(text='FaceBoxes', parent=algorithm_gb)
        box_group_layout.addWidget(faceboxes_btn)
        bg.addButton(faceboxes_btn)

        s3fd_btn = qwt.QRadioButton(text='S3FD', parent=algorithm_gb)
        s3fd_btn.setChecked(True)
        box_group_layout.addWidget(s3fd_btn)
        bg.addButton(s3fd_btn)

        left_part_layout.addWidget(algorithm_selection_wgt)

        input_directory_wgt = qwt.QWidget()
        input_directory_wgt_layout = qwt.QHBoxLayout()
        input_directory_wgt.setLayout(input_directory_wgt_layout)
        input_directory_wgt_layout.addWidget(
            qwt.QLabel(text='Directory for input faces')
        )
        input_directory_wgt_layout.addItem(HorizontalSpacer)
        select_input_faces_directory_btn = qwt.QPushButton(text='Select')
        select_input_faces_directory_btn.clicked.connect(
            lambda: self.select_faces_directory(DATA_TYPE.INPUT)
        )
        select_input_faces_directory_btn.setFixedWidth(120)
        input_directory_wgt_layout.addWidget(select_input_faces_directory_btn)
        left_part_layout.addWidget(input_directory_wgt)

        output_directory_wgt = qwt.QWidget()
        output_directory_wgt_layout = qwt.QHBoxLayout()
        output_directory_wgt.setLayout(output_directory_wgt_layout)
        output_directory_wgt_layout.addWidget(
            qwt.QLabel(text='Directory for output faces')
        )
        output_directory_wgt_layout.addItem(HorizontalSpacer)
        select_output_faces_directory_btn = qwt.QPushButton(text='Select')
        select_output_faces_directory_btn.clicked.connect(
            lambda: self.select_faces_directory(DATA_TYPE.OUTPUT)
        )
        select_output_faces_directory_btn.setFixedWidth(120)
        output_directory_wgt_layout.addWidget(
            select_output_faces_directory_btn)
        left_part_layout.addWidget(output_directory_wgt)

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
        # self.sort_btn.setFixedWidth(150)
        self.sort_btn.clicked.connect(self._sort)
        button_row_wgt_layout.addWidget(self.sort_btn)

        self.save_sorted_btn = qwt.QPushButton(text='Save sorted')
        self.save_sorted_btn.clicked.connect(self._save_sorted)
        button_row_wgt_layout.addWidget(self.save_sorted_btn)

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

        right_part_layout.addWidget(qwt.QLabel(
            text='Preview of the detected faces in input and output data'))

        tab_wgt = qwt.QTabWidget()
        tab_wgt.currentChanged.connect(self._tab_changed)

        self.input_image_viewer_sorter_wgt = ImageViewerSorter()
        self.input_image_viewer_sorter_wgt \
            .image_viewer_images_ok \
            .images_added_sig.connect(
                self._images_added_to_image_viewer
            )
        self.output_image_viewer_sorter_wgt = ImageViewerSorter()

        tab_wgt.addTab(self.input_image_viewer_sorter_wgt, 'Input faces')
        tab_wgt.addTab(self.output_image_viewer_sorter_wgt, 'Output faces')

        right_part_layout.addWidget(tab_wgt)

        layout.addWidget(left_part)
        # line dividing left anf right part of the window
        line = qwt.QFrame()
        line.setFrameShape(qwt.QFrame.VLine)
        line.setFrameShadow(qwt.QFrame.Sunken)
        layout.addWidget(line)
        layout.addWidget(right_part)
        self.setLayout(layout)

    def add_signals(self):
        """Adds input picture viewer and output picture viewer signals
        to message worker so detected faces can be shown in gui.
        """
        # message for input picture viewer
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

        # message for output picture viewer
        msg = Message(
            MESSAGE_TYPE.REQUEST,
            MESSAGE_STATUS.OK,
            SIGNAL_OWNER.DETECTION_ALGORITHM_TAB,
            SIGNAL_OWNER.MESSAGE_WORKER,
            Body(
                JOB_TYPE.ADD_SIGNAL,
                {
                    BODY_KEY.SIGNAL_OWNER:
                    SIGNAL_OWNER.DETECTION_ALGORITHM_TAB_OUTPUT_PICTURE_VIEWER,
                    BODY_KEY.SIGNAL: self.output_picture_added_sig,
                }
            )
        )
        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

    @qtc.pyqtSlot(list)
    def _images_added_to_image_viewer(self, images: List[Face]):
        if len(images) > 0:
            self.enable_widget(self.sort_btn, True)

    @qtc.pyqtSlot(int)
    def _tab_changed(self, index: int):
        """Tracks index of the current tab.

        Args:
            index (int): tab index
        """
        self._current_tab = index

    @property
    def current_tab_ivs(self) -> ImageViewerSorter:
        if self._current_tab == 0:
            return self.input_image_viewer_sorter_wgt
        return self.output_image_viewer_sorter_wgt

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
        self.current_tab_ivs.sort_sig.emit(eps)

    @qtc.pyqtSlot()
    def _save_sorted(self):
        directory = qwt.QFileDialog.getExistingDirectory(
            self,
            'getExistingDirectory',
            './',
        )
        if not directory:
            logger.warning('No directory selected.')
            return
        logger.info(f'Selected: {directory} for sorted metadata objects.')

        data: List[Face] = self.current_tab_ivs \
            .image_viewer_images_ok \
            .get_all_data(
            StandardItem.FaceRole
        )
        self.thread = qtc.QThread()
        self.worker = IOWorker(data, directory)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @qtc.pyqtSlot(int)
    def algorithm_selected(self, id: int) -> None:
        """Face detection algorithm selection changed.

        Parameters
        ----------
        id : int
            if of the button selected
        """
        if id == -2:
            self.algorithm_selected_value = FACE_DETECTION_ALGORITHM.MTCNN
        elif id == -3:
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

    def select_faces_directory(self, data_type: DATA_TYPE) -> None:
        """Selects input or output faces directory.

        Parameters
        ----------
        data_type : DATA_TYPE
            input or output data directory
        """
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self,
        #     'getExistingDirectory',
        #     './',
        # )
        directory = r'C:\Users\kuzmi\Documents\deepfake\data\face_A\metadata_sorted'
        if not directory:
            logger.warning('No directory selected.')
            return

        image_paths = get_file_paths_from_dir(directory, ['p'])
        data_type = data_type.value.lower()
        # if path exists and some face metadata exists in this folder,
        # update preview
        if image_paths and len(image_paths) > 0:
            logger.info(
                'Found faces metadata in selected folder. You ' +
                'can now sort this metadata and save which ' +
                'suits you.'
            )
            faces = [FaceSerializer.load(i_p) for i_p in image_paths]
            # get input or output ImageViewerSorter
            ivs: ImageViewerSorter = getattr(
                self,
                f'{data_type}_image_viewer_sorter_wgt'
            )
            ivs.faces_cache = faces
            # get images_ok ImageViewer on appropriate ImageViewerSorter
            viewer: ImageViewer = ivs.image_viewer_images_ok
            viewer.images_added_sig.emit(faces)

        setattr(
            self,
            f'{data_type}_faces_directory',
            directory,
        )

        logger.info(
            'Selected faces directory for ' +
            f'{data_type} data: {directory}.'
        )

    def start_detection(self) -> None:
        """Sends message with faces directories to make deepfake page.
        """
        msg = Message(
            MESSAGE_TYPE.REQUEST,
            MESSAGE_STATUS.OK,
            SIGNAL_OWNER.DETECTION_ALGORITHM_TAB,
            SIGNAL_OWNER.MAKE_DEEPFAKE_PAGE_DETECT_FACES,
            Body(
                JOB_TYPE.FACE_DETECTION,
                {
                    BODY_KEY.INPUT_FACES_DIRECTORY: self.input_faces_directory,
                    BODY_KEY.OUTPUT_FACES_DIRECTORY:
                    self.output_faces_directory,
                    BODY_KEY.ALGORITHM: self.algorithm_selected_value,
                }
            )
        )
        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)
