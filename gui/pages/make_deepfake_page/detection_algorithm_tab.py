from typing import Dict, Optional

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import (
    BODY_KEY,
    CONSOLE_MESSAGE_TYPE,
    DATA_TYPE,
    FACE_DETECTION_ALGORITHM,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
)

from gui.widgets.base_widget import BaseWidget
from gui.widgets.picture_viewer import PictureViewer

from message.message import Body, Message, Messages

from resources.icons import icons


class DetectionAlgorithmTab(BaseWidget):

    input_picture_added_sig = qtc.pyqtSignal(Message)
    output_picture_added_sig = qtc.pyqtSignal(Message)

    def __init__(self,
                 signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict()):
        super().__init__(signals)

        self.input_faces_directory = None
        self.output_faces_directory = None
        self.model_path = None
        self.algorithm_selected_value = FACE_DETECTION_ALGORITHM.S3FD

        self.input_picture_added_sig.connect(self.input_picture_added)
        self.output_picture_added_sig.connect(self.output_picture_added)

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

        algorithm_selection_wgt = qwt.QWidget()
        algorithm_selection_wgt_layout = qwt.QHBoxLayout()
        algorithm_selection_wgt.setLayout(algorithm_selection_wgt_layout)
        size_policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Minimum, qwt.QSizePolicy.Maximum)
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
            qwt.QLabel(text='Directory for input faces'))
        spacer = qwt.QSpacerItem(
            40, 20, qwt.QSizePolicy.Fixed, qwt.QSizePolicy.Fixed)
        input_directory_wgt_layout.addItem(spacer)
        select_input_faces_directory_btn = qwt.QPushButton(text='Select')
        select_input_faces_directory_btn.clicked.connect(lambda:
                                                         self.select_faces_directory(DATA_TYPE.INPUT))
        select_input_faces_directory_btn.setFixedWidth(120)
        input_directory_wgt_layout.addWidget(select_input_faces_directory_btn)
        left_part_layout.addWidget(input_directory_wgt)

        output_directory_wgt = qwt.QWidget()
        output_directory_wgt_layout = qwt.QHBoxLayout()
        output_directory_wgt.setLayout(output_directory_wgt_layout)
        output_directory_wgt_layout.addWidget(
            qwt.QLabel(text='Directory for output faces'))
        spacer = qwt.QSpacerItem(
            40, 20, qwt.QSizePolicy.Fixed, qwt.QSizePolicy.Fixed)
        output_directory_wgt_layout.addItem(spacer)
        select_output_faces_directory_btn = qwt.QPushButton(text='Select')
        select_output_faces_directory_btn.clicked.connect(lambda:
                                                          self.select_faces_directory(DATA_TYPE.OUTPUT))
        select_output_faces_directory_btn.setFixedWidth(120)
        output_directory_wgt_layout.addWidget(
            select_output_faces_directory_btn)
        left_part_layout.addWidget(output_directory_wgt)

        model_path_wgt = qwt.QWidget()
        model_path_wgt_layout = qwt.QHBoxLayout()
        model_path_wgt.setLayout(model_path_wgt_layout)
        model_path_wgt_layout.addWidget(qwt.QLabel(text='Model path'))
        spacer = qwt.QSpacerItem(
            40, 20, qwt.QSizePolicy.Fixed, qwt.QSizePolicy.Fixed)
        model_path_wgt_layout.addItem(spacer)
        select_model_path_btn = qwt.QPushButton(text='Select')
        select_model_path_btn.clicked.connect(self.select_model_path)
        select_model_path_btn.setFixedWidth(120)
        model_path_wgt_layout.addWidget(select_model_path_btn)
        left_part_layout.addWidget(model_path_wgt)

        spacer = qwt.QSpacerItem(
            40, 20, qwt.QSizePolicy.Expanding,
            qwt.QSizePolicy.MinimumExpanding
        )
        left_part_layout.addItem(spacer)

        start_detection_btn = qwt.QPushButton(text='Start detection')
        start_detection_btn.clicked.connect(self.start_detection)
        start_detection_btn.setFixedWidth(150)
        start_detection_btn.setIcon(qtg.QIcon(qtg.QPixmap(':/play.svg')))
        left_part_layout.addWidget(start_detection_btn, 0, qtc.Qt.AlignHCenter)

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

        self.input_picture_viewer = PictureViewer()
        self.output_picture_viewer = PictureViewer()

        tab_wgt.addTab(self.input_picture_viewer, 'Input faces')
        tab_wgt.addTab(self.output_picture_viewer, 'Output faces')

        right_part_layout.addWidget(tab_wgt)

        layout.addWidget(left_part)

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

    @qtc.pyqtSlot(Message)
    def input_picture_added(self, msg: Message):
        """New face pictures added to input picture viewer.

        Parameters
        ----------
        msg : Message
            message containing faces
        """
        data = msg.body.data
        image = data[BODY_KEY.FILE]
        self.input_picture_viewer.pictures_added_sig.emit([image])

    @qtc.pyqtSlot(Message)
    def output_picture_added(self, msg: Message):
        """New face pictures added to output picture viewer.

        Parameters
        ----------
        msg : Message
            message containing faces
        """
        data = msg.body.data
        image = data[BODY_KEY.FILE]
        self.output_picture_viewer.pictures_added_sig.emit([image])

    @qtc.pyqtSlot(int)
    def algorithm_selected(self, id: int):
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

    def select_faces_directory(self, data_type: DATA_TYPE):
        """Selects input or output faces directory.

        Parameters
        ----------
        data_type : DATA_TYPE
            input or output data directory
        """
        directory = qwt.QFileDialog.getExistingDirectory(
            self, 'getExistingDirectory', './')
        if directory:
            if data_type == DATA_TYPE.INPUT:
                self.input_faces_directory = directory
            else:
                self.output_faces_directory = directory

            msg = Messages.CONSOLE_PRINT(
                CONSOLE_MESSAGE_TYPE.LOG,
                'Selected faces directory for ' +
                f'{data_type.value.lower()}: {directory}.'
            )

        else:
            msg = Messages.DIRECTORY_NOT_SELECTED()

        self.signals[SIGNAL_OWNER.CONSOLE].emit(msg)

    def select_model_path(self):
        """Selects where model for face detection is located.
        """
        options = qwt.QFileDialog.Options()
        options |= qwt.QFileDialog.DontUseNativeDialog
        model_path, _ = qwt.QFileDialog.getOpenFileName(
            self,
            'Select model file',
            "data/weights",
            "Model file (*.pth)",
            options=options)

        if model_path:
            self.model_path = model_path

            msg = Messages.CONSOLE_PRINT(
                CONSOLE_MESSAGE_TYPE.LOG,
                f'Selected model: {model_path}.'
            )

        else:
            msg = Messages.FILE_NOT_SELECTED()

        self.signals[SIGNAL_OWNER.CONSOLE].emit(msg)

    def start_detection(self):
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
                    BODY_KEY.OUTPUT_FACES_DIRECTORY: self.output_faces_directory,
                    BODY_KEY.MODEL_PATH: self.model_path,
                    BODY_KEY.ALGORITHM: self.algorithm_selected_value,
                }
            )
        )
        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)
