from typing import Dict, Optional

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.page import Page
from gui.pages.make_deepfake_page.data_tab import DataTab
from gui.pages.make_deepfake_page.detection_algorithm_tab \
    import DetectionAlgorithmTab
from gui.templates.make_deepfake_page import Ui_make_deepfake_page
from gui.widgets.video_player import VideoPlayer
from gui.widgets.picture_viewer import PictureViewer

from message.message import Message


from enums import (
    BODY_KEY,
    CONSOLE_MESSAGE_TYPE,
    DATA_TYPE,
    FACE_DETECTION_ALGORITHM,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
)

from names import (
    MAKE_DEEPFAKE_PAGE_NAME,
    MAKE_DEEPFAKE_PAGE_TITLE,
)

from utils import get_file_paths_from_dir

from resources.icons import icons


class MakeDeepfakePage(Page, Ui_make_deepfake_page):

    input_data_directory_sig = qtc.pyqtSignal(str)
    output_data_directory_sig = qtc.pyqtSignal(str)
    extract_frames_sig = qtc.pyqtSignal(Message)

    def __init__(self,
                 parent,
                 signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
                 * args,
                 **kwargs):
        super().__init__(parent, signals, page_name=MAKE_DEEPFAKE_PAGE_NAME, *args, **kwargs)

        self.input_data_directory = None
        self.output_data_directory = None

        self.input_data_directory_sig.connect(
            self.input_data_directory_selected)
        self.output_data_directory_sig.connect(
            self.output_data_directory_selected)
        self.extract_frames_sig.connect(self.extract_frames)

        self.setupUi(self)
        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.picture_viewer_tab_1 = PictureViewer()
        self.preview_widget.addWidget(self.picture_viewer_tab_1)

        # --- video widget with faces directory selection ---
        self.video_player = VideoPlayer()

        self.central_widget = qwt.QWidget()
        central_layout = qwt.QHBoxLayout()
        self.central_widget.setLayout(central_layout)
        central_layout.addWidget(self.video_player)

        self.frame_extraction_gb = qwt.QGroupBox()
        self.frame_extraction_gb.setTitle('Select destination directory ' +
                                          'for extracted frames from video')
        size_policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Maximum, qwt.QSizePolicy.Minimum)
        self.frame_extraction_gb.setSizePolicy(size_policy)
        # self.frame_extraction_gb.setMaximumWidth(200)

        frame_extraction_part_layout = qwt.QVBoxLayout(
            self.frame_extraction_gb)

        select_frames_directory = qwt.QPushButton(text='Select')
        select_frames_directory.clicked.connect(self.select_frames_directory)
        frame_extraction_part_layout.addWidget(select_frames_directory)

        # --- resize frames part of the window ---
        resize_frames_chk = qwt.QCheckBox(text='Resize frames')
        frame_extraction_part_layout.addWidget(resize_frames_chk)
        self.dim_wgt = qwt.QWidget()
        self.enable_widget(self.dim_wgt, False)
        dim_wgt_layout = qwt.QHBoxLayout()
        self.dim_wgt.setLayout(dim_wgt_layout)
        dim_wgt_layout.addWidget(qwt.QLabel(text='Biggest frame dimension: '))
        frame_extraction_part_layout.addWidget(self.dim_wgt)

        central_layout.addWidget(self.frame_extraction_gb)
        self.preview_widget.addWidget(self.central_widget)

        # until pictures or video is selected, page with face detection
        # is disabled
        self.enable_detection_algorithm_tab(False)

        self.select_pictures_btn.clicked.connect(self.select_pictures)
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.start_detection_btn.setIcon(qtg.QIcon(qtg.QPixmap(':/play.svg')))
        self.enable_widget(self.start_detection_btn, False)
        self.select_faces_directory_btn.clicked.connect(
            self.select_faces_directory
        )

        self.picture_viewer_tab_2 = PictureViewer()
        self.image_viewer_layout.addWidget(self.picture_viewer_tab_2)

        data_tab_signals = {
            SIGNAL_OWNER.CONSOLE: self.signals[SIGNAL_OWNER.CONSOLE],
            SIGNAL_OWNER.INPUT_DATA_DIRECTORY: self.input_data_directory_sig,
            SIGNAL_OWNER.OUTPUT_DATA_DIRECTORY: self.output_data_directory_sig,
            SIGNAL_OWNER.FRAMES_EXTRACTION: self.extract_frames_sig,
        }
        data_tab = DataTab(data_tab_signals)

        self.tab_widget.addTab(data_tab, 'Data')
        self.tab_widget.addTab(DetectionAlgorithmTab(), 'Detection algorithm')

    @qtc.pyqtSlot(str)
    def input_data_directory_selected(self, directory: str):
        self.input_data_directory = directory

    @qtc.pyqtSlot(str)
    def output_data_directory_selected(self, directory: str):
        self.output_data_directory = directory

    def progress_value_changed(self, value: int):
        if value == 100:
            msg = qwt.QMessageBox(self)
            msg.setIcon(qwt.QMessageBox.Information)
            msg.setText("Face extraction successful.")
            msg.setInformativeText("Extracted faces are shown below.")
            msg.setWindowTitle("Face extraction information")
            msg.setStandardButtons(qwt.QMessageBox.Ok)
            msg.exec_()

    def enable_detection_algorithm_tab(self, enable: bool):
        self.tab_widget.setTabEnabled(1, enable)

    def start_detection(self):
        """Initiates process of face detection and extraction from
        selected directory.
        """
        # msg = Message(
        #     MESSAGE_TYPE.REQUEST,
        #     FaceDetectionMessageBody(
        #         self.faces_directory,
        #         'C:\\Users\\tonkec\\Documents\\deepfake\\data\\weights\\s3fd\\s3fd.pth',
        #         FACE_DETECTION_ALGORITHM.S3FD
        #     )
        # )
        # self.send_message(msg)
        ...

    @qtc.pyqtSlot(Message)
    def extract_frames(self, msg: Message):
        """Starts process of extracting frames from video.
        """
        if msg.body.data[BODY_KEY.DATA_TYPE] == DATA_TYPE.INPUT:
            msg.body.data[BODY_KEY.DATA_DIRECTORY] = self.input_data_directory
        else:
            msg.body.data[BODY_KEY.DATA_DIRECTORY] = self.output_data_directory

        msg.sender = SIGNAL_OWNER.FRAMES_EXTRACTION
        msg.recipient = SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER

        self.send_message(msg)

    def select_faces_directory(self):
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self, "getExistingDirectory", "./")
        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\gen_faces"
        if directory:
            self.selected_faces_directory_label.setText(directory)
            self.enable_widget(self.start_detection_btn, True)

            # msg = Message(
            #     MESSAGE_TYPE.REQUEST,
            #     ConsolePrintMessageBody(
            #         CONSOLE_MESSAGE_TYPE.INFO,
            #         f'Selected directory for extracted faces: {directory}.'
            #     )
            # )

            self.faces_directory = directory

        else:
            ...

        #     msg = no_foler_selected_msg

        # self.send_message(msg)

    def select_frames_directory(self):
        """Select directory in which extracted frames from video will go.
        """
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self,
        #     "getExistingDirectory",
        #     "./")

        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\gen_faces"

        if directory:
            self.data_directory = directory
            # msg = Message(
            #     MESSAGE_TYPE.REQUEST,
            #     ConsolePrintMessageBody(
            #         CONSOLE_MESSAGE_TYPE.INFO,
            #         'Directory in which extracted frames will go ' +
            #         f'selected: {directory}.'
            #     )
            # )

            self.enable_widget(self.extract_frames_btn, True)

        else:
            # msg = no_foler_selected_msg
            ...

        # self.send_message(msg)

    def select_pictures(self):
        """Selecting directory with faces which would be used for face
        extraction process.
        """
        # directory = qwt.QFileDialog.getExistingDirectory(
        #     self,
        #     "getExistingDirectory",
        #     "./"
        # )

        directory = "C:\\Users\\tonkec\\Documents\\deepfake\\dummy_pics"

        if directory:
            self.data_directory = directory
            self.preview_widget.setCurrentWidget(self.picture_viewer_tab_1)

            image_paths = get_file_paths_from_dir(directory)
            if len(image_paths) == 0:
                # msg = Message(
                #     MESSAGE_TYPE.REQUEST,
                #     ConsolePrintMessageBody(
                #         CONSOLE_MESSAGE_TYPE.WARNING,
                #         f'No images were found in: {directory}.'
                #     )
                # )

                self.enable_detection_algorithm_tab(False)

            else:
                self.picture_viewer_tab_1.pictures_added_sig.emit(image_paths)

                # msg = Message(
                #     MESSAGE_TYPE.REQUEST,
                #     ConsolePrintMessageBody(
                #         CONSOLE_MESSAGE_TYPE.INFO,
                #         'Loaded: {} images from: {}.'.format(
                #             len(image_paths),
                #             directory
                #         )
                #     )
                # )

                self.enable_detection_algorithm_tab(True)

        else:
            ...
            # msg = no_foler_selected_msg

        # self.send_message(msg)
