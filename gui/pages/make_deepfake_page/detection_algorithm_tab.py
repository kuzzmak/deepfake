import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import FACE_DETECTION_ALGORITHM

from gui.widgets.picture_viewer import PictureViewer

from resources.icons import icons


class DetectionAlgorithmTab(qwt.QWidget):

    algorithm_selected_sig = qtc.pyqtSignal(FACE_DETECTION_ALGORITHM)

    def __init__(self):
        super().__init__()

        self.init_ui()

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
        bg.idPressed.connect(self.face_detection_algorithm_selected)

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
        select_output_faces_directory_btn.setFixedWidth(120)
        output_directory_wgt_layout.addWidget(
            select_output_faces_directory_btn)
        left_part_layout.addWidget(output_directory_wgt)

        spacer = qwt.QSpacerItem(
            40, 20, qwt.QSizePolicy.Expanding,
            qwt.QSizePolicy.MinimumExpanding
        )
        left_part_layout.addItem(spacer)

        start_detection_btn = qwt.QPushButton(text='Start detection')
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

        input_picture_viewer = PictureViewer()
        output_picture_viewer = PictureViewer()

        tab_wgt.addTab(input_picture_viewer, 'Input faces')
        tab_wgt.addTab(output_picture_viewer, 'Output faces')

        right_part_layout.addWidget(tab_wgt)

        layout.addWidget(left_part)

        line = qwt.QFrame()
        line.setFrameShape(qwt.QFrame.VLine)
        line.setFrameShadow(qwt.QFrame.Sunken)
        layout.addWidget(line)

        layout.addWidget(right_part)

        self.setLayout(layout)

    @qtc.pyqtSlot(int)
    def face_detection_algorithm_selected(self, id: int):
        if id == -2:
            self.algorithm_selected_sig.emit(
                FACE_DETECTION_ALGORITHM.MTCNN)
        elif id == -3:
            self.algorithm_selected_sig.emit(
                FACE_DETECTION_ALGORITHM.FACEBOXES)
        else:
            self.algorithm_selected_sig.emit(
                FACE_DETECTION_ALGORITHM.S3FD)
# self.start_detection_btn.setIcon(qtg.QIcon(qtg.QPixmap(':/play.svg')))
