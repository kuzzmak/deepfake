from enums import FACE_DETECTION_ALGORITHM
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt


class DetectionAlgorithmTab(qwt.QWidget):

    algorithm_selected_sig = qtc.pyqtSignal(FACE_DETECTION_ALGORITHM)

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = qwt.QVBoxLayout()

        algorithm_selection_wgt = qwt.QWidget()
        algorithm_selection_wgt_layout = qwt.QHBoxLayout()
        algorithm_selection_wgt.setLayout(algorithm_selection_wgt_layout)
        size_policy = qwt.QSizePolicy(
            qwt.QSizePolicy.Minimum, qwt.QSizePolicy.Maximum)
        algorithm_selection_wgt.setSizePolicy(size_policy)

        box_group = qwt.QGroupBox()
        box_group.setTitle('Available face detection algorithms')
        box_group_layout = qwt.QHBoxLayout(box_group)

        algorithm_selection_wgt_layout.addWidget(box_group)

        bg = qwt.QButtonGroup(box_group)
        bg.idPressed.connect(self.face_detection_algorithm_selected)

        mtcnn_btn = qwt.QRadioButton(text='MTCNN', parent=box_group)
        box_group_layout.addWidget(mtcnn_btn)
        bg.addButton(mtcnn_btn)

        faceboxes_btn = qwt.QRadioButton(text='FaceBoxes', parent=box_group)
        box_group_layout.addWidget(faceboxes_btn)
        bg.addButton(faceboxes_btn)

        s3fd_btn = qwt.QRadioButton(text='S3FD', parent=box_group)
        s3fd_btn.setChecked(True)
        box_group_layout.addWidget(s3fd_btn)
        bg.addButton(s3fd_btn)

        spacer = qwt.QSpacerItem(
            40, 20, qwt.QSizePolicy.Expanding, qwt.QSizePolicy.Minimum)
        box_group_layout.addItem(spacer)

        start_detection_btn = qwt.QPushButton(text='Start detection')
        start_detection_btn.setMinimumWidth(120)
        box_group_layout.addWidget(start_detection_btn)

        layout.addWidget(algorithm_selection_wgt, 0, qtc.Qt.AlignTop)

        self.group_box = qwt.QGroupBox(title='Directory which will be used ' +
                                       'for storing extracted frames')

        layout.addWidget(self.group_box)

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
