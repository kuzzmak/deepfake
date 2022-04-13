# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'make_deepfake_page.ui'
#
# Created by: PyQt6 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_make_deepfake_page(object):
    def setupUi(self, make_deepfake_page):
        make_deepfake_page.setObjectName("make_deepfake_page")
        make_deepfake_page.resize(864, 773)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(make_deepfake_page)
        self.verticalLayout_7.setSpacing(20)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.tab_widget = QtWidgets.QTabWidget(make_deepfake_page)
        self.tab_widget.setObjectName("tab_widget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.tab_1)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.line = QtWidgets.QFrame(self.tab_1)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.select_video_btn = QtWidgets.QPushButton(self.tab_1)
        self.select_video_btn.setMinimumSize(QtCore.QSize(200, 0))
        self.select_video_btn.setMaximumSize(QtCore.QSize(300, 16777215))
        self.select_video_btn.setObjectName("select_video_btn")
        self.horizontalLayout.addWidget(self.select_video_btn)
        self.select_pictures_btn = QtWidgets.QPushButton(self.tab_1)
        self.select_pictures_btn.setMinimumSize(QtCore.QSize(200, 0))
        self.select_pictures_btn.setMaximumSize(QtCore.QSize(300, 16777215))
        self.select_pictures_btn.setObjectName("select_pictures_btn")
        self.horizontalLayout.addWidget(self.select_pictures_btn)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line_2 = QtWidgets.QFrame(self.tab_1)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.preview_label = QtWidgets.QLabel(self.tab_1)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.preview_label.setFont(font)
        self.preview_label.setText("")
        self.preview_label.setObjectName("preview_label")
        self.verticalLayout.addWidget(self.preview_label)
        self.preview_widget = QtWidgets.QStackedWidget(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.preview_widget.sizePolicy().hasHeightForWidth())
        self.preview_widget.setSizePolicy(sizePolicy)
        self.preview_widget.setObjectName("preview_widget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.preview_widget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.preview_widget.addWidget(self.page_2)
        self.verticalLayout.addWidget(self.preview_widget)
        self.tab_widget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.detection_algorithm_tab_layout = QtWidgets.QVBoxLayout(self.tab_2)
        self.detection_algorithm_tab_layout.setObjectName(
            "detection_algorithm_tab_layout")
        self.available_algorithms_gb = QtWidgets.QGroupBox(self.tab_2)
        self.available_algorithms_gb.setObjectName("available_algorithms_gb")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(
            self.available_algorithms_gb)
        self.horizontalLayout_2.setSpacing(15)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.mtcnn_chk_btn = QtWidgets.QRadioButton(
            self.available_algorithms_gb)
        self.mtcnn_chk_btn.setObjectName("mtcnn_chk_btn")
        self.horizontalLayout_2.addWidget(self.mtcnn_chk_btn)
        self.faceboxes_chk_btn = QtWidgets.QRadioButton(
            self.available_algorithms_gb)
        self.faceboxes_chk_btn.setObjectName("faceboxes_chk_btn")
        self.horizontalLayout_2.addWidget(self.faceboxes_chk_btn)
        self.s3fd_chk_btn = QtWidgets.QRadioButton(
            self.available_algorithms_gb)
        self.s3fd_chk_btn.setChecked(True)
        self.s3fd_chk_btn.setObjectName("s3fd_chk_btn")
        self.horizontalLayout_2.addWidget(self.s3fd_chk_btn)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.start_detection_btn = QtWidgets.QPushButton(
            self.available_algorithms_gb)
        self.start_detection_btn.setMinimumSize(QtCore.QSize(120, 0))
        self.start_detection_btn.setMaximumSize(
            QtCore.QSize(16777215, 16777215))
        self.start_detection_btn.setBaseSize(QtCore.QSize(0, 0))
        self.start_detection_btn.setObjectName("start_detection_btn")
        self.horizontalLayout_2.addWidget(self.start_detection_btn)
        self.detection_algorithm_tab_layout.addWidget(
            self.available_algorithms_gb)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2, 0, QtCore.Qt.AlignLeft)
        self.selected_faces_directory_label = QtWidgets.QLabel(self.groupBox)
        self.selected_faces_directory_label.setObjectName(
            "selected_faces_directory_label")
        self.horizontalLayout_3.addWidget(
            self.selected_faces_directory_label, 0, QtCore.Qt.AlignLeft)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.select_faces_directory_btn = QtWidgets.QPushButton(self.groupBox)
        self.select_faces_directory_btn.setObjectName(
            "select_faces_directory_btn")
        self.verticalLayout_3.addWidget(
            self.select_faces_directory_btn, 0, QtCore.Qt.AlignLeft)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.detection_algorithm_tab_layout.addLayout(self.verticalLayout_2)
        self.image_viewer_layout = QtWidgets.QVBoxLayout()
        self.image_viewer_layout.setObjectName("image_viewer_layout")
        self.detection_algorithm_tab_layout.addLayout(self.image_viewer_layout)
        self.tab_widget.addTab(self.tab_2, "")
        self.verticalLayout_7.addWidget(self.tab_widget)
        spacerItem3 = QtWidgets.QSpacerItem(
            20, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
        self.verticalLayout_7.addItem(spacerItem3)

        self.retranslateUi(make_deepfake_page)
        self.tab_widget.setCurrentIndex(1)
        self.preview_widget.setCurrentIndex(1)
        QtCore.QMetaObject.connectpyqtSlotsByName(make_deepfake_page)

    def retranslateUi(self, make_deepfake_page):
        _translate = QtCore.QCoreApplication.translate
        make_deepfake_page.setWindowTitle(
            _translate("make_deepfake_page", "Form"))
        self.label.setText(
            _translate(
                "make_deepfake_page",
                "Please select your training data. Possible choices are video and already existing images of faces."))
        self.select_video_btn.setText(
            _translate(
                "make_deepfake_page",
                "Select video"))
        self.select_pictures_btn.setText(
            _translate(
                "make_deepfake_page",
                "Select pictures"))
        self.tab_widget.setTabText(
            self.tab_widget.indexOf(
                self.tab_1), _translate(
                "make_deepfake_page", "Data"))
        self.available_algorithms_gb.setTitle(
            _translate(
                "make_deepfake_page",
                "Available face detection algorithms"))
        self.mtcnn_chk_btn.setText(_translate("make_deepfake_page", "MTCNN"))
        self.faceboxes_chk_btn.setText(
            _translate("make_deepfake_page", "FaceBoxes"))
        self.s3fd_chk_btn.setText(_translate("make_deepfake_page", "S3FD"))
        self.start_detection_btn.setText(
            _translate(
                "make_deepfake_page",
                "Start detection"))
        self.groupBox.setTitle(
            _translate(
                "make_deepfake_page",
                "Folder which will be used for storing extracted frames"))
        self.label_2.setText(_translate("make_deepfake_page", "Selected: "))
        self.selected_faces_directory_label.setText(
            _translate("make_deepfake_page", "NOTHING SELECTED"))
        self.select_faces_directory_btn.setText(
            _translate("make_deepfake_page", "Select"))
        self.tab_widget.setTabText(
            self.tab_widget.indexOf(
                self.tab_2), _translate(
                "make_deepfake_page", "Detection algorithm"))
