# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'make_deepfake_page_2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_make_deepfake_page(object):
    def setupUi(self, make_deepfake_page):
        make_deepfake_page.setObjectName("make_deepfake_page")
        make_deepfake_page.resize(817, 804)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(make_deepfake_page)
        self.verticalLayout_7.setSpacing(20)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label = QtWidgets.QLabel(make_deepfake_page)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_7.addWidget(self.label)
        self.line = QtWidgets.QFrame(make_deepfake_page)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_7.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.select_video_btn = QtWidgets.QPushButton(make_deepfake_page)
        self.select_video_btn.setMinimumSize(QtCore.QSize(200, 0))
        self.select_video_btn.setMaximumSize(QtCore.QSize(300, 16777215))
        self.select_video_btn.setObjectName("select_video_btn")
        self.horizontalLayout.addWidget(self.select_video_btn)
        self.select_pictures_btn = QtWidgets.QPushButton(make_deepfake_page)
        self.select_pictures_btn.setMinimumSize(QtCore.QSize(200, 0))
        self.select_pictures_btn.setMaximumSize(QtCore.QSize(300, 16777215))
        self.select_pictures_btn.setObjectName("select_pictures_btn")
        self.horizontalLayout.addWidget(self.select_pictures_btn)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.line_2 = QtWidgets.QFrame(make_deepfake_page)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_7.addWidget(self.line_2)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setObjectName("main_layout")
        self.label_2 = QtWidgets.QLabel(make_deepfake_page)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.main_layout.addWidget(self.label_2)
        self.verticalLayout_7.addLayout(self.main_layout)
        self.preview_widget = QtWidgets.QStackedWidget(make_deepfake_page)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preview_widget.sizePolicy().hasHeightForWidth())
        self.preview_widget.setSizePolicy(sizePolicy)
        self.preview_widget.setObjectName("preview_widget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.preview_widget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.preview_widget.addWidget(self.page_2)
        self.verticalLayout_7.addWidget(self.preview_widget)
        spacerItem1 = QtWidgets.QSpacerItem(20, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
        self.verticalLayout_7.addItem(spacerItem1)

        self.retranslateUi(make_deepfake_page)
        QtCore.QMetaObject.connectSlotsByName(make_deepfake_page)

    def retranslateUi(self, make_deepfake_page):
        _translate = QtCore.QCoreApplication.translate
        make_deepfake_page.setWindowTitle(_translate("make_deepfake_page", "Form"))
        self.label.setText(_translate("make_deepfake_page", "Please select your training data. Possible choices are video and already existing images of faces. \n"
"After data is selected, algorithm for face detection needs to be selected."))
        self.select_video_btn.setText(_translate("make_deepfake_page", "Select video"))
        self.select_pictures_btn.setText(_translate("make_deepfake_page", "Select pictures"))
        self.label_2.setText(_translate("make_deepfake_page", "Preview"))
