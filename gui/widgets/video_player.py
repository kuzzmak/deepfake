from constants import MIN_VIDEO_HEIGHT, MIN_VIDEO_WIDTH
import PyQt5.QtWidgets as qwt
from PyQt5 import QtCore as qtc
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget


class VideoPlayer(qwt.QWidget):

    video_selection = qtc.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mediaPlayer = QMediaPlayer(
            None, QMediaPlayer.VideoSurface)

        videoWidget = QVideoWidget()

        self.playButton = qwt.QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(
            self.style().standardIcon(qwt.QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = qwt.QSlider(qtc.Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        # wid = qwt.QWidget(self)

        controlLayout = qwt.QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = qwt.QVBoxLayout()
        # layout.addWidget(qwt.QLabel(text='labela'))  # izbrisati
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        # spacerItem1 = qwt.QSpacerItem(
        #     20, 40, qwt.QSizePolicy.Expanding, qwt.QSizePolicy.Expanding)
        # layout.addItem(spacerItem1)

        # lay = qwt.QHBoxLayout()
        # lay.addLayout(layout)
        # spacerItem2 = qwt.QSpacerItem(
        #     20, 40, qwt.QSizePolicy.Expanding, qwt.QSizePolicy.Expanding)
        # lay.addItem(spacerItem2)
        # # lay.addWidget(qwt.QPushButton(text='pritisni'))

        # self.setLayout(lay)

        self.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

        sizePolicy = qwt.QSizePolicy(
            qwt.QSizePolicy.Expanding, qwt.QSizePolicy.Expanding)
        self.setSizePolicy(sizePolicy)
        videoWidget.setMinimumSize(MIN_VIDEO_WIDTH, 250)

        self.video_selection.connect(self.video_selected)

    @qtc.pyqtSlot(str)
    def video_selected(self, video_path: str):
        self.mediaPlayer.setMedia(
            QMediaContent(qtc.QUrl.fromLocalFile(video_path)))
        self.playButton.setEnabled(True)

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(qwt.QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                self.style().standardIcon(qwt.QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
