import PyQt6.QtWidgets as qwt
from PyQt6 import QtCore as qtc
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget


class VideoPlayer(qwt.QWidget):

    video_selection = qtc.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.media_player = QMediaPlayer()

        video_widget = QVideoWidget()

        self.play_button = qwt.QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.setIcon(
            self.style().standardIcon(qwt.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_button.clicked.connect(self.play)

        self.position_slider = qwt.QSlider(qtc.Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)

        control_layout = qwt.QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.position_slider)

        layout = qwt.QVBoxLayout()
        layout.addWidget(video_widget)
        layout.addLayout(control_layout)

        self.setLayout(layout)

        self.media_player.setVideoOutput(video_widget)
        self.media_player.playbackStateChanged.connect(
            self.playback_state_changed
        )
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

        self.video_selection.connect(self.video_selected)

    @qtc.pyqtSlot(str)
    def video_selected(self, video_path: str):
        self.media_player.setSource(qtc.QUrl.fromLocalFile(video_path))
        self.play_button.setEnabled(True)

    def play(self):
        if self.media_player.playbackState() == \
                QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def playback_state_changed(self, state):
        if self.media_player.playbackState() == \
                QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(
                self.style().standardIcon(
                    qwt.QStyle.StandardPixmap.SP_MediaPause)
            )
        else:
            self.play_button.setIcon(
                self.style().standardIcon(
                    qwt.QStyle.StandardPixmap.SP_MediaPlay))

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def set_position(self, position):
        self.media_player.setPosition(position)
