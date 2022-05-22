from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from core.worker import Worker
from enums import JOB_TYPE, SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    ApplyIcon,
    Button,
    CancelIconButton,
    DeviceWidget,
    HWidget,
    HorizontalSpacer,
    NoMarginLayout,
    VerticalSpacer,
)


class EventTypes:
    """Stores a string name for each event type.

    With PySide2 str() on the event type gives a nice string name,
    but with PyQt5 it does not. So this method works with both systems.
    """

    def __init__(self):
        """Create mapping for all known event types."""
        self.string_name = {}
        for name in vars(qtc.QEvent.Type):
            attribute = getattr(qtc.QEvent.Type, name)
            if isinstance(attribute, qtc.QEvent.Type):
                self.string_name[attribute] = name

    def as_string(self, event: qtc.QEvent.Type) -> str:
        """Return the string name for this event."""
        try:
            return self.string_name[event]
        except KeyError:
            return f"UnknownEvent:{event}"


class ImageLabel(qwt.QLabel):

    image_path_sig = qtc.pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._init_ui()

        self._image_path = None

        self.image_path_sig.connect(self._set_image)

    @property
    def image_path(self) -> Union[Path, None]:
        if self._image_path is not None:
            return Path(self._image_path)
        return None

    def _init_ui(self) -> None:
        self.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')
        self.setAcceptDrops(True)

    def setPixmap(self, image):
        super().setPixmap(image)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(qtc.Qt.DropAction.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self._set_image(file_path)
            event.accept()
        else:
            event.ignore()

    @qtc.pyqtSlot(str)
    def _set_image(self, file_path: str):
        pixmap = qtg.QPixmap(file_path)
        pixmap = pixmap.scaled(
            128,
            128,
            qtc.Qt.AspectRatioMode.KeepAspectRatio,
            qtc.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class InferDFDetectorWidget(BaseWidget):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self._threads: Dict[JOB_TYPE, Tuple[qtc.QThread, Worker]] = dict()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)

        self.devices = DeviceWidget()
        layout.addWidget(self.devices)

        self.image_lbl = ImageLabel()
        layout.addWidget(self.image_lbl)
        self.image_lbl.installEventFilter(self)

        select_model_row = HWidget()
        layout.addWidget(select_model_row)
        self._select_model_btn = Button('select')
        select_model_row.layout().addWidget(self._select_model_btn)
        self._select_model_btn.clicked.connect(self._select_model)
        self._model_loaded_ibtn = CancelIconButton()
        select_model_row.layout().addWidget(self._model_loaded_ibtn)
        self.model_loaded_lbl = qwt.QLabel(text='model NOT loaded')
        select_model_row.layout().addWidget(self.model_loaded_lbl)
        select_model_row.layout().addItem(HorizontalSpacer())

        self._start_inference_btn = Button('start inference')
        layout.addWidget(self._start_inference_btn)
        self._start_inference_btn.clicked.connect(self._start_inference)

        self.setMaximumWidth(400)

        layout.addItem(VerticalSpacer())

    @qtc.pyqtSlot()
    def _start_inference(self) -> None:
        ...

    @qtc.pyqtSlot()
    def _select_model(self) -> None:
        path = qwt.QFileDialog.getOpenFileName(self, 'Select model')
        if path != ('', ''):
            path = path[0]
        else:
            return
        self._model_loaded_ibtn.setIcon(ApplyIcon())
        self._select_model_btn.setText('model loaded')

    def _select_image(self) -> None:
        path = qwt.QFileDialog.getOpenFileName(self, 'Select an image')
        if path != ('', ''):
            path = path[0]
        else:
            return
        self.image_lbl.image_path_sig.emit(path)

    def eventFilter(self, source, event: qtc.QEvent):
        if event.type() == qtc.QEvent.Type.Enter:
            self.setCursor(qtg.QCursor(qtc.Qt.CursorShape.PointingHandCursor))
        elif event.type() == qtc.QEvent.Type.Leave:
            self.setCursor(qtg.QCursor(qtc.Qt.CursorShape.ArrowCursor))
        elif event.type() == qtc.QEvent.Type.MouseButtonPress:
            self._select_image()
        return super().eventFilter(source, event)
