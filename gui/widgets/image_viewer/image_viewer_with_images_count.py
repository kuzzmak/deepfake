from pathlib import Path
from typing import Dict, List, Optional

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt
from config import APP_CONFIG

from enums import SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import HWidget, HorizontalSpacer, MinimalSizePolicy
from gui.widgets.image_viewer.image_viewer import ImageViewer


class ImageViewerWithImageCount(BaseWidget):

    label_value_sig = qtc.pyqtSignal(int)
    data_paths_sig = qtc.pyqtSignal(list)  # list of str

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        """Widget containing `ImageViewer` and label which show how many
        images are in this `ImageViewer`.
        """
        super().__init__(signals)
        self._init_ui()
        self.label_value_sig.connect(self._label_value_changed)
        self.data_paths_sig.connect(self._data_paths_changed)
        self.image_viewer.current_page_sig.connect(self._current_page_changed)

    def _init_ui(self):
        """Constructs widget with `ImageViever` and label describing how many
        images are in this `ImageViewer`.
        """
        layout = qwt.QVBoxLayout()
        label_row_wgt = qwt.QWidget()
        label_row_wgt_layout = qwt.QHBoxLayout()
        label_row_wgt.setLayout(label_row_wgt_layout)
        label = qwt.QLabel(text='images: ')
        label.setSizePolicy(MinimalSizePolicy)
        label_row_wgt_layout.addWidget(label)
        self.label_value = qwt.QLabel(text='0')
        label_row_wgt_layout.addWidget(self.label_value)

        label_row_wgt_layout.addWidget(qwt.QLabel(text='images per page'))
        self.num_of_images_in_viewer = qwt.QComboBox()
        label_row_wgt_layout.addWidget(self.num_of_images_in_viewer)
        for val in APP_CONFIG \
                .app \
                .gui \
                .widgets \
                .image_viewer_sorter \
                .images_per_page_options:
            self.num_of_images_in_viewer.addItem(str(val), val)
        self.num_of_images_in_viewer.currentTextChanged.connect(
            self._images_per_page_changed
        )

        self.previous_page_btn = qwt.QPushButton(text='previous page')
        label_row_wgt_layout.addWidget(self.previous_page_btn)
        self.previous_page_btn.clicked.connect(self._previous_page_changed)

        self.next_page_btn = qwt.QPushButton(text='next page')
        label_row_wgt_layout.addWidget(self.next_page_btn)
        self.next_page_btn.clicked.connect(self._next_page_changed)

        self.image_viewer = ImageViewer(
            signals={
                SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                    SIGNAL_OWNER.MESSAGE_WORKER
                ]
            }
        )
        self.image_viewer.number_of_images_sig.connect(
            self._label_value_changed
        )
        self.image_viewer.images_loading_sig.connect(
            self._images_loading_changed
        )
        layout.addWidget(label_row_wgt)
        layout.addWidget(self.image_viewer)

        page_row = HWidget()
        layout.addWidget(page_row)
        page_row.layout().setContentsMargins(0, 0, 0, 0)
        page_row.layout().addItem(HorizontalSpacer)
        page_label = qwt.QLabel(text='page ')
        page_row.layout().addWidget(page_label)
        self.current_page_label = qwt.QLabel(text='0')
        page_row.layout().addWidget(self.current_page_label)
        divider_label = qwt.QLabel(text='/')
        page_row.layout().addWidget(divider_label)
        self.total_pages_label = qwt.QLabel(text='0')
        page_row.layout().addWidget(self.total_pages_label)
        page_row.layout().addItem(HorizontalSpacer)

        self.setLayout(layout)

    @qtc.pyqtSlot(int)
    def _current_page_changed(self, page: int) -> None:
        self.current_page_label.setText(str(page))

    @qtc.pyqtSlot(str)
    def _images_per_page_changed(self, text: str) -> None:
        """Slot that triggers when user selected some other number of
        images per page.

        Args:
            text (str): number of images per page
        """
        self.image_viewer.images_per_page.emit(int(text))

    @qtc.pyqtSlot(int)
    def _label_value_changed(self, value: int):
        """Changes value of the label which show how many images are in the
        `ImageViewer`.

        Args:
            value (int): new value
        """
        self.label_value.setText(str(value))

    @qtc.pyqtSlot(list)
    def _data_paths_changed(self, data_paths: List[Path]) -> None:
        """Slot that triggers when new images are being loaded into
        the `ImageViewer`.

        Args:
            data_paths (List[Path]): list of paths
        """
        self.image_viewer.data_paths_sig.emit(data_paths)

    @qtc.pyqtSlot()
    def _next_page_changed(self) -> None:
        """Triggers when next page of images should load.
        """
        self.image_viewer.next_page_sig.emit()

    @qtc.pyqtSlot()
    def _previous_page_changed(self) -> None:
        """Triggers when previous page of images should load.
        """
        self.image_viewer.previous_page_sig.emit()

    @qtc.pyqtSlot(bool)
    def _images_loading_changed(self, status: bool) -> None:
        """Slot that triggers when image loading is in process.

        Args:
            status (bool): `True` if images are loading, `False` otherwise
        """
        self.enable_widget(self.previous_page_btn, not status)
        self.enable_widget(self.next_page_btn, not status)
        self.enable_widget(self.num_of_images_in_viewer, not status)
