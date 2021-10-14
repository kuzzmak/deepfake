from typing import List

from matplotlib.animation import TimedAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt
import torch

from gui.widgets.base_widget import BaseWidget
from utils import tensor_to_np_image


class CustomFigCanvas(FigureCanvasQTAgg, TimedAnimation):

    def __init__(self, n_images: int):

        self._new_data: List[torch.Tensor] = []

        subplot_titles = ['input', 'AA']
        self.n_images = n_images
        self.n_cols = len(subplot_titles)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.axes = [self.fig.add_subplot(self.n_images, self.n_cols, i + 1)
                     for i in range(self.n_images * self.n_cols)]
        [self._hide_axes(axe) for axe in self.axes]
        [self.axes[index].set_title(title)
            for index, title in enumerate(subplot_titles)]

        FigureCanvasQTAgg.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=50, blit=True)

    @staticmethod
    def _hide_axes(axe):
        axe.axes.get_xaxis().set_visible(False)
        axe.axes.get_yaxis().set_visible(False)

    def new_frame_seq(self):
        return iter(range(self.n_images))

    def _step(self, *args):
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            print('err')
            print(e)
            TimedAnimation._stop(self)

    def _draw_frame(self, framedata):
        while len(self._new_data) > 0:
            images = self._new_data[0]
            for row in range(self.n_images):
                axe_input = self.axes[row * self.n_cols]
                img_input = tensor_to_np_image(images[0][row])
                axe_input.imshow(img_input)

                axe_output = self.axes[row * self.n_cols + 1]
                img_output = tensor_to_np_image(images[1][row])
                axe_output.imshow(img_output)
            del self._new_data[0]

    def add_data(self, data: List[torch.Tensor]):
        self._new_data.append(data)


class Preview(BaseWidget):

    refresh_data_sig = qtc.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._init_ui()
        self.refresh_data_sig.connect(self._refresh_data)

    def _init_ui(self):
        layout = qwt.QVBoxLayout()
        self.preview = CustomFigCanvas(4)
        layout.addWidget(self.preview)
        self.setLayout(layout)

    @qtc.pyqtSlot(list)
    def _refresh_data(self, data: List[torch.Tensor]):
        self.preview.add_data(data)
