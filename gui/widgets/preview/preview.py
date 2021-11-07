import sys
import traceback
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

    def __init__(self, num_of_rows: int):
        self._new_data: List[torch.Tensor] = []
        self.subplot_titles = [
            'Face A',
            'A->A',
            'A->B',
            'Face B',
            'B->B',
            'B->A',
        ]
        self.num_of_rows = num_of_rows
        self.num_of_cols = len(self.subplot_titles)
        fig = Figure(figsize=(5, 5), dpi=100)
        self.axes = [fig.add_subplot(self.num_of_rows, self.num_of_cols, i + 1)
                     for i in range(self.num_of_rows * self.num_of_cols)]
        self._hide_axes()
        self._refresh_axes_titles()

        FigureCanvasQTAgg.__init__(self, fig)
        TimedAnimation.__init__(self, fig, interval=200, blit=True)

    def _hide_axes(self):
        """Hides axes and axe values from every single image displayed.
        """
        for axis in self.axes:
            axis.axes.get_xaxis().set_visible(False)
            axis.axes.get_yaxis().set_visible(False)

    def new_frame_seq(self):
        return iter(range(self.num_of_rows))

    def _step(self, *args):
        try:
            TimedAnimation._step(self, *args)
        except Exception:
            print(traceback.format_exc())
            TimedAnimation._stop(self)
            sys.exit(0)

    def _refresh_axes_titles(self):
        """Sets titles on axes after they have been refreshed.
        """
        for i in range(len(self.subplot_titles)):
            self.axes[i].set_title(self.subplot_titles[i])

    def _draw_frame(self, framedata):
        while len(self._new_data) > 0:
            # get oldest data sent to preview widget
            images = self._new_data[0]
            # iterate through every row
            for row in range(self.num_of_rows):
                # through every column
                for col in range(self.num_of_cols):
                    # get corresponding axe from an array
                    axe = self.axes[row * self.num_of_cols + col]
                    try:
                        # convert tensor from model or model input to "normal"
                        # image which can be displayed in matplotlib canvas
                        img_input = tensor_to_np_image(images[col][row])
                        axe.cla()
                        axe.imshow(img_input, animated=True)
                    except IndexError:
                        ...
            del self._new_data[0]

            self._refresh_axes_titles()

    def add_data(self, data: List[torch.Tensor]):
        self._new_data.append(data)


class Preview(BaseWidget):

    refresh_data_sig = qtc.pyqtSignal(list)

    def __init__(self, num_of_rows: int):
        """Vidget used to display progress of the training process. It
        includes original face A, face A through decoder A, face A through
        decoder B, original face B, face B through decoder A and face B
        through decoder B.

        Args:
            num_of_rows (int): how many rows of pictures on preview
        """
        super().__init__()
        self._init_ui(num_of_rows)
        self.refresh_data_sig.connect(self._refresh_data)

    def _init_ui(self, num_of_rows: int):
        """Initiates preview widget with `num_of_rows` rows of images.

        Args:
            num_of_rows (int): how many rows of images
        """
        layout = qwt.QVBoxLayout()
        self.preview = CustomFigCanvas(num_of_rows)
        layout.addWidget(self.preview)
        self.setLayout(layout)

    @qtc.pyqtSlot(list)
    def _refresh_data(self, data: List[torch.Tensor]):
        """Sends data to preview widget.

        Args:
            data (List[torch.Tensor]): list of tensor images
        """
        self.preview.add_data(data)
