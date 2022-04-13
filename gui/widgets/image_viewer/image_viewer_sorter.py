from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from core.face import Face
from core.sort import sort_faces_by_image_hash
from enums import SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.image_viewer.image_viewer import (
    ImageViewer,
    ImageViewerAction,
    StandardItem,
)
from gui.widgets.image_viewer.image_viewer_with_images_count import \
    ImageViewerWithImageCount


class ImageViewerSorter(BaseWidget):

    sort_sig = qtc.pyqtSignal(int)
    data_paths_sig = qtc.pyqtSignal(list)

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        """Widget containing og two `ImageViewer` widgets where one on the
        left side serves as an `ImageViewer` where `Face` metadata objects
        reside which will be used for training. `ImageViewer` on the left
        contains `Face` object which will be removed/deleted/not used.
        """
        super().__init__(signals)
        self._init_ui()
        self.sort_sig.connect(self._sort)
        self.data_paths_sig.connect(self._data_paths_changed)

    def _init_ui(self):
        """Construct ui with two `ImaveViewer`'s. Adds new action to the
        context menu where one can move images from one viewer to another one.
        """
        layout = qwt.QVBoxLayout()

        viewers_wgt = qwt.QWidget()
        viewers_wgt_layout = qwt.QHBoxLayout()
        viewers_wgt.setLayout(viewers_wgt_layout)
        signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER
            ]
        }
        self.left_viewer_wgt = ImageViewerWithImageCount(signals)
        viewers_wgt_layout.addWidget(self.left_viewer_wgt)

        self.right_viewer_wgt = ImageViewerWithImageCount(signals)
        viewers_wgt_layout.addWidget(self.right_viewer_wgt)
        layout.addWidget(viewers_wgt)
        self.image_viewer_images_ok.images_changed_sig.connect(
            self._clear_images_not_ok_viewer
        )

        self.setLayout(layout)

        def _move_to_side(side: str) -> None:
            """Moves selected faces to other `ImageViewer`.

            Args:
                side (str): move selected faces to which side
            """
            viewers = (
                self.image_viewer_images_not_ok,
                self.image_viewer_images_ok,
            )
            if side == 'right':
                viewers = viewers[::-1]
            # viewer where selection is made, viewer where to send selected
            # faces
            from_viewer, to_viewer = viewers
            selected_faces = from_viewer.get_data_from_selected_indices(
                StandardItem.FaceRole,
            )[1]
            # make list of the selected faces
            if isinstance(selected_faces, Face):
                selected_faces = [selected_faces]
            else:
                selected_faces = list(selected_faces)
            # send selected faces to opposite imageViewer
            to_viewer.images_added_sig.emit(selected_faces)
            # remove selected faces from ImageViewer containing them
            from_viewer.remove_selected()

        self.image_viewer_images_ok.add_actions_to_context_menu(
            [
                ImageViewerAction(
                    'Move to right',
                    lambda: _move_to_side('right'),
                )
            ]
        )
        self.image_viewer_images_not_ok.add_actions_to_context_menu(
            [
                ImageViewerAction(
                    'Move to left',
                    lambda: _move_to_side('left'),
                )
            ]
        )

    @property
    def image_viewer_images_ok(self) -> ImageViewer:
        """Getter for the left `ImageViewe` which contains `Face` metadata
        objects which will be used for training.

        Returns:
            ImageViewer: left side `ImageViewer`
        """
        return self.left_viewer_wgt.image_viewer

    @property
    def image_viewer_images_not_ok(self) -> ImageViewer:
        """Getter for the right `ImageViewer` which contains `Face` metadata
        objects which will not be used for training.

        Returns:
            ImageViewer: right side `ImageViewer`
        """
        return self.right_viewer_wgt.image_viewer

    def _clear_images_not_ok_viewer(self) -> None:
        """Clears viewer with images not ok when some data change happens.
        """
        self.image_viewer_images_not_ok.clear()

    @qtc.pyqtSlot(int)
    def _sort(self, eps: int) -> None:
        """Sorts `Face` metadata objects by some image sorting method.

        Args:
            eps (int): [description]
        """
        ok_images = self.image_viewer_images_ok.get_all_data(
            StandardItem.FaceRole
        )
        not_ok_images = self.image_viewer_images_not_ok.get_all_data(
            StandardItem.FaceRole
        )
        all_data = [*ok_images, *not_ok_images]
        # TODO implement other sorting methods and make this more generic
        indices_ok, indices_not_ok = sort_faces_by_image_hash(
            all_data,
            eps,
        )
        self.image_viewer_images_ok.clear()
        self.image_viewer_images_not_ok.clear()
        if len(indices_ok) > 0:
            faces_ok = itemgetter(*indices_ok)(all_data)
            if not isinstance(faces_ok, tuple):
                faces_ok = [faces_ok]
            else:
                faces_ok = list(faces_ok)
            self.image_viewer_images_ok.images_added_sig.emit(faces_ok)
        if len(indices_not_ok) > 0:
            faces_not_ok = itemgetter(*indices_not_ok)(all_data)
            if not isinstance(faces_not_ok, tuple):
                faces_not_ok = [faces_not_ok]
            else:
                faces_not_ok = list(faces_not_ok)
            self.image_viewer_images_not_ok.images_added_sig.emit(faces_not_ok)

    @qtc.pyqtSlot(list)
    def _data_paths_changed(self, data_paths: List[Path]) -> None:
        self.left_viewer_wgt.data_paths_sig.emit(data_paths)
