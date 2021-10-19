from operator import itemgetter
from typing import List

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from core.face import Face
from core.sort import sort_faces_by_image_hash
from enums import INDEX_TYPE
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import MinimalSizePolicy
from gui.widgets.picture_viewer import ImageViewer, ImageViewerAction


class ImageViewerWithImageCount(qwt.QWidget):

    label_value_sig = qtc.pyqtSignal(int)

    def __init__(self, label: str):
        """Widget containing `ImageViewer` and label which show how many
        images are in this `ImageViewer`.

        Args:
            label (str): label describing which `ImageViewer` is this, shows
                before image count
        """
        super().__init__()
        self.label = label
        self._init_ui()
        self.label_value_sig.connect(self._label_value_changed)

    def _init_ui(self):
        """Constructs widget with `ImageViever` and label describing how many
        images are in this `ImageViewer`.
        """
        layout = qwt.QVBoxLayout()
        label_row_wgt = qwt.QWidget()
        label_row_wgt_layout = qwt.QHBoxLayout()
        label_row_wgt.setLayout(label_row_wgt_layout)
        label = qwt.QLabel(text=self.label)
        label.setSizePolicy(MinimalSizePolicy)
        label_row_wgt_layout.addWidget(label)
        self.label_value = qwt.QLabel(text='0')
        label_row_wgt_layout.addWidget(self.label_value)
        self.image_viewer = ImageViewer()
        self.image_viewer.number_of_images_sig.connect(
            self._label_value_changed
        )
        layout.addWidget(label_row_wgt)
        layout.addWidget(self.image_viewer)
        self.setLayout(layout)

    @qtc.pyqtSlot(int)
    def _label_value_changed(self, value: int):
        """Changes value of the label which show how many images are in the 
        `ImageViewer`.

        Args:
            value (int): new value
        """
        self.label_value.setText(str(value))


class ImageViewerSorter(BaseWidget):

    sort_sig = qtc.pyqtSignal(int)

    def __init__(self):
        """Widget containing og two `ImageViewer` widgets where one on the
        left side serves as an `ImageViewer` where `Face` metadata objects
        reside which will be used for training. `ImageViewer` on the left
        contains `Face` object which will be removed/deleted/not used.
        """
        super().__init__()
        self._faces_cache: List[Face] = []
        self._init_ui()
        self.sort_sig.connect(self._sort)

    def _init_ui(self):
        """Construct ui with two `ImaveViewer`'s. Adds new action to the
        context menu where one can move images from one viewer to another one.
        """
        layout = qwt.QVBoxLayout()

        viewers_wgt = qwt.QWidget()
        viewers_wgt_layout = qwt.QHBoxLayout()
        viewers_wgt.setLayout(viewers_wgt_layout)

        self.left_viewer_wgt = ImageViewerWithImageCount(
            'Number of images_ok: '
        )
        viewers_wgt_layout.addWidget(self.left_viewer_wgt)

        self.right_viewer_wgt = ImageViewerWithImageCount(
            'Number of images_not_ok: '
        )
        viewers_wgt_layout.addWidget(self.right_viewer_wgt)
        layout.addWidget(viewers_wgt)

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
            selected = from_viewer.get_selected_indices(INDEX_TYPE.INT)
            # tuple of selected Face metadata objects
            selected_faces = itemgetter(*selected)(self.faces_cache)
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
    def faces_cache(self) -> List[Face]:
        """List of `Face` metadata objects, used for easier sorting and then
        later, easier saving of metadata objects which will be used for
        training.

        Returns:
            List[Face]: list of `Face` objects metadata
        """
        return self._faces_cache

    @faces_cache.setter
    def faces_cache(self, faces: List[Face]) -> None:
        """Sets `faces_metadata` property.

        Args:
            faces (List[Face]): list of `Face` objects metadata
        """
        self._faces_cache = faces

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

    @qtc.pyqtSlot(int)
    def _sort(self, eps: int) -> None:
        """Sorts `Face` metadata objects by some image sorting method.

        Args:
            eps (int): [description]
        """
        # TODO implement other sorting methods and make this more generic
        indices_ok, indices_not_ok = sort_faces_by_image_hash(
            self.faces_cache,
            eps,
        )
        self.image_viewer_images_ok.clear()
        self.image_viewer_images_not_ok.clear()
        if len(indices_ok) > 0:
            faces_ok = itemgetter(*indices_ok)(self.faces_cache)
            self.image_viewer_images_ok.images_added_sig.emit(list(faces_ok))
        if len(indices_not_ok) > 0:
            faces_not_ok = itemgetter(*indices_not_ok)(self.faces_cache)
            self.image_viewer_images_not_ok.images_added_sig.emit(
                list(faces_not_ok)
            )
