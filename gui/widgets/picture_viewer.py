from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from core.face import Face
from core.face_alignment.face_aligner import FaceAligner
from common_structures import DialogMessages
from enums import INDEX_TYPE
from gui.widgets.dialog import Dialog
from utils import np_array_to_qicon

DEFAULT_ROLE = qtc.Qt.UserRole + 1


@dataclass
class ImageViewerAction:
    """Class for specifying action for the context menu of the `ImageViewer`
    class. This context menu opens on right button click.

    Args:
        name (str): what is this action called in context menu
        callback (Callable): function which executes on action click
    """
    name: str
    callback: Callable


class StandardItem(qtg.QStandardItem):

    DataRole = DEFAULT_ROLE
    NameRole = qtc.Qt.UserRole + 2
    FaceRole = qtc.Qt.UserRole + 3

    def __init__(self) -> None:
        """Single item displayed in `ImageViewer`. Serves also as a container
        for `Face` metadata. To get data of a particular type from the item, 
        appropriate `qtc.Qt.ItemDataRole` role must be passed to the `data`
        function.

        `Face` metadata contains detected face which is aligned and
        then displayed in the `ImageViewer` and also image name which shows
        below every item.

        Later, when sorting is done and images are moved from one side to
        other, these items are fetched and their `Face` objects are extracted
        and then these "filtered" `Face` objects can be saved to desired
        location.
        """
        super().__init__()
        self.name = ''
        self.image = ''
        self.face = None

    def setData(self, value, role: qtc.Qt.ItemDataRole = DEFAULT_ROLE) -> None:
        if role == StandardItem.NameRole:
            self.name = value
        elif role == StandardItem.DataRole:
            self.image = value
        elif role == StandardItem.FaceRole:
            self.face = value
        else:
            qtg.QStandardItem.setData(self, value, role)

    def data(self, role: qtc.Qt.ItemDataRole = DEFAULT_ROLE):
        if role == StandardItem.NameRole:
            return self.name
        elif role == StandardItem.DataRole:
            return self.image
        elif role == StandardItem.FaceRole:
            return self.face
        return qtg.QStandardItem.data(self, role)

    def type(self):
        return qtc.Qt.UserType


class StandardItemModel(qtg.QStandardItemModel):

    def __init__(
        self,
        icon_size: Tuple[int, int] = (64, 64),
    ) -> None:
        """Model class for the `qwt.QListView` widget.

        Args:
            icon_size (Tuple[int, int], optional): size of the item in
                `ImageViewer`. Defaults to (64, 64).
        """
        super().__init__()
        self.icon_size = icon_size
        self.setItemPrototype(StandardItem())

    def data(self, index, role: qtc.Qt.ItemDataRole = qtc.Qt.DisplayRole):
        if role == qtc.Qt.DecorationRole:
            it = self.itemFromIndex(index)
            item = it.data(qtc.Qt.DecorationRole)
            if item is None:
                item_data = it.data(StandardItem.DataRole)
                it.setData(np_array_to_qicon(item_data), qtc.Qt.DecorationRole)
                name = it.data(StandardItem.NameRole)
                it.setText(name)
            return item
        elif role == qtc.Qt.SizeHintRole:
            icon_size = self.icon_size
            return qtc.QSize(icon_size[0] + 6, icon_size[1] + 16)
        else:
            return qtg.QStandardItemModel.data(self, index, role)


class ContextMenuEventFilter(qtc.QObject):

    def __init__(self, parent: qwt.QWidget) -> None:
        """Event filter for the context menu. Does basic functionality of
        closing context menu when button leaves bounds of the context menu.

        Args:
            parent (qwt.QWidget): context menu
        """
        super(ContextMenuEventFilter, self).__init__()
        self.parent = parent

    def eventFilter(self, source: qtc.QObject, event: qtc.QEvent) -> bool:
        if event.type() == qtc.QEvent.Type.Leave:
            self.parent.close()
        return super().eventFilter(source, event)


class ImageViewer(qwt.QWidget):

    number_of_images_sig = qtc.pyqtSignal(int)
    images_added_sig = qtc.pyqtSignal(list)

    def __init__(self, icon_size: Tuple[int, int] = (64, 64)) -> None:
        """Widget for displaying some sort of the images. Images can be sent
        to this widget in a form of the `np.ndarray` or like path to the image
        on disk.

        Functionality:
            - scrolling
            - single and multiple select
            - deletion of selected images
            - renaming of selected image #TODO

        Args:
            icon_size (Tuple[int, int], optional): size of the icons displayed
                in widget. Defaults to (64, 64).
        """
        super().__init__()
        self.icon_size = icon_size
        self._number_of_images = 0
        self.images_added_sig.connect(self._images_added)
        self._init_ui()

    def _init_ui(self) -> None:
        """Constructs `ImageViewer` ui.
        """
        self.ui_image_viewer = qwt.QListView()
        self.ui_image_viewer.setSelectionMode(
            qwt.QAbstractItemView.ExtendedSelection
        )
        self.ui_image_viewer.viewport().installEventFilter(self)
        self.ui_image_viewer.setSpacing(5)
        self.ui_image_viewer.setViewMode(qwt.QListView.IconMode)
        self.ui_image_viewer.setResizeMode(qwt.QListView.Adjust)
        self.ui_image_viewer.setEditTriggers(
            qwt.QAbstractItemView.NoEditTriggers
        )
        self.ui_image_viewer.setIconSize(qtc.QSize(*self.icon_size))
        self.ui_image_viewer.setMovement(qwt.QListView.Static)
        self.ui_image_viewer.setModel(StandardItemModel(self.icon_size))

        grid = qwt.QVBoxLayout()
        grid.addWidget(self.ui_image_viewer)
        self.setLayout(grid)

        sizePolicy = qwt.QSizePolicy(
            qwt.QSizePolicy.Preferred,
            qwt.QSizePolicy.MinimumExpanding,
        )
        self.setSizePolicy(sizePolicy)
        self._init_context_menu()

    def _init_context_menu(self) -> None:
        """Consctructs context menu which opens on right button click when
        item is selected in `ImageViewer`.
        """
        self.context_menu = qwt.QMenu()
        self.cmef = ContextMenuEventFilter(self.context_menu)
        self.context_menu.installEventFilter(self.cmef)

        rename = self.context_menu.addAction("Rename")
        rename.triggered.connect(self._rename_selected_picture)

        delete = self.context_menu.addAction("Delete")
        delete.triggered.connect(self._remove_selected_images)

    @property
    def number_of_images(self) -> int:
        """How many images are in the `ImageViewer`.

        Returns:
            int: number of images
        """
        return self._number_of_images

    @number_of_images.setter
    def number_of_images(self, value: int) -> None:
        """Sets number of images in the `ImageViewer`.

        Args:
            value (int): number of images to set
        """
        self.number_of_images_sig.emit(value)
        self._number_of_images = value

    def get_all_data(
        self,
        role: qtc.Qt.ItemDataRole,
    ) -> Union[List[np.ndarray], List[Face]]:
        """Getter for all the data in particular `ImageViewer`.

        Args:
            role (qtc.Qt.ItemDataRole): what type of data is requested, see
            `StandardItem` roles

        Returns:
            Union[List[np.ndarray], List[Face]]: list of `Face` or `np.ndarray`
                objects
        """
        model = self.ui_image_viewer.model()
        # number of items in particular ImageViewer
        total = model.rowCount()
        # construct QModelIndex for every item and then get item on that index
        # with specific role
        data = [model.data(model.index(row, 0), role) for row in range(total)]
        return data

    def add_actions_to_context_menu(
        self,
        actions: List[ImageViewerAction],
    ) -> None:
        """Adds additional action to the context menu which opens on right
        button click when one or more items are selected in `ImageViwer`.

        Args:
            actions (List[ImageViewerAction]): list of actions to add
        """
        for action in actions:
            act = self.context_menu.addAction(action.name)
            act.triggered.connect(action.callback)

    def clear(self) -> None:
        """Removes all data from `ImageViewer`.
        """
        self.ui_image_viewer.model().removeRows(
            0,
            self.ui_image_viewer.model().rowCount(),
        )

    def get_selected_indices(
        self,
        index_type: INDEX_TYPE = INDEX_TYPE.QMODELINDEX,
    ) -> List[Union[qtc.QModelIndex, int]]:
        """Getter for the selected indices inside `QListView`.

        Returns:
            List[Union[qtc.QModelIndex, int]]: selected indices
        """
        indices = self.ui_image_viewer.selectedIndexes()
        if index_type == INDEX_TYPE.INT:
            indices = list(map(lambda idx: idx.row(), indices))
        return indices

    def get_data_from_selected_indices(
        self,
        role: qtc.Qt.ItemDataRole = StandardItem.DataRole,
    ) -> Tuple[List[qtc.QModelIndex], Any]:
        """Getter for the data of the selected indices. Data is the name of
        the image if image path was sent to the `PictureViewer` or newly
        generated name (number of the newly inserted image) if `np.ndarray`
        was sent to the `PictureViewer`.

        Args:
            role (qtc.Qt.ItemDataRole, optional): type of the data which is
                being fetched. Defaults to StandardItem.PathRole.

        Returns:
            Tuple[List[qtc.QModelIndex], Any]: list of indices, requested data
        """
        indices = self.get_selected_indices()
        data = [
            self.ui_image_viewer.model().data(index, role) for index in indices
        ]
        return indices, data

    def _remove_item_from_viewer(self, row: int) -> None:
        """Removes item from the model of the viewer on selected index.

        Args:
            row (int): index on which to remove item
        """
        self.ui_image_viewer.model().removeRow(row)

    def remove_selected(self) -> None:
        """Removes selected images from the `ImageViewer`.
        """
        indices = self.get_selected_indices()
        # indices need to be sorted from the biggest to the lowest and
        # removed in that order, that way indices before don't "move"
        indices_sorted = sorted(
            indices,
            key=lambda index: index.row(),
            reverse=True,
        )
        [self._remove_item_from_viewer(index.row())
         for index in indices_sorted]

    def _remove_selected_images(self) -> None:
        """Function for removing selected images from the `PictureViewer` and
        from the disk.
        """
        def remove_fn(remove: bool) -> None:
            if remove:
                self.remove_selected()

                # op = IO_OP(IO_OPERATION_TYPE.DELETE, path)
                # self.app.io_op_sig.emit(op)

        dialog_msg = DialogMessages.DELETE(
            'Do you really want to delete selected images?'
        )
        dialog = Dialog(dialog_msg, self)
        dialog.remove_sig.connect(remove_fn)
        dialog.exec()

    def _rename_selected_picture(self):
        ...

    @qtc.pyqtSlot(list)
    def _images_added(self, images: List[Union[np.ndarray, Face]]):
        """`qtc.pyqtSlot` which triggers when new images in form of an `np.ndarray`
        or `Face` object are emitted to show in `ImageViewer`.

        Parameters
        ----------
        images : List[Union[np.ndarray, Face]]
            list of `np.ndarray` images of `Face` faces
        """
        for image in images:
            item = StandardItem()
            if isinstance(image, Face):
                name = image.raw_image.name
                item.setData(image, StandardItem.FaceRole)
                item.setData(name, StandardItem.NameRole)
                FaceAligner.align_face(image, 64)
                item.setData(
                    image.aligned_image,
                    StandardItem.DataRole,
                )
            else:
                name = str(self.ui_image_viewer.model().rowCount())
                item.setData(name, StandardItem.NameRole)
                item.setData(image, StandardItem.DataRole)
            # add image to ImageViewer
            self.ui_image_viewer.model().appendRow(item)
            # update number of images of the ImageViewer
            self.number_of_images = self.ui_image_viewer.model().rowCount()

    def eventFilter(self, source: qtc.QObject, event: qtc.QEvent) -> bool:
        if source == self.ui_image_viewer.viewport():
            if event.type() == qtc.QEvent.MouseButtonPress:
                self.context_menu.close()
                if event.button() == qtc.Qt.MouseButton.RightButton:
                    selected = self.get_selected_indices()
                    if selected:
                        self._show_context_menu(event)

        return super().eventFilter(source, event)

    def _show_context_menu(self, event: qtc.QEvent):
        self.context_menu.exec_(self.mapToGlobal(event.pos()))
