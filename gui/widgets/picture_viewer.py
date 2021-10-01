import os
from typing import Any, List, Tuple, Union

import numpy as np
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from common_structures import DialogMessages
from gui.widgets.dialog import Dialog
from utils import np_array_to_qicon, qicon_from_path

DEFAULT_ROLE = qtc.Qt.UserRole + 1


class StandardItem(qtg.QStandardItem):

    DataRole = DEFAULT_ROLE
    NameRole = qtc.Qt.UserRole + 2
    PathRole = qtc.Qt.UserRole + 3

    def __init__(self, *args, **kwargs) -> None:
        super(StandardItem, self).__init__(*args, **kwargs)
        self.name = ''
        self.path = ''
        self.image = ''

    def setData(self, value, role=DEFAULT_ROLE):
        if role == StandardItem.PathRole:
            self.path = value
        elif role == StandardItem.NameRole:
            self.name = value
        elif role == StandardItem.DataRole:
            self.image = value
        else:
            qtg.QStandardItem.setData(self, value, role)

    def data(self, role=DEFAULT_ROLE):
        if role == StandardItem.PathRole:
            return self.path
        elif role == StandardItem.NameRole:
            return self.name
        elif role == StandardItem.DataRole:
            return self.image
        return qtg.QStandardItem.data(self, role)

    def type(self):
        return qtc.Qt.UserType


class StandardItemModel(qtg.QStandardItemModel):

    def __init__(
        self,
        icon_size: Tuple[int, int] = (64, 64),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.icon_size = icon_size

        self.setItemPrototype(StandardItem())

    def data(self, index, role=qtc.Qt.DisplayRole):
        if role == qtc.Qt.DecorationRole:

            it = self.itemFromIndex(index)
            item = it.data(qtc.Qt.DecorationRole)

            if item is None:

                item_data = it.data(StandardItem.DataRole)
                path = it.data(StandardItem.DataRole)

                if isinstance(item_data, str):
                    image = qicon_from_path(path)
                elif isinstance(item_data, np.ndarray):
                    image = np_array_to_qicon(item_data)

                it.setData(image, qtc.Qt.DecorationRole)
                name = it.data(StandardItem.NameRole)
                it.setText(name)

            return item

        elif role == qtc.Qt.SizeHintRole:
            icon_size = self.icon_size
            return qtc.QSize(icon_size[0] + 6, icon_size[1] + 12)
        else:
            return qtg.QStandardItemModel.data(self, index, role)


class ContextMenuEventFilter(qtc.QObject):

    def __init__(self, parent) -> None:
        super(ContextMenuEventFilter, self).__init__()
        self.parent = parent

    def eventFilter(self, source: qtc.QObject, event: qtc.QEvent) -> bool:
        if event.type() == qtc.QEvent.Type.Leave:
            self.parent.close()
        return super().eventFilter(source, event)


class ImageViewer(qwt.QWidget):

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

        self.images_added_sig.connect(self._images_added)

        self._init_ui()

    def _init_ui(self) -> None:
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

        self.init_context_menu()

    def init_context_menu(self):
        self.context_menu = qwt.QMenu()
        self.cmef = ContextMenuEventFilter(self.context_menu)
        self.context_menu.installEventFilter(self.cmef)

        rename = self.context_menu.addAction("Rename")
        rename.triggered.connect(self._rename_selected_picture)

        delete = self.context_menu.addAction("Delete")
        delete.triggered.connect(self._remove_selected_images)

    def _get_selected_indices(self) -> List[qtc.QModelIndex]:
        """Getter for the selected indices inside `QListView`.

        Returns:
            List[qtc.QModelIndex]: selected indices
        """
        return self.ui_image_viewer.selectedIndexes()

    def _get_data_from_selected_indices(
        self,
        role=StandardItem.PathRole,
    ) -> Tuple[List[qtc.QModelIndex], Any]:
        """Getter for the data of the selected indices. Data is the name of
        the image if image path was sent to the `PictureViewer` or newly
        generated name (number of the newly inserted image) if `np.ndarray`
        was sent to the `PictureViewer`.

        Args:
            role (qtc.ItemDataRole, optional): type of the data which is being
                fetched. Defaults to StandardItem.PathRole.

        Returns:
            Tuple[List[qtc.QModelIndex], Any]: list of indices, requested data
        """
        indices = self._get_selected_indices()
        data = [self.ui_image_viewer.model().data(index, role)
                for index in indices]
        return indices, data

    def _remove_item_from_viewer(self, row: int):
        """Removes item from the model of the viewer on selected index.

        Args:
            row (int): index on which to remove item
        """
        self.ui_image_viewer.model().removeRow(row)

    def _remove_selected_images(self):
        """Function for removing selected images from the `PictureViewer` and
        from the disk.
        """
        indices, data = self._get_data_from_selected_indices()

        def remove_fn(remove: bool):
            if remove:
                # indices need to be sorted from the biggest to the lowest and
                # removed in that order, that way indices before don't "move"
                indices_sorted = sorted(
                    indices,
                    key=lambda index: index.row(),
                    reverse=True,
                )

                [self._remove_item_from_viewer(index.row())
                 for index in indices_sorted]

                # op = IO_OP(IO_OPERATION_TYPE.DELETE, path)
                # self.app.io_op_sig.emit(op)

        dialog_msg = DialogMessages.DELETE(
            f'Do you really want to delete: \n{data}'
        )
        dialog = Dialog(dialog_msg, self)
        dialog.remove_sig.connect(remove_fn)
        dialog.exec()

    def _rename_selected_picture(self):
        ...

    @qtc.pyqtSlot(list)
    def _images_added(self, images: List[Union[str, np.ndarray]]):
        """pyqtSlot which triggers when new image paths or images
        in form of an `np.ndarray` are emitted to show in `ImageViewer`.

        Parameters
        ----------
        images : List[Union[str, np.ndarray]]
            list of image paths or `np.ndarray`s
        """
        for image in images:
            item = StandardItem()
            if isinstance(image, str):
                name = os.path.splitext(os.path.basename(image))[0]
                item.setData(image, StandardItem.PathRole)
            else:
                name = str(self.ui_image_viewer.model().rowCount())
                item.setData(name, StandardItem.PathRole)
            item.setData(image, StandardItem.DataRole)
            item.setData(name, StandardItem.NameRole)
            self.ui_image_viewer.model().appendRow(item)

    def eventFilter(self, source: qtc.QObject, event: qtc.QEvent) -> bool:
        if source == self.ui_image_viewer.viewport():
            if event.type() == qtc.QEvent.MouseButtonPress:
                self.context_menu.close()
                if event.button() == qtc.Qt.MouseButton.RightButton:
                    selected = self._get_selected_indices()
                    if selected:
                        self._show_context_menu(event)

        return super().eventFilter(source, event)

    def _show_context_menu(self, event: qtc.QEvent):
        action = self.context_menu.exec_(self.mapToGlobal(event.pos()))
