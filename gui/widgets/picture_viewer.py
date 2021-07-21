import os
from typing import List, Union

import numpy as np

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.widgets.dialog import Dialog

from utils import np_array_to_qicon, qicon_from_path

from common_structures import DialogMessages, IO_OP

from enums import IO_OP_TYPE

DEFAULT_ROLE = qtc.Qt.UserRole + 1


class StandardItem(qtg.QStandardItem):

    DataRole = DEFAULT_ROLE
    NameRole = qtc.Qt.UserRole + 2
    PathRole = qtc.Qt.UserRole + 3

    def __init__(self, *args, **kwargs):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            return qtc.QSize(300, 200)
        else:
            return qtg.QStandardItemModel.data(self, index, role)


class ContextMenuEventFilter(qtc.QObject):

    def __init__(self, parent):
        super(ContextMenuEventFilter, self).__init__()
        self.parent = parent

    def eventFilter(self, source: qtc.QObject, event: qtc.QEvent) -> bool:
        if event.type() == qtc.QEvent.Type.Leave:
            self.parent.close()
        return super().eventFilter(source, event)


class PictureViewer(qwt.QWidget):

    pictures_added_sig = qtc.pyqtSignal(list)

    def __init__(self):
        super(PictureViewer, self).__init__()

        self.pictures_added_sig.connect(self.pictures_added)

        self.init_ui()

    def init_ui(self):
        self.ui_image_viewer = qwt.QListView()
        self.ui_image_viewer.viewport().installEventFilter(self)
        self.ui_image_viewer.setSpacing(10)
        self.ui_image_viewer.setViewMode(qwt.QListView.IconMode)
        self.ui_image_viewer.setResizeMode(qwt.QListView.Adjust)
        self.ui_image_viewer.setEditTriggers(
            qwt.QAbstractItemView.NoEditTriggers)
        self.ui_image_viewer.setIconSize(qtc.QSize(300, 150))
        self.ui_image_viewer.setMovement(qwt.QListView.Static)
        self.ui_image_viewer.setModel(StandardItemModel())

        grid = qwt.QVBoxLayout()
        grid.addWidget(self.ui_image_viewer)
        self.setLayout(grid)

        sizePolicy = qwt.QSizePolicy(
            qwt.QSizePolicy.Preferred, qwt.QSizePolicy.MinimumExpanding)
        self.setSizePolicy(sizePolicy)

        self.init_context_menu()

    def init_context_menu(self):
        self.context_menu = qwt.QMenu()
        self.cmef = ContextMenuEventFilter(self.context_menu)
        self.context_menu.installEventFilter(self.cmef)

        rename = self.context_menu.addAction("Rename")
        rename.triggered.connect(self.rename_selected_picture)

        delete = self.context_menu.addAction("Delete")
        delete.triggered.connect(self.remove_selected_picture)

    def get_selected_index(self) -> qtc.QModelIndex:
        return self.ui_image_viewer.selectionModel().currentIndex()

    def get_data_from_selected_item(self, role=StandardItem.PathRole):
        index = self.get_selected_index()
        data = self.ui_image_viewer.model().data(index, role)
        return index, data

    def remove_item_from_viewer(self, row: int):
        self.ui_image_viewer.model().removeRow(row)

    def remove_selected_picture(self):
        index, path = self.get_data_from_selected_item()

        def remove_fn(remove: bool):
            if remove:
                self.remove_item_from_viewer(index.row())
                op = IO_OP(IO_OP_TYPE.DELETE, path)
                # self.app.io_op_sig.emit(op)

        dialog_msg = DialogMessages.DELETE(
            f'Do you really want to delete: \n{path}')
        dialog = Dialog(dialog_msg, self)

        dialog.remove_sig.connect(remove_fn)

        dialog.exec()

    def rename_selected_picture(self):
        ...

    @qtc.pyqtSlot(list)
    def pictures_added(self, images: List[Union[str, np.ndarray]]):
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
                    selected = self.ui_image_viewer.selectedIndexes()
                    if selected:
                        self.show_context_menu(event)
        return super().eventFilter(source, event)

    def show_context_menu(self, event):
        action = self.context_menu.exec_(self.mapToGlobal(event.pos()))
