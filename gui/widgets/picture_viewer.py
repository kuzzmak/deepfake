import os
import sys
from typing import List
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt


class StandardItem(qtg.QStandardItem):
    PathRole = qtc.Qt.UserRole + 1

    def __init__(self, *args, **kwargs):
        super(StandardItem, self).__init__(*args, **kwargs)
        self.path = ''

    def setData(self, value, role=qtc.Qt.UserRole + 1):
        if role == StandardItem.PathRole:
            self.path = value
        else:
            qtg.QStandardItem.setData(self, value, role)

    def data(self, role=qtc.Qt.UserRole + 1):
        if role == StandardItem.PathRole:
            return self.path
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
            value = it.data(qtc.Qt.DecorationRole)
            if value is None:
                path = it.data(StandardItem.PathRole)
                value = qtg.QIcon(qtg.QPixmap(path))
                it.setData(value, qtc.Qt.DecorationRole)
            return value
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

        self.ui_image_viewer = qwt.QListView()
        self.ui_image_viewer.viewport().installEventFilter(self)
        self.ui_image_viewer.setSpacing(10)
        # self.ui_image_viewer.installEventFilter(self)
        # self.ui_image_viewer.mousePressEvent.connect(self.item_clicked)
        # self.ui_image_viewer.clicked.connect(self.item_clicked)
        self.ui_image_viewer.setViewMode(qwt.QListView.IconMode)
        self.ui_image_viewer.setResizeMode(qwt.QListView.Adjust)
        self.ui_image_viewer.setEditTriggers(
            qwt.QAbstractItemView.NoEditTriggers)
        self.ui_image_viewer.setIconSize(qtc.QSize(300, 150))
        self.ui_image_viewer.setMovement(qwt.QListView.Static)
        self.ui_image_viewer.setModel(StandardItemModel())

        grid = qwt.QVBoxLayout()
        # grid.setContentsMargins(10, 10, 10, 10)
        grid.addWidget(self.ui_image_viewer)

        self.setLayout(grid)
        sizePolicy = qwt.QSizePolicy(
            qwt.QSizePolicy.Preferred, qwt.QSizePolicy.MinimumExpanding)
        self.setSizePolicy(sizePolicy)

        self.context_menu = qwt.QMenu()
        self.cmef = ContextMenuEventFilter(self.context_menu)
        self.context_menu.installEventFilter(self.cmef)
        newAct = self.context_menu.addAction("New")
        openAct = self.context_menu.addAction("Open")
        quitAct = self.context_menu.addAction("Quit")

    @qtc.pyqtSlot(list)
    def pictures_added(self, img_paths: List[str]):
        for img_path in img_paths:
            name = os.path.splitext(os.path.basename(img_path))[0]
            item = StandardItem(name)
            item.setData(img_path)
            self.ui_image_viewer.model().appendRow(item)

    def eventFilter(self, source, event) -> bool:
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
