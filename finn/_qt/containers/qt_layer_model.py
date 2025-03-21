import typing

from qtpy.QtCore import QModelIndex, QSize, Qt
from qtpy.QtGui import QImage

from finn import current_viewer
from finn._qt.containers.qt_list_model import QtListModel
from finn.layers import Layer
from finn.settings import get_settings
from finn.utils.translations import trans

ThumbnailRole = Qt.UserRole + 2
LoadedRole = Qt.UserRole + 3


class QtLayerListModel(QtListModel[Layer]):
    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        if not index.isValid():
            return None
        layer = self.getItem(index)
        viewer = current_viewer()
        layer_loaded = layer.loaded
        # Playback with async slicing causes flickering between the thumbnail
        # and loading animation in some cases due quick changes in the loaded
        # state, so report as unloaded in that case to avoid that.
        if get_settings().experimental.async_ and (viewer := current_viewer()):
            viewer_playing = viewer.window._qt_viewer.dims.is_playing
            layer_loaded = layer.loaded and not viewer_playing
        if role == Qt.ItemDataRole.DisplayRole:  # used for item text
            return layer.name
        if role == Qt.ItemDataRole.TextAlignmentRole:  # alignment of the text
            return Qt.AlignCenter
        if role == Qt.ItemDataRole.EditRole:
            # used to populate line edit when editing
            return layer.name
        if role == Qt.ItemDataRole.ToolTipRole:  # for tooltip
            layer_source_info = layer.get_source_str()
            if layer_loaded:
                return layer_source_info
            return trans._("{source} (loading)", source=layer_source_info)
        if role == Qt.ItemDataRole.CheckStateRole:  # the "checked" state of this item
            return Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.SizeHintRole:  # determines size of item
            return QSize(200, 34)
        if role == ThumbnailRole:  # return the thumbnail
            thumbnail = layer.thumbnail
            return QImage(
                thumbnail,
                thumbnail.shape[1],
                thumbnail.shape[0],
                QImage.Format_RGBA8888,
            )
        if role == LoadedRole:
            return layer_loaded
        # normally you'd put the icon in DecorationRole, but we do that in the
        # # LayerDelegate which is aware of the theme.
        # if role == Qt.ItemDataRole.DecorationRole:  # icon to show
        #     pass
        return super().data(index, role)

    def setData(
        self,
        index: QModelIndex,
        value: typing.Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role == Qt.ItemDataRole.CheckStateRole:
            # The item model stores a Qt.CheckState enum value that can be
            # partially checked, but we only use the unchecked and checked
            # to correspond to the layer's visibility.
            # https://doc.qt.io/qt-5/qt.html#CheckState-enum
            self.getItem(index).visible = Qt.CheckState(value) == Qt.CheckState.Checked
        elif role == Qt.ItemDataRole.EditRole:
            self.getItem(index).name = value
            role = Qt.ItemDataRole.DisplayRole
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True

    def all_loaded(self):
        """Return if all the layers are loaded."""
        return all(self.index(row, 0).data(LoadedRole) for row in range(self.rowCount()))

    def _process_event(self, event):
        # The model needs to emit `dataChanged` whenever data has changed
        # for a given index, so that views can update themselves.
        # Here we convert native events to the dataChanged signal.
        if not hasattr(event, "index"):
            return
        role = {
            "thumbnail": ThumbnailRole,
            "visible": Qt.ItemDataRole.CheckStateRole,
            "name": Qt.ItemDataRole.DisplayRole,
            "loaded": LoadedRole,
        }.get(event.type)
        roles = [role] if role is not None else []
        row = self.index(event.index)
        self.dataChanged.emit(row, row, roles)
