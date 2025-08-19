# do not put the from __future__ import annotations as it breaks the injection


from typing import NamedTuple

import numpy as np
from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

import finn
from finn.layers.tracks.tracks import Tracks
from finn.track_data_views.views.tree_view.flip_axes_widget import FlipTreeWidget
from finn.track_data_views.views.tree_view.navigation_widget import NavigationWidget
from finn.track_data_views.views.tree_view.tree_plot import TreePlot
from finn.track_data_views.views.tree_view.tree_view_feature_widget import (
    TreeViewFeatureWidget,
)
from finn.track_data_views.views.tree_view.tree_view_mode_widget import TreeViewModeWidget
from finn.track_data_views.views_coordinator.tracks_viewer import TracksViewer


class VertexData(NamedTuple):
    marker: str
    node: int
    time: float
    area: float
    trackid: int


class TreeWidget(QWidget):
    """pyqtgraph-based widget for lineage tree visualization and navigation"""

    def __init__(self, viewer: finn.Viewer | None = None):
        super().__init__()
        self.tracks_viewer = TracksViewer.get_instance(viewer)
        self.mode = "all"  # options: "all", "lineage"
        self.feature = "tree"  # options: "tree", "area"
        self.view_direction = "vertical"  # options: "horizontal", "vertical"

        self.lineages = []
        self.selected_nodes_data = {}

        self.tracks_viewer.tracks_updated.connect(self._update_track_data)

        # Construct the tree view pyqtgraph widget
        self.layout = QVBoxLayout()

        # Add radiobuttons for switching between different display modes
        self.mode_widget = TreeViewModeWidget()
        self.mode_widget.change_mode.connect(self._set_mode)

        # Add buttons to change which feature to display
        self.feature_widget = TreeViewFeatureWidget()
        self.feature_widget.change_feature.connect(self._set_feature)

        # Add widget to flip the axes
        self.flip_widget = FlipTreeWidget()
        self.flip_widget.flip_tree.connect(self._flip_axes)

        self.tree_plot: TreePlot = TreePlot(
            self.tracks_viewer.colormap,
            self.tracks_viewer.selected_nodes,
            self.selected_nodes_data,
        )
        self.tree_plot.set_event_handler(self.my_handler)
        self.tree_plot.init()

        # Add navigation widget
        self.navigation_widget = NavigationWidget(
            # navigation_widget.py/get_next_track_node and friends are now
            # implemented in tree_plot.py/select_next_cell
            self.lineages,
            self.tree_plot.displayed_lineages,
            self.tracks_viewer.selected_nodes,
            self.selected_nodes_data,
            self.tree_plot.get_view_direction,
            self.tree_plot.get_feature,
        )

        # Construct a toolbar and set main layout
        panel_layout = QHBoxLayout()
        panel_layout.addWidget(self.mode_widget)
        panel_layout.addWidget(self.feature_widget)
        panel_layout.addWidget(self.navigation_widget)
        panel_layout.addWidget(self.flip_widget)
        panel_layout.setSpacing(0)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        panel.setLayout(panel_layout)
        panel.setMaximumWidth(930)
        panel.setMaximumHeight(82)

        # Make a collapsible for TreeView widgets
        collapsable_widget = QCollapsible("Show/Hide Tree View Controls")
        collapsable_widget.layout().setContentsMargins(0, 0, 0, 0)
        collapsable_widget.layout().setSpacing(0)
        collapsable_widget.addWidget(panel)
        collapsable_widget.collapse(animate=False)

        self.layout.addWidget(collapsable_widget)
        self.layout.addWidget(self.tree_plot)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

    def recurse_tree(self, node, lineage):
        """returns a list of lists (trees) of lists (tracks)"""
        tracks = self.tracks_viewer.tracks
        time = tracks.get_time(node)
        area = tracks.get_area(node)
        trackid = tracks.get_track_id(node)
        _lineage = [VertexData("square", node, time, area, trackid)]
        succs = tracks.successors(node)
        while len(succs) == 1:
            node = succs[0]
            time = tracks.get_time(node)
            area = tracks.get_area(node)
            trackid = tracks.get_track_id(node)
            _lineage.append(VertexData("circle", node, time, area, trackid))
            succs = tracks.successors(node)
        if len(succs) == 0:
            _lineage[-1] = VertexData("cross", *_lineage[-1][1:])
            lineage.append(_lineage)
            return lineage
        if len(succs) > 1:
            lineage = self.recurse_tree(succs[0], lineage)
            _lineage[-1] = VertexData("triangle", *_lineage[-1][1:])
            lineage.append(_lineage)
            return self.recurse_tree(succs[1], lineage)
        return "just to make ruff CI happy"

    def _update_track_data(self):
        """transform tracks into a structure more amenable to plotting"""
        if self.tracks_viewer.tracks is not None:
            self.lineages.clear()
            nodes = self.tracks_viewer.tracks.nodes()
            in_degrees = self.tracks_viewer.tracks.in_degree(nodes)
            for iprogenitor in np.nonzero(in_degrees == 0)[0]:
                self.lineages.append(self.recurse_tree(nodes[iprogenitor].item(), []))
            self.selected_nodes_data.clear()
            self.tree_plot.set_lineages(self.lineages)
            self.tree_plot.init()

    def my_handler(self, arg):
        if arg["event_type"] == "char" and arg["char_str"] == "q":
            self.mode_widget._toggle_display_mode()
        if arg["event_type"] == "char" and arg["char_str"] == "w":
            self.feature_widget._toggle_feature_mode()
        if arg["event_type"] == "char" and arg["char_str"] == "f":
            self._flip_axes()
        if arg["event_type"] == "key_down" and arg["key"] == "x":
            self.tree_plot.only_x()
        if arg["event_type"] == "key_down" and arg["key"] == "y":
            self.tree_plot.only_y()
        if arg["event_type"] == "key_up":
            self.tree_plot.both_xy()
        if arg["event_type"] == "double_click":
            self.tree_plot.reset_fov()
        if arg["event_type"] == "key_up":
            if arg["key"] == "ArrowLeft":
                self.navigation_widget.move_left()
            if arg["key"] == "ArrowRight":
                self.navigation_widget.move_right()
            if arg["key"] == "ArrowUp":
                self.navigation_widget.move_up()
            if arg["key"] == "ArrowDown":
                self.navigation_widget.move_down()

    def _flip_axes(self):
        """Flip the axes of the plot"""
        if self.view_direction == "horizontal":
            self.view_direction = "vertical"
        else:
            self.view_direction = "horizontal"
        self.tree_plot.set_view_direction(self.view_direction)
        self.tree_plot.actuate_view_direction()

    def _update_selected(self):
        """Called whenever the selection list is updated."""

    def refresh(self, tracks: Tracks) -> None:
        """Called when the TracksViewer emits the tracks_updated signal, indicating
        that a new set of tracks should be viewed.
        """

    def _set_mode(self, mode: str) -> None:
        """Set the display mode to all or lineage view. Currently, linage
        view is always horizontal and all view is always vertical.

        Args:
            mode (str): The mode to set the view to. Options are "all" or "lineage"

        """
        self.tree_plot.set_mode(mode)
        if mode == "all":
            self.view_direction = "vertical"
        else:
            self.view_direction = "horizontal"
        self.tree_plot.set_view_direction(self.view_direction)
        self.tree_plot.update()

    def _set_feature(self, feature: str) -> None:
        """Set the feature mode to 'tree' or 'area'. For this the view is always
        horizontal.

        Args:
            feature (str): The feature to plot. Options are "tree" or "area"

        """
        if feature not in ["tree", "area"]:
            raise ValueError(f"Feature must be 'tree' or 'area', got {feature}")
        self.tree_plot.set_feature(feature)
        self.tree_plot.update()

    def _update_lineage_df(self) -> None:
        """Subset dataframe to include only nodes belonging to the current lineage"""
