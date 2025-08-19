from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

if TYPE_CHECKING:
    from finn.track_data_views import NodeSelectionList


class NameData(NamedTuple):
    itree: int
    itrack: int
    icell: int


class NavigationWidget(QWidget):
    def __init__(
        self,
        lineages: list,
        displayed_lineages: list,
        selected_nodes: NodeSelectionList,
        selected_nodes_data: dict,
        get_view_direction,
        get_feature,
    ):
        """Widget for controlling navigation in the tree widget

        Args:
            selected_nodes (NodeSelectionList): The list of selected nodes.
        """

        super().__init__()
        self.lineages = lineages
        self.displayed_lineages = displayed_lineages
        self.selected_nodes = selected_nodes
        self.selected_nodes_data = selected_nodes_data
        self.get_view_direction = get_view_direction
        self.get_feature = get_feature

        navigation_box = QGroupBox("Navigation [\u2b05 \u27a1 \u2b06 \u2b07]")
        navigation_layout = QHBoxLayout()
        left_button = QPushButton("\u2b05")
        right_button = QPushButton("\u27a1")
        up_button = QPushButton("\u2b06")
        down_button = QPushButton("\u2b07")

        left_button.clicked.connect(lambda: self.move_left())
        right_button.clicked.connect(lambda: self.move_right())
        up_button.clicked.connect(lambda: self.move_up())
        down_button.clicked.connect(lambda: self.move_down())

        navigation_layout.addWidget(left_button)
        navigation_layout.addWidget(right_button)
        navigation_layout.addWidget(up_button)
        navigation_layout.addWidget(down_button)
        navigation_box.setLayout(navigation_layout)
        navigation_box.setMaximumWidth(250)
        navigation_box.setMaximumHeight(60)

        layout = QHBoxLayout()
        layout.addWidget(navigation_box)

        self.setLayout(layout)

    def move_left(self) -> None:
        if self.get_view_direction() == "horizontal":
            self.select_prev_cell()
        else:
            if self.get_feature() == "tree":
                self.select_prev_lineage()
            else:
                self.select_prev_feature()

    def move_right(self) -> None:
        if self.get_view_direction() == "horizontal":
            self.select_next_cell()
        else:
            if self.get_feature() == "tree":
                self.select_next_lineage()
            else:
                self.select_next_feature()

    def move_up(self) -> None:
        if self.get_view_direction() == "vertical":
            self.select_prev_cell()
        else:
            if self.get_feature() == "tree":
                self.select_next_lineage()
            else:
                self.select_next_feature()

    def move_down(self) -> None:
        if self.get_view_direction() == "vertical":
            self.select_next_cell()
        else:
            if self.get_feature() == "tree":
                self.select_prev_lineage()
            else:
                self.select_prev_feature()

    def select_next_cell(self):
        """adds the cell at the next timepoint in the track to selected_nodes"""
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        track = self.lineages[nd.itree][nd.itrack]
        icell = vi + nd.icell + 1
        if icell < len(track):
            if icell == len(track) - 1:
                iy = icell
                vi = 0
            else:
                iy = 1
                vi = icell - 1
            newnode = self.lineages[nd.itree][nd.itrack][icell].node
            self.selected_nodes_data[newnode] = (NameData(nd.itree, nd.itrack, iy), vi)
            self.selected_nodes.add(newnode, False)
        elif track[-1].marker == "triangle":
            iprev = nd.itrack - 1
            while self.lineages[nd.itree][iprev][0].time - 1 != track[-1].time:
                iprev -= 1
            track = self.lineages[nd.itree][iprev]
            newnode = track[0].node
            self.selected_nodes_data[newnode] = (NameData(nd.itree, iprev, 0), 0)
            self.selected_nodes.add(newnode, False)

    def select_prev_cell(self):
        """adds the cell at the previous timepoint in the track to selected_nodes"""
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        track = self.lineages[nd.itree][nd.itrack]
        icell = vi + nd.icell - 1
        if icell >= 0:
            if icell == 0:
                iy = 0
                vi = 0
            else:
                iy = 1
                vi = icell - 1
            newnode = self.lineages[nd.itree][nd.itrack][icell].node
            self.selected_nodes_data[newnode] = (NameData(nd.itree, nd.itrack, iy), vi)
            self.selected_nodes.add(newnode, False)
        elif track[0].marker == "square":
            iprev = nd.itrack - 1
            while (
                iprev >= 0 and self.lineages[nd.itree][iprev][-1].time + 1 > track[0].time
            ):
                iprev -= 1
            if (
                iprev >= 0
                and self.lineages[nd.itree][iprev][-1].time + 1 == track[0].time
            ):
                iparent = iprev
            else:
                inext = nd.itrack + 1
                while (
                    inext < len(self.lineages[nd.itree])
                    and self.lineages[nd.itree][inext][-1].time + 1 > track[0].time
                ):
                    inext += 1
                if (
                    inext < len(self.lineages[nd.itree])
                    and self.lineages[nd.itree][inext][-1].time + 1 == track[0].time
                ):
                    iparent = inext
                else:
                    return
            track = self.lineages[nd.itree][iparent]
            newnode = track[-1].node
            self.selected_nodes_data[newnode] = (
                NameData(nd.itree, iparent, len(track) - 1),
                0,
            )
            self.selected_nodes.add(newnode, False)

    def select_next_lineage(self):
        """
        adds the cell at the same timepoint in the right neighboring track
        to selected_nodes
        """
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        track = self.lineages[nd.itree][nd.itrack]
        time = track[vi + nd.icell].time
        itree, itrack = nd.itree, nd.itrack + 1
        itree = self.displayed_lineages.index(self.lineages[itree])
        if itrack == len(self.displayed_lineages[itree]):
            itrack = 0
            itree += 1
        found = False
        while itree < len(self.displayed_lineages) and itrack < len(
            self.displayed_lineages[itree]
        ):
            times = [x.time for x in self.displayed_lineages[itree][itrack]]
            if time in times:
                icell = times.index(time)
                found = True
                break
            itrack += 1
            if itrack == len(self.displayed_lineages[itree]):
                itrack = 0
                itree += 1
        if found:
            track = self.displayed_lineages[itree][itrack]
            if icell == 0:
                iy = 0
                vi = 0
            elif icell == len(track) - 1:
                iy = icell
                vi = 0
            else:
                iy = 1
                vi = icell - 1
            newnode = track[vi + iy].node
            itree = self.lineages.index(self.displayed_lineages[itree])
            self.selected_nodes_data[newnode] = (NameData(itree, itrack, iy), vi)
            self.selected_nodes.add(newnode, False)

    def select_prev_lineage(self):
        """
        adds the cell at the same timepoint in the left neighboring track
        to selected_nodes
        """
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        track = self.lineages[nd.itree][nd.itrack]
        time = track[vi + nd.icell].time
        itree, itrack = nd.itree, nd.itrack - 1
        itree = self.displayed_lineages.index(self.lineages[itree])
        if itrack == -1:
            itree -= 1
            itrack = len(self.displayed_lineages[itree]) - 1
        found = False
        while itree >= 0 and itrack >= 0:
            times = [x.time for x in self.displayed_lineages[itree][itrack]]
            if time in times:
                icell = times.index(time)
                found = True
                break
            itrack -= 1
            if itrack == -1:
                itree -= 1
                itrack = len(self.displayed_lineages[itree]) - 1
        if found:
            track = self.displayed_lineages[itree][itrack]
            if icell == 0:
                iy = 0
                vi = 0
            elif icell == len(track) - 1:
                iy = len(track) - 1
                vi = 0
            else:
                iy = 1
                vi = icell - 1
            newnode = track[vi + iy].node
            itree = self.lineages.index(self.displayed_lineages[itree])
            self.selected_nodes_data[newnode] = (NameData(itree, itrack, iy), vi)
            self.selected_nodes.add(newnode, False)

    def select_next_feature(self):
        """
        adds the cell at the same timepoint in the above neighboring track
        to selected_nodes
        """
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        track = self.lineages[nd.itree][nd.itrack]
        icell = vi + nd.icell
        time = track[icell].time
        feature = track[icell].area
        feature_next = np.inf
        found = False
        for itree in range(len(self.displayed_lineages)):
            for itrack in range(len(self.displayed_lineages[itree])):
                times = [x.time for x in self.displayed_lineages[itree][itrack]]
                if time in times:
                    icell = times.index(time)
                    thisfeature = self.displayed_lineages[itree][itrack][icell].area
                    if feature < thisfeature < feature_next:
                        itree_next, itrack_next, icell_next = itree, itrack, icell
                        feature_next = thisfeature
                        found = True
        if found:
            track = self.displayed_lineages[itree_next][itrack_next]
            if icell_next == 0:
                iy = 0
                vi = 0
            elif icell_next == len(track) - 1:
                iy = len(track) - 1
                vi = 0
            else:
                iy = 1
                vi = icell_next - 1
            newnode = track[vi + iy].node
            itree_next = self.lineages.index(self.displayed_lineages[itree_next])
            self.selected_nodes_data[newnode] = (
                NameData(itree_next, itrack_next, iy),
                vi,
            )
            self.selected_nodes.add(newnode, False)

    def select_prev_feature(self):
        """
        adds the cell at the same timepoint in the below neighboring track
        to selected_nodes
        """
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        track = self.lineages[nd.itree][nd.itrack]
        icell = vi + nd.icell
        time = track[icell].time
        feature = track[icell].area
        feature_prev = 0
        found = False
        for itree in range(len(self.displayed_lineages)):
            for itrack in range(len(self.displayed_lineages[itree])):
                times = [x.time for x in self.displayed_lineages[itree][itrack]]
                if time in times:
                    icell = times.index(time)
                    thisfeature = self.displayed_lineages[itree][itrack][icell].area
                    if feature > thisfeature > feature_prev:
                        itree_prev, itrack_prev, icell_prev = itree, itrack, icell
                        feature_prev = thisfeature
                        found = True
        if found:
            track = self.displayed_lineages[itree_prev][itrack_prev]
            if icell_prev == 0:
                iy = 0
                vi = 0
            elif icell_prev == len(track) - 1:
                iy = len(track) - 1
                vi = 0
            else:
                iy = 1
                vi = icell_prev - 1
            newnode = track[vi + iy].node
            itree_prev = self.lineages.index(self.displayed_lineages[itree_prev])
            self.selected_nodes_data[newnode] = (
                NameData(itree_prev, itrack_prev, iy),
                vi,
            )
            self.selected_nodes.add(newnode, False)
