import copy
from typing import NamedTuple

import numpy as np
import pygfx as gfx
import pylinalg as la
from qtpy.QtWidgets import QVBoxLayout, QWidget
from wgpu.gui.auto import WgpuCanvas


class NameData(NamedTuple):
    itree: int
    itrack: int
    icell: int


class TreePlot(QWidget):
    """The actual vispy (or pygfx) tree plot"""

    def __init__(self, colormap, selected_nodes, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.lineages = []

        self.colors = colormap
        self.selected_nodes = selected_nodes
        self.selected_nodes_data = {}
        self.displayed_lineages = []
        self.time_range = 0
        self.feature_range = 0
        self.selected_points = None
        self.start_points = None
        self.middle_points = None
        self.end_points = None
        self.vertical_lines = None
        self.diagonal_lines = None
        self.mode = "all"  # options: "all", "lineage"
        self.feature = "tree"  # options: "tree", "area"
        self.view_direction = "vertical"  # options: "horizontal", "vertical"

        self.label_fontsize = 15

        self.canvas = WgpuCanvas()
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = gfx.OrthographicCamera(110, 110, maintain_aspect=False)
        self.controller_xy = gfx.PanZoomController(register_events=self.renderer)
        self.controller_xy.add_camera(self.camera)
        self.controller_x = gfx.PanZoomController(
            register_events=self.renderer, enabled=False
        )
        self.controller_x.add_camera(self.camera, include_state={"x", "width"})
        self.controller_y = gfx.PanZoomController(
            register_events=self.renderer, enabled=False
        )
        self.controller_y.add_camera(self.camera, include_state={"y", "height"})

        self.layout.addWidget(self.canvas)

        self.selected_nodes.list_updated.connect(self.__select_nodes)

        self.canvas.request_draw(self.animate)

    def set_lineages(self, lineages):
        """called by TreeWidget when tracks change"""
        self.lineages = lineages

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(100)
        return hint

    def animate(self):
        self.renderer.render(self.scene, self.camera)

    def set_event_handler(self, f):
        self.canvas.add_event_handler(f, "*")

    def both_xy(self):
        """enables zooming on both axes"""
        self.controller_xy.enabled = True
        self.controller_x.enabled = False
        self.controller_y.enabled = False

    def only_x(self):
        """zoom only horizontally"""
        self.controller_xy.enabled = False
        self.controller_x.enabled = True

    def only_y(self):
        """zoom only vertically"""
        self.controller_xy.enabled = False
        self.controller_y.enabled = True

    def __select_nodes(self):
        """called when a node is selected in the data view"""
        if len(self.selected_nodes) > 0:
            node = self.selected_nodes[-1]
            if node not in self.selected_nodes_data:
                for itree in range(len(self.lineages)):
                    for itrack in range(len(self.lineages[itree])):
                        for icell in range(len(self.lineages[itree][itrack])):
                            if node == self.lineages[itree][itrack][icell].node:
                                self.selected_nodes_data[node] = (
                                    NameData(itree, itrack, icell),
                                    0,
                                )
                                break
        self.draw_selected_nodes()
        if len(self.selected_nodes)>0:
            self.camera.show_object(self.selected_points)

    def _select_nodes(self, event):
        """called when a node is selected in the tree view"""
        nd, vi = (event.pick_info["world_object"].name, event.pick_info["vertex_index"])
        node = self.lineages[nd.itree][nd.itrack][nd.icell + vi].node
        if node in self.selected_nodes:
            del self.selected_nodes_data[node]
        else:
            if "Shift" not in event.modifiers:
                self.selected_nodes_data = {}
            self.selected_nodes_data[node] = (nd, vi)
        self.selected_nodes.add(node, "Shift" in event.modifiers)

    def draw_selected_nodes(self):
        """called whenever a node is (de-)selected"""
        for i, n in enumerate(self.selected_nodes):
            nd, vi = self.selected_nodes_data[n]
            track = self.lineages[nd.itree][nd.itrack]
            ilineage = (
                sum(
                    [
                        len(x)
                        for x in self.lineages[0 : nd.itree]
                        if x in self.displayed_lineages
                    ]
                )
                + nd.itrack
            )
            self.selected_points.geometry.colors.data[i,3] = 1
            self.selected_points.geometry.colors.update_range(i)
            self.selected_points.geometry.positions.data[i, 0] = (
                ilineage * 10 if self.feature == "tree" else track[vi + nd.icell].area
            )
            self.selected_points.geometry.positions.data[i, 1] = -track[
                vi + nd.icell
            ].time
            self.selected_points.geometry.positions.update_range(i)
        for i in range(len(self.selected_nodes), 100):
            self.selected_points.geometry.colors.data[i,3] = 0
            self.selected_points.geometry.colors.update_range(i)
            self.selected_points.geometry.positions.data[i, :] = \
                    self.selected_points.geometry.positions.data[0, :]
            self.selected_points.geometry.positions.update_range(i)
        self.canvas.request_draw()

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

    def set_mode(self, mode):
        self.mode = mode

    def set_feature(self, feature):
        self.feature = feature

    def set_view_direction(self, direction):
        self.view_direction = direction

    def get_feature(self):
        return self.feature

    def get_view_direction(self):
        return self.view_direction

    def actuate_view_direction(self):
        """rotates entire plot by 90 degrees"""
        if self.view_direction == "horizontal":
            self.scene.local.rotation = la.quat_from_axis_angle(
                [0.0, 0.0, 1.0], 3.14159 / 2
            )
            self.camera.width = self.time_range + 1
            self.camera.height = self.feature_range + 1
            self.label_time.anchor_offset = 20
            self.label_time.anchor = "top-center"
            self.label_feature.anchor_offset = -60
            self.label_feature.anchor = "middle-right"
        else:
            self.scene.local.rotation = [0.0, 0.0, 0.0, 1.0]
            self.camera.width = self.feature_range + 1
            self.camera.height = self.time_range + 1
            self.label_time.anchor_offset = 40
            self.label_time.anchor = "middle-right"
            self.label_feature.anchor_offset = 20
            self.label_feature.anchor = "top-center"
        if self.lineages:
            self.camera.show_object(self.scene)
        self.camera_state0 = copy.deepcopy(self.camera.get_state())
        self.canvas.update()
        if self.lineages:
            self.ruler_time.update(self.camera, self.canvas.get_logical_size())
            self.ruler_feature.update(self.camera, self.canvas.get_logical_size())
        self.canvas.update()

    def reset_fov(self):
        """adjusts axes to fill the plot window"""
        self.controller_xy.enabled = False
        self.camera.set_state(self.camera_state0)
        self.controller_xy.enabled = True

    def init(self):
        """instantiates all the plot widgets and glyphs etc"""
        self.scene.clear()

        # selected markers
        self.selected_points = gfx.Points(
            gfx.Geometry(
                positions=[(0, 0, 0) for _ in range(100)],
                colors=[[0.68, 0.85, 0.90, 0] for _ in range(100)], # light blue
            ),
            gfx.PointsMarkerMaterial(
                marker="circle", color_mode="vertex", size=15, size_space="screen"
            ),
            render_order=0,
        )
        self.scene.add(self.selected_points)

        self.start_points = []
        self.middle_points = []
        self.end_points = []
        self.vertical_lines = []
        self.diagonal_lines = []

        ilineage = 0
        for itree in range(len(self.lineages)):
            for itrack in range(len(self.lineages[itree])):
                track = self.lineages[itree][itrack]
                trackid = track[0].trackid

                # start markers
                if len(track) > 1:
                    points = gfx.Points(
                        gfx.Geometry(
                            positions=[(0, 0, 0)], edge_colors=[self.colors.map(trackid)]
                        ),
                        gfx.PointsMarkerMaterial(
                            marker=track[0].marker,
                            color="black",
                            edge_color_mode="vertex",
                            edge_width=4,
                            pick_write=True,
                        ),
                        name=NameData(itree, itrack, 0),
                        render_order=2,
                    )
                    self.scene.add(points)
                    self.start_points.append(points)

                    @points.add_event_handler("pointer_down")
                    def select_nodes(event):
                        self._select_nodes(event)
                else:
                    self.start_points.append(None)

                # middle markers
                if len(track) > 2:
                    points = gfx.Points(
                        gfx.Geometry(
                            positions=[(0, 0, 0) for _ in track[1:-1]],
                            edge_colors=[self.colors.map(trackid) for _ in track[1:-1]],

                        ),
                        gfx.PointsMarkerMaterial(
                            marker=track[1].marker,
                            color="black",
                            edge_color_mode="vertex",
                            edge_width=4,
                            pick_write=True,
                        ),
                        name=NameData(itree, itrack, 1),
                    )
                    self.scene.add(points)
                    self.middle_points.append(points)

                    @points.add_event_handler("pointer_down")
                    def select_nodes(event):
                        self._select_nodes(event)
                else:
                    self.middle_points.append(None)

                # end markers
                marker = (
                    track[-1].marker + "_up"
                    if track[-1].marker == "triangle"
                    else track[-1].marker
                )
                points = gfx.Points(
                    gfx.Geometry(
                        positions=[(0, 0, 0)], edge_colors=[self.colors.map(trackid)]
                    ),
                    gfx.PointsMarkerMaterial(
                        marker=marker,
                        edge_width=4,
                        size=14,
                        color="black",
                        edge_mode="inner",
                        edge_color_mode="vertex",
                        pick_write=True,
                    ),
                    name=NameData(itree, itrack, len(track) - 1),
                )
                self.scene.add(points)
                self.end_points.append(points)

                @points.add_event_handler("pointer_down")
                def select_nodes(event):
                    self._select_nodes(event)

                # vertical track lines
                line = gfx.Line(
                    gfx.Geometry(positions=[(0, 0, 0) for _ in track]),
                    gfx.LineMaterial(thickness=2.0, color=self.colors.map(trackid)),
                    render_order=4,
                )
                self.scene.add(line)
                self.vertical_lines.append(line)

                # diagonal division lines
                if track[-1].marker == "triangle":
                    line = gfx.Line(
                        gfx.Geometry(
                            positions=[[0, 0, 0] for _ in range(3)],
                            colors=[(1, 1, 1, 1) for _ in range(3)],
                        ),
                        gfx.LineMaterial(thickness=2.0),
                        render_order=4,
                    )
                    self.scene.add(line)
                    self.diagonal_lines.append(line)
                else:
                    self.diagonal_lines.append(None)

                ilineage += 1

        # needs https://github.com/pygfx/pygfx/pull/1130
        self.ruler_time = gfx.Ruler(tick_side="right", start_value=0)
        self.scene.add(self.ruler_time)
        self.label_time = gfx.Text(
            text="time",
            font_size=self.label_fontsize,
            screen_space=True,
            material=gfx.TextMaterial(color="#ffffff"),
        )
        self.scene.add(self.label_time)

        self.ruler_feature = gfx.Ruler(tick_side="right", start_value=0)
        self.scene.add(self.ruler_feature)
        self.label_feature = gfx.Text(
            text="area",
            font_size=self.label_fontsize,
            screen_space=True,
            material=gfx.TextMaterial(color="#ffffff"),
        )
        self.scene.add(self.label_feature)

        self.update()

    def update(self):
        """sets the position and color of all the plot widgets and glyphs etc"""
        ilineage = idisplayed = 0
        self.displayed_lineages = []
        for itree in range(len(self.lineages)):
            # skip if not selected
            skip = False
            if self.mode == "lineage":
                skip = True
                for itrack in range(len(self.lineages[itree])):
                    for node in self.selected_nodes:
                        nd, vi = self.selected_nodes_data[node]
                        if nd.itree == itree and nd.itrack == itrack:
                            skip = False
                            break
                        if not skip:
                            break
            if not skip:
                self.displayed_lineages.append(self.lineages[itree])

            for itrack in range(len(self.lineages[itree])):
                track = self.lineages[itree][itrack]

                # start markers
                if len(track) > 1:
                    if not skip:
                        self.start_points[ilineage].visible = True
                        self.start_points[ilineage].geometry.positions.data[0, 0] = (
                            idisplayed * 10 if self.feature == "tree" else track[0].area
                        )
                        self.start_points[ilineage].geometry.positions.data[
                            0, 1
                        ] = -track[0].time
                        self.start_points[ilineage].geometry.positions.data[0, 2] = 0
                    else:
                        self.start_points[ilineage].visible = False
                        self.start_points[ilineage].geometry.positions.data[0, 0] = 0
                        self.start_points[ilineage].geometry.positions.data[0, 1] = 0
                        self.start_points[ilineage].geometry.positions.data[0, 2] = -10
                    self.start_points[ilineage].geometry.positions.update_range(0)

                # middle markers
                if len(track) > 2:
                    for i, t in enumerate(track[1:-1]):
                        if not skip:
                            self.middle_points[ilineage].visible = True
                            self.middle_points[ilineage].geometry.positions.data[i, 0] = (
                                idisplayed * 10 if self.feature == "tree" else t.area
                            )
                            self.middle_points[ilineage].geometry.positions.data[
                                i, 1
                            ] = -t.time
                            self.middle_points[ilineage].geometry.positions.data[i, 2] = 0
                        else:
                            self.middle_points[ilineage].visible = False
                            self.middle_points[ilineage].geometry.positions.data[i, 0] = 0
                            self.middle_points[ilineage].geometry.positions.data[i, 1] = 0
                            self.middle_points[ilineage].geometry.positions.data[
                                i, 2
                            ] = -10
                        self.middle_points[ilineage].geometry.positions.update_range(i)

                # end markers
                if not skip:
                    self.end_points[ilineage].visible = True
                    self.end_points[ilineage].geometry.positions.data[0, 0] = (
                        idisplayed * 10 if self.feature == "tree" else track[-1].area
                    )
                    self.end_points[ilineage].geometry.positions.data[0, 1] = -track[
                        -1
                    ].time
                    self.end_points[ilineage].geometry.positions.data[0, 2] = 0
                else:
                    self.end_points[ilineage].visible = False
                    self.end_points[ilineage].geometry.positions.data[0, 0] = 0
                    self.end_points[ilineage].geometry.positions.data[0, 1] = 0
                    self.end_points[ilineage].geometry.positions.data[0, 2] = -10
                if self.end_points[ilineage].material.marker.startswith("triangle"):
                    self.end_points[ilineage].material.size = 14
                    if self.view_direction == "horizontal":
                        self.end_points[ilineage].material.marker = "triangle_left"
                    else:
                        self.end_points[ilineage].material.marker = "triangle_up"
                else:
                    self.end_points[ilineage].material.size = 10
                self.end_points[ilineage].geometry.positions.update_range(0)

                # vertical track lines
                for i, t in enumerate(track):
                    if not skip:
                        self.vertical_lines[ilineage].visible = True
                        self.vertical_lines[ilineage].geometry.positions.data[i, 0] = (
                            idisplayed * 10 if self.feature == "tree" else t.area
                        )
                        self.vertical_lines[ilineage].geometry.positions.data[
                            i, 1
                        ] = -t.time
                    else:
                        self.vertical_lines[ilineage].visible = False
                        self.vertical_lines[ilineage].geometry.positions.data[i, 0] = 0
                        self.vertical_lines[ilineage].geometry.positions.data[i, 1] = 0
                    self.vertical_lines[ilineage].geometry.positions.data[i, 2] = 0
                    self.vertical_lines[ilineage].geometry.positions.update_range(i)

                # diagonal division lines
                if track[-1].marker == "triangle":
                    iprev = itrack - 1
                    while (
                        self.lineages[itree][iprev][0].time - 1 != track[-1].time
                        and iprev > 0
                    ):
                        iprev -= 1
                    inext = itrack + 1
                    while (
                        self.lineages[itree][inext][0].time - 1 != track[-1].time
                        and inext < len(self.lineages[itree]) - 1
                    ):
                        inext += 1
                    if not skip:
                        self.diagonal_lines[ilineage].visible = True
                        for i in range(3):
                            self.diagonal_lines[ilineage].geometry.colors.data[i, 3] = 1
                            self.diagonal_lines[ilineage].geometry.colors.update_range(i)
                        self.diagonal_lines[ilineage].geometry.positions.data[0, 0] = (
                            (idisplayed - (itrack - iprev)) * 10
                            if self.feature == "tree"
                            else self.lineages[itree][iprev][0].area
                        )
                        self.diagonal_lines[ilineage].geometry.positions.data[
                            0, 1
                        ] = -self.lineages[itree][iprev][0].time
                        self.diagonal_lines[ilineage].geometry.positions.data[1, 0] = (
                            idisplayed * 10 if self.feature == "tree" else track[-1].area
                        )
                        self.diagonal_lines[ilineage].geometry.positions.data[
                            1, 1
                        ] = -track[-1].time
                        self.diagonal_lines[ilineage].geometry.positions.data[2, 0] = (
                            (idisplayed - (itrack - inext)) * 10
                            if self.feature == "tree"
                            else self.lineages[itree][inext][0].area
                        )
                        self.diagonal_lines[ilineage].geometry.positions.data[
                            2, 1
                        ] = -self.lineages[itree][inext][0].time
                    else:
                        self.diagonal_lines[ilineage].visible = False
                        for i in range(3):
                            self.diagonal_lines[ilineage].geometry.positions.data[
                                i, 0
                            ] = 0
                            self.diagonal_lines[ilineage].geometry.positions.data[
                                i, 1
                            ] = 0
                    for i in range(3):
                        self.diagonal_lines[ilineage].geometry.positions.data[i, 2] = 0
                        self.diagonal_lines[ilineage].geometry.positions.update_range(i)

                ilineage += 1
                if not skip:
                    idisplayed += 1

        self.draw_selected_nodes()

        if len(self.vertical_lines) > 0:
            f = [vl.geometry.positions.data[:, 0] for vl in self.vertical_lines]
            f = [y for x in f for y in x]
            t = [vl.geometry.positions.data[:, 1] for vl in self.vertical_lines]
            t = [y for x in t for y in x]
            self.feature_range = np.max(f) - np.min(f)
            self.time_range = np.max(t) - np.min(t)

            self.ruler_time.start_pos = (-0.1 * self.feature_range, 0, 0)
            self.ruler_time.end_pos = (-0.1 * self.feature_range, -self.time_range, 0)
            self.label_time.local.position = (
                -0.1 * self.feature_range,
                -self.time_range / 2,
                0,
            )

        if self.feature == "tree":
            self.ruler_feature.start_pos = (0, -1.1 * self.time_range, 0)
            self.ruler_feature.end_pos = (0, -1.1 * self.time_range, 0)
        else:
            self.ruler_feature.start_pos = (0, -1.1 * self.time_range, 0)
            self.ruler_feature.end_pos = (self.feature_range, -1.1 * self.time_range, 0)
        self.label_feature.local.position = (
            self.feature_range / 2,
            -1.1 * self.time_range,
            0,
        )

        if idisplayed == 0:
            self.label_time.font_size = 0
            self.ruler_time._line.material.color = (1, 1, 1, 0)
            self.ruler_time._points.material.color = (1, 1, 1, 0)
        else:
            self.label_time.font_size = self.label_fontsize
            self.ruler_time._line.material.color = (1, 1, 1, 1)
            self.ruler_time._points.material.color = (1, 1, 1, 1)

        if idisplayed == 0 or self.feature == "tree":
            self.label_feature.font_size = 0
            self.ruler_feature._line.material.color = (1, 1, 1, 0)
            self.ruler_feature._points.material.color = (1, 1, 1, 0)
        else:
            self.label_feature.font_size = self.label_fontsize
            self.ruler_feature._line.material.color = (1, 1, 1, 1)
            self.ruler_feature._points.material.color = (1, 1, 1, 1)

        self.actuate_view_direction()
