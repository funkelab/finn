from qtpy.QtWidgets import QVBoxLayout, QWidget

from wgpu.gui.auto import WgpuCanvas
import pygfx as gfx
import distinctipy
import numpy as np
from collections import namedtuple
import pylinalg as la
import copy

class TreePlot(QWidget):
    """The actual vispy (or pygfx) tree plot"""

    def __init__(self, lineages, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.lineages = lineages

        self.colors = [(*x, 1) for x in distinctipy.get_colors(sum([len(x) for x in self.lineages]))]
        self.selected_nodes = []
        self.displayed_lineages = []
        self.time_range = 0
        self.feature_range = 0
        self.selected_geometry = None
        self.start_geometries = None
        self.middle_geometries = None
        self.end_geometries = None
        self.vertical_geometries = None
        self.diagonal_geometries = None
        self.mode = "all"  # options: "all", "lineage"
        self.feature = "tree"  # options: "tree", "area"
        self.view_direction = "vertical"  # options: "horizontal", "vertical"

        self.NameData = namedtuple('NameData', 'x, node, offset, time, area, itree, itrack')

        self.canvas = WgpuCanvas()
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = gfx.OrthographicCamera(110, 110, maintain_aspect=False)
        self.controller_xy = gfx.PanZoomController(register_events=self.renderer)
        self.controller_xy.add_camera(self.camera)
        self.controller_x = gfx.PanZoomController(register_events=self.renderer, enabled=False)
        self.controller_x.add_camera(self.camera, include_state={"x", "width"})
        self.controller_y = gfx.PanZoomController(register_events=self.renderer, enabled=False)
        self.controller_y.add_camera(self.camera, include_state={"y", "height"})

        self.layout.addWidget(self.canvas)

        self.canvas.request_draw(self.animate)

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(100)
        return hint

    def animate(self):
        self.renderer.render(self.scene, self.camera)

    def set_event_handler(self, f):
        self.canvas.add_event_handler(f, "*")

    def both_xy(self):
        self.controller_xy.enabled=True
        self.controller_x.enabled=False
        self.controller_y.enabled=False

    def only_x(self):
        self.controller_xy.enabled=False
        self.controller_x.enabled=True

    def only_y(self):
        self.controller_xy.enabled=False
        self.controller_y.enabled=True

    def _select_nodes(self, event):
        if self.mode!="all":  return
        info = (event.pick_info["world_object"].name, event.pick_info["vertex_index"])
        if info in self.selected_nodes:
            self.selected_nodes.remove(info)
        else:
            if 'Shift' not in event.modifiers:
                self.selected_nodes = []
            self.selected_nodes.append(info)
        self.draw_selected_nodes()

    def draw_selected_nodes(self):
        if self.mode=="all":
            for i,(nd,vi) in enumerate(self.selected_nodes):
                self.selected_geometry.colors.data[i] = [0.68,0.85,0.90,1]  # light blue
                self.selected_geometry.colors.update_range(i)
                self.selected_geometry.positions.data[i,0] = nd.x*10 if self.feature == "tree" else nd.area[vi+nd.offset]
                self.selected_geometry.positions.data[i,1] = nd.time[vi+nd.offset]
                self.selected_geometry.positions.update_range(i)
        for i in range(len(self.selected_nodes) if self.mode=="all" else 0, 100):
            self.selected_geometry.colors.data[i] = [0,0,0,0]
            self.selected_geometry.colors.update_range(i)
            self.selected_geometry.positions.data[i,:] = 0
            self.selected_geometry.positions.update_range(i)
        self.canvas.request_draw()

    def select_next_cell(self):
        node = self.selected_nodes[-1]
        track = self.displayed_lineages[node[0].itree][node[0].itrack]
        if node[1] + node[0].offset < len(track)-1:
            icell = node[1] + node[0].offset + 1
            if icell == len(track)-1:
                offset = len(track)-1
                vi = 0
            else:
                offset = 1
                vi = icell-1
            time, area = [-t.time for t in track], [t.area for t in track]
            self.selected_nodes[-1] = (
                self.NameData(node[0].x, track[0].node, offset, time, area,
                              node[0].itree, node[0].itrack),
                vi
                )
            self.draw_selected_nodes()
        elif track[-1].marker=="triangle_up":
            iprev = node[0].itrack-1
            while self.displayed_lineages[node[0].itree][iprev][0].time-1 != track[-1].time:
                iprev -= 1
            track = self.displayed_lineages[node[0].itree][iprev]
            x = node[0].x - (node[0].itrack - iprev)
            time, area = [-t.time for t in track], [t.area for t in track]
            self.selected_nodes[-1] = (
                self.NameData(x, track[0].node, 0, time, area, node[0].itree, iprev),
                0
                )
            self.draw_selected_nodes()

    def select_prev_cell(self):
        node = self.selected_nodes[-1]
        track = self.displayed_lineages[node[0].itree][node[0].itrack]
        if node[1] + node[0].offset > 0:
            icell = node[1] + node[0].offset - 1
            if icell == 0:
                offset = 0
                vi = 0
            else:
                offset = 1
                vi = icell-1
            time, area = [-t.time for t in track], [t.area for t in track]
            self.selected_nodes[-1] = (
                self.NameData(node[0].x, track[0].node, offset,
                              time, area, node[0].itree, node[0].itrack),
                vi
                )
            self.draw_selected_nodes()
        elif track[0].marker=="square":
            iprev = node[0].itrack-1
            while iprev>=0 and self.displayed_lineages[node[0].itree][iprev][-1].time+1 > track[0].time:
                iprev -= 1
            if iprev>=0 and self.displayed_lineages[node[0].itree][iprev][-1].time+1 == track[0].time:
                iparent = iprev
            else:
                inext = node[0].itrack+1
                while inext<len(self.displayed_lineages[node[0].itree]) and \
                        self.displayed_lineages[node[0].itree][inext][-1].time+1 > track[0].time:
                    inext += 1
                if inext<len(self.displayed_lineages[node[0].itree]) and \
                        self.displayed_lineages[node[0].itree][inext][-1].time+1 == track[0].time:
                    iparent = inext
                else:
                    return
            track = self.displayed_lineages[node[0].itree][iparent]
            x = node[0].x - (node[0].itrack - iparent)
            time, area = [-t.time for t in track], [t.area for t in track]
            self.selected_nodes[-1] = (
                self.NameData(x, track[0].node, len(time)-1, time, area, node[0].itree, iparent),
                0
                )
            self.draw_selected_nodes()

    def select_next_lineage(self):
        node = self.selected_nodes[-1]
        itree, itrack = node[0].itree, node[0].itrack+1
        if itrack == len(self.displayed_lineages[itree]):
            itrack = 0
            itree += 1
        found = False
        dx = 1
        while itree < len(self.displayed_lineages) and itrack < len(self.displayed_lineages[itree]):
            time = -node[0].time[node[1]+node[0].offset]
            times = [x.time for x in self.displayed_lineages[itree][itrack]]
            if time in times:
                icell = times.index(time)
                found = True
                break
            dx += 1
            itrack += 1
            if itrack == len(self.displayed_lineages[itree]):
                itrack = 0
                itree += 1
        if found:
            track = self.displayed_lineages[itree][itrack]
            if icell == 0:
                offset = 0
                vi = 0
            elif icell==len(track)-1:
                offset = len(track)-1
                vi = 0
            else:
                offset = 1
                vi = icell-1
            time, area = [-t.time for t in track], [t.area for t in track]
            self.selected_nodes[-1] = (
                self.NameData(node[0].x+dx, track[0].node, offset, time, area, itree, itrack),
                vi
                )
            self.draw_selected_nodes()

    def select_prev_lineage(self):
        node = self.selected_nodes[-1]
        itree, itrack = node[0].itree, node[0].itrack-1
        if itrack == -1:
            itree -= 1
            itrack = len(self.displayed_lineages[itree])-1
        found = False
        dx = 1
        while itree >= 0 and itrack >= 0:
            time = -node[0].time[node[1]+node[0].offset]
            times = [x.time for x in self.displayed_lineages[itree][itrack]]
            if time in times:
                icell = times.index(time)
                found = True
                break
            dx += 1
            itrack -= 1
            if itrack == -1:
                itree -= 1
                itrack = len(self.displayed_lineages[itree])-1
        if found:
            track = self.displayed_lineages[itree][itrack]
            if icell == 0:
                offset = 0
                vi = 0
            elif icell==len(track)-1:
                offset = len(track)-1
                vi = 0
            else:
                offset = 1
                vi = icell-1
            time, area = [-t.time for t in track], [t.area for t in track]
            self.selected_nodes[-1] = (
                self.NameData(node[0].x-dx, track[0].node, offset, time, area, itree, itrack),
                vi
                )
            self.draw_selected_nodes()

    def select_next_feature(self):
        node = self.selected_nodes[-1]
        icell = node[1] + node[0].offset
        time = -node[0].time[icell]
        feature = node[0].area[icell]
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
            x = sum([len(x) for x in self.displayed_lineages[0:itree_next]]) + itrack_next
            if icell_next == 0:
                offset = 0
                vi = 0
            elif icell_next==len(track)-1:
                offset = len(track)-1
                vi = 0
            else:
                offset = 1
                vi = icell_next-1
            time, area = [-t.time for t in track], [t.area for t in track]
            self.selected_nodes[-1] = (
                self.NameData(x, track[0].node, offset, time, area, itree_next, itrack_next),
                vi
                )
            self.draw_selected_nodes()

    def select_prev_feature(self):
        node = self.selected_nodes[-1]
        icell = node[1] + node[0].offset
        time = -node[0].time[icell]
        feature = node[0].area[icell]
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
            x = sum([len(x) for x in self.displayed_lineages[0:itree_prev]]) + itrack_prev
            if icell_prev == 0:
                offset = 0
                vi = 0
            elif icell_prev==len(track)-1:
                offset = len(track)-1
                vi = 0
            else:
                offset = 1
                vi = icell_prev-1
            time, area = [-t.time for t in track], [t.area for t in track]
            self.selected_nodes[-1] = (
                self.NameData(x, track[0].node, offset, time, area, itree_prev, itrack_prev),
                vi
                )
            self.draw_selected_nodes()

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
        if self.view_direction == "horizontal":
            self.scene.local.rotation = la.quat_from_axis_angle([0., 0., 1.], 3.14159/2)
            self.camera.width = self.time_range
            self.camera.height = self.feature_range
        else:
            self.scene.local.rotation = [0., 0., 0., 1.]
            self.camera.width = self.feature_range
            self.camera.height = self.time_range
        self.camera.show_object(self.scene)
        self.camera_state0 = copy.deepcopy(self.camera.get_state())
        self.canvas.update()

    def reset_fov(self):
        self.controller_xy.enabled=False
        self.camera.set_state(self.camera_state0)
        self.controller_xy.enabled=True

    def init(self):
        self.scene.clear()

        # selected markers
        self.selected_geometry = gfx.Geometry(positions=[(0,0,0) for _ in range(100)],
                                              colors=[[0,0,0,0] for _ in range(100)])
        points = gfx.Points(
            self.selected_geometry,
            gfx.PointsMarkerMaterial(marker="circle",
                                     color_mode="vertex",
                                     size=15,
                                     size_space="screen"),
            render_order=3)
        self.scene.add(points)

        self.start_geometries = []
        self.middle_geometries = []
        self.end_geometries = []
        self.vertical_geometries = []
        self.diagonal_geometries = []

        ilineage = 0
        iselected_tree = 0
        for itree in range(len(self.lineages)):
            for itrack in range(len(self.lineages[itree])):
                track = self.lineages[itree][itrack]
                time, area =  [-t.time for t in track], [t.area for t in track],

                # start markers
                self.start_geometries.append(gfx.Geometry(positions=[(0, 0, 0)],
                                                          edge_colors=[self.colors[ilineage]]))
                points = gfx.Points(
                    self.start_geometries[-1],
                    gfx.PointsMarkerMaterial(marker=track[0].marker,
                                             color="black",
                                             edge_color_mode="vertex",
                                             edge_width=4,
                                             pick_write=True),
                    name=self.NameData(ilineage, track[0].node, 0, time, area, iselected_tree, itrack),
                    render_order=2)
                self.scene.add(points)

                @points.add_event_handler("pointer_down")
                def select_nodes(event):  self._select_nodes(event)

                # middle markers
                if len(track)>2:
                    self.middle_geometries.append(gfx.Geometry(
                            positions=[(0, 0, 0) for _ in track[1:-1]],
                            edge_colors=[self.colors[ilineage] for _ in track[1:-1]]))
                    points = gfx.Points(
                        self.middle_geometries[-1],
                        gfx.PointsMarkerMaterial(marker=track[1].marker,
                                                 color="black",
                                                 edge_color_mode="vertex",
                                                 edge_width=4,
                                                 pick_write=True),
                        name=self.NameData(ilineage, track[0].node, 1, time, area, iselected_tree, itrack))
                    self.scene.add(points)

                    @points.add_event_handler("pointer_down")
                    def select_nodes(event):  self._select_nodes(event)
                else:
                    self.middle_geometries.append(None)

                # end markers
                self.end_geometries.append(gfx.Geometry(positions=[(0, 0, 0)],
                                                        edge_colors=[self.colors[ilineage]]))
                points = gfx.Points(
                    self.end_geometries[-1],
                    gfx.PointsMarkerMaterial(marker=track[-1].marker,
                                             edge_width=4,
                                             size=10,
                                             color="black",
                                             edge_mode="inner",
                                             edge_color_mode="vertex",
                                             pick_write=True),
                    name=self.NameData(ilineage, track[0].node, len(time)-1, time, area, iselected_tree, itrack))
                self.scene.add(points)

                @points.add_event_handler("pointer_down")
                def select_nodes(event):  self._select_nodes(event)

                # vertical track lines
                self.vertical_geometries.append(gfx.Geometry(positions=[(0, 0, 0) for _ in track]))
                line = gfx.Line(
                    self.vertical_geometries[-1],
                    gfx.LineMaterial(thickness=2.0, color=self.colors[ilineage]),
                    render_order=4,
                )
                self.scene.add(line)

                # diagonal division lines
                if track[-1].marker=="triangle_up":
                    self.diagonal_geometries.append(
                        gfx.Geometry(positions=[[0, 0, 0] for _ in range(3)],
                                     colors=[(1,1,1,1) for _ in range(3)]))
                    line = gfx.Line(
                        self.diagonal_geometries[-1],
                        gfx.LineMaterial(thickness=2.0, color_mode="vertex"),
                        render_order=4,
                    )
                    self.scene.add(line)
                else:
                    self.diagonal_geometries.append(None)

                ilineage += 1
            iselected_tree += 1

        self.update()

    def update(self):
        ilineage = idisplayed = 0
        for itree in range(len(self.lineages)):

            # skip if not selected
            skip=False
            if self.mode=="lineage":
                skip=True
                for itrack in range(len(self.lineages[itree])):
                    for selected in self.selected_nodes:
                        if selected[0].node == self.lineages[itree][itrack][0].node:
                            skip=False
                            break
                        if not skip: break
            if not skip:
                self.displayed_lineages.append(self.lineages[itree])

            for itrack in range(len(self.lineages[itree])):
                track = self.lineages[itree][itrack]
                time, area =  [-t.time for t in track], [t.area for t in track],

                # start markers
                if not skip:
                    self.start_geometries[ilineage].edge_colors.data[0,3] = 1
                    self.start_geometries[ilineage].edge_colors.update_range(0)
                    self.start_geometries[ilineage].positions.data[0,0] = \
                            idisplayed*10 if self.feature == "tree" else track[0].area
                    self.start_geometries[ilineage].positions.data[0,1] = -track[0].time
                else:
                    self.start_geometries[ilineage].edge_colors.data[0,3] = 0
                    self.start_geometries[ilineage].edge_colors.update_range(0)
                    self.start_geometries[ilineage].positions.data[0,0] = 0
                    self.start_geometries[ilineage].positions.data[0,1] = 0
                self.start_geometries[ilineage].positions.data[0,2] = 0
                self.start_geometries[ilineage].positions.update_range(0)

                # middle markers
                if len(track)>2:
                    for i,t in enumerate(track[1:-1]):
                        if not skip:
                            self.middle_geometries[ilineage].edge_colors.data[i,3] = 1
                            self.middle_geometries[ilineage].edge_colors.update_range(i)
                            self.middle_geometries[ilineage].positions.data[i,0] = \
                                idisplayed*10 if self.feature == "tree" else t.area
                            self.middle_geometries[ilineage].positions.data[i,1] = -t.time
                        else:
                            self.middle_geometries[ilineage].edge_colors.data[i,3] = 0
                            self.middle_geometries[ilineage].edge_colors.update_range(i)
                            self.middle_geometries[ilineage].positions.data[i,0] = 0
                            self.middle_geometries[ilineage].positions.data[i,1] = 0
                        self.middle_geometries[ilineage].positions.data[i,2] = 0
                        self.middle_geometries[ilineage].positions.update_range(i)

                # end markers
                if not skip:
                    self.end_geometries[ilineage].edge_colors.data[0,3] = 1
                    self.end_geometries[ilineage].edge_colors.update_range(0)
                    self.end_geometries[ilineage].positions.data[0,0] = \
                            idisplayed*10 if self.feature == "tree" else track[-1].area
                    self.end_geometries[ilineage].positions.data[0,1] = -track[-1].time
                else:
                    self.end_geometries[ilineage].edge_colors.data[0,3] = 0
                    self.end_geometries[ilineage].edge_colors.update_range(0)
                    self.end_geometries[ilineage].positions.data[0,0] = 0
                    self.end_geometries[ilineage].positions.data[0,1] = 0
                self.end_geometries[ilineage].positions.data[0,2] = 0
                self.end_geometries[ilineage].positions.update_range(0)

                # vertical track lines
                for i,t in enumerate(track):
                    if not skip:
                        self.vertical_geometries[ilineage].positions.data[i,0] = \
                                idisplayed*10 if self.feature == "tree" else t.area
                        self.vertical_geometries[ilineage].positions.data[i,1] = -t.time
                    else:
                        self.vertical_geometries[ilineage].positions.data[i,0] = 0
                        self.vertical_geometries[ilineage].positions.data[i,1] = 0
                    self.vertical_geometries[ilineage].positions.data[i,2] = 0
                    self.vertical_geometries[ilineage].positions.update_range(i)

                # diagonal division lines
                if track[-1].marker=="triangle_up":
                    iprev = itrack-1
                    while self.lineages[itree][iprev][0].time-1 != track[-1].time and iprev>0:
                        iprev -= 1
                    inext = itrack+1
                    while self.lineages[itree][inext][0].time-1 != track[-1].time and inext<len(self.lineages[itree])-1:
                        inext += 1
                    if not skip:
                        for i in range(3):
                            self.diagonal_geometries[ilineage].colors.data[i,3] = 1
                            self.diagonal_geometries[ilineage].colors.update_range(i)
                        self.diagonal_geometries[ilineage].positions.data[0,0] = \
                                (idisplayed-(itrack-iprev))*10 if self.feature == "tree" else self.lineages[itree][iprev][0].area
                        self.diagonal_geometries[ilineage].positions.data[0,1] = \
                                 -self.lineages[itree][iprev][0].time
                        self.diagonal_geometries[ilineage].positions.data[1,0] = \
                                 idisplayed*10 if self.feature == "tree" else track[-1].area
                        self.diagonal_geometries[ilineage].positions.data[1,1] = \
                                  -track[-1].time
                        self.diagonal_geometries[ilineage].positions.data[2,0] = \
                                 (idisplayed-(itrack-inext))*10 if self.feature == "tree" else self.lineages[itree][inext][0].area
                        self.diagonal_geometries[ilineage].positions.data[2,1] = \
                                  -self.lineages[itree][inext][0].time
                    else:
                        for i in range(3):
                            self.diagonal_geometries[ilineage].colors.data[i,3] = 0
                            self.diagonal_geometries[ilineage].colors.update_range(i)
                            self.diagonal_geometries[ilineage].positions.data[i,0] = 0
                            self.diagonal_geometries[ilineage].positions.data[i,1] = 0
                    for i in range(3):
                        self.diagonal_geometries[ilineage].positions.data[i,2] = 0
                        self.diagonal_geometries[ilineage].positions.update_range(i)

                ilineage += 1
                if not skip: idisplayed += 1

        self.draw_selected_nodes()

        f = [l.positions.data[:,0] for l in self.vertical_geometries]
        f = [y for x in f for y in x]
        t = [l.positions.data[:,1] for l in self.vertical_geometries]
        t = [y for x in t for y in x]
        self.feature_range = np.max(f) - np.min(f)
        self.time_range = np.max(t) - np.min(t)

        self.actuate_view_direction()
