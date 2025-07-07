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

    def __init__(self, lineages, colormap, selected_nodes, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.lineages = lineages

        self.colors = colormap
        self.selected_nodes = selected_nodes
        self.selected_nodes_data = {}
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

        self.NameData = namedtuple('NameData', 'x, iy, nodes, times, areas, itree, itrack')

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

        self.selected_nodes.list_updated.connect(self.draw_selected_nodes)

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
        nd, vi = (event.pick_info["world_object"].name, event.pick_info["vertex_index"])
        node = nd.nodes[nd.iy + vi]
        if node in self.selected_nodes:
            del self.selected_nodes_data[node]
        else:
            if 'Shift' not in event.modifiers:
                self.selected_nodes_data = {}
            self.selected_nodes_data[node] = (nd, vi)
        self.selected_nodes.add(node, 'Shift' in event.modifiers)

    def draw_selected_nodes(self):
        if self.mode=="all":
            for i,n in enumerate(self.selected_nodes):
                nd, vi = self.selected_nodes_data[n]
                self.selected_geometry.colors.data[i] = [0.68,0.85,0.90,1]  # light blue
                self.selected_geometry.colors.update_range(i)
                self.selected_geometry.positions.data[i,0] = nd.x*10 if self.feature == "tree" else nd.areas[vi+nd.iy]
                self.selected_geometry.positions.data[i,1] = nd.times[vi+nd.iy]
                self.selected_geometry.positions.update_range(i)
        for i in range(len(self.selected_nodes) if self.mode=="all" else 0, 100):
            self.selected_geometry.colors.data[i] = [0,0,0,0]
            self.selected_geometry.colors.update_range(i)
            self.selected_geometry.positions.data[i,:] = 0
            self.selected_geometry.positions.update_range(i)
        self.canvas.request_draw()

    def select_next_cell(self):
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        track = self.displayed_lineages[nd.itree][nd.itrack]
        icell = vi + nd.iy + 1
        if icell < len(track):
            if icell == len(track)-1:
                iy = icell
                vi = 0
            else:
                iy = 1
                vi = icell-1
            newnode = nd.nodes[icell]
            self.selected_nodes_data[newnode] = (
                self.NameData(nd.x, iy, nd.nodes, nd.times, nd.areas, nd.itree, nd.itrack),
                vi
                )
            self.selected_nodes.add(newnode, False)
        elif track[-1].marker=="triangle_up":
            iprev = nd.itrack-1
            while self.displayed_lineages[nd.itree][iprev][0].time-1 != track[-1].time:
                iprev -= 1
            track = self.displayed_lineages[nd.itree][iprev]
            x = nd.x - (nd.itrack - iprev)
            nodes, times, areas = [t.node for t in track], [-t.time for t in track], [t.area for t in track]
            newnode = track[0].node
            self.selected_nodes_data[newnode] = (
                self.NameData(x, 0, nodes, times, areas, nd.itree, iprev),
                0
                )
            self.selected_nodes.add(newnode, False)

    def select_prev_cell(self):
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        track = self.displayed_lineages[nd.itree][nd.itrack]
        icell = vi + nd.iy - 1
        if icell >= 0:
            if icell == 0:
                iy = 0
                vi = 0
            else:
                iy = 1
                vi = icell-1
            newnode = nd.nodes[icell]
            self.selected_nodes_data[newnode] = (
                self.NameData(nd.x, iy, nd.nodes, nd.times, nd.areas, nd.itree, nd.itrack),
                vi
                )
            self.selected_nodes.add(newnode, False)
        elif track[0].marker=="square":
            iprev = nd.itrack-1
            while iprev>=0 and self.displayed_lineages[nd.itree][iprev][-1].time+1 > track[0].time:
                iprev -= 1
            if iprev>=0 and self.displayed_lineages[nd.itree][iprev][-1].time+1 == track[0].time:
                iparent = iprev
            else:
                inext = nd.itrack+1
                while inext<len(self.displayed_lineages[nd.itree]) and \
                        self.displayed_lineages[nd.itree][inext][-1].time+1 > track[0].time:
                    inext += 1
                if inext<len(self.displayed_lineages[nd.itree]) and \
                        self.displayed_lineages[nd.itree][inext][-1].time+1 == track[0].time:
                    iparent = inext
                else:
                    return
            track = self.displayed_lineages[nd.itree][iparent]
            x = nd.x - (nd.itrack - iparent)
            nodes, times, areas = [t.node for t in track], [-t.time for t in track], [t.area for t in track]
            newnode = track[-1].node
            self.selected_nodes_data[newnode] = (
                self.NameData(x, len(times)-1, nodes, times, areas, nd.itree, iparent),
                0
                )
            self.selected_nodes.add(newnode, False)

    def select_next_lineage(self):
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        itree, itrack = nd.itree, nd.itrack+1
        if itrack == len(self.displayed_lineages[itree]):
            itrack = 0
            itree += 1
        found = False
        dx = 1
        while itree < len(self.displayed_lineages) and itrack < len(self.displayed_lineages[itree]):
            time = -nd.times[vi+nd.iy]
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
                iy = 0
                vi = 0
            elif icell==len(track)-1:
                iy = icell
                vi = 0
            else:
                iy = 1
                vi = icell-1
            nodes, times, areas = [t.node for t in track], [-t.time for t in track], [t.area for t in track]
            newnode = track[vi+iy].node
            self.selected_nodes_data[newnode] = (
                self.NameData(nd.x+dx, iy, nodes, times, areas, itree, itrack),
                vi
                )
            self.selected_nodes.add(newnode, False)

    def select_prev_lineage(self):
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        itree, itrack = nd.itree, nd.itrack-1
        if itrack == -1:
            itree -= 1
            itrack = len(self.displayed_lineages[itree])-1
        found = False
        dx = 1
        while itree >= 0 and itrack >= 0:
            time = -nd.times[vi+nd.iy]
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
                iy = 0
                vi = 0
            elif icell==len(track)-1:
                iy = len(track)-1
                vi = 0
            else:
                iy = 1
                vi = icell-1
            nodes, times, areas = [t.node for t in track], [-t.time for t in track], [t.area for t in track]
            newnode = track[vi+iy].node
            self.selected_nodes_data[newnode] = (
                self.NameData(nd.x-dx, iy, nodes, times, areas, itree, itrack),
                vi
                )
            self.selected_nodes.add(newnode, False)

    def select_next_feature(self):
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        icell = vi + nd.iy
        time = -nd.times[icell]
        feature = nd.areas[icell]
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
                iy = 0
                vi = 0
            elif icell_next==len(track)-1:
                iy = len(track)-1
                vi = 0
            else:
                iy = 1
                vi = icell_next-1
            nodes, times, areas = [t.node for t in track], [-t.time for t in track], [t.area for t in track]
            newnode = track[vi+iy].node
            self.selected_nodes_data[newnode] = (
                self.NameData(x, iy, nodes, times, areas, itree_next, itrack_next),
                vi
                )
            self.selected_nodes.add(newnode, False)

    def select_prev_feature(self):
        node = self.selected_nodes[-1]
        nd, vi = self.selected_nodes_data[node]
        icell = vi + nd.iy
        time = -nd.times[icell]
        feature = nd.areas[icell]
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
                iy = 0
                vi = 0
            elif icell_prev==len(track)-1:
                iy = len(track)-1
                vi = 0
            else:
                iy = 1
                vi = icell_prev-1
            nodes, times, areas = [t.node for t in track], [-t.time for t in track], [t.area for t in track]
            newnode = track[vi+iy].node
            self.selected_nodes_data[newnode] = (
                self.NameData(x, iy, nodes, times, areas, itree_prev, itrack_prev),
                vi
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
        if self.view_direction == "horizontal":
            self.scene.local.rotation = la.quat_from_axis_angle([0., 0., 1.], 3.14159/2)
            self.camera.width = self.time_range
            self.camera.height = self.feature_range
            self.label.anchor_offset=20
            self.label.anchor="top-center"
        else:
            self.scene.local.rotation = [0., 0., 0., 1.]
            self.camera.width = self.feature_range
            self.camera.height = self.time_range
            self.label.anchor_offset=40
            self.label.anchor="middle-right"
        self.camera.show_object(self.scene)
        self.camera_state0 = copy.deepcopy(self.camera.get_state())
        self.canvas.update()
        self.ruler.update(self.camera, self.canvas.get_logical_size())
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
                nodes, times, areas =  [t.node for t in track], [-t.time for t in track], [t.area for t in track]
                trackid = track[0].trackid

                # start markers
                self.start_geometries.append(gfx.Geometry(positions=[(0, 0, 0)],
                                                          edge_colors=[self.colors.map(trackid)]))
                points = gfx.Points(
                    self.start_geometries[-1],
                    gfx.PointsMarkerMaterial(marker=track[0].marker,
                                             color="black",
                                             edge_color_mode="vertex",
                                             edge_width=4,
                                             pick_write=True),
                    name=self.NameData(ilineage, 0, nodes, times, areas, iselected_tree, itrack),
                    render_order=2)
                self.scene.add(points)

                @points.add_event_handler("pointer_down")
                def select_nodes(event):  self._select_nodes(event)

                # middle markers
                if len(track)>2:
                    self.middle_geometries.append(gfx.Geometry(
                            positions=[(0, 0, 0) for _ in track[1:-1]],
                            edge_colors=[self.colors.map(trackid) for _ in track[1:-1]]))
                    points = gfx.Points(
                        self.middle_geometries[-1],
                        gfx.PointsMarkerMaterial(marker=track[1].marker,
                                                 color="black",
                                                 edge_color_mode="vertex",
                                                 edge_width=4,
                                                 pick_write=True),
                        name=self.NameData(ilineage, 1, nodes, times, areas, iselected_tree, itrack))
                    self.scene.add(points)

                    @points.add_event_handler("pointer_down")
                    def select_nodes(event):  self._select_nodes(event)
                else:
                    self.middle_geometries.append(None)

                # end markers
                self.end_geometries.append(gfx.Geometry(positions=[(0, 0, 0)],
                                                        edge_colors=[self.colors.map(trackid)]))
                points = gfx.Points(
                    self.end_geometries[-1],
                    gfx.PointsMarkerMaterial(marker=track[-1].marker,
                                             edge_width=4,
                                             size=10,
                                             color="black",
                                             edge_mode="inner",
                                             edge_color_mode="vertex",
                                             pick_write=True),
                    name=self.NameData(ilineage, len(times)-1, nodes, times, areas, iselected_tree, itrack))
                self.scene.add(points)

                @points.add_event_handler("pointer_down")
                def select_nodes(event):  self._select_nodes(event)

                # vertical track lines
                self.vertical_geometries.append(gfx.Geometry(positions=[(0, 0, 0) for _ in track]))
                line = gfx.Line(
                    self.vertical_geometries[-1],
                    gfx.LineMaterial(thickness=2.0, color=self.colors.map(trackid)),
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

        # needs https://github.com/pygfx/pygfx/pull/1130
        self.ruler = gfx.Ruler(tick_side="right", start_value=0)
        self.scene.add(self.ruler)
        self.label = gfx.Text(
            text="time",
            font_size=15,
            screen_space=True,
            material=gfx.TextMaterial(color="#ffffff"),
        )
        self.scene.add(self.label)

        self.update()

    def update(self):
        if self.mode=="lineage" and len(self.selected_nodes)==0:  return
        ilineage = idisplayed = 0
        for itree in range(len(self.lineages)):

            # skip if not selected
            skip=False
            if self.mode=="lineage":
                skip=True
                for itrack in range(len(self.lineages[itree])):
                    for node in self.selected_nodes:
                        nd, vi = self.selected_nodes_data[node]
                        if nd.nodes[0] == self.lineages[itree][itrack][0].node:
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

        self.ruler.start_pos = (-0.1*self.feature_range, 0, 0)
        self.ruler.end_pos = (-0.1*self.feature_range, -self.time_range, 0)
        self.label.local.position = (-0.1*self.feature_range, -self.time_range/2, 0)

        self.actuate_view_direction()
