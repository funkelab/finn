from __future__ import annotations

from typing import TYPE_CHECKING

from finn.experimental import link_layers, unlink_layers
from finn.track_data_views.views.layers.track_graph import TrackGraph
from finn.track_data_views.views.layers.track_labels import TrackLabels
from finn.track_data_views.views.layers.track_points import TrackPoints

if TYPE_CHECKING:
    from finn.track_data_views.views_coordinator.project_viewer import ProjectViewer


class TracksLayerGroup:
    def __init__(
        self,
        project_viewer: ProjectViewer,
    ):
        self.finn_viewer = project_viewer.viewer
        self.project_viewer = project_viewer
        self.project = project_viewer.project
        self.name = self.project.name
        self.tracks_layer: TrackGraph | None = None
        self.points_layer: TrackPoints | None = None
        self.seg_layer: TrackLabels | None = None

        # Create new layers
        if self.project is not None and self.project.segmentation is not None:
            self.seg_layer = TrackLabels(
                viewer=self.finn_viewer,
                data=self.project.segmentation,
                name=self.name + "_seg",
                opacity=0.6,
                project_viewer=self.project_viewer,
            )

        if len(self.project.cand_graph) != 0:
            self.points_layer = TrackPoints(
                name=self.name + "_points",
                project_viewer=self.project_viewer,
                show_cands=True,
            )
        # TODO: Update this when a solution becomes available
        if len(self.project.solution) != 0:
            self.tracks_layer = TrackGraph(
                name=self.name + "_tracks",
                project_viewer=self.project_viewer,
            )
        self.add_finn_layers()

    def add_finn_layers(self) -> None:
        """Add new tracking layers to the viewer"""

        if self.tracks_layer is not None:
            self.finn_viewer.add_layer(self.tracks_layer)
        if self.seg_layer is not None:
            self.finn_viewer.add_layer(self.seg_layer)
        if self.points_layer is not None:
            self.finn_viewer.add_layer(self.points_layer)
        self.link_clipping_planes()

    def link_clipping_planes(self):
        """Link the clipping planes of all tracking layers"""

        track_layers = []
        if self.tracks_layer is not None:
            track_layers.append(self.tracks_layer)
        if self.seg_layer is not None:
            track_layers.append(self.seg_layer)
        if self.points_layer is not None:
            track_layers.append(self.points_layer)

        if all(layer.ndim >= 3 for layer in track_layers):
            link_layers(track_layers, ("clipping_planes",))

    def unlink_clipping_planes(self):
        """Unlink the clipping planes of all tracking layers"""

        track_layers = []
        if self.tracks_layer is not None:
            track_layers.append(self.tracks_layer)
        if self.seg_layer is not None:
            track_layers.append(self.seg_layer)
        if self.points_layer is not None:
            track_layers.append(self.points_layer)
        unlink_layers(track_layers, ("clipping_planes",))

    def _refresh(self) -> None:
        """Refresh the tracking layers with new tracks info"""
        if self.tracks_layer is not None:
            self.tracks_layer._refresh()
        if self.seg_layer is not None:
            self.seg_layer._refresh()
        if self.points_layer is not None:
            self.points_layer._refresh()

    def update_visible(self, visible_tracks: list[int], visible_nodes: list[int]):
        if self.seg_layer is not None:
            self.seg_layer.update_label_colormap(visible_nodes)
        if self.points_layer is not None:
            self.points_layer.update_point_outline(visible_tracks)
        if self.tracks_layer is not None:
            self.tracks_layer.update_track_visibility(visible_tracks)

    def center_view(self, node):
        """Adjust the current_step and camera center of the viewer to jump to the node
        location, if the node is not already in the field of view"""

        if self.seg_layer is None or self.seg_layer.mode == "pan_zoom":
            graph = self.project.cand_graph
            time = graph.get_time(node)
            pos = self.project.cand_graph.get_position(node)
            location = [time, *pos]
            assert len(location) == self.finn_viewer.dims.ndim, (
                f"Location {location} does not match viewer number of dims "
                f"{self.finn_viewer.dims.ndim}"
            )

            step = list(self.finn_viewer.dims.current_step)
            for dim in self.finn_viewer.dims.not_displayed:
                step[dim] = int(
                    location[dim] + 0.5
                )  # use the world location, since the 'step' in viewer.dims.range
                # already in world units
            self.finn_viewer.dims.current_step = step

            # check whether the new coordinates are inside or outside the field of view,
            # then adjust the camera if needed
            example_layer = (
                self.points_layer
            )  # the points layer is always in world units,
            # because it directly reads the scaled coordinates. Therefore, no rescaling
            # is necessary to compute the camera center
            corner_coordinates = example_layer.corner_pixels

            # check which dimensions are shown, the first dimension is displayed on the
            # x axis, and the second on the y_axis
            dims_displayed = self.finn_viewer.dims.displayed

            # Note: This centering does not work in 3D. What we should do instead is take
            # the view direction vector, start at the point, and move backward along the
            # vector a certain amount to put the point in view.
            # Note #2: Points already does centering when you add the first point, and it
            # works in 3D. We can look at that to see what logic they use.

            # self.viewer.dims.displayed_order
            x_dim = dims_displayed[-1]
            y_dim = dims_displayed[-2]

            # find corner pixels for the displayed axes
            _min_x = corner_coordinates[0][x_dim]
            _max_x = corner_coordinates[1][x_dim]
            _min_y = corner_coordinates[0][y_dim]
            _max_y = corner_coordinates[1][y_dim]

            # check whether the node location falls within the corner spatial range
            if not (
                (location[x_dim] > _min_x and location[x_dim] < _max_x)
                and (location[y_dim] > _min_y and location[y_dim] < _max_y)
            ):
                camera_center = self.finn_viewer.camera.center

                # set the center y and x to the center of the node, by using the index
                # of the currently displayed dimensions
                self.finn_viewer.camera.center = (
                    camera_center[0],
                    location[y_dim],
                    # camera center is calculated in scaled coordinates, and the optional
                    # labels layer is scaled by the layer.scale attribute
                    location[x_dim],
                )
