from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

import finn

if TYPE_CHECKING:
    from funtracks import Project

    from finn.track_data_views.views_coordinator.project_viewer import (
        ProjectViewer,
    )


def update_finn_tracks(
    project: Project,
):
    """Function to take a networkx graph with assigned track_ids and return the data
    needed to add to a finn tracks layer.

    Args:
        tracks (SolutionTracks): tracks that have track_ids and have a tree structure

    Returns:
        data: array (N, D+1)
            Coordinates for N points in D+1 dimensions. ID,T,(Z),Y,X. The first
            axis is the integer ID of the track. D is either 3 or 4 for planar
            or volumetric timeseries respectively.
        graph: dict {int: list}
            Graph representing associations between tracks. Dictionary defines the
            mapping between a track ID and the parents of the track. This can be
            one (the track has one parent, and the parent has >=1 child) in the
            case of track splitting, or more than one (the track has multiple
            parents, but only one child) in the case of track merging.
    """
    ndim = project.ndim - 1
    graph = project.solution
    cand_graph = project.cand_graph
    finn_data = np.zeros((len(graph), ndim + 2))
    finn_edges = {}

    parents = [node for node in graph.nodes if graph.out_degree(node) >= 2]
    intertrack_edges = []

    # Remove all intertrack edges from a copy of the original graph
    for parent in parents:
        daughters = graph.successors(parent)
        for daughter in daughters:
            # graph.remove_edge(parent, daughter)
            intertrack_edges.append((parent, daughter))

    for index, node in enumerate(graph.nodes):
        node_id = node
        location = cand_graph.get_position(node_id)
        finn_data[index] = [
            cand_graph.get_track_id(node_id),
            cand_graph.get_time(node_id),
            *location,
        ]

    for parent, child in intertrack_edges:
        parent_track_id = cand_graph.get_track_id(parent)
        child_track_id = cand_graph.get_track_id(child)
        if child_track_id in finn_edges:
            finn_edges[child_track_id].append(parent_track_id)
        else:
            finn_edges[child_track_id] = [parent_track_id]

    return finn_data, finn_edges


class TrackGraph(finn.layers.Tracks):
    """Extended tracks layer that holds the track information and emits and responds
    to dynamics visualization signals"""

    def __init__(
        self,
        name: str,
        project_viewer: ProjectViewer,
    ):
        self.project_viewer = project_viewer
        track_data, track_edges = update_finn_tracks(
            self.project_viewer.project,
        )

        super().__init__(
            data=track_data,
            graph=track_edges,
            name=name,
            tail_length=3,
            color_by="track_id",
        )

        self.colormaps_dict["track_id"] = self.project_viewer.colormap
        self.tracks_layer_graph = copy.deepcopy(self.graph)  # for restoring graph later
        # just to 'refresh' the track_id colormap, we do not actually use turbo
        self.colormap = "turbo"

    def _refresh(self):
        """Refreshes the displayed tracks based on the graph in the current
        tracks_viewer.tracks
        """

        track_data, track_edges = update_finn_tracks(
            self.project_viewer.project,
        )

        self.data = track_data
        self.graph = track_edges
        self.tracks_layer_graph = copy.deepcopy(self.graph)
        self.colormaps_dict["track_id"] = self.project_viewer.colormap
        # just to 'refresh' the track_id colormap, we do not actually use turbo
        self.colormap = "turbo"

    def update_track_visibility(self, visible: list[int] | str) -> None:
        """Optionally show only the tracks of a current lineage"""

        if visible == "all":
            self.track_colors[:, 3] = 1
            self.graph = self.tracks_layer_graph
        else:
            track_id_mask = np.isin(
                self.properties["track_id"],
                visible,
            )
            self.graph = {
                key: self.tracks_layer_graph[key]
                for key in visible
                if key in self.tracks_layer_graph
            }

            self.track_colors[:, 3] = 0
            self.track_colors[track_id_mask, 3] = 1
            # empty dicts to not trigger update (bug?) so disable the graph entirely as a
            # workaround
            if len(self.graph.items()) == 0:
                self.display_graph = False
            else:
                self.display_graph = True
