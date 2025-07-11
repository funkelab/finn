import argparse
from pathlib import Path

from funtracks.data_model import SolutionTracks
from geff.networkx.io import read as read_nx

import finn
from finn.track_data_views.views.tree_view import TreeWidget
from finn.track_data_views.views_coordinator.tracks_viewer import TracksViewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("geff_path_1", type=Path)
    parser.add_argument("geff_path_2", type=Path)
    args = parser.parse_args()

    nx_graph_1 = read_nx(args.geff_path_1)
    tracks1 = SolutionTracks(nx_graph_1, time_attr="t", pos_attr=("z", "y", "x"), ndim=4)

    nx_graph_2 = read_nx(args.geff_path_2)
    tracks2 = SolutionTracks(nx_graph_2, time_attr="t", pos_attr=("z", "y", "x"), ndim=4)

    viewer = finn.Viewer()
    tracks_viewer1 = TracksViewer(viewer, tracks1, None)
    tracks_viewer2 = TracksViewer(viewer, tracks2, None)
    tree_widget1 = TreeWidget(tracks_viewer1)
    tree_widget2 = TreeWidget(tracks_viewer2)
    viewer.window.add_dock_widget(tree_widget1)
    viewer.window.add_dock_widget(tree_widget2)

    # Start the finn GUI event loop
    finn.run()
