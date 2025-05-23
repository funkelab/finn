from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from finn.utils.geometry import (
    project_points_onto_plane,
)

if TYPE_CHECKING:
    from finn.layers.image.image import Image
    from finn.layers.labels.labels import Labels
    from finn.track_data_views.views.layers.track_labels import TrackLabels


def displayed_plane_from_nd_line_segment(
    start_point: npt.NDArray,
    end_point: npt.NDArray,
    dims_displayed: list[int] | npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Get the plane defined by start_point and the normal vector that goes
    from start_point to end_point.

    Note the start_point and end_point are nD and
    the returned plane is in the displayed dimensions (i.e., 3D).

    Parameters
    ----------
    start_point : np.ndarray
        The start point of the line segment in nD coordinates.
    end_point : np.ndarray
        The end point of the line segment in nD coordinates..
    dims_displayed : Union[List[int], np.ndarray]
        The dimensions of the data array currently in view.

    Returns
    -------
    plane_point : np.ndarray
        The point on the plane that intersects the click ray. This is returned
        in data coordinates with only the dimensions that are displayed.
    plane_normal : np.ndarray
        The normal unit vector for the plane. It points in the direction of the click
        in data coordinates.
    """
    plane_point = start_point[dims_displayed]
    end_position_view = end_point[dims_displayed]
    ray_direction = end_position_view - plane_point
    plane_normal = ray_direction / np.linalg.norm(ray_direction)
    return plane_point, plane_normal


def drag_data_to_projected_distance(
    start_position: npt.NDArray,
    end_position: npt.NDArray,
    view_direction: npt.NDArray,
    vector: npt.NDArray,
) -> npt.NDArray:
    """Calculate the projected distance between two mouse events.

    Project the drag vector between two mouse events onto a 3D vector
    specified in data coordinates.

    The general strategy is to
    1) find mouse drag start and end positions, project them onto a
       pseudo-canvas (a plane aligned with the canvas) in data coordinates.
    2) project the mouse drag vector onto the (normalised) vector in data
       coordinates
    Parameters
    ----------
    start_position : np.ndarray
        Starting point of the drag vector in data coordinates
    end_position : np.ndarray
        End point of the drag vector in data coordinates
    view_direction : np.ndarray
        Vector defining the plane normal of the plane onto which the drag
        vector is projected.
    vector : np.ndarray
        (3,) unit vector or (n, 3) array thereof on which to project the drag
        vector from start_event to end_event. This argument is defined in data
        coordinates.
    Returns
    -------
    projected_distance : (1, ) or (n, ) np.ndarray of float
    """
    # enforce at least 2d input
    vector = np.atleast_2d(vector)

    # Store the start and end positions in world coordinates
    start_position = np.asarray(start_position)
    end_position = np.asarray(end_position)

    # Project the start and end positions onto a pseudo-canvas, a plane
    # parallel to the rendered canvas in data coordinates.
    end_position_canvas, _ = project_points_onto_plane(
        end_position, start_position, view_direction
    )
    # Calculate the drag vector on the pseudo-canvas.
    drag_vector_canvas = np.squeeze(end_position_canvas - start_position)

    # Project the drag vector onto the specified vector(s), return the distance
    return np.einsum("j, ij -> i", drag_vector_canvas, vector).squeeze()


def orient_clipping_plane_normals(layer: Image | Labels | TrackLabels, orientation: str):
    if not layer.ndim >= 3:
        return

    if orientation == "x":
        layer.clipping_planes[0].normal = (0, 0, 1)
        layer.clipping_planes[1].normal = (0, 0, -1)

    elif orientation == "y":
        layer.clipping_planes[0].normal = (0, 1, 0)
        layer.clipping_planes[1].normal = (0, -1, 0)

    elif orientation == "z":
        layer.clipping_planes[0].normal = (1, 0, 0)
        layer.clipping_planes[1].normal = (-1, 0, 0)


def nd_line_segment_to_displayed_data_ray(
    start_point: np.ndarray,
    end_point: np.ndarray,
    dims_displayed: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the start and end point of the line segment of a mouse click ray
    intersecting a data cube to a ray (i.e., start position and direction) in
    displayed data coordinates

    Note: the ray starts 0.1 data units outside of the data volume.

    Parameters
    ----------
    start_point : np.ndarray
        The start position of the ray used to interrogate the data.
    end_point : np.ndarray
        The end position of the ray used to interrogate the data.
    dims_displayed : List[int]
        The indices of the dimensions currently displayed in the Viewer.

    Returns
    -------
    start_position : np.ndarray
        The start position of the ray in displayed data coordinates
    ray_direction : np.ndarray
        The unit vector describing the ray direction.
    """
    # get the ray in the displayed data coordinates
    start_position = start_point[dims_displayed]
    end_position = end_point[dims_displayed]
    ray_direction = end_position - start_position
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    # step the start position back a little bit to be able to detect shapes
    # that contain the start_position
    start_position = start_position - 0.1 * ray_direction
    return start_position, ray_direction
