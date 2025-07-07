import os

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from finn.track_application_menus.welcome.create_project import (
    DialogValueError,
    check_df_seg_match,
    create_fp_array,
    has_duplicate_ids,
    read_tracks_df,
)


def test_create_fp_array_seg(tmp_path):
    # Create a small 3D dask array
    arr = np.arange(24, dtype=np.uint32).reshape((2, 3, 4))
    darr = da.from_array(arr, chunks=arr.shape)
    axes = {
        "time": {"axis_name": "t", "step_size": 1.0, "unit": "frame", "size": 2},
        "y": {"axis_name": "y", "step_size": 1.0, "unit": "um", "size": 3},
        "x": {"axis_name": "x", "step_size": 1.0, "unit": "um", "size": 4},
    }
    out_path = tmp_path / "test.zarr"
    # Call the function
    fpds = create_fp_array(
        str(out_path),
        darr,
        axes=axes,
        dtype=arr.dtype,
    )
    # Check that the output exists and has correct shape/dtype and that the values match
    assert os.path.exists(str(out_path))
    assert fpds.shape == (2, 3, 4)
    assert fpds.dtype == np.uint32
    np.testing.assert_array_equal(fpds[:], arr)


def test_has_duplicate_ids_dask():
    arr = np.zeros((2, 3, 3), dtype=np.uint32)
    arr[0, 0, 0] = 1
    arr[1, 1, 1] = 1
    darr = da.from_array(arr, chunks=(1, 3, 3))
    assert has_duplicate_ids(darr) is True

    arr = np.zeros((2, 3, 3), dtype=np.uint32)
    arr[0, 0, 0] = 1
    arr[1, 1, 1] = 2
    darr = da.from_array(arr, chunks=(1, 3, 3))
    assert has_duplicate_ids(darr) is False


def test_read_tracks_df_scaling(tmp_path):
    df = pd.DataFrame(
        {
            "t": [0, 1],
            "z": [10, 20],
            "y": [30, 40],
            "x": [50, 60],
            "id": [1, 2],
            "parent_id": [np.nan, 1],
        }
    )
    csv_path = tmp_path / "tracks.csv"
    df.to_csv(csv_path, index=False)
    axes = {
        "z": {"step_size": 2.0},
        "y": {"step_size": 1.0},
        "x": {"step_size": 0.5},
        "time": {"size": 2},
    }
    column_mapping = {
        "t": "t",
        "z": "z",
        "y": "y",
        "x": "x",
        "id": "id",
        "parent_id": "parent_id",
    }
    # with convert_pixel_units True, should scale the coordinates
    out_df = read_tracks_df(str(csv_path), column_mapping, True, axes)
    assert np.allclose(out_df["z"], df["z"] * 2.0)
    assert np.allclose(out_df["x"], df["x"] * 0.5)
    # with convert_pixel_units False, should not scale the coordinates and assume they are
    # world coordinates already
    out_df = read_tracks_df(str(csv_path), column_mapping, False, axes)
    assert np.allclose(out_df["z"], df["z"])
    assert np.allclose(out_df["x"], df["x"])


def test_read_tracks_df_nonunique_id(tmp_path):
    df = pd.DataFrame(
        {
            "t": [0, 1],
            "z": [10, 20],
            "y": [30, 40],
            "x": [50, 60],
            "id": [1, 1],  # duplicate id
            "parent_id": [np.nan, 1],
        }
    )
    csv_path = tmp_path / "tracks.csv"
    df.to_csv(csv_path, index=False)
    axes = {
        "z": {"step_size": 1.0},
        "y": {"step_size": 1.0},
        "x": {"step_size": 1.0},
        "time": {"size": 2},
    }
    column_mapping = {
        "t": "t",
        "z": "z",
        "y": "y",
        "x": "x",
        "id": "id",
        "parent_id": "parent_id",
    }
    with pytest.raises(DialogValueError):
        read_tracks_df(str(csv_path), column_mapping, False, axes)


def test_df_seg_match_success():
    df = pd.DataFrame(
        {
            "t": [0],
            "z": [1],
            "y": [2],
            "x": [3],
            "seg_id": [42],
        }
    )
    seg_np = np.zeros((1, 4, 5, 6), dtype=np.uint32)
    seg_np[0, 1, 2, 3] = 42
    seg = da.from_array(seg_np, chunks=seg_np.shape)
    axes = {
        "z": {"step_size": 1.0},
        "y": {"step_size": 1.0},
        "x": {"step_size": 1.0},
    }
    # Should not raise
    check_df_seg_match(df, seg, axes, mapping={"seg_id": "seg_id"})


def test_df_seg_match_rescale_success():
    df = pd.DataFrame(
        {
            "t": [0],
            "z": [
                2
            ],  # world coordinate means pixel coordinate (1) times scaling factor (2)
            "y": [2],
            "x": [3],
            "seg_id": [42],
        }
    )
    seg_np = np.zeros((1, 4, 5, 6), dtype=np.uint32)
    seg_np[0, 1, 2, 3] = 42
    seg = da.from_array(seg_np, chunks=seg_np.shape)
    axes = {
        "z": {"step_size": 2.0},
        "y": {"step_size": 1.0},
        "x": {"step_size": 1.0},
    }
    # Should not raise
    check_df_seg_match(df, seg, axes, mapping={"seg_id": "seg_id"})


def test_df_seg_match_fail():
    df = pd.DataFrame(
        {
            "t": [0],
            "z": [1],
            "y": [2],
            "x": [3],
            "seg_id": [99],
        }
    )
    seg_np = np.zeros((1, 4, 5, 6), dtype=np.uint32)
    seg_np[0, 1, 2, 3] = 42
    seg = da.from_array(seg_np, chunks=seg_np.shape)
    axes = {
        "z": {"step_size": 1.0},
        "y": {"step_size": 1.0},
        "x": {"step_size": 1.0},
    }
    with pytest.raises(DialogValueError):
        check_df_seg_match(df, seg, axes, mapping={"seg_id": "seg_id"})
