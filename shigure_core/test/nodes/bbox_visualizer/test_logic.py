import numpy as np

from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.node_bbox_visualizer import (
    _build_marker_to_camera,
    _camera_to_marker,
    _get_median_depth,
)


def test_get_median_depth_returns_median_of_valid_pixels():
    depth_img = np.zeros((40, 40), dtype=np.uint16)
    depth_img[10:30, 10:30] = 1000
    depth_img[15:25, 15:25] = 2000

    bbox = BoundingBox(0, 0, 40, 40)

    result = _get_median_depth(depth_img, bbox)

    assert result == 1000.0


def test_get_median_depth_returns_zero_when_no_valid_pixels():
    depth_img = np.zeros((40, 40), dtype=np.uint16)
    bbox = BoundingBox(0, 0, 40, 40)

    result = _get_median_depth(depth_img, bbox)

    assert result == 0.0


def test_build_marker_to_camera_identity_rotation():
    rvec = np.zeros((3, 1))
    tvec = np.array([[1.0], [2.0], [3.0]])

    T = _build_marker_to_camera(rvec, tvec)

    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])


def test_camera_to_marker_round_trip():
    rvec = np.zeros((3, 1))
    tvec = np.array([[1.0], [2.0], [3.0]])
    T_marker_to_cam = _build_marker_to_camera(rvec, tvec)

    p_cam = np.array([1.0, 2.0, 3.0])

    p_marker = _camera_to_marker(p_cam, T_marker_to_cam)

    assert np.allclose(p_marker, [0.0, 0.0, 0.0])
