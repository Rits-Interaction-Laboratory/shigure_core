"""object_tracking の3Dコライダー生成（深度レンジ算出＋透視逆投影）.

追跡そのもの（2D）とは独立した「2D→3D復元」の責務をここに集約する。
ROS非依存の純関数群とし、マスクの復号(cv_bridge)などROS境界はノード側の責務とする
（ここには decode 済みの numpy 配列を渡す）。

将来的に透視逆投影 pixel→3D は util へ昇格し people_tracking 系と共有する想定。
"""
from typing import Optional, Tuple

import cv2
import numpy as np
from shigure_core_msgs.msg import Cube


def compute_depth_range(depth_roi: np.ndarray, mask_roi: Optional[np.ndarray]) -> Tuple[float, float]:
    """物体領域の深度レンジ (min, max) を求める.

    セグメンテーションマスク内部（縁から5px内側に収縮）の有効深度の 5%/95% 点を返す。
    マスクが無い / 内部に有効深度が無い場合は bbox 内の有効深度の min/max にフォールバックし、
    それも無ければ (0.0, 0.0) を返す。

    :param depth_roi: bbox 内に切り出した深度画像 (float32)
    :param mask_roi: depth_roi と同形の物体マスク（None 可）
    :return: (depth_min, depth_max)
    """
    valid = None
    if mask_roi is not None and mask_roi.shape == depth_roi.shape and mask_roi.any():
        binary = (mask_roi > 0).astype(np.uint8)
        # 縁から5pxより内部のみを残す (11x11カーネルで縁を5px収縮)
        kernel = np.ones((11, 11), np.uint8)
        interior_mask = cv2.erode(binary, kernel)
        interior = (interior_mask > 0) & (depth_roi != 0.0)
        if np.count_nonzero(interior) > 0:
            valid = depth_roi[interior]

    if valid is None or valid.size == 0:
        # フォールバック: bbox 内の有効深度の最小/最大を用いる
        masked = np.ma.masked_equal(depth_roi, 0.0, copy=False)
        if masked.count() == 0:
            return 0.0, 0.0
        return float(masked.min()), float(masked.max())

    depth_min = float(np.percentile(valid, 5))
    depth_max = float(np.percentile(valid, 95))
    return depth_min, depth_max


def build_collider(bounding_box, depth_min: float, depth_max: float, k_inv: np.ndarray) -> Cube:
    """2D bbox と深度レンジから3Dコライダー(Cube)を作る.

    bbox の左上・右下を近面深度 depth_min で透視逆投影して x/y/幅/高さを求め、
    z=depth_min・depth=depth_max-depth_min を厚みとする（角基準＋寸法表現）。

    :param bounding_box: x/y/width/height を持つ2D bbox
    :param depth_min: 近面深度
    :param depth_max: 遠面深度
    :param k_inv: カメラ内部行列 K の逆行列（呼び出し側で1回だけ計算して渡す）
    :return: Cube collider
    """
    s1 = np.asarray([[bounding_box.x, bounding_box.y, 1]]).T
    s2 = np.asarray([[bounding_box.x + bounding_box.width,
                      bounding_box.y + bounding_box.height, 1]]).T

    m1 = (depth_min * np.matmul(k_inv, s1)).T
    m2 = (depth_min * np.matmul(k_inv, s2)).T

    collider = Cube()
    collider.x, collider.y = float(m1[0, 0]), float(m1[0, 1])
    collider.width, collider.height = float(m2[0, 0] - m1[0, 0]), float(m2[0, 1] - m1[0, 1])
    collider.z = float(depth_min)
    collider.depth = float(depth_max - depth_min)
    return collider
