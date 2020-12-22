from typing import Tuple

import cv2
import numpy as np
from openpose_ros2_msgs.msg import PoseKeyPointsList


class PeopleDetectionLogic:
    """人物検出ロジック."""

    @staticmethod
    def execute(img: np.ndarray, pose: PoseKeyPointsList, is_debug_mode: bool) -> Tuple[np.ndarray, np.ndarray]:
        # 人物候補のpixelをマーク
        candidate_img: np.ndarray = np.zeros_like(img, np.uint8)

        for pose_key_points in pose.pose_key_points_list:
            max_depth = 0
            min_depth = 999999
            for pose_key_point in pose_key_points.pose_key_points:
                x = int(pose_key_point.x)
                y = int(pose_key_point.y)

                if x == 0 and y == 0:
                    continue

                depth = img[y, x]
                if depth == 0:
                    continue

                max_depth = max(max_depth, depth)
                min_depth = min(min_depth, depth)

            candidate_img = np.where(min_depth <= img <= max_depth, 255, candidate_img)

        # 連続領域をラベリングしkey_pointがあった領域を人物領域とする
        result_img = np.zeros_like(img, np.uint8)
        debug_img = None
        if is_debug_mode:
            debug_img = cv2.cvtColor(np.zeros_like(img, np.uint8), cv2.COLOR_GRAY2BGR)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_img, connectivity=8)

        for pose_key_points in pose.pose_key_points_list:
            for pose_key_point in pose_key_points.pose_key_points:
                x = int(pose_key_point.x)
                y = int(pose_key_point.y)

                if x == 0 and y == 0:
                    continue

                label_id = labels[y, x]
                # 背景は無視
                if label_id == 0:
                    continue

                result_img = np.where(labels == label_id, 255, result_img)
                if is_debug_mode:
                    row = stats[label_id]
                    x = row[cv2.CC_STAT_LEFT]
                    y = row[cv2.CC_STAT_TOP]
                    height = row[cv2.CC_STAT_HEIGHT]
                    width = row[cv2.CC_STAT_WIDTH]

                    debug_img[np.where(labels == label_id)] = [100, 255, 225]
                    cv2.rectangle(debug_img, (x, y), (x + width, y + height), (255, 255, 100))

        return result_img, debug_img
