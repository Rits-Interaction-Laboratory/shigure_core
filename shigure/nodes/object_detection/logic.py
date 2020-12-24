from typing import Tuple

import numpy as np


class ObjectDetectionLogic:
    """物体検出ロジッククラス"""

    @staticmethod
    def execute(subtraction_analyzed_img: np.ndarray, people_mask: np.ndarray,
                known_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        物体検出ロジック

        :param subtraction_analyzed_img:
        :param people_mask:
        :param known_mask:
        :return: 検出マスク, 更新された既知マスク
        """

        # 人物領域を0としたい
        people_mask_inv = (255 - people_mask) // 255

        object_detection_img = subtraction_analyzed_img * people_mask_inv * known_mask

        # 既知マスク更新
        # In の場合はマスクに追加
        known_mask = np.where(object_detection_img == 255, 0, known_mask)
        # Out の場合はマスクから削除
        known_mask = np.where(object_detection_img == 255, 1, known_mask)

        return object_detection_img, known_mask
