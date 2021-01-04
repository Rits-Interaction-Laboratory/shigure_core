import cv2
import numpy as np


class ObjectExtractionLogic:
    """
    物体抽出ロジック
    """

    @staticmethod
    def execute(color_img: np.ndarray, object_detection_mask: np.ndarray, result_buffer: np.ndarray, min_size: int):
        """
        物体を画像から抽出します

        :param color_img:
        :param object_detection_mask:
        :param min_size:
        :return:
        """

        binary_img = np.where(object_detection_mask != 0, 255, 0).astype(np.uint8)

        result_buffer = np.where(object_detection_mask == 255, 1, result_buffer)
        result_buffer = np.where(object_detection_mask == 128, 0, result_buffer)

        # ラベリング処理(移行予定)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

        result_img = np.zeros_like(color_img, np.uint8)
        for j in range(color_img.shape[2]):
            result_img[:, :, j] = color_img[:, :, j] * result_buffer

        for i, row in enumerate(stats):
            area = row[cv2.CC_STAT_AREA]
            x = row[cv2.CC_STAT_LEFT]
            y = row[cv2.CC_STAT_TOP]
            height = row[cv2.CC_STAT_HEIGHT]
            width = row[cv2.CC_STAT_WIDTH]

            if i != 0 and area >= min_size:
                # for j in range(color_img.shape[2]):
                #     result_img[:, :, j] += np.where(labels == i, color_img[:, :, j], 0).astype(np.uint8)
                cv2.rectangle(result_img, (x, y), (x + width, y + height), (255, 255, 100))

        return result_img, result_buffer
