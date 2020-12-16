import numpy as np
from shigure.nodes.object_detection.subtraction_frames import SubtractionFrames


class ObjectDetectionLogic:
    """物体検知ロジック."""

    def execute(self, subtraction_frames: SubtractionFrames) -> (bool, np.ndarray):
        """
        物体検知をします.

        :param subtraction_frames: 背景差分データ
        :return: result, data <br> result: 物体検知できたか <br> data: 物体検知データ
        """
        if not subtraction_frames.is_full():
            return False, np.zeros(shape=1)

        # 有効フレーム数を超えているpixelのみを対象にする
        valid_pixel = subtraction_frames.get_valid_pixel()
        data: np.ndarray = valid_pixel * 255
        return True, data
