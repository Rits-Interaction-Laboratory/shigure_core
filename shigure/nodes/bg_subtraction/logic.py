import numpy as np
from shigure.nodes.bg_subtraction.depth_frames import DepthFrames


class BgSubtractionLogic:
    """背景差分取得ロジック."""

    def execute(self, depth_frames: DepthFrames, current_frame: np.ndarray) -> (bool, np.ndarray):
        """
        背景差分を取得します.

        :param depth_frames: depthデータ
        :param current_frame: 現在のframeデータ
        :return: result, data <br> result: 背景差分が取得できたか <br> data: 背景差分データ
        """
        if not depth_frames.is_full():
            return False, np.zeros(shape=1)

        data = (np.abs(depth_frames.get_average() - current_frame) > 250) * 255
        return True, data
