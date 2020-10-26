import numpy as np
from shigure.nodes.bg_subtraction.depth_frames import DepthFrames


class BgSubtractionLogic:
    """背景差分取得ロジック."""

    def execute(self, depth_frames: DepthFrames, current_frame: np.ndarray, depth: int = 100) -> (bool, np.ndarray):
        """
        背景差分を取得します.

        :param depth_frames: depthデータ
        :param current_frame: 現在のframeデータ
        :param depth: 抽出するdepth差[mm]
        :return: result, data <br> result: 背景差分が取得できたか <br> data: 背景差分データ
        """
        if not depth_frames.is_full():
            return False, np.zeros(shape=1)

        # 各ピクセルが平均値からのL1誤差
        subtraction_pixel = np.abs(depth_frames.get_average() - current_frame)
        # 差分による対象pixelを取得(平均)
        valid_pixel_of_avg = subtraction_pixel > depth
        # 差分による対象pixelを取得(分散)
        valid_pixel_of_var = subtraction_pixel > 2 * depth_frames.get_var()
        # 有効フレーム数を超えているpixelのみを対象にする
        valid_pixel = np.array([valid_pixel_of_avg, depth_frames.get_valid_pixel()]).all(axis=0)
        data: np.ndarray = valid_pixel * 255
        return True, data
