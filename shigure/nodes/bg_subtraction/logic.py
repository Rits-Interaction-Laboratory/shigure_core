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
        avg = depth_frames.get_average()
        subtraction_pixel = np.abs(avg - current_frame)
        # 差分による対象pixelを取得(平均)
        valid_pixel_of_avg = subtraction_pixel > depth
        # 標準偏差による対象pixelを取得(分散)
        standard_deviation = depth_frames.get_standard_deviation()
        valid_pixel_of_sd = np.divide(subtraction_pixel, standard_deviation,
                                      out=np.zeros_like(subtraction_pixel), where=standard_deviation != 0) > 10
        # 有効フレーム数を超えているpixelのみを対象にする
        valid_pixel = np.array([valid_pixel_of_avg, valid_pixel_of_sd, depth_frames.get_valid_pixel()]).all(axis=0)
        data: np.ndarray = valid_pixel * 255
        return True, data
