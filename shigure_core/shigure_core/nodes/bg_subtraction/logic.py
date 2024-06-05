import numpy as np
from shigure_core.nodes.bg_subtraction.depth_frames import DepthFrames


class BgSubtractionLogic:
    """背景差分取得ロジック."""

    @staticmethod
    def execute(depth_frames: DepthFrames, current_frame: np.ndarray, depth_threshold: int = 25,
                sd_threshold: int = 3) -> (bool, np.ndarray):
        """
        背景差分を取得します.

        :param depth_frames: depthデータ
        :param current_frame: 現在のframeデータ
        :param depth_threshold: 抽出するdepth差[mm]
        :param sd_threshold: 抽出するsdの値
        :return: result, data <br> result: 背景差分が取得できたか <br> data: 背景差分データ
        """
        if not depth_frames.is_full():
            return False, np.zeros(shape=1)

        # 各ピクセルが平均値からのL1誤差
        avg = depth_frames.get_average()
        diff_pixel = avg - current_frame
        subtraction_pixel = np.abs(diff_pixel)
        # 差分による対象pixelを取得(平均)
        valid_pixel_of_avg = subtraction_pixel > depth_threshold
        # 標準偏差による対象pixelを取得(分散)
        standard_deviation = depth_frames.get_standard_deviation()
        valid_pixel_of_sd = np.divide(subtraction_pixel, standard_deviation, out=np.zeros_like(subtraction_pixel),
                                      where=standard_deviation != 0) > sd_threshold
        # 有効フレーム数を超えているpixelのみを対象にする
        valid_pixel = np.array([valid_pixel_of_avg, valid_pixel_of_sd, depth_frames.get_valid_pixel()]).all(axis=0)
        # pixelの値を求める
        signed_pixel = np.sign(diff_pixel) * 255
        pixel_value = np.where(signed_pixel >= 0, signed_pixel, 128)

        data: np.ndarray = valid_pixel * pixel_value
        return True, data.astype(np.uint8)
