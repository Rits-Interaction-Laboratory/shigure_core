import numpy as np
from shigure_core.nodes.subtraction_analysis.subtraction_frames import SubtractionFrames


class SubtractionAnalysisLogic:
    """背景差分解析ロジック."""

    @staticmethod
    def execute(subtraction_frames: SubtractionFrames) -> (bool, np.ndarray):
        """
        背景差分を解析します.
        あらかじめ指定されたフレーム数すべてで差分が検知されているピクセルが白色になります.

        :param subtraction_frames: 背景差分データ
        :return: result, data <br> result: 物体検知できたか <br> data: 物体検知データ
        """
        if not subtraction_frames.is_full():
            return False, np.zeros(shape=1)

        # すべてのフレームで差分が取得されたpixelのみを対象にする
        valid_pixel = subtraction_frames.get_valid_pixel()
        data: np.ndarray = valid_pixel * subtraction_frames.get_top_frame()
        return True, data.astype(np.uint8)
