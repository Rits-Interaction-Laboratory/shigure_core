from typing import List

import numpy as np

from shigure_core.nodes.subtraction_analysis.subtraction_frame import SubtractionFrame


class SubtractionFrames:
    """複数枚の背景差分を解析するクラス."""

    frames: List[SubtractionFrame]
    valid_frame_count: np.ndarray
    frame_size: int

    def __init__(self, frame_size: int = 60):
        """
        コンストラクタ.

        :param frame_size: 判定に利用する差分のフレーム数
        """
        self.frame_size = frame_size

        self.frames = []

    def add_frame(self, frame: SubtractionFrame) -> None:
        """
        フレームを保存します.

        フレームが最大フレーム数に達している場合は、最古のフレームが削除されます.

        :param frame: 保存するフレーム
        :return: None
        """
        if self.is_full():
            delete_frame = self.frames[0]
            delete_frame_img = delete_frame.synthesized_img
            self.valid_frame_count -= (delete_frame_img > 0) * 1
            self.frames = self.frames[1:]

        synthesized_img = frame.synthesized_img
        if hasattr(self, 'valid_frame_count'):
            self.valid_frame_count += (synthesized_img > 0) * 1
        else:
            self.valid_frame_count = (synthesized_img > 0) * 1

        self.frames.append(frame)

    def is_full(self) -> bool:
        """
        保存フレーム数が最大フレーム数かどうか判定します.

        :return: 保存フレーム数が最大フレーム数であれば true
        """
        return len(self.frames) == self.frame_size

    def get_valid_pixel(self) -> np.ndarray:
        """
        有効なピクセルを取得します.

        :return: 有効なpixelであれば True、そうでなければ False
        """
        return self.frame_size == self.valid_frame_count

    def get_timestamp(self) -> (int, int):
        """
        解析している画像のタイムスタンプを取得します.

        :return:
        """
        return self.frames[0].timestamp.timestamp

    def get_top_frame(self) -> SubtractionFrame:
        """
        解析している画像を取得します.

        :return:
        """
        return self.frames[0]

    def get_top_frame_img(self) -> np.ndarray:
        """
        解析している画像を取得します.

        :return:
        """
        return self.frames[0].subtraction_img
