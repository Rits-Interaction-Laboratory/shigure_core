import copy
from typing import List

import numpy as np

from shigure.nodes.subtraction_analysis.timestamp import Timestamp


class SubtractionFrames:
    """複数枚の背景差分を解析するクラス."""

    frames: list
    timestamps: List[Timestamp]
    valid_frame_count: np.ndarray
    frame_size: int

    def __init__(self, frame_size: int = 15):
        """
        コンストラクタ.

        :param frame_size: 判定に利用する差分のフレーム数
        """
        self.frame_size = frame_size

        self.frames = []
        self.timestamps = []

    def add_frame(self, frame: np.ndarray, timestamp: Timestamp) -> None:
        """
        フレームを保存します.

        フレームが最大フレーム数に達している場合は、最古のフレームが削除されます.

        :param frame: 保存するフレーム
        :param timestamp: フレームのタイムスタンプ
        :return: None
        """
        if self.is_full():
            delete_frame = self.frames[0]
            self.valid_frame_count -= delete_frame > 0
            self.frames = self.frames[1:]
            self.timestamps = self.timestamps[1:]

        if hasattr(self, 'valid_frame_count'):
            self.valid_frame_count += (frame > 0) * 1
        else:
            self.valid_frame_count = (frame > 0) * 1

        self.frames.append(frame.copy())
        self.timestamps.append(copy.copy(timestamp))

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
        解析しているタイムスタンプを取得します.

        :return:
        """
        return self.timestamps[0].get_timestamp()
