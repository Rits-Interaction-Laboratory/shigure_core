import numpy as np


class DepthFrames:
    """背景差分を抽出するためのフレームを保存するクラス."""

    frames: np.ndarray
    sum_each_pixel: np.ndarray
    max_frame_size: int

    def __init__(self, max_frame_size: int = 200):
        """
        コンストラクタ.

        :param max_frame_size: 最大保存フレーム数
        """
        self.max_frame_size = max_frame_size

    def add_frame(self, frame: np.ndarray) -> None:
        """
        フレームを保存します.

        フレームが最大フレーム数に達している場合は、最古のフレームが削除されます.

        :param frame: 保存するフレーム
        :return: None
        """
        if self.is_full():
            delete_frame = self.frames[0]
            self.sum_each_pixel -= delete_frame
            self.frames = self.frames[1:]

        if hasattr(self, 'sum_each_pixel'):
            self.sum_each_pixel += frame
        else:
            self.sum_each_pixel = frame

        if hasattr(self, 'frames'):
            self.frames = np.append(self.frames, [frame], axis=0)
        else:
            self.frames = np.array([frame])

    def is_full(self) -> bool:
        """
        保存フレーム数が最大フレーム数かどうか判定します.

        :return: 保存フレーム数が最大フレーム数であれば true
        """
        if not hasattr(self, 'frames'):
            return False
        return len(self.frames) == self.max_frame_size

    def get_average(self) -> np.ndarray:
        """
        pixelごとの平均値を取得します.

        :return: 各pixelの平均値
        """
        return self.sum_each_pixel / self.max_frame_size

    def get_var(self) -> np.ndarray:
        """
        pixelごとの標準偏差を取得します.

        :return: 各pixelの標準偏差
        """
        return np.var(self.frames, axis=0)
