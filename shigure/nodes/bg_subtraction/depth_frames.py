import numpy as np


class DepthFrames:
    """背景差分を抽出するためのフレームを保存するクラス."""

    # 測定有効範囲の深度しきい値
    THRESHOLD: int = 5000

    frames: list
    buffer_frames: list
    sum_each_pixel: np.ndarray
    sum_each_pixel_square: np.ndarray
    valid_frame_count: np.ndarray
    max_frame_size: int
    min_frame_size: int
    buffer_frame_size: int

    def __init__(self, max_frame_size: int = 30, min_frame_size: int = 25, buffer_frame_size: int = 100):
        """
        コンストラクタ.

        :param max_frame_size: 最大保存フレーム数
        :param min_frame_size: 演算対象最小フレーム数
        :param buffer_frame_size: 演算するまでの保持フレーム数
        """
        self.max_frame_size = max_frame_size
        self.min_frame_size = min_frame_size
        self.buffer_frame_size = buffer_frame_size
        self.frames = []
        self.buffer_frames = []

    def add_frame(self, frame: np.ndarray) -> None:
        """
        フレームを保存します.

        フレームが最大フレーム数に達している場合は、最古のフレームが削除されます.

        :param frame: 保存するフレーム
        :return: None
        """
        frame = frame.astype(np.float32)

        if self.is_buffer_full():
            # 最新のフレームはbufferの最後へ
            self.buffer_frames.append(frame.copy())
            # bufferの先頭をframeとする
            frame = self.buffer_frames[0]
            self.buffer_frames = self.buffer_frames[1:]
        else:
            # bufferにためてreturn
            self.buffer_frames.append(frame.copy())
            return

        if self.is_full():
            delete_frame = self.frames[0]
            self.sum_each_pixel -= delete_frame
            self.sum_each_pixel_square -= delete_frame * delete_frame
            self.valid_frame_count -= delete_frame > 0
            self.frames = self.frames[1:]

        if hasattr(self, 'sum_each_pixel'):
            self.sum_each_pixel += frame
        else:
            self.sum_each_pixel = frame

        if hasattr(self, 'sum_each_pixel_square'):
            self.sum_each_pixel_square += frame * frame
        else:
            self.sum_each_pixel_square = frame * frame

        if hasattr(self, 'valid_frame_count'):
            self.valid_frame_count += (frame > 0) * 1
        else:
            self.valid_frame_count = (frame > 0) * 1

        self.frames.append(frame.copy())

    def is_full(self) -> bool:
        """
        保存フレーム数が最大フレーム数かどうか判定します.

        :return: 保存フレーム数が最大フレーム数であれば true
        """
        return len(self.frames) == self.max_frame_size

    def is_buffer_full(self) -> bool:
        """
        バッファフレーム数が最大フレーム数かどうか判定します.

        :return: 保存フレーム数が最大フレーム数であれば true
        """
        return len(self.buffer_frames) == self.buffer_frame_size

    def get_average(self) -> np.ndarray:
        """
        pixelごとの平均値を取得します.

        有効フレーム数条件は考慮しません.

        :return: 各pixelの平均値
        """
        sum_of_each_pixel = self.sum_each_pixel
        valid_frame_count = self.valid_frame_count
        # 0除算はNaNとなるため0で返却
        return np.divide(sum_of_each_pixel, valid_frame_count,
                         out=np.zeros_like(sum_of_each_pixel), where=valid_frame_count != 0)

    def get_average_of_square(self):
        """
        pixelごとの2乗の平均値を取得します.

        有効フレーム数条件は考慮しません.

        :return: 各pixelの2乗の平均値
        """
        # 0除算はNaNとなるため0で返却
        sum_of_each_pixel_square = self.sum_each_pixel_square
        valid_frame_count = self.valid_frame_count
        # 0除算はNaNとなるため0で返却
        return np.divide(sum_of_each_pixel_square, valid_frame_count,
                         out=np.zeros_like(sum_of_each_pixel_square), where=valid_frame_count != 0)

    def get_var(self) -> np.ndarray:
        """
        pixelごとの分散を取得します.

        :return: 各pixelの分散
        """
        avg = self.get_average()
        return self.get_average_of_square() - (avg * avg)

    def get_standard_deviation(self):
        """
        pixelごとの標準偏差を取得します.

        :return: 各pixelの標準偏差
        """
        var = self.get_var()
        # 0除算はNaNとなるため0で返却
        return np.sqrt(var, out=np.zeros_like(var), where=var > 0)

    def get_valid_pixel(self) -> np.ndarray:
        """
        有効なピクセルを取得します.

        :return: 有効なpixelであれば True、そうでなければ False
        """
        # 0は値が取得できていないため、無効なデータとみなす
        return self.valid_frame_count >= self.min_frame_size

