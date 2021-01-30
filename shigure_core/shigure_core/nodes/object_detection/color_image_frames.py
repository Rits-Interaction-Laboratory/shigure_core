from typing import List

from shigure_core.nodes.object_detection.color_image_frame import ColorImageFrame


class ColorImageFrames:
    def __init__(self, buffer_size: int = 100):
        self._buffer_size = buffer_size
        self._buffer: List[ColorImageFrame] = []

    def is_full(self) -> bool:
        """
        バッファがいっぱいかどうか判定します.

        :return:
        """
        return len(self._buffer) == self._buffer_size

    def add(self, frame: ColorImageFrame) -> None:
        """
        フレームを追加します.

        :param frame:
        :return:
        """
        if self.is_full():
            self._buffer: List[ColorImageFrame] = self._buffer[1:]
        self._buffer.append(frame)

    def get(self, index: int) -> ColorImageFrame:
        return self._buffer[index]

    @property
    def top_frame(self) -> ColorImageFrame:
        return self._buffer[0]
