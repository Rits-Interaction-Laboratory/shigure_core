from typing import List

import numpy as np

from shigure_core.nodes.people_mask_buffer.people_mask_buffer_frame import PeopleMaskBufferFrame


class PeopleMaskBufferFrames:
    _mask_count: np.ndarray

    def __init__(self, buffer_size: int = 100):
        self._buffer_size = buffer_size
        self._buffer: List[PeopleMaskBufferFrame] = []

    def is_full(self) -> bool:
        """
        バッファがいっぱいか判定します.

        :return:
        """
        return len(self._buffer) == self._buffer_size

    def add(self, frame: PeopleMaskBufferFrame) -> None:
        """
        バッファにフレームを追加します.

        :param frame: 追加するフレーム
        :return:
        """

        if not hasattr(self, '_mask_count'):
            self._mask_count = np.zeros(frame.mask.shape[:2])

        if self.is_full():
            delete_mask = self._buffer[0].mask
            self._mask_count -= delete_mask > 0
            self._buffer = self._buffer[1:]

        self._buffer.append(frame)
        self._mask_count += frame.mask > 0

    def get_people_mask(self) -> np.ndarray:
        """
        pixelが人物領域ならtrueであるmaskを取得します.

        :return:
        """
        return self._mask_count > 0
