import numpy as np

from shigure_core.nodes.common_model.timestamp import Timestamp


class TopColorImage:
    """持ち去り時のためのカラー画像"""

    def __init__(self, timestamp: Timestamp, image: np.ndarray):
        self._timestamp = timestamp
        self._image = image

    @property
    def timestamp(self) -> Timestamp:
        return self._timestamp

    @property
    def image(self) -> np.ndarray:
        return self._image
