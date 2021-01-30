import numpy as np

from shigure_core.nodes.common_model.timestamp import Timestamp


class ColorImageFrame:
    """ある時間のカラー画像"""

    def __init__(self, timestamp: Timestamp, old_image: np.ndarray, new_image: np.ndarray):
        self._timestamp = timestamp
        self._old_image = old_image
        self._new_image = new_image

    @property
    def timestamp(self) -> Timestamp:
        return self._timestamp

    @property
    def old_image(self) -> np.ndarray:
        return self._old_image

    @property
    def new_image(self) -> np.ndarray:
        return self._new_image

    @new_image.setter
    def new_image(self, new_image: np.ndarray):
        self._new_image = new_image
