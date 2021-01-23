import numpy as np

from shigure_core.nodes.common_model.timestamp import Timestamp


class SubtractionFrame:
    def __init__(self, subtraction_img: np.ndarray, people_mask: np.ndarray, timestamp: Timestamp):
        # 人物領域を0としたい
        people_mask_inv = (255 - people_mask) // 255
        self._synthesized_img = subtraction_img * people_mask_inv
        self._subtraction_img = subtraction_img
        self._people_mask = people_mask
        self._timestamp = timestamp

    @property
    def synthesized_img(self) -> np.ndarray:
        return self._synthesized_img

    @property
    def subtraction_img(self) -> np.ndarray:
        return self._subtraction_img

    @property
    def people_mask(self) -> np.ndarray:
        return self._people_mask

    @property
    def timestamp(self) -> Timestamp:
        return self._timestamp
