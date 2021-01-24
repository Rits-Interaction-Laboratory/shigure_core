import numpy as np

from shigure_core.nodes.common_model.timestamp import Timestamp


class PeopleMaskBufferFrame:
    def __init__(self, mask: np.ndarray, timestamp: Timestamp):
        self._mask = mask
        self._timestamp = timestamp

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def timestamp(self) -> Timestamp:
        return self._timestamp
