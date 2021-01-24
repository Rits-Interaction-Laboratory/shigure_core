from typing import Tuple

import numpy as np

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.object_detection.bounding_box import BoundingBox


class FrameObjectItem:

    def __init__(self, action: DetectedObjectActionEnum, bounding_box: BoundingBox, size: int,
                 mask: np.ndarray, detected_at: Timestamp):
        self._action = action
        self._bounding_box = bounding_box
        self._size = size
        self._mask = mask
        self._detected_at = detected_at

    def is_match(self, other) -> Tuple[bool, int]:
        if self.action != other.action:
            return False, 0

        left: BoundingBox = self.bounding_box
        right: BoundingBox = other.bounding_box

        x1 = max(left.x, right.x)
        y1 = max(left.y, right.y)
        x2 = min(left.x + left.width, right.x + right.width)
        y2 = min(left.y + left.height, right.y + right.height)

        result = x1 <= x2 and y1 <= y2
        size = (x2 - x1) * (y2 - y1) if result else 0
        return result, size

    @property
    def items(self) -> Tuple[DetectedObjectActionEnum, BoundingBox, int, np.ndarray, Timestamp]:
        return self._action, self._bounding_box, self._size, self._mask, self._detected_at

    @property
    def action(self) -> DetectedObjectActionEnum:
        return self._action

    @property
    def bounding_box(self) -> BoundingBox:
        return self._bounding_box

    @property
    def size(self) -> int:
        return self._size

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def detected_at(self) -> Timestamp:
        return self._detected_at
