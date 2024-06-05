from typing import Tuple

import numpy as np

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.common_model.bounding_box import BoundingBox


class FrameObjectItem:

    def __init__(self, action: DetectedObjectActionEnum, bounding_box: BoundingBox, size: int,
                 mask: np.ndarray, detected_at: Timestamp, class_id: str):
        self._action = action
        self._bounding_box = bounding_box
        self._size = size
        self._mask = mask
        self._detected_at = detected_at
        self._class_id = class_id

    def is_match(self, other) -> Tuple[bool, int]:
        if self.action != other.action:
            return False, 0

        left: BoundingBox = self.bounding_box
        right: BoundingBox = other.bounding_box

        return left.is_collided(right)

    @property
    def items(self) -> Tuple[DetectedObjectActionEnum, BoundingBox, int, np.ndarray, Timestamp]:
        return self._action, self._bounding_box, self._size, self._mask, self._detected_at,self._class_id

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
        
