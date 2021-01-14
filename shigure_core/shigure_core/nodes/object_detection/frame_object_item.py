from typing import Tuple

import numpy as np

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.object_detection.bounding_box import BoundingBox
from shigure_core.nodes.object_detection.top_color_image import TopColorImage


class FrameObjectItem:

    def __init__(self, action: DetectedObjectActionEnum, bounding_box: BoundingBox, size: int,
                 mask: np.ndarray, top_color_image: TopColorImage):
        self._action = action
        self._bounding_box = bounding_box
        self._size = size
        self._mask = mask
        self._top_color_image = top_color_image

    def is_match(self, other) -> Tuple[bool, int]:
        if self.get_action() != other.get_action():
            return False, 0

        left: BoundingBox = self.get_bounding_box()
        right: BoundingBox = other.get_bounding_box()

        x1 = max(left.get_x(), right.get_x())
        y1 = max(left.get_y(), right.get_y())
        x2 = min(left.get_x() + left.get_width(), right.get_x() + right.get_width())
        y2 = min(left.get_y() + left.get_height(), right.get_y() + right.get_height())

        result = x1 <= x2 and y1 <= y2
        size = (x2 - x1) * (y2 - y1) if result else 0
        return result, size

    def get_items(self) -> Tuple[DetectedObjectActionEnum, BoundingBox, int, np.ndarray, TopColorImage]:
        return self._action, self._bounding_box, self._size, self._mask, self._top_color_image

    def get_action(self) -> DetectedObjectActionEnum:
        return self._action

    def get_bounding_box(self) -> BoundingBox:
        return self._bounding_box

    def get_size(self) -> int:
        return self._size

    def get_mask(self) -> np.ndarray:
        return self._mask

    def get_top_color_image(self) -> TopColorImage:
        return self._top_color_image