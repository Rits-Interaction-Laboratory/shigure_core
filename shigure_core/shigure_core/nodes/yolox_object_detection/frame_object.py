from typing import Tuple

from shigure_core.nodes.object_detection.frame_object_item import FrameObjectItem
from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum


class FrameObject:

    def __init__(self, item: FrameObjectItem, allow_empty_frame_count: int):
        self._item = item
        self._allow_empty_frame_count = allow_empty_frame_count
        self._empty_count = 0

    def add_empty_frame(self) -> None:
        self._empty_count += 1

    def is_finished(self) -> bool:
        #self._item.action = DetectedObjectActionEnum.TAKE_OUT
        return self._allow_empty_frame_count < self._empty_count

    @property
    def item(self) -> FrameObjectItem:
        return self._item
