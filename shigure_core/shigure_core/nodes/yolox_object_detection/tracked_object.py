from enum import Enum, auto
from typing import List, Optional

import numpy as np

from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.yolox_object_detection.detection import Detection


class TrackedState(Enum):
    WAITING = auto()
    CONFIRMED = auto()


class TrackedObject:
    def __init__(self, detection: Detection):
        self.bbox: BoundingBox = detection.bbox
        self.class_id: str = detection.class_id
        self.mask: np.ndarray = detection.mask
        self.found_at: Timestamp = detection.found_at
        self.state: TrackedState = TrackedState.WAITING
        self.fhist: List[bool] = []

    def record(self, found: Optional[bool], fhist_size: int) -> None:
        """None = 骨格隠蔽によりスキップ"""
        if found is None:
            return
        self.fhist.append(found)
        self.trim_history(fhist_size)

    @property
    def found_rate(self) -> float:
        if not self.fhist:
            return 0.0
        return sum(self.fhist) / len(self.fhist)

    def is_history_full(self, fhist_size: int) -> bool:
        return len(self.fhist) >= fhist_size

    def should_confirm(self, fhist_size: int, confirm_threshold: float) -> bool:
        return self.is_history_full(fhist_size) and self.found_rate > confirm_threshold

    def should_dismiss(self, fhist_size: int, dismiss_threshold: float) -> bool:
        return self.is_history_full(fhist_size) and self.found_rate < dismiss_threshold

    def should_take_out(self, fhist_size: int, take_out_threshold: float) -> bool:
        return self.is_history_full(fhist_size) and self.found_rate < take_out_threshold

    def trim_history(self, fhist_size: int) -> None:
        if len(self.fhist) > fhist_size:
            self.fhist = self.fhist[-fhist_size:]

    def matches(self, detection: Detection, match_dist_threshold: int) -> bool:
        bbox_x = abs(self.bbox._x - detection.bbox._x)
        bbox_y = abs(self.bbox._y - detection.bbox._y)
        bbox_width = abs(self.bbox._width - detection.bbox._width)
        if (self.class_id == detection.class_id) and \
           (bbox_x < match_dist_threshold) and \
           (bbox_y < match_dist_threshold) and \
           (bbox_width < match_dist_threshold):
            self.found_at = detection.found_at
            return True
        return False
