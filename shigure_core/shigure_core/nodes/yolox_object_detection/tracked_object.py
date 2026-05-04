from enum import Enum, auto
from typing import List, Optional

import numpy as np

from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.yolox_object_detection.detection import Detection

FHIST_SIZE = 10


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

    def record(self, found: Optional[bool]) -> None:
        """None = 骨格隠蔽によりスキップ"""
        if found is None:
            return
        self.fhist.append(found)

    @property
    def found_rate(self) -> float:
        if not self.fhist:
            return 0.0
        return sum(self.fhist) / len(self.fhist)

    @property
    def is_history_full(self) -> bool:
        return len(self.fhist) >= FHIST_SIZE

    def should_confirm(self) -> bool:
        return self.is_history_full and self.found_rate > 0.6

    def should_dismiss(self) -> bool:
        return self.is_history_full and self.found_rate < 0.5

    def should_take_out(self) -> bool:
        return self.is_history_full and self.found_rate < 0.5

    def trim_history(self) -> None:
        self.fhist = self.fhist[-(FHIST_SIZE - 1):]

    def matches(self, detection: Detection) -> bool:
        bbox_x = abs(self.bbox._x - detection.bbox._x)
        bbox_y = abs(self.bbox._y - detection.bbox._y)
        bbox_width = abs(self.bbox._width - detection.bbox._width)
        if (self.class_id == detection.class_id) and (bbox_x < 10) and (bbox_y < 10) and (bbox_width < 10):
            self.found_at = detection.found_at
            return True
        return False
