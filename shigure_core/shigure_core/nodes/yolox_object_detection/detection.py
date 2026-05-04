from dataclasses import dataclass

import numpy as np

from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.common_model.timestamp import Timestamp


@dataclass
class Detection:
    bbox: BoundingBox
    class_id: str
    mask: np.ndarray
    found_at: Timestamp
