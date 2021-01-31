import numpy as np
from shigure_core.nodes.common_model.bounding_box import BoundingBox


class Event:
    """イベントを表すクラス"""

    def __init__(self, people_id: str, object_id: str, action: str, people_bounding_box: BoundingBox,
                 object_bounding_box: BoundingBox):
        self._people_id = people_id
        self._object_id = object_id
        self._action = action
        self._people_bounding_box = people_bounding_box
        self._object_bounding_box = object_bounding_box

    @property
    def people_id(self) -> str:
        return self._people_id

    @property
    def object_id(self) -> str:
        return self._object_id

    @property
    def action(self) -> str:
        return self._action

    @property
    def people_bounding_box(self) -> BoundingBox:
        return self._people_bounding_box

    @property
    def object_bounding_box(self) -> BoundingBox:
        return self._object_bounding_box
