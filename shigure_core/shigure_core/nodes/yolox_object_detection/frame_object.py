from shigure_core.nodes.yolox_object_detection.frame_object_item import FrameObjectItem


class FrameObject:

    def __init__(self, item: FrameObjectItem):
        self._item = item

    @property
    def item(self) -> FrameObjectItem:
        return self._item
