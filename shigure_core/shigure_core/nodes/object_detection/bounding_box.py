from typing import Tuple


class BoundingBox:
    def __init__(self, x: int, y: int, width: int, height: int):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def get_items(self) -> Tuple[int, int, int, int]:
        return self._x, self._y, self._width, self._height

    def get_x(self) -> int:
        return self._x

    def get_y(self) -> int:
        return self._y

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height
