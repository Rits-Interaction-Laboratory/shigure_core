from typing import Tuple


class BoundingBox:
    def __init__(self, x: int, y: int, width: int, height: int):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def is_collided(self, other) -> Tuple[bool, int]:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        result = x1 <= x2 and y1 <= y2
        size = (x2 - x1) * (y2 - y1) if result else 0
        return result, size

    @property
    def items(self) -> Tuple[int, int, int, int]:
        return self._x, self._y, self._width, self._height

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height
