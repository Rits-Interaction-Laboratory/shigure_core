class Cube:
    def __init__(self, x: float, y: float, z: float, width: float, height: float, depth: float):
        self._x = x
        self._y = y
        self._z = z
        self._width = width
        self._height = height
        self._depth = depth

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def depth(self):
        return self._depth

    def is_collided(self, other):
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        z1 = max(self.z, other.z)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        z2 = min(self.z + self.depth, other.z + other.depth)

        result = x1 <= x2 and y1 <= y2 and z1 <= z2
        volume = (x2 - x1) * (y2 - y1) * (z2 - z1) if result else 0
        return result, volume
