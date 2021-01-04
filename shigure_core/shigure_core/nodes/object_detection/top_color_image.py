import numpy as np


class TopColorImage:
    """持ち去り時のためのカラー画像"""

    def __init__(self, sec: int, nano_sec: int, image: np.ndarray):
        self._sec = sec
        self._nano_sec = nano_sec
        self._image = image

    def get_sec(self) -> int:
        return self._sec

    def get_nano_sec(self) -> int:
        return self._nano_sec

    def get_image(self) -> np.ndarray:
        return self._image

    def is_old(self, top_color_image) -> bool:
        if self.get_sec() < top_color_image.get_sec():
            return True
        if self.get_sec() > top_color_image.get_sec():
            return False
        return self.get_nano_sec() < top_color_image.get_nano_sec()
