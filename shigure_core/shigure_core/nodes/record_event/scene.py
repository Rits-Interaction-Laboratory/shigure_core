from typing import List

import numpy as np
from shigure_core.enum.contact_action_enum import ContactActionEnum

from shigure_core.nodes.record_event.event import Event


class Scene:
    """あるシーンを保存するためのクラス"""

    def __init__(self, frame_num: int, k: np.ndarray, event: Event, color_img_list: List[np.ndarray],
                 depth_img_list: List[np.ndarray]):
        self._frame_num = frame_num
        self._k_inv = np.linalg.inv(k)
        self._event = event
        self._color_img_list = color_img_list
        self._depth_imd_list = depth_img_list

    def is_full(self) -> bool:
        return len(self._color_img_list) == self._frame_num

    def add_frame(self, color_img: np.ndarray, depth_img: np.ndarray) -> None:
        self._color_img_list.append(color_img.copy())
        self._depth_imd_list.append(depth_img.copy())

    @property
    def k_inv(self) -> np.ndarray:
        return self._k_inv

    @property
    def event(self) -> Event:
        return self._event

    @property
    def color_img_list(self) -> List[np.ndarray]:
        return self._color_img_list

    @property
    def depth_img_list(self) -> List[np.ndarray]:
        return self._depth_imd_list

    @property
    def color_img_for_icon(self) -> np.ndarray:
        return self._color_img_list[self._frame_num - 1]
        # if ContactActionEnum.value_of(self.event.action) == ContactActionEnum.BRING_IN:
        #     return self._color_img_list[self._frame_num // 2 + 30]
        # else:
        #     return self._color_img_list[self._frame_num // 2 - 30]
