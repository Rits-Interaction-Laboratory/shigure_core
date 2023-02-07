from typing import Tuple

import numpy as np

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.common_model.bounding_box import BoundingBox


class BboxObject:

    def __init__(self,  bounding_box:BoundingBox, size:int, mask_img:np.ndarray, found_at:Timestamp,class_id:str):
        self._bounding_box = bounding_box
        self._size = size
        self._mask = mask_img
        self._found_at = found_at
        self._class_id = class_id
        self.fhist = [] # 各フレームでそのbboxが見つかったかどうかの履歴(True or False)
        self.is_exist_start = False
        self.is_exist_wait = False
        self.is_exist_bring = False
        

    def is_match(self, other):
    	bbox_x = abs(self._bounding_box._x - other._bounding_box._x)
    	bbox_y = abs(self._bounding_box._y - other._bounding_box._y)
    	bbox_width = abs(self._bounding_box._width - other._bounding_box._width)
    	bbox_height = abs(self._bounding_box._height - other._bounding_box._height)
    	if (self._class_id==other._class_id)and(bbox_x < 10) and (bbox_y < 10)and(bbox_width < 10): #& bbox_width < 30 & bbox_height < 30:
    		self._found_at = other._found_at
    		return True
    	else:
    		
    		return False
    		
    
