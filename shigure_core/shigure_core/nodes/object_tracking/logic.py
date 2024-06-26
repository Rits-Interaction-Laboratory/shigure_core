from operator import itemgetter
from typing import Tuple, List

import numpy as np
from shigure_core_msgs.msg import DetectedObjectList, DetectedObject

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.object_tracking.tracking_info import TrackingInfo


class ObjectTrackingLogic:
    """物体追跡ロジック."""

    @staticmethod
    def execute(depth_img: np.ndarray, detected_object_list: DetectedObjectList,
                tracking_info: TrackingInfo) -> TrackingInfo:
        bring_in_list = []
        obj_move_list = []
        take_out_list = []

        detected_object: DetectedObject
        for detected_object in detected_object_list.object_list:
            action = DetectedObjectActionEnum.value_of(detected_object.action)
            bounding_box = ObjectTrackingLogic.convert_to_bounding_box(detected_object, depth_img.shape[:2])

            if action == DetectedObjectActionEnum.BRING_IN :
                bring_in_list.append((detected_object, bounding_box))

            elif action == DetectedObjectActionEnum.OBJ_MOVE:
                obj_move_list.append((detected_object, bounding_box))

            else:    
                take_out_list.append((detected_object, bounding_box))

        return ObjectTrackingLogic.tracking(bring_in_list, obj_move_list, take_out_list, tracking_info)

    @staticmethod
    def tracking(bring_in_list: list,obj_move_list: list, take_out_list: list, tracking_info: TrackingInfo) -> TrackingInfo:
        previous_object_dict = tracking_info.object_dict
        current_object_dict = {}
        buffer_obj_id = str


        

        linked_list: List[Tuple[str, Tuple[DetectedObject, BoundingBox], Tuple[DetectedObject, BoundingBox], int]] = []
        for object_id, prev_item in list(previous_object_dict.items()):
            prev_object, prev_bounding_box = prev_item

            # 持ち出しは無視
            action = DetectedObjectActionEnum.value_of(prev_object.action)
            if action == DetectedObjectActionEnum.TAKE_OUT:
                buffer_obj_id = object_id
                del previous_object_dict[object_id]
                continue

            for take_out_item in take_out_list:
                _, take_out_bounding_box = take_out_item

                result, size = take_out_bounding_box.is_collided(prev_bounding_box)
                if result:
                    linked_list.append((object_id, prev_item, take_out_item, size))

        linked_list = sorted(linked_list, key=itemgetter(3), reverse=True)
        for object_id, prev_item, take_out_item, _ in linked_list:
            if object_id not in current_object_dict.keys() and take_out_item in take_out_list:
                _, prev_bounding_box = prev_item
                _, take_out_bounding_box = take_out_item

                new_bounding_box = ObjectTrackingLogic.marge_bounding_box(prev_bounding_box, take_out_bounding_box)
                take_out_item[0].bounding_box.x = float(new_bounding_box.x)
                take_out_item[0].bounding_box.y = float(new_bounding_box.y)
                take_out_item[0].bounding_box.width = float(new_bounding_box.width)
                take_out_item[0].bounding_box.height = float(new_bounding_box.height)

                current_object_dict[object_id] = take_out_item
                del previous_object_dict[object_id]
                take_out_list.remove(take_out_item)

        for object_id, item in previous_object_dict.items():
            item[0].action = "stay"
            current_object_dict[object_id] = item

        
        for obj_move_object in obj_move_list:
            current_object_dict[tracking_info.old_object_id()] = obj_move_object

        # 持ち込みは新規登録
        for bring_in_object in bring_in_list:
            current_object_dict[tracking_info.new_object_id()] = bring_in_object


        tracking_info.object_dict = current_object_dict

        return tracking_info

    @staticmethod
    def convert_to_bounding_box(detected_object: DetectedObject, shape: Tuple[int, int]) -> BoundingBox:
        max_height, max_width = shape
        x = min(int(detected_object.bounding_box.x), max_width - 1)
        y = min(int(detected_object.bounding_box.y), max_height - 1)
        width = int(detected_object.bounding_box.width)
        height = int(detected_object.bounding_box.height)

        return BoundingBox(x, y, width, height)

    @staticmethod
    def marge_bounding_box(left: BoundingBox, right: BoundingBox) -> BoundingBox:
        x = min(left.x, right.x)
        y = min(left.y, right.y)
        width = max(left.x + left.width,
                    right.x + right.width) - x
        height = max(left.y + left.height,
                     right.y + right.height) - y

        return BoundingBox(x, y, width, height)
