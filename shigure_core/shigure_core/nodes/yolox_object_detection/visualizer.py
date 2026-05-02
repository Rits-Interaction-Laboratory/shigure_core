from typing import List

import cv2
import numpy as np
from bboxes_ex_msgs.msg import BoundingBoxes
from shigure_core_msgs.msg import PoseKeyPointsList

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.yolox_object_detection.bbox_object import BboxObject
from shigure_core.nodes.yolox_object_detection.frame_object import FrameObject

_POSE_PAIRS = [
    [1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [8, 12], [9, 10], [10, 11], [11, 22], [11, 24],
    [12, 13], [13, 14], [14, 19], [14, 21],
    [15, 0], [15, 17], [16, 0], [16, 18],
    [19, 20], [22, 11], [22, 23],
]


class YoloxVisualizer:

    @staticmethod
    def draw(
        color_img: np.ndarray,
        yolox_bboxes: BoundingBoxes,
        wait_item_list: List[BboxObject],
        bring_in_list: List[BboxObject],
        people: PoseKeyPointsList,
        frame_object_list: List[FrameObject],
    ) -> None:
        img_height, img_width = color_img.shape[:2]
        result_img = color_img.copy()
        yolox_img = color_img.copy()

        for w_item in wait_item_list:
            x, y, w, h = w_item._bounding_box.items
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 204, 102), thickness=3)
            cv2.putText(result_img, 'WAIT', (x + 2, y + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 204, 102), thickness=3)

        for b_item in bring_in_list:
            x, y, w, h = b_item._bounding_box.items
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 102), thickness=3)
            cv2.putText(result_img, 'BRING', (x + 2, y + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 102), thickness=3)

        for bbox in yolox_bboxes.bounding_boxes:
            cv2.rectangle(yolox_img, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (102, 204, 51), thickness=3)

        for person in people.pose_key_points_list:
            for part in range(len(person.point_data)):
                pp = person.point_data[part].pixel_point
                x = np.clip(int(pp.x), 0, img_width - 1)
                y = np.clip(int(pp.y), 0, img_height - 1)
                cv2.circle(result_img, (x, y), 5, (255, 0, 0), thickness=-1)

            for pair in _POSE_PAIRS:
                pA = person.point_data[pair[0]].pixel_point
                pB = person.point_data[pair[1]].pixel_point
                if pA.x > 0 and pA.y > 0 and pB.x > 0 and pB.y > 0:
                    cv2.line(result_img, (int(pA.x), int(pA.y)), (int(pB.x), int(pB.y)), (0, 255, 255), 3)
                    segment = [pA.x, pA.y, pB.x, pB.y]
                    for b_item in bring_in_list:
                        bx, by, bw, bh = b_item._bounding_box.items
                        if YoloxVisualizer._intersects([bx, by, bw, bh], segment):
                            cv2.putText(result_img, 'OVERLAP', (bx, bh + by),
                                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 75, 0), thickness=3)

        for frame_obj in frame_object_list:
            action = frame_obj._item._action
            action_str = 'TAKE_OUT' if action == DetectedObjectActionEnum.TAKE_OUT else 'BRING_IN'
            x = frame_obj._item._bounding_box._x
            y = frame_obj._item._bounding_box._y
            cv2.putText(result_img, action_str, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness=3)

        tile_img = cv2.hconcat([yolox_img, result_img])
        cv2.namedWindow('yolox_object_detection', cv2.WINDOW_NORMAL)
        cv2.imshow('yolox_object_detection', tile_img)
        cv2.waitKey(1)

    @staticmethod
    def _intersects(rectangle, segment) -> bool:
        rx, ry, rw, rh = rectangle
        tl, tr, bl, br = (rx, ry), (rx + rw, ry), (rx, ry + rh), (rx + rw, ry + rh)
        s, e = (segment[0], segment[1]), (segment[2], segment[3])
        if YoloxVisualizer._point_in_rect(s, rectangle) or YoloxVisualizer._point_in_rect(e, rectangle):
            return True
        return (YoloxVisualizer._seg_intersect(s, e, tl, tr) or
                YoloxVisualizer._seg_intersect(s, e, tr, br) or
                YoloxVisualizer._seg_intersect(s, e, br, bl) or
                YoloxVisualizer._seg_intersect(s, e, bl, tl))

    @staticmethod
    def _point_in_rect(point, rect) -> bool:
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    @staticmethod
    def _seg_intersect(p1, p2, p3, p4) -> bool:
        a1, b1, c1 = YoloxVisualizer._line_eq(p1, p2)
        a2, b2, c2 = YoloxVisualizer._line_eq(p3, p4)
        try:
            denom = a1 * b2 - a2 * b1
            x = (b1 * c2 - b2 * c1) / denom
            y = (a2 * c1 - a1 * c2) / denom
        except ZeroDivisionError:
            return False
        return (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and
                min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and
                min(p3[0], p4[0]) <= x <= max(p3[0], p4[0]) and
                min(p3[1], p4[1]) <= y <= max(p3[1], p4[1]))

    @staticmethod
    def _line_eq(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return y2 - y1, x1 - x2, x2 * y1 - x1 * y2
