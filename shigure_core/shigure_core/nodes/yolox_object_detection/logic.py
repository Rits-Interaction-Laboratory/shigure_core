from typing import List, Tuple
import cv2
import numpy as np
from shigure_core_msgs.msg import PoseKeyPointsList
from bboxes_ex_msgs.msg import Segments

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.yolox_object_detection.color_image_frame import ColorImageFrame
from shigure_core.nodes.yolox_object_detection.color_image_frames import ColorImageFrames
from shigure_core.nodes.yolox_object_detection.frame_object import FrameObject
from shigure_core.nodes.yolox_object_detection.frame_object_item import FrameObjectItem
from shigure_core.nodes.yolox_object_detection.detection import Detection
from shigure_core.nodes.yolox_object_detection.tracked_object import TrackedObject, TrackedState


class YoloxObjectDetectionLogic:
    """物体検出ロジッククラス"""

    def __init__(self):
        self._frame_object_list: List[FrameObject] = []
        self._tracked_objects: List[TrackedObject] = []
        self._take_out_people_id: str = ""
        self._take_out_obj_class_id: str = ""
        self._buffer_size: int = 25
        self._color_img_buffer: List[np.ndarray] = []
        self._color_img_frames = ColorImageFrames()

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self._buffer_size = value
        self._color_img_buffer = self._color_img_buffer[-value:]

    @property
    def frame_object_list(self) -> List[FrameObject]:
        return self._frame_object_list

    def consume_frame_object_list(self) -> List[FrameObject]:
        items = list(self._frame_object_list)
        self._frame_object_list.clear()
        return items

    @property
    def bring_in_list(self) -> List[TrackedObject]:
        return [t for t in self._tracked_objects if t.state == TrackedState.CONFIRMED]

    @property
    def wait_item_list(self) -> List[TrackedObject]:
        return [t for t in self._tracked_objects if t.state == TrackedState.WAITING]

    def execute(self, segments: Segments, sec: int, nanosec: int, people: PoseKeyPointsList,
                color_img: np.ndarray, fhist_size: int, confirm_threshold: float, dismiss_threshold: float,
                take_out_threshold: float, match_dist_threshold: int, probability_threshold: float,
                stuffed_toy_threshold: float) -> None:
        started_at = Timestamp(sec, nanosec)

        if len(self._color_img_buffer) > self._buffer_size:
            self._color_img_buffer = self._color_img_buffer[1:]
            self._color_img_frames.get(-self._buffer_size).new_image = color_img
        self._color_img_buffer.append(color_img)
        frame = ColorImageFrame(started_at, self._color_img_buffer[0], color_img)
        self._color_img_frames.add(frame)

        detections = self._parse_detections(segments, color_img, started_at, probability_threshold,
                                           stuffed_toy_threshold)

        matched: set = set()
        frame_object_item_list = self._update_confirmed(detections, matched, people, fhist_size, take_out_threshold,
                                                        match_dist_threshold)
        frame_object_item_list += self._update_waiting(detections, matched, people, fhist_size, confirm_threshold,
                                                       dismiss_threshold, match_dist_threshold)
        self._register_new(detections, matched)

        self._frame_object_list = [FrameObject(item) for item in frame_object_item_list]

    def _update_confirmed(self, detections: List[Detection], matched: set, people: PoseKeyPointsList,
                          fhist_size: int, take_out_threshold: float, match_dist_threshold: int) -> List[FrameObjectItem]:
        """CONFIRMED 物体を現フレームと照合し TAKE_OUT を判定する"""
        frame_object_items: List[FrameObjectItem] = []
        to_remove: List[TrackedObject] = []

        for obj in [o for o in self._tracked_objects if o.state == TrackedState.CONFIRMED]:
            found = False
            for det in detections:
                if obj.matches(det, match_dist_threshold):
                    obj.record(True, fhist_size)
                    matched.add(id(det))
                    found = True
                    print("bring in %s" % obj.class_id)
                    break

            if not found:
                occluded = YoloxObjectDetectionLogic._is_occluded_by_people(obj.bbox, people)
                if len(detections) == 0 or not occluded:
                    obj.record(False, fhist_size)
                    print("not found %s" % obj.class_id)
                else:
                    print("hide judge %s" % obj.class_id)

            print("len(%s.fhist) : %s" % (obj.class_id, len(obj.fhist)))
            if obj.is_history_full(fhist_size):
                if obj.should_take_out(fhist_size, take_out_threshold):
                    print("take out found rate:", obj.found_rate)
                    self._take_out_people_id = YoloxObjectDetectionLogic._find_take_out_person_id(
                        obj.bbox, people
                    )
                    self._take_out_obj_class_id = obj.class_id
                    to_remove.append(obj)
                    frame_object_items.append(FrameObjectItem(
                        DetectedObjectActionEnum.TAKE_OUT,
                        obj.bbox,
                        obj.bbox.width * obj.bbox.height,
                        obj.mask,
                        obj.found_at,
                        obj.class_id,
                    ))
                obj.trim_history(fhist_size)
                print("%s.fhist: %s" % (obj.class_id, obj.fhist))

        if to_remove:
            print("del: %s" % [o.class_id for o in to_remove])
            for obj in to_remove:
                self._tracked_objects.remove(obj)

        return frame_object_items

    def _update_waiting(self, detections: List[Detection], matched: set, people: PoseKeyPointsList,
                        fhist_size: int, confirm_threshold: float, dismiss_threshold: float,
                        match_dist_threshold: int) -> List[FrameObjectItem]:
        """WAITING 物体を現フレームと照合し BRING_IN / OBJ_MOVE を判定する"""
        frame_object_items: List[FrameObjectItem] = []
        to_remove: List[TrackedObject] = []

        for obj in [o for o in self._tracked_objects if o.state == TrackedState.WAITING]:
            for det in detections:
                if id(det) in matched:
                    continue
                if obj.matches(det, match_dist_threshold):
                    obj.record(True, fhist_size)
                    matched.add(id(det))
                    break
            else:
                obj.record(False, fhist_size)

            if obj.is_history_full(fhist_size):
                samepeople_judge = any(
                    p.people_id == self._take_out_people_id
                    for p in people.pose_key_points_list
                )
                print("wait_item.fist:", obj.fhist)
                if obj.should_dismiss(fhist_size, dismiss_threshold):
                    to_remove.append(obj)
                elif obj.should_confirm(fhist_size, confirm_threshold):
                    print("bring in found rate:", obj.found_rate)
                    if obj.class_id == self._take_out_obj_class_id and samepeople_judge:
                        action = DetectedObjectActionEnum.OBJ_MOVE
                    else:
                        action = DetectedObjectActionEnum.BRING_IN
                    frame_object_items.append(FrameObjectItem(
                        action,
                        obj.bbox,
                        obj.bbox.width * obj.bbox.height,
                        obj.mask,
                        obj.found_at,
                        obj.class_id,
                    ))
                    obj.state = TrackedState.CONFIRMED
                obj.trim_history(fhist_size)

        print("del_idx_list:", len(to_remove))
        for obj in to_remove:
            self._tracked_objects.remove(obj)

        return frame_object_items

    def _register_new(self, detections: List[Detection], matched: set) -> None:
        """どの追跡物体にも一致しなかった検出を WAITING として新規登録する"""
        for det in detections:
            if id(det) not in matched:
                self._tracked_objects.append(TrackedObject(det))

    @staticmethod
    def _is_occluded_by_people(bbox: BoundingBox, people: PoseKeyPointsList) -> bool:
        """物体の bbox が骨格セグメントと重なっているかどうかを判定する"""
        POSE_PAIRS: List[List[int]] = [
            [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[8,12],
            [9,10],[10,11],[11,22],[11,24],[12,13],[13,14],[14,19],[14,21],
            [15,0],[15,17],[16,0],[16,18],[19,20],[22,11],[22,23],
        ]
        rectangle = [bbox._x, bbox._y, bbox._width, bbox._height]
        for person in people.pose_key_points_list:
            for pair in POSE_PAIRS:
                partA, partB = pair
                if (person.point_data[partA].pixel_point.x > 0 and
                        person.point_data[partA].pixel_point.y > 0 and
                        person.point_data[partB].pixel_point.x > 0 and
                        person.point_data[partB].pixel_point.y > 0):
                    segment = [
                        person.point_data[partA].pixel_point.x,
                        person.point_data[partA].pixel_point.y,
                        person.point_data[partB].pixel_point.x,
                        person.point_data[partB].pixel_point.y,
                    ]
                    if YoloxObjectDetectionLogic._intersects_bbox_and_segment(rectangle, segment):
                        return True
        return False

    @staticmethod
    def _find_take_out_person_id(bbox: BoundingBox, people: PoseKeyPointsList) -> str:
        """TAKE_OUT 時に物体を隠している人物の ID を返す。なければ空文字列を返す"""
        POSE_PAIRS: List[List[int]] = [
            [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[8,12],
            [9,10],[10,11],[11,22],[11,24],[12,13],[13,14],[14,19],[14,21],
            [15,0],[15,17],[16,0],[16,18],[19,20],[22,11],[22,23],
        ]
        rectangle = [bbox._x, bbox._y, bbox._width, bbox._height]
        for person in people.pose_key_points_list:
            for pair in POSE_PAIRS:
                partA, partB = pair
                if (person.point_data[partA].pixel_point.x > 0 and
                        person.point_data[partA].pixel_point.y > 0 and
                        person.point_data[partB].pixel_point.x > 0 and
                        person.point_data[partB].pixel_point.y > 0):
                    segment = [
                        person.point_data[partA].pixel_point.x,
                        person.point_data[partA].pixel_point.y,
                        person.point_data[partB].pixel_point.x,
                        person.point_data[partB].pixel_point.y,
                    ]
                    if YoloxObjectDetectionLogic._intersects_bbox_and_segment(rectangle, segment):
                        return person.people_id
        return ""
    
    @staticmethod
    def _parse_detections(segments: Segments, color_img: np.ndarray, started_at: Timestamp,
                         probability_threshold: float, stuffed_toy_threshold: float) -> List[Detection]:
        """YOLO11 ノードからの Segments を Detection のリストに変換する.

        Segment.x_masks には物体領域の y 座標の頂点が, Segment.y_masks には
        x 座標の頂点が格納されており, インデックスでペアになっている.
        """
        detections = []
        img_height, img_width = color_img.shape[:2]
        for segment in segments.segments:
            x = segment.xmin
            y = segment.ymin
            width = segment.xmax - x
            height = segment.ymax - y
            class_id = segment.class_id
            probability = segment.probability
            if not YoloxObjectDetectionLogic.is_unknown_object(class_id, probability, probability_threshold,
                                                               stuffed_toy_threshold):
                continue

            # セグメンテーション結果からポリゴンマスクを生成する
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            if len(segment.x_masks) >= 3 and len(segment.x_masks) == len(segment.y_masks):
                # x_masks = y座標, y_masks = x座標 (頂点はインデックスでペア)
                polygon = np.array(
                    [[px, py] for px, py in zip(segment.y_masks, segment.x_masks)],
                    dtype=np.int32,
                )
                cv2.fillPoly(full_mask, [polygon], 255)
            else:
                # マスク頂点が無い場合は bbox 矩形で代替する
                full_mask[y:y + height, x:x + width] = 255

            mask_img = full_mask[y:y + height, x:x + width]
            detections.append(Detection(BoundingBox(x, y, width, height), class_id, mask_img, started_at))
        return detections

    @staticmethod
    def is_unknown_object(class_id: str, probability: float, probability_threshold: float,
                          stuffed_toy_threshold: float) -> bool:
        """
        物体と思われるものの規定の物体でないものかどうか調べる関数
        :param class_id: 物体のクラス名
        :param probability: 物体かどうかの確からしさ（max 1）
        :param probability_threshold: 物体と判定するしきい値（max 1）
        :param stuffed_toy_threshold: 物体と判定するしきい値（max 1, ぬいぐるみ専用の閾値）
        :return: 物体と思われるものの規定の物体でないものかどうか
        """
        # YOLO11 (COCO) のクラス名で記述した, 持ち込み/持ち出し判定の対象外とする物体
        DEFAULT_OBJECTS = [
            'person', 'dog', 'cat', 'chair', 'laptop', 'tv', 'microwave', 'refrigerator',
            'potted plant', 'cup', 'keyboard', 'couch', 'mouse', 'sink', 'dining table',
            'skateboard', 'book', 'banana', 'backpack',
        ]
        # YOLO11 ではぬいぐるみが 'teddy bear' として検出される (旧 YOLOX の 'stuffed toy')
        if class_id == 'teddy bear':
            is_object = probability > stuffed_toy_threshold
        else:
            is_object = probability > probability_threshold
        return is_object and class_id not in DEFAULT_OBJECTS

    @staticmethod
    def _intersects_bbox_and_segment(rectangle, segment):
	# 矩形の頂点を取得
        rect_x, rect_y, rect_width, rect_height = rectangle
        rect_top_left = (rect_x, rect_y)
        rect_top_right = (rect_x + rect_width, rect_y)
        rect_bottom_left = (rect_x, rect_y + rect_height)
        rect_bottom_right = (rect_x + rect_width, rect_y + rect_height)

        # 線分の端点を取得
        seg_x1, seg_y1, seg_x2, seg_y2 = segment
        seg_start = (seg_x1, seg_y1)
        seg_end = (seg_x2, seg_y2)

        # 線分が矩形の内部にあるかチェック
        if YoloxObjectDetectionLogic.point_in_rectangle(seg_start, rectangle) or YoloxObjectDetectionLogic.point_in_rectangle(seg_end, rectangle):
            return True

        # 線分と矩形の各辺との交差判定
        if YoloxObjectDetectionLogic.line_segment_intersect(seg_start, seg_end, rect_top_left, rect_top_right):
            return True
        if YoloxObjectDetectionLogic.line_segment_intersect(seg_start, seg_end, rect_top_right, rect_bottom_right):
            return True
        if YoloxObjectDetectionLogic.line_segment_intersect(seg_start, seg_end, rect_bottom_right, rect_bottom_left):
            return True
        if YoloxObjectDetectionLogic.line_segment_intersect(seg_start, seg_end, rect_bottom_left, rect_top_left):
            return True

        return False


    @staticmethod
    def point_in_rectangle(point, rectangle):
        x, y = point
        rect_x, rect_y, rect_width, rect_height = rectangle
        return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height

    @staticmethod
    def line_segment_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
        # 2つの線分の方程式の係数を計算
        a1, b1, c1 = YoloxObjectDetectionLogic.line_equation(seg1_start, seg1_end)
        a2, b2, c2 = YoloxObjectDetectionLogic.line_equation(seg2_start, seg2_end)
        

        # 交差点の座標を計算
        try:
            x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
            y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
        except ZeroDivisionError:
            x = 0
            y = 0

        # 交差点が線分の内部にあるかチェック
        if (min(seg1_start[0], seg1_end[0]) <= x <= max(seg1_start[0], seg1_end[0]) and
                min(seg1_start[1], seg1_end[1]) <= y <= max(seg1_start[1], seg1_end[1]) and
                min(seg2_start[0], seg2_end[0]) <= x <= max(seg2_start[0], seg2_end[0]) and
                min(seg2_start[1], seg2_end[1]) <= y <= max(seg2_start[1], seg2_end[1])):
            return True

        return False
    @staticmethod
    def line_equation(start_point, end_point):
        x1, y1 = start_point
        x2, y2 = end_point
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c      

            
    
