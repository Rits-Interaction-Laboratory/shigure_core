from typing import List

import cv2
import numpy as np

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.object_detection.bounding_box import BoundingBox
from shigure_core.nodes.object_detection.frame_object import FrameObject
from shigure_core.nodes.object_detection.frame_object_item import FrameObjectItem
from shigure_core.nodes.object_detection.top_color_image import TopColorImage
from shigure_core.nodes.object_detection.union_find_tree import UnionFindTree


class ObjectDetectionLogic:
    """物体検出ロジッククラス"""

    @staticmethod
    def execute(subtraction_analyzed_img: np.ndarray, people_mask: np.ndarray, top_color_image: TopColorImage,
                frame_object_list: List[FrameObject], min_size: int, max_size: int,
                allow_empty_frame_count: int) -> List[FrameObject]:
        """
        物体検出ロジック

        :param subtraction_analyzed_img:
        :param people_mask:
        :param top_color_image:
        :param frame_object_list:
        :param min_size:
        :param max_size:
        :param allow_empty_frame_count:
        :return: 検出したObjectリスト, 更新された既知マスク
        """

        # 人物領域を0としたい
        people_mask_inv = (255 - people_mask) // 255

        object_detection_img = subtraction_analyzed_img * people_mask_inv

        # ラベリング処理
        binary_img = np.where(object_detection_img != 0, 255, 0).astype(np.uint8)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

        prev_frame_object_dict = {frame_object.get_item(): frame_object for frame_object in frame_object_list}
        frame_object_item_list = []
        union_find_tree: UnionFindTree[FrameObjectItem] = UnionFindTree[FrameObjectItem]()

        for i, row in enumerate(stats):
            area = row[cv2.CC_STAT_AREA]
            x = row[cv2.CC_STAT_LEFT]
            y = row[cv2.CC_STAT_TOP]
            height = row[cv2.CC_STAT_HEIGHT]
            width = row[cv2.CC_STAT_WIDTH]

            if i != 0 and min_size <= area <= max_size:
                mask_img: np.ndarray = object_detection_img[y:y + height, x:x + width]

                # モード選択
                unique, freq = np.unique(mask_img, return_counts=True)
                not_zero_index = np.where(unique != 0)
                unique = unique[not_zero_index]
                freq = freq[not_zero_index]
                mode = unique[np.argmax(freq)]

                action = DetectedObjectActionEnum.BRING_IN if mode == 255 else DetectedObjectActionEnum.TAKE_OUT

                bounding_box = BoundingBox(x, y, width, height)
                item = FrameObjectItem(action, bounding_box, area, mask_img, top_color_image)
                frame_object_item_list.append(item)

                for prev_item, frame_object in prev_frame_object_dict.items():
                    is_matched, size = prev_item.is_match(item)
                    if is_matched:
                        if not union_find_tree.has_item(prev_item):
                            union_find_tree.add(prev_item)
                            frame_object_list.remove(frame_object)
                        if not union_find_tree.has_item(item):
                            union_find_tree.add(item)
                            frame_object_item_list.remove(item)
                        union_find_tree.unite(prev_item, item)

        result = []

        # リンクした範囲を1つにまとめる
        groups = union_find_tree.all_group_members().values()
        for items in groups:
            new_item: FrameObjectItem = items[0]
            mask_img: np.ndarray = ObjectDetectionLogic.update_mask_image(np.zeros(object_detection_img.shape[:2]),
                                                                          new_item)
            for item in items[1:]:
                x = min(new_item.get_bounding_box().get_x(), item.get_bounding_box().get_x())
                y = min(new_item.get_bounding_box().get_y(), item.get_bounding_box().get_y())
                width = max(new_item.get_bounding_box().get_x() + new_item.get_bounding_box().get_width(),
                            item.get_bounding_box().get_x() + item.get_bounding_box().get_width()) - x
                height = max(new_item.get_bounding_box().get_y() + new_item.get_bounding_box().get_height(),
                             item.get_bounding_box().get_y() + item.get_bounding_box().get_height()) - y
                mask_img = ObjectDetectionLogic.update_mask_image(mask_img, item)
                size = np.count_nonzero(mask_img[y:y + height, x:x + width])

                new_bounding_box = BoundingBox(x, y, width, height)
                top_color_image = new_item.get_top_color_image() if new_item.get_top_color_image().is_old(
                    item.get_top_color_image()) else item.get_top_color_image()
                new_item = FrameObjectItem(new_item.get_action(), new_bounding_box, size,
                                           mask_img[y:y + height, x:x + width], top_color_image)

            result.append(FrameObject(new_item, allow_empty_frame_count))

        # リンクしなかったframe_objectはからのフレームを挟む
        for frame_object in frame_object_list:
            frame_object.add_empty_frame()
            result.append(frame_object)

        # リンクしなかったframe_object_itemは新たなframe_objectとして登録
        for frame_object_item in frame_object_item_list:
            frame_object = FrameObject(frame_object_item, allow_empty_frame_count)
            result.append(frame_object)

        return result

    @staticmethod
    def update_mask_image(mask_img: np.ndarray, item: FrameObjectItem) -> np.ndarray:
        _, bounding_box, _, mask, _ = item.get_items()
        x, y, width, height = bounding_box.get_items()
        mask_img[y:y + height, x:x + width] = np.where(mask > 0, mask, mask_img[y:y + height, x:x + width])
        return mask_img
