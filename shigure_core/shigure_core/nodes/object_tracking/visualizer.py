import random
import re
from typing import Dict

import cv2
import numpy as np

# 物体IDごとの表示色（起動時に一度だけ生成）。IDの連番を255で割った余りで引く。
_COLORS = [tuple(random.randint(128, 192) for _ in range(3)) for _ in range(255)]


class ObjectTrackingVisualizer:
    """object_tracking のデバッグ描画（追跡中の物体bboxとIDラベル）を担う."""

    @staticmethod
    def draw(color_img: np.ndarray, object_dict: Dict, frame_count: int, fps: float) -> None:
        """追跡中の物体をbbox＋IDラベルで描画し、object_tracking ウィンドウに表示する."""
        height, width = color_img.shape[:2]

        for object_id, item in object_dict.items():
            stay_object, _ = item
            bounding_box = stay_object.bounding_box
            left = min(int(bounding_box.x), width - 1)
            top = min(int(bounding_box.y), height - 1)
            right = min(int(bounding_box.x + bounding_box.width), width - 1)
            bottom = min(int(bounding_box.y + bounding_box.height), height - 1)

            object_id_num = int(re.sub('.*_', '', object_id))
            color = _COLORS[object_id_num % 255]
            cv2.rectangle(color_img, (left, top), (right, bottom), color, thickness=3)
            text_w, text_h = cv2.getTextSize(f'ID : {object_id_num}',
                                             cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
            cv2.rectangle(color_img, (left, top), (left + text_w, top - text_h), color, -1)
            cv2.putText(color_img, f'ID : {object_id_num}({stay_object.action})', (left, top),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2)

        ObjectTrackingVisualizer._draw_fps(color_img, frame_count, fps)

        cv2.namedWindow('object_tracking', cv2.WINDOW_NORMAL)
        cv2.imshow('object_tracking', color_img)
        cv2.waitKey(1)

    @staticmethod
    def _draw_fps(img: np.ndarray, frame_count: int, fps: float) -> None:
        """フレーム数とFPSを左上に印字する（ImagePreviewNode.print_fps と同じ体裁）."""
        cv2.putText(img, 'frame = ' + str(frame_count), (0, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv2.putText(img, 'FPS: {:.2f}'.format(fps), (0, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
