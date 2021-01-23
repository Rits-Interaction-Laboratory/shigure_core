import datetime
from itertools import chain
from typing import List

import cv2
import message_filters
import numpy as np
import rclpy
from people_detection_ros2_msg.msg import People
from sensor_msgs.msg import CompressedImage
from shigure_core_msgs.msg import DetectedObjectList, DetectedObject, BoundingBox

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_detection.frame_object import FrameObject
from shigure_core.nodes.object_detection.judge_params import JudgeParams
from shigure_core.nodes.object_detection.logic import ObjectDetectionLogic


class ObjectDetectionNode(ImagePreviewNode):

    def __init__(self):
        super().__init__("object_detection_node")

        self.detection_publisher = self.create_publisher(DetectedObjectList, '/shigure/object_detection', 10)

        # people_subscriber = message_filters.Subscriber(self, People, '/people_detection')
        subtraction_analysis_subscriber = message_filters.Subscriber(self, CompressedImage,
                                                                     '/shigure/subtraction_analysis')
        color_subscriber = message_filters.Subscriber(self, CompressedImage, '/rs/color/compressed')

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [subtraction_analysis_subscriber, color_subscriber], 30000)
        self.time_synchronizer.registerCallback(self.callback)

        self.object_detection_logic = ObjectDetectionLogic()

        self.frame_object_list: List[FrameObject] = []
        self._color_img_buffer: List[np.ndarray] = []
        self._buffer_size = 90

        self._judge_params = JudgeParams(200, 5000, 5)

    def callback(self, subtraction_analysis_src: CompressedImage, color_img_src: CompressedImage):
        self.frame_count_up()

        subtraction_analysis_img = self.bridge.compressed_imgmsg_to_cv2(subtraction_analysis_src)
        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
        people_mask_img = np.zeros(color_img.shape[:2], dtype=np.uint8)  # self.bridge.compressed_imgmsg_to_cv2(people.mask)

        if len(self._color_img_buffer) > 30:
            self._color_img_buffer = self._color_img_buffer[1:]
        self._color_img_buffer.append(color_img)

        timestamp = Timestamp(color_img_src.header.stamp.sec, color_img_src.header.stamp.nanosec)
        frame_object_dict = self.object_detection_logic.execute(subtraction_analysis_img, people_mask_img,
                                                                self._color_img_buffer[0], color_img, timestamp,
                                                                self.frame_object_list, self._judge_params)

        self.frame_object_list = list(chain.from_iterable(frame_object_dict.values()))

        result_img = cv2.cvtColor(subtraction_analysis_img, cv2.COLOR_GRAY2BGR)

        for key, frame_object_list in sorted(frame_object_dict.items(), key=lambda item: item[0]):
            # 検知開始時間が同じのフレームオブジェクトが検知済でなければ終わり
            if not all([frame_object.is_finished() for frame_object in frame_object_list]):
                break

            detected_object_list = DetectedObjectList()
            sec, nano_sec = frame_object_list[0].item.detected_at.timestamp
            detected_object_list.header.stamp.sec = sec
            detected_object_list.header.stamp.nanosec = nano_sec
            for frame_object in frame_object_list:
                action, bounding_box_src, size, mask_img, item_color_img, _ = frame_object.item.items
                x, y, width, height = bounding_box_src.items

                detected_object = DetectedObject()
                detected_object.action = action.value
                detected_object.mask = self.bridge.cv2_to_compressed_imgmsg(mask_img, 'png')

                bounding_box = BoundingBox()
                bounding_box.x = float(x)
                bounding_box.y = float(y)
                bounding_box.width = float(width)
                bounding_box.height = float(height)
                detected_object.bounding_box = bounding_box

                detected_object_list.object_list.append(detected_object)

                self.frame_object_list.remove(frame_object)

                if self.is_debug_mode:
                    color = (0, 255, 0) if action == DetectedObjectActionEnum.BRING_IN else (0, 0, 255)
                    print('オブジェクトが検出されました(',
                          f'action: {action.value}, x: {x}, y: {y}, width: {width}, height: {height}, size: {size})')
                    cv2.rectangle(result_img, (x, y), (x + width, y + height), color, thickness=1)
                    cv2.imshow(f'Result{x}{y}{width}{height}', item_color_img[y:y + height, x:x + width, :])

            if len(detected_object_list.object_list) > 0:
                self.detection_publisher.publish(detected_object_list)

        if self.is_debug_mode:
            img = self.print_fps(result_img)
            tile_img = cv2.hconcat([cv2.cvtColor(subtraction_analysis_img, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(people_mask_img, cv2.COLOR_GRAY2BGR), img])
            cv2.imshow("Result", tile_img)
            cv2.waitKey(1)
        else:
            print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')


def main(args=None):
    rclpy.init(args=args)

    object_detection_node = ObjectDetectionNode()

    try:
        rclpy.spin(object_detection_node)

    except KeyboardInterrupt:
        pass

    finally:
        print()
        # 終了処理
        object_detection_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
