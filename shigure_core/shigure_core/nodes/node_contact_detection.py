import datetime
import re
from typing import List, Tuple
from copy import deepcopy

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import PoseKeyPointsList, TrackedObjectList, ContactedList, Contacted

from shigure_core.enum.contact_action_enum import ContactActionEnum
from shigure_core.enum.tracked_object_action_enum import TrackedObjectActionEnum
from shigure_core.nodes.contact_detection.id_manager import IdManager
from shigure_core.nodes.contact_detection.logic import ContactDetectionLogic
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_detection.frame_object import FrameObject


class ContactDetectionNode(ImagePreviewNode):
    action_list: list

    def __init__(self):
        super().__init__("contact_detection_node")

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.LEFT_HAND_INDEX = 4
        self.RIGHT_HAND_INDEX = 7

        # publisher, subscriber
        self._publisher = self.create_publisher(
            ContactedList, 
            '/shigure/contacted', 
            10
        )
        object_subscriber = message_filters.Subscriber(
            self, 
            TrackedObjectList, 
            '/shigure/object_tracking', 
            qos_profile=shigure_qos
        )
        people_subscriber = message_filters.Subscriber(
            self, 
            PoseKeyPointsList, 
            '/shigure/people_detection', 
            qos_profile=shigure_qos
        )
        color_subscriber = message_filters.Subscriber(
            self, 
            CompressedImage, 
            '/rs/color/compressed',
            qos_profile=shigure_qos
        )
        depth_camera_info_subscriber = message_filters.Subscriber(
            self, 
            CameraInfo, 
            '/rs/aligned_depth_to_color/cameraInfo', 
            qos_profile=shigure_qos
        )

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [object_subscriber, people_subscriber, color_subscriber, depth_camera_info_subscriber], 1000)
        self.time_synchronizer.registerCallback(self.callback)

        self.contact_detection_logic = ContactDetectionLogic()

        self.frame_object_list: List[FrameObject] = []
        self._color_img_buffer: List[np.ndarray] = []
        self._buffer_size = 90

        self.hand_collider_distance = 300  # 手の当たり判定の距離

        self.is_not_touch = False
        self.action_index = 0
        self._id_manager = IdManager()

    def callback(self, object_list: TrackedObjectList, people: PoseKeyPointsList, color_img_src: CompressedImage, camera_info: CameraInfo):
        self.frame_count_up()

        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)

        result_list, self.is_not_touch = self.contact_detection_logic.execute(object_list, people)

        publish_msg = ContactedList()
        publish_msg.header.stamp = people.header.stamp
        publish_msg.header.frame_id = camera_info.header.frame_id
        for hand, object_item in result_list:
            person, _, _ = hand
            tracked_object, _ = object_item

            action = ContactActionEnum.from_tracked_object_action_enum(
                TrackedObjectActionEnum.value_of(tracked_object.action)
            )

            if action != ContactActionEnum.TOUCH:
                contacted = Contacted()
                contacted.event_id = self._id_manager.new_event_id()
                contacted.people_id = person.people_id
                contacted.object_id = tracked_object.object_id
                contacted.action = action.value
                contacted.people_bounding_box = person.bounding_box
                contacted.object_bounding_box = tracked_object.bounding_box
                contacted.object_cube = tracked_object.collider
                publish_msg.contacted_list.append(contacted)
            else:
                # TOUCHの場合はevent idを追加せずにpublish
                contacted = Contacted()
                contacted.people_id = person.people_id
                contacted.object_id = tracked_object.object_id
                contacted.action = action.value
                contacted.people_bounding_box = person.bounding_box
                contacted.object_bounding_box = tracked_object.bounding_box
                contacted.object_cube = tracked_object.collider
                publish_msg.contacted_list.append(contacted)

            print(f'PeopleId: {person.people_id}, ObjectId: {tracked_object.object_id}, Action: {action.value}')

        self._publisher.publish(publish_msg)

        if self.is_debug_mode:
            height, width = color_img.shape[:2]

            event_frame = deepcopy(color_img)

            if not hasattr(self, 'action_list'):
                self.action_list = []
                self.event_frame_list = []
                black_img = np.zeros_like(color_img)
                for i in range(4):
                    self.action_list.append(cv2.resize(black_img.copy(), (width // 2, height // 2)))
                    self.event_frame_list.append(cv2.resize(black_img.copy(), (width // 2, height // 2)))

            # すべての人物領域を書く
            for person in people.pose_key_points_list:
                bounding_box = person.bounding_box
                left = np.clip(int(bounding_box.x), 0, width - 1)
                top = np.clip(int(bounding_box.y), 0, height - 1)
                right = np.clip(int(bounding_box.x + bounding_box.width), 0, width - 1)
                bottom = np.clip(int(bounding_box.y + bounding_box.height), 0, height - 1)

                # 対象の手首の位置
                pixel_point = person.point_data[self.LEFT_HAND_INDEX].pixel_point
                x = np.clip(int(pixel_point.x), 0, width - 1)
                y = np.clip(int(pixel_point.y), 0, height - 1)
                cv2.circle(color_img, (x, y), 5, (255, 0, 0), thickness=-1)
                cv2.circle(event_frame, (x, y), 5, (255, 0, 0), thickness=-1)
                pixel_point = person.point_data[self.RIGHT_HAND_INDEX].pixel_point
                x = np.clip(int(pixel_point.x), 0, width - 1)
                y = np.clip(int(pixel_point.y), 0, height - 1)
                cv2.circle(color_img, (x, y), 5, (255, 0, 0), thickness=-1)
                cv2.circle(event_frame, (x, y), 5, (255, 0, 0), thickness=-1)

                cv2.rectangle(color_img, (left, top), (right, bottom), (255, 0, 0), thickness=3)
                cv2.rectangle(event_frame, (left, top), (right, bottom), (255, 0, 0), thickness=3)
                text_w, text_h = cv2.getTextSize(f'ID : {re.sub(".*_", "", person.people_id)}',
                                                 cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
                cv2.rectangle(color_img, (left, top), (left + text_w, top - text_h), (255, 0, 0), -1)
                cv2.rectangle(event_frame, (left, top), (left + text_w, top - text_h), (255, 0, 0), -1)
                cv2.putText(color_img, f'ID : {re.sub(".*_", "", person.people_id)}', (left, top),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2)
                cv2.putText(event_frame, f'ID : {re.sub(".*_", "", person.people_id)}', (left, top),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2)
                

            # すべての物体領域を書く
            for tracked_object in object_list.tracked_object_list:
                bounding_box = tracked_object.bounding_box
                left = np.clip(int(bounding_box.x), 0, width - 1)
                top = np.clip(int(bounding_box.y), 0, height - 1)
                right = np.clip(int(bounding_box.x + bounding_box.width), 0, width - 1)
                bottom = np.clip(int(bounding_box.y + bounding_box.height), 0, height - 1)

                cv2.rectangle(color_img, (left, top), (right, bottom), (0, 128, 255), thickness=3)
                cv2.putText(color_img, f'ID : {re.sub(".*_", "", tracked_object.object_id)}', (left, top),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)

                if tracked_object.action == "bring_in" or tracked_object.action == "take_out":
                    cv2.rectangle(event_frame, (left, top), (right, bottom), (0, 128, 255), thickness=3)
                    cv2.putText(event_frame, f'ID : {re.sub(".*_", "", tracked_object.object_id)}', (left, top),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                    
                    # 右側ウインドウへの検知物体の拡大表示
                    # 物体画像は, 上下左右にexpansion_paramだけ大きく切り取る
                    object_image = []
                    expansion_param = 20
                    left_expansion = left - expansion_param
                    top_expansion = top - expansion_param
                    right_expansion = right + expansion_param
                    bottom_expansion = bottom + expansion_param
                    if (0 <= left_expansion) and (right_expansion <= width) and (0 <= top_expansion) and (bottom_expansion <= height):
                        object_image = event_frame[top_expansion : bottom_expansion, left_expansion : right_expansion]
                    else:
                        object_image = event_frame[top : bottom, left : right]
                    event_frame = self.overlay_image(overlapping_img=object_image, underlying_img=event_frame, shift=(0, 0), resize_scale=(3, 3), is_frame_line=True)

            for hand, object_item in result_list:
                person, _, index = hand
                tracked_object, _ = object_item

                action = ContactActionEnum.from_tracked_object_action_enum(
                    TrackedObjectActionEnum.value_of(tracked_object.action)
                )

                self.is_not_touch = action != ContactActionEnum.TOUCH

                color = self.get_color_from_action(action)

                # Actionの表示
                pixel_point = person.point_data[index].pixel_point
                x = np.clip(int(pixel_point.x), 0, width - 1)
                y = np.clip(int(pixel_point.y), 0, height - 1)
                cv2.putText(color_img, f'Action : {action.value}', (x - 40, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, color, thickness=2)
                cv2.rectangle(event_frame, (0, height - 100), (width//4 + 40, height), (0,0,0), -1)
                cv2.putText(event_frame, f'{action.value}', (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, thickness=5)

            if len(result_list) > 0:
                cv2.putText(color_img, 'Detected', (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

            color_img = self.print_fps(color_img)
            # cv2.imshow('color', color_img)

            if self.is_not_touch:
                print("is_not_touch", self.is_not_touch)
                # cv2.imshow(f'{people.header.stamp.sec}.{people.header.stamp.nanosec}', color_img)
                self.action_list[self.action_index] = cv2.resize(event_frame, (width // 2, height // 2))
                self.event_frame_list = deepcopy(self.action_list)
                self.event_frame_list[self.action_index] = self.draw_outer_frame_line(self.event_frame_list[self.action_index], band_width=10, color= [255, 0, 255])
                self.action_index = (self.action_index + 1) % 4
            tile_img = cv2.hconcat([
                color_img,
                cv2.vconcat([
                cv2.hconcat([self.event_frame_list[0], self.event_frame_list[1]]),
                cv2.hconcat([self.event_frame_list[2], self.event_frame_list[3]])
                ])
            ])
            cv2.namedWindow('contact_detection', cv2.WINDOW_NORMAL)
            cv2.imshow('contact_detection', tile_img)

            cv2.waitKey(1)
        else:
            print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')

    @staticmethod
    def get_color_from_action(action: ContactActionEnum) -> Tuple[int, int, int]:
        if action == ContactActionEnum.BRING_IN:
            return 128, 255, 255
        if action == ContactActionEnum.TOUCH:
            return 255, 128, 128
        return 255, 128, 255


def main(args=None):
    rclpy.init(args=args)

    contact_detection_node = ContactDetectionNode()

    try:
        rclpy.spin(contact_detection_node)

    except KeyboardInterrupt:
        pass

    finally:
        print()
        # 終了処理
        contact_detection_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()