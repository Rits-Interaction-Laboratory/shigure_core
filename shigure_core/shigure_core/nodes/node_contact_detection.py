import datetime
from operator import itemgetter
from typing import List

import cv2
import message_filters
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum

from shigure_core.util import compressed_depth_util
from shigure_core_msgs.msg import DetectedObjectList, DetectedObject, BoundingBox, PoseKeyPointsList, PoseKeyPoints, \
    Point, PointData

from shigure_core.nodes.common_model.projection_matrix import ProjectionMatrix
from shigure_core.nodes.contact_detection.cube import Cube
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_detection.frame_object import FrameObject
from shigure_core.nodes.object_detection.logic import ObjectDetectionLogic


class ContactDetectionNode(ImagePreviewNode):
    def __init__(self):
        super().__init__("contact_detection_node")

        self.LEFT_HAND_INDEX = 4
        self.RIGHT_HAND_INDEX = 7

        object_subscriber = message_filters.Subscriber(self, DetectedObjectList, '/shigure/object_detection')
        people_subscriber = message_filters.Subscriber(self, PoseKeyPointsList, '/shigure/people_detection')
        depth_subscriber = message_filters.Subscriber(self, CompressedImage,
                                                      '/rs/aligned_depth_to_color/compressedDepth')
        depth_camera_info_subscriber = message_filters.Subscriber(self, CameraInfo,
                                                                  '/rs/aligned_depth_to_color/cameraInfo')
        color_subscriber = message_filters.Subscriber(self, CompressedImage, '/rs/color/compressed')

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [object_subscriber, people_subscriber, depth_subscriber, depth_camera_info_subscriber, color_subscriber],
            50000)
        self.time_synchronizer.registerCallback(self.callback)

        self.object_detection_logic = ObjectDetectionLogic()

        self.frame_object_list: List[FrameObject] = []
        self._color_img_buffer: List[np.ndarray] = []
        self._buffer_size = 90

        self.hand_collider_distance = 50  # 手の当たり判定の距離

    def callback(self, object_list: DetectedObjectList, people: PoseKeyPointsList, depth_img_src: CompressedImage,
                 camera_info: CameraInfo, color_img_src: CompressedImage):
        print('データを取得')
        depth_img: np.ndarray = compressed_depth_util.convert_compressed_depth_img_to_cv2(depth_img_src)
        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)

        # 焦点距離取得
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        k = camera_info.k.reshape((3, 3))

        # 透視変換のためのオブジェクトを作成
        projection_matrix = ProjectionMatrix(k)

        hand_cube_list = []
        person: PoseKeyPoints
        for person in people.pose_key_points_list:
            people_id = person.people_id
            left_hand: PointData = person.point_data[self.LEFT_HAND_INDEX]
            if int(left_hand.pixel_point.x) != 0 and int(left_hand.pixel_point.y) != 0:
                hand_cube_list.append((people_id, self.convert_point_to_cube(left_hand.projection_point),
                                       person.bounding_box, left_hand.pixel_point))
            right_hand: PointData = person.point_data[self.RIGHT_HAND_INDEX]
            if int(right_hand.pixel_point.x) != 0 and int(right_hand.pixel_point.y) != 0:
                hand_cube_list.append((people_id, self.convert_point_to_cube(right_hand.projection_point),
                                       person.bounding_box, right_hand.pixel_point))

        object_cube_list = []
        detected_object: DetectedObject
        img_height, img_width = color_img.shape[:2]
        for detected_object in object_list.object_list:
            bounding_box: BoundingBox = detected_object.bounding_box
            x, y = int(bounding_box.x), int(bounding_box.y)
            width, height = int(bounding_box.width), int(bounding_box.height)

            left_top_x, left_top_y, _ = projection_matrix.projection_to_view(x, y, depth_img[y, x])
            right_bottom_x_pixel, right_bottom_y_pixel = min(x + width, img_width - 1), min(y + height, img_height - 1)
            right_bottom_x, right_bottom_y, _ = projection_matrix \
                .projection_to_view(right_bottom_x_pixel,
                                    right_bottom_y_pixel,
                                    depth_img[right_bottom_y_pixel, right_bottom_x_pixel])

            z_min = depth_img[y:right_bottom_y_pixel, x:right_bottom_x_pixel].min()
            z_max = depth_img[y:right_bottom_y_pixel, x:right_bottom_x_pixel].max()

            object_cube_list.append((Cube(left_top_x, left_top_y, z_min,
                                          right_bottom_x - left_top_x,
                                          right_bottom_y - left_top_y,
                                          z_max - z_min), detected_object))

        linked_list = []
        object_cube: Cube
        hand_cube: Cube
        for object_item in object_cube_list:
            object_cube, _ = object_item
            for hand in hand_cube_list:
                _, hand_cube, _, _ = hand
                result, volume = object_cube.is_collided(hand_cube)
                if result:
                    linked_list.append((hand, object_item, volume))

        linked_list = sorted(linked_list, key=itemgetter(2), reverse=True)
        for hand, object_item, _ in linked_list:
            object_cube, detected_object = object_item
            people_id, hand_cube, people_bounding_box, pixel_point = hand
            if hand in hand_cube_list and object_item in object_cube_list:
                hand_cube_list.remove(hand)
                object_cube_list.remove(object_item)

                print(f'{people_id}が接触しました')

                if self.is_debug_mode:
                    action: DetectedObjectActionEnum = DetectedObjectActionEnum.value_of(detected_object.action)
                    color = (0, 255, 0) if action == DetectedObjectActionEnum.BRING_IN else (0, 0, 255)

                    # Objectのバウンディングボックス
                    x, y = int(detected_object.bounding_box.x), int(detected_object.bounding_box.y)
                    width, height = int(detected_object.bounding_box.width), int(detected_object.bounding_box.height)
                    cv2.rectangle(color_img, (x, y), (x + width, y + height), color, thickness=1)
                    cv2.putText(color_img, f'Action : {action.value}', (x, y), cv2.FONT_HERSHEY_PLAIN, 1,
                                color)

                    # Peopleのバウンディングボックス
                    width, height = color_img.shape[:2]
                    left = min(int(people_bounding_box.x), width - 1)
                    top = min(int(people_bounding_box.y), height - 1)
                    right = min(int(people_bounding_box.x + people_bounding_box.width), width - 1)
                    bottom = min(int(people_bounding_box.y + people_bounding_box.height), height - 1)
                    cv2.rectangle(color_img, (left, top), (right, bottom), (255, 0, 0), thickness=1)
                    cv2.putText(color_img, f'ID : {people_id}', (left, top), cv2.FONT_HERSHEY_PLAIN, 1,
                                (255, 0, 0))

                    # 対象の手首の位置
                    x = min(int(pixel_point.x), width - 1)
                    y = min(int(pixel_point.y), height - 1)
                    cv2.circle(color_img, (x, y), 5, (255, 0, 0), thickness=-1)

                    cv2.imshow(f'Result{x}{y}{width}{height}', color_img)

        if self.is_debug_mode:
            cv2.imshow('color', color_img)
            cv2.waitKey(1)
        else:
            print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')

    def convert_point_to_cube(self, point: Point) -> Cube:
        x = point.x - self.hand_collider_distance
        y = point.y - self.hand_collider_distance
        z = point.z - self.hand_collider_distance
        return Cube(x, y, z,
                    x + self.hand_collider_distance * 2,
                    y + self.hand_collider_distance * 2,
                    z + self.hand_collider_distance * 2)


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
