import random
import re

import cv2
import message_filters
import numpy as np
import rclpy
from openpose_ros2_msgs.msg import PoseKeyPointsList as OpenPosePoseKeyPointsList
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import PoseKeyPointsList as ShigurePoseKeyPointsList, PoseKeyPoints

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.people_tracking.logic import PeopleTrackingLogic
from shigure_core.nodes.people_tracking.tracking_info import TrackingInfo
from shigure_core.util import compressed_depth_util, pose_key_points_util


class PeopleTrackingNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('people_tracking_node')

        self._publisher = self.create_publisher(ShigurePoseKeyPointsList, '/shigure/people_detection', 10)

        depth_subscriber = message_filters.Subscriber(self, CompressedImage,
                                                      '/rs/aligned_depth_to_color/compressedDepth')
        depth_camera_info_subscriber = message_filters.Subscriber(self, CameraInfo,
                                                                  '/rs/aligned_depth_to_color/cameraInfo')
        key_points_subscriber = message_filters.Subscriber(self, OpenPosePoseKeyPointsList, '/openpose/pose_key_points')

        if not self.is_debug_mode:
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [depth_subscriber, key_points_subscriber, depth_camera_info_subscriber], 30000)
            self.time_synchronizer.registerCallback(self.callback)
        else:
            color_subscriber = message_filters.Subscriber(self, CompressedImage, '/rs/color/compressed')
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [depth_subscriber, key_points_subscriber, depth_camera_info_subscriber, color_subscriber], 400000)
            self.time_synchronizer.registerCallback(self.callback_debug)

        self.tracking_info = TrackingInfo()
        self.people_tracking_logic = PeopleTrackingLogic()

        self._colors = []
        for i in range(255):
            self._colors.append(tuple([random.randint(128, 192) for _ in range(3)]))

    def callback(self, depth_src: CompressedImage, key_points_list: OpenPosePoseKeyPointsList, camera_info: CameraInfo):
        self.frame_count_up()

        depth_img = compressed_depth_util.convert_compressed_depth_img_to_cv2(depth_src)
        depth_img = depth_img.astype(np.float32)

        # 焦点距離取得
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        k = camera_info.k.reshape((3, 3))

        self.tracking_info = self.people_tracking_logic.execute(depth_img, key_points_list, self.tracking_info, k)

        # publish
        publish_msg = ShigurePoseKeyPointsList()
        publish_msg.header = key_points_list.header
        publish_msg.header.frame_id = camera_info.header.frame_id
        for people_id, people in self.tracking_info.get_people_dict().items():
            _, _, _, openpose_pose_key_points = people
            shigure_pose_key_points = pose_key_points_util.convert_openpose_to_shigure(openpose_pose_key_points,
                                                                                       depth_img, people_id, k)
            publish_msg.pose_key_points_list.append(shigure_pose_key_points)
        self._publisher.publish(publish_msg)

        return publish_msg

    def callback_debug(self, depth_src: CompressedImage, key_points_list: OpenPosePoseKeyPointsList,
                       camera_info: CameraInfo, color_src: CompressedImage):
        published_msg: ShigurePoseKeyPointsList = self.callback(depth_src, key_points_list, camera_info)
        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_src)
        height, width = color_img.shape[:2]

        pose_key_points: PoseKeyPoints
        for pose_key_points in published_msg.pose_key_points_list:
            people_id = pose_key_points.people_id
            _, _, _, current_pose_key_points = self.tracking_info.get_people_dict()[people_id]

            neck_point = current_pose_key_points.pose_key_points[1]

            x = int(neck_point.x) if neck_point.x < width else width - 1
            y = int(neck_point.y) if neck_point.y < height else height - 1

            bounding_box = pose_key_points.bounding_box
            left = int(bounding_box.x) if bounding_box.x < width else width - 1
            top = int(bounding_box.y) if bounding_box.y < height else height - 1
            right = int(
                bounding_box.x + bounding_box.width) if bounding_box.x + bounding_box.width < width else width - 1
            bottom = int(
                bounding_box.y + bounding_box.height) if bounding_box.y + bounding_box.height < height else height - 1

            people_id_num = int(re.sub(".*_", "", people_id))
            color = self._colors[people_id_num % 255]
            cv2.circle(color_img, (x, y), 5, color, thickness=-1)
            cv2.rectangle(color_img, (left, top), (right, bottom), color, thickness=3)
            text_w, text_h = cv2.getTextSize(f'ID : {people_id_num}',
                                             cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
            cv2.rectangle(color_img, (left, top), (left + text_w, top - text_h), color, -1)
            cv2.putText(color_img, f'ID : {people_id_num}', (left, top),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2)

        self.print_fps(color_img)
        cv2.namedWindow('people_tracking', cv2.WINDOW_NORMAL)
        cv2.imshow('people_tracking', color_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    people_tracking_node = PeopleTrackingNode()

    try:
        rclpy.spin(people_tracking_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        people_tracking_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
