import cv2
import message_filters
import numpy as np
import rclpy
from openpose_ros2_msgs.msg import PoseKeyPointsList
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import CompressedImage

from shigure.nodes.node_image_preview import ImagePreviewNode
from shigure.nodes.people_tracking.logic import PeopleTrackingLogic
from shigure.nodes.people_tracking.tracking_info import TrackingInfo
from shigure.util import compressed_depth_util


class PeopleTrackingNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('people_tracking_node')

        # ROS params
        focal_length_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                      description='Focal length [mm] of depth camera.')
        self.declare_parameter('focal_length', 1.0, focal_length_descriptor)
        self.focal_length: float = self.get_parameter("focal_length").get_parameter_value().double_value

        depth_subscriber = message_filters.Subscriber(self, CompressedImage,
                                                      '/rs/aligned_depth_to_color/compressedDepth')
        key_points_subscriber = message_filters.Subscriber(self, PoseKeyPointsList, '/openpose/pose_key_points')

        if not self.is_debug_mode:
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [depth_subscriber, key_points_subscriber], 10000)
            self.time_synchronizer.registerCallback(self.callback)
        else:
            color_subscriber = message_filters.Subscriber(self, CompressedImage, '/rs/color/compressed')
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [depth_subscriber, key_points_subscriber, color_subscriber], 10000)
            self.time_synchronizer.registerCallback(self.callback_debug)

        self.tracking_info = TrackingInfo()
        self.people_tracking_logic = PeopleTrackingLogic()

    def callback(self, depth_src: CompressedImage, key_points_list: PoseKeyPointsList):
        self.frame_count_up()

        depth_img = compressed_depth_util.convert_compressed_depth_img_to_cv2(depth_src)
        depth_img = depth_img.astype(np.float32)

        self.tracking_info = self.people_tracking_logic.execute(depth_img, key_points_list, self.tracking_info,
                                                                self.focal_length)

    def callback_debug(self, depth_src: CompressedImage, key_points_list: PoseKeyPointsList,
                       color_src: CompressedImage):
        self.callback(depth_src, key_points_list)
        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_src)
        height, width = color_img.shape[:2]

        for people_id, people in self.tracking_info.get_people_dict().items():
            people_x, people_y, _ = people

            x = int(people_x * self.focal_length) if people_x * self.focal_length < width else width - 1
            y = int(people_y * self.focal_length) if people_y * self.focal_length < height else height - 1

            cv2.circle(color_img, (x, y), 5, (0, 0, 255), thickness=-1)
            cv2.putText(color_img, f'ID : {people_id}', (x + 10, y + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        self.print_fps(color_img)
        cv2.imshow('Result', color_img)
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
